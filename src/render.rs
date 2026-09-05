use crate::camera::Camera;
use crate::layout;
use crate::project::{CameraViewLaunch, DEPTH_KEY_BITS, project_splats};
use crate::raster::{map_isects, rasterize_kernel};
use crate::scan::{ScanScratch, exclusive_scan_gather};
use crate::sort::{RadixScratch, bits_for, radix_argsort_with};
use crate::tensor::GpuTensor;
use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

// Safety cap: 2 * max_tiles_per_dim * max_splats
const INTERSECTS_UPPER_BOUND: usize = 2 * 512 * 65535;

// Web builds compile the rasterizer without the shared-memory early exit (see
// rasterize_kernel); native keeps it.
const EARLY_EXIT: bool = !cfg!(target_arch = "wasm32");

/// Host-side splat payload produced by the PLY/SOG parsers, uploaded once by
/// `Splats::new`. Field-major layout: see the `PLANE_*` planes in `layout`.
#[derive(Debug)]
pub struct CpuSplats {
    pub attributes: Vec<f32>,
    pub sh_coeffs: Vec<f32>,
}

impl CpuSplats {
    /// Upload to the GPU, keeping the parsers client-free.
    pub fn upload(self, client: &ComputeClient<WgpuRuntime>) -> Splats {
        Splats::new(self.attributes, self.sh_coeffs, client)
    }
}

/// Per-frame GPU buffers, reused across frames so steady-state rendering
/// allocates nothing new. Rebuild when `matches` returns false.
pub struct RenderScratch {
    depth_order: GpuTensor,
    depth_keys: GpuTensor,
    projected: GpuTensor,
    counters: GpuTensor,
    tile_counts: GpuTensor,
    tile_bbox: GpuTensor,
    tile_ids: GpuTensor,
    gaussian_ids: GpuTensor,
    bitmap: GpuTensor,
    // Ping-pong scratch for the two sorts. The tile sort must not share the
    // depth sort's: an odd pass count returns buffers aliasing the scratch.
    sort_depth: RadixScratch,
    sort_tile: RadixScratch,
    scan: ScanScratch,
    img_size: glam::UVec2,
    // Last frame's raw intersection emission, stashed for the next frame's
    // capacity check — growth only ever happens between frames.
    last_isects_raw: u32,
}

impl RenderScratch {
    pub fn new(client: &ComputeClient<WgpuRuntime>, total: usize, img_size: glam::UVec2) -> Self {
        let tile_bounds = img_size.map(|c| c.div_ceil(layout::TILE_WIDTH));
        let num_tiles = (tile_bounds.x * tile_bounds.y) as usize;
        let isect_capacity = (num_tiles.saturating_mul(total).min(1 << 22)).next_power_of_two();
        // 256-byte rows: wgpu buffer→texture copies align rows to
        // COPY_BYTES_PER_ROW_ALIGNMENT; texture.rs copies with this stride.
        let row_stride = (img_size.x * 4).next_multiple_of(256) / 4;
        Self {
            depth_order: GpuTensor::empty(client, [total]),
            depth_keys: GpuTensor::empty(client, [total]),
            projected: GpuTensor::empty(client, [total, layout::PROJ_FLOATS]),
            counters: GpuTensor::empty(client, [2]),
            tile_counts: GpuTensor::empty(client, [total]),
            tile_bbox: GpuTensor::empty(client, [total, 2]),
            tile_ids: GpuTensor::empty(client, [isect_capacity]),
            gaussian_ids: GpuTensor::empty(client, [isect_capacity]),
            bitmap: GpuTensor::empty(client, [img_size.y as usize, row_stride as usize]),
            sort_depth: RadixScratch::new(client, total),
            sort_tile: RadixScratch::new(client, isect_capacity),
            scan: ScanScratch::new(client, total),
            img_size,
            last_isects_raw: 0,
        }
    }

    fn matches(&self, total: usize, img_size: glam::UVec2) -> bool {
        self.depth_order.shape[0] == total && self.img_size == img_size
    }

    /// Grow the isect buffers when last frame's raw emission exceeded their
    /// capacity. Between frames, so a frame always runs on one buffer set.
    fn grow_isects(&mut self, client: &ComputeClient<WgpuRuntime>, raw: u32) {
        let new_cap = (raw as usize)
            .next_power_of_two()
            .min(INTERSECTS_UPPER_BOUND);
        if new_cap > self.tile_ids.shape[0] {
            eprintln!(
                "intersection capacity {} reached ({raw} emitted); growing to {new_cap}",
                self.tile_ids.shape[0]
            );
            self.tile_ids = GpuTensor::empty(client, [new_cap]);
            self.gaussian_ids = GpuTensor::empty(client, [new_cap]);
            self.sort_tile = RadixScratch::new(client, new_cap);
        }
    }
}

pub struct Splats {
    pub attributes: GpuTensor,
    pub sh_coeffs: GpuTensor,
    pub bounds: (glam::Vec3, glam::Vec3),
}

impl Splats {
    pub fn new(
        attributes: Vec<f32>,
        sh_coeffs: Vec<f32>,
        client: &ComputeClient<WgpuRuntime>,
    ) -> Self {
        let n = attributes.len() / layout::ATTR_PLANES;
        assert!(n > 0, "Splats::new: zero splats");
        let n_coeffs = sh_coeffs.len() / n;

        let mut min = glam::Vec3::splat(f32::MAX);
        let mut max = glam::Vec3::splat(f32::MIN);
        for i in 0..n {
            let p = glam::vec3(
                attributes[layout::PLANE_X * n + i],
                attributes[layout::PLANE_Y * n + i],
                attributes[layout::PLANE_Z * n + i],
            );
            min = min.min(p);
            max = max.max(p);
        }

        Self {
            attributes: GpuTensor::from(client, [n, layout::ATTR_PLANES], attributes),
            sh_coeffs: GpuTensor::from(client, [n, n_coeffs / 3, 3], sh_coeffs),
            bounds: (min, max),
        }
    }

    /// Render one frame into `scratch`'s buffers; the returned bitmap aliases
    /// `scratch.bitmap` and is valid until the next `render_with` on it.
    ///
    /// A `scratch` that doesn't match the splat count or image size is rebuilt
    /// in place.
    pub async fn render_with(
        &self,
        scratch: &mut RenderScratch,
        camera: &Camera,
        img_size: glam::UVec2,
    ) -> GpuTensor {
        let client = &self.attributes.client;
        let total = self.attributes.shape[0];
        if !scratch.matches(total, img_size) {
            *scratch = RenderScratch::new(client, total, img_size);
        }
        // Grow from last frame's raw emission, before anything is in flight.
        scratch.grow_isects(client, scratch.last_isects_raw);
        let tile_bounds = img_size.map(|c| c.div_ceil(layout::TILE_WIDTH));
        let num_tiles = (tile_bounds.x * tile_bounds.y) as usize;
        let max_isects = scratch.tile_ids.shape[0] as u32;
        let cube_dim = CubeDim::new_1d(layout::TILE_SIZE);

        let sh_per_ch = self.sh_coeffs.shape[1] as u32;
        // World-to-camera rotation rows + translation, passed as kernel scalars.
        let w2c = camera.w2c();
        let rot = w2c.matrix3.transpose();
        let view = CameraViewLaunch::new(
            rot.x_axis.into(),
            rot.y_axis.into(),
            rot.z_axis.into(),
            w2c.translation.into(),
            camera.focal(img_size).into(),
            camera.position.into(),
            img_size.as_vec2().into(),
            tile_bounds.as_vec2().into(),
        );

        // Stream-ordered before the project launch; saves a kernel dispatch.
        scratch.counters.write([0u32, 0]);

        project_splats::launch::<WgpuRuntime>(
            client,
            calculate_cube_count_elemwise(client, total, cube_dim),
            cube_dim,
            view,
            self.attributes.as_buffer_arg(),
            self.sh_coeffs.as_buffer_arg(),
            sh_per_ch,
            scratch.depth_order.as_buffer_arg(),
            scratch.depth_keys.as_buffer_arg(),
            scratch.projected.as_buffer_arg(),
            scratch.counters.as_buffer_arg(),
            scratch.tile_counts.as_buffer_arg(),
            scratch.tile_bbox.as_buffer_arg(),
        );

        let [num_isects_raw, num_visible] = scratch.counters.read_pair().await;
        let num_isects = num_isects_raw.min(max_isects);
        let (_sorted_keys, depth_order) = radix_argsort_with(
            &scratch.depth_keys,
            &scratch.depth_order,
            num_visible,
            DEPTH_KEY_BITS,
            &scratch.sort_depth,
        );

        // Emission offsets in depth order; dead depth_keys doubles as the
        // offsets scratch.
        exclusive_scan_gather(
            &depth_order,
            &scratch.tile_counts,
            &scratch.depth_keys,
            num_visible,
            &scratch.scan,
        );

        map_isects::launch::<WgpuRuntime>(
            client,
            calculate_cube_count_elemwise(client, num_visible as usize, cube_dim),
            cube_dim,
            depth_order.as_buffer_arg(),
            scratch.tile_bbox.as_buffer_arg(),
            scratch.depth_keys.as_buffer_arg(),
            tile_bounds.x,
            max_isects,
            scratch.tile_ids.as_buffer_arg(),
            scratch.gaussian_ids.as_buffer_arg(),
            num_visible,
        );

        // Isects are emitted in depth order, so this single stable sort by
        // tile id yields tile-major, depth-minor order — no separate gid sort
        // or composite key needed.
        let tile_bits = bits_for(num_tiles as u32);
        let (tile_ids, gaussian_ids) = radix_argsort_with(
            &scratch.tile_ids,
            &scratch.gaussian_ids,
            num_isects,
            tile_bits,
            &scratch.sort_tile,
        );

        let row_stride = scratch.bitmap.shape[1] as u32;
        rasterize_kernel::launch::<WgpuRuntime>(
            client,
            CubeCount::new_2d(tile_bounds.x, tile_bounds.y),
            CubeDim::new_2d(layout::TILE_WIDTH, layout::TILE_WIDTH),
            img_size.x,
            img_size.y,
            row_stride,
            num_isects,
            tile_ids.as_buffer_arg(),
            gaussian_ids.as_buffer_arg(),
            scratch.projected.as_buffer_arg(),
            scratch.bitmap.as_buffer_arg(),
            EARLY_EXIT,
        );
        scratch.last_isects_raw = num_isects_raw;
        scratch.bitmap.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Golden from current rasterizer; regenerate only with documented behavior change.
    const GOLDEN_CENTER: [i32; 3] = [127, 127, 127];

    /// Field-major attributes for `n` opaque splats at the origin XY, `z` per
    /// splat. [x | y | z | qw | qx | qy | qz | sx | sy | sz | opacity]
    fn stacked_attributes(n: usize, z: impl Fn(usize) -> f32) -> Vec<f32> {
        let mut a = vec![0f32; n * layout::ATTR_PLANES];
        for i in 0..n {
            a[2 * n + i] = z(i);
            a[3 * n + i] = 1.0; // qw
            a[7 * n + i] = -2.0; // sx
            a[8 * n + i] = -2.0; // sy
            a[9 * n + i] = -2.0; // sz
            a[10 * n + i] = 8.0; // opacity
        }
        a
    }

    /// Unpack a packed RGBA8 pixel (r | g<<8 | b<<16 | a<<24).
    fn rgba(px: u32) -> [i32; 4] {
        let [r, g, b, a] = px.to_le_bytes();
        [r as i32, g as i32, b as i32, a as i32]
    }

    #[test]
    fn test_render_stacked_opaque_splats() {
        let (_gpu, client) = crate::tensor::test_client();
        // Five opaque splats stacked in depth at the same XY, nearest at z = 1.0.
        let n = 5usize;
        let attributes = stacked_attributes(n, |i| 1.0 + i as f32 * 0.5);
        let splats = Splats::new(attributes, vec![0.0; n * 3], &client);
        let camera = crate::camera::Camera::default();
        let mut scratch = RenderScratch::new(&client, 5, glam::uvec2(32, 32));
        let bitmap =
            pollster::block_on(splats.render_with(&mut scratch, &camera, glam::uvec2(32, 32)));
        let px: Vec<u32> = bitmap.read_vec();
        let stride = bitmap.shape[1] as usize;
        let [r, g, b, a] = rgba(px[16 * stride + 16]);
        let corner = px[0];
        // quantize_u8 truncates; f32 residual transmittance leaves 255-1 LSB.
        assert!(
            a >= 254,
            "opaque stack must saturate alpha (truncating quantizer: {a})"
        );
        assert_eq!(
            corner & 0xFF00_0000,
            0,
            "uncovered pixel must be transparent"
        );
        assert!((r - GOLDEN_CENTER[0]).abs() <= 1);
        assert!((g - GOLDEN_CENTER[1]).abs() <= 1);
        assert!((b - GOLDEN_CENTER[2]).abs() <= 1);
    }

    #[test]
    fn test_render_rotated_translated_camera() {
        let (_gpu, client) = crate::tensor::test_client();
        // Five opaque splats stacked at the world origin, seen by a camera
        // rotated 90° around Y and translated to (-2, 0, 0): it looks along
        // +X, so the stack must land dead center. A transposed rotation row
        // or a flipped/dropped translation sends it behind the camera
        // instead — the identity-camera test can't distinguish those.
        let n = 5usize;
        let attributes = stacked_attributes(n, |_| 0.0);
        let splats = Splats::new(attributes, vec![0.0; n * 3], &client);
        let camera = crate::camera::Camera {
            position: glam::Vec3::new(-2.0, 0.0, 0.0),
            rotation: glam::Quat::from_rotation_y(core::f32::consts::FRAC_PI_2),
            ..Camera::default()
        };
        let mut scratch = RenderScratch::new(&client, n, glam::uvec2(32, 32));
        let bitmap =
            pollster::block_on(splats.render_with(&mut scratch, &camera, glam::uvec2(32, 32)));
        let px: Vec<u32> = bitmap.read_vec();
        let stride = bitmap.shape[1] as usize;
        let [r, g, b, a] = rgba(px[16 * stride + 16]);
        assert!(
            a >= 254,
            "yawed camera must still see the origin splat (a={a})"
        );
        assert_eq!(px[0] & 0xFF00_0000, 0, "corner stays transparent");
        for (got, want) in [r, g, b].iter().zip(GOLDEN_CENTER) {
            assert!((got - want).abs() <= 1, "center color {r},{g},{b}");
        }
    }

    #[test]
    fn test_render_reuses_scratch() {
        let (_gpu, client) = crate::tensor::test_client();
        let attributes: Vec<f32> = (0..50 * 11)
            .map(|i| (i as f32 * 0.13).sin() * 0.5)
            .collect();
        let sh = vec![0.1f32; 50 * 3];
        let splats = Splats::new(attributes, sh, &client);
        let camera = crate::camera::Camera {
            position: glam::Vec3::new(0.0, -1.0, 0.0),
            ..Camera::default()
        };
        let mut scratch = RenderScratch::new(&client, 50, glam::uvec2(64, 64));
        let b = pollster::block_on(splats.render_with(&mut scratch, &camera, glam::uvec2(64, 64)))
            .read_vec::<u32>();
        let c = pollster::block_on(splats.render_with(&mut scratch, &camera, glam::uvec2(64, 64)))
            .read_vec::<u32>();
        assert_eq!(b, c);
    }

    #[test]
    fn test_scratch_stops_allocating_after_first_frame() {
        let (_gpu, client) = crate::tensor::test_client();
        let attributes: Vec<f32> = (0..100 * 11)
            .map(|i| (i as f32 * 0.07).sin() * 0.4)
            .collect();
        let splats = Splats::new(attributes, vec![0.0; 100 * 3], &client);
        let camera = crate::camera::Camera {
            position: glam::Vec3::new(0.0, -1.2, 0.0),
            ..Camera::default()
        };
        let mut scratch = RenderScratch::new(&client, 100, glam::uvec2(64, 64));
        pollster::block_on(splats.render_with(&mut scratch, &camera, glam::uvec2(64, 64)));
        // The runtime caches one client per device, so other tests' deferred
        // frees and pool reclaim land inside the measurement window and can
        // spike a single reading. Fail only on sustained growth — systematic
        // per-frame allocation trips every attempt.
        let mut prev = client.memory_usage().unwrap().bytes_in_use;
        let mut steady = false;
        for _ in 0..5 {
            pollster::block_on(splats.render_with(&mut scratch, &camera, glam::uvec2(64, 64)));
            let cur = client.memory_usage().unwrap().bytes_in_use;
            if cur <= prev + 65_536 {
                steady = true;
                break;
            }
            prev = cur;
        }
        assert!(steady, "steady-state frames keep allocating memory");
    }

    /// Test-only probe of the rasterizer's per-tile range lookup, sharing
    /// the production `lower_bound`.
    #[cube(launch)]
    fn probe_tile_range(ids: &[u32], num_isects: u32, tile: u32, range: &mut [u32]) {
        if UNIT_POS == 0 {
            range[0] = crate::raster::lower_bound(ids, num_isects, tile);
            range[1] = crate::raster::lower_bound(ids, num_isects, tile + 1u32);
        }
    }

    #[test]
    fn test_tile_ranges_binary_search_matches_cpu() {
        let (_gpu, client) = crate::tensor::test_client();
        // Sorted tile ids: tiles 0, 1, 2, 5 have intersections; 3, 4, 6, 7
        // are empty. Every tile's rasterizer range must match a CPU
        // lower-bound pair over the same ids.
        let ids: Vec<u32> = vec![0, 0, 1, 2, 2, 2, 5, 5];
        let ids_t = GpuTensor::from(&client, [ids.len()], &ids[..]);

        let lower_bound_host = |target: u32| ids.partition_point(|&k| k < target) as u32;

        for tile in 0..8u32 {
            let range = GpuTensor::empty(&client, [2]);
            probe_tile_range::launch::<WgpuRuntime>(
                &client,
                CubeCount::new_single(),
                CubeDim::new_1d(layout::TILE_SIZE),
                ids_t.as_buffer_arg(),
                ids.len() as u32,
                tile,
                range.as_buffer_arg(),
            );
            let r: Vec<u32> = range.read_vec();
            let expect = [lower_bound_host(tile), lower_bound_host(tile + 1)];
            assert_eq!(r, expect, "tile {tile}: rasterizer range");
        }
    }
}
