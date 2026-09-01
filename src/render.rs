use crate::camera::Camera;
use crate::helpers;
use crate::sort::{bits_for, radix_argsort};
use crate::tensor::{GpuTensor, cube_count_1d};
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

// Safety cap: 2 * max_tiles_per_dim * max_splats
const INTERSECTS_UPPER_BOUND: usize = 2 * 512 * 65535;

// z ∈ (0.1, 1e4) keeps the top 4 mantissa-bits constant; dropping them saves a radix pass.
const DEPTH_KEY_BITS: u32 = 28;

/// Host-side splat payload produced by the PLY/SOG parsers, uploaded once by `Splats::new`.
#[derive(Debug, Clone, Default)]
pub struct CpuSplats {
    pub attributes: Vec<f32>,
    pub sh_coeffs: Vec<f32>,
}

/// Per-frame GPU buffers, reused across frames so steady-state rendering
/// allocates nothing new. Rebuild when `matches` returns false.
#[derive(Debug)]
pub struct RenderScratch {
    depth_order: GpuTensor,
    depth_keys: GpuTensor,
    projected: GpuTensor,
    counters: GpuTensor,
    tile_ids: GpuTensor,
    gaussian_ids: GpuTensor,
    tile_ranges: GpuTensor,
    bitmap: GpuTensor,
    total: usize,
    img_size: glam::UVec2,
    num_tiles: usize,
    isect_capacity: usize,
}

impl RenderScratch {
    pub fn new(client: &ComputeClient<WgpuRuntime>, total: usize, img_size: glam::UVec2) -> Self {
        let tile_bounds = img_size.map(|c| c.div_ceil(helpers::TILE_WIDTH));
        let num_tiles = (tile_bounds.x * tile_bounds.y) as usize;
        let isect_capacity = num_tiles
            .saturating_mul(total)
            .min(1 << 22)
            .min(INTERSECTS_UPPER_BOUND);
        let row_stride = (img_size.x * 4).next_multiple_of(256) / 4;
        Self {
            depth_order: GpuTensor::empty(client, [total]),
            depth_keys: GpuTensor::empty(client, [total]),
            projected: GpuTensor::empty(client, [total, 9]),
            counters: GpuTensor::empty(client, [2]),
            tile_ids: GpuTensor::empty(client, [isect_capacity]),
            gaussian_ids: GpuTensor::empty(client, [isect_capacity]),
            tile_ranges: GpuTensor::empty(client, [num_tiles * 2]),
            bitmap: GpuTensor::empty(client, [img_size.y as usize, row_stride as usize]),
            total,
            img_size,
            num_tiles,
            isect_capacity,
        }
    }

    pub fn matches(&self, total: usize, img_size: glam::UVec2) -> bool {
        self.total == total && self.img_size == img_size
    }
}

#[derive(Debug, Clone)]
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
        let n = attributes.len() / 11;
        assert!(n > 0, "Splats::new: zero splats");
        let n_coeffs = sh_coeffs.len() / n;

        let mut min = glam::Vec3::splat(f32::MAX);
        let mut max = glam::Vec3::splat(f32::MIN);
        for attr in attributes.chunks_exact(11) {
            let p = glam::Vec3::from_slice(&attr[..3]);
            min = min.min(p);
            max = max.max(p);
        }

        Self {
            attributes: GpuTensor::from(client, [n, 11], attributes),
            sh_coeffs: GpuTensor::from(client, [n, n_coeffs / 3, 3], sh_coeffs),
            bounds: (min, max),
        }
    }

    /// Render pipeline (3D Gaussian Splatting):
    ///
    /// 1. **Project** 3D Gaussians into image space — compute 2D mean, conic, color (SH),
    ///    opacity, and emit (`tile_id`, `gaussian_id`) pairs for every tile each Gaussian covers.
    /// 2. **Tile & replicate** — divide image into 16×16 tiles; replicate Gaussians that
    ///    span multiple tiles, assigning each copy a tile ID.
    /// 3. **Sort** — depth-sort Gaussians, then stable-sort intersection pairs by tile ID
    ///    (equivalent to the paper's single composite-key sort).
    /// 4. **Rasterize** — render sorted Gaussians per tile in parallel; each pixel
    ///    alpha-blends front-to-back through its tile's Gaussian list.
    ///
    /// One-shot render builds a fresh scratch at the initial isect capacity; views
    /// exceeding it render truncated (warned) for that call — use `render_with`
    /// for the self-healing path.
    pub async fn render(&self, camera: &Camera, img_size: glam::UVec2) -> GpuTensor {
        let mut scratch =
            RenderScratch::new(&self.attributes.client, self.attributes.shape[0], img_size);
        self.render_with(&mut scratch, camera, img_size).await
    }

    /// Render one frame into `scratch`'s buffers; the returned bitmap aliases
    /// `scratch.bitmap` and is valid until the next `render_with` on it.
    ///
    /// `scratch` must satisfy `matches(self.attributes.shape[0], img_size)`
    /// (debug_asserted); a mismatched scratch in release yields garbage-clamped
    /// output, not UB.
    pub async fn render_with(
        &self,
        scratch: &mut RenderScratch,
        camera: &Camera,
        img_size: glam::UVec2,
    ) -> GpuTensor {
        let client = &self.attributes.client;
        let total = self.attributes.shape[0];
        debug_assert!(scratch.matches(total, img_size));
        let tile_bounds = img_size.map(|c| c.div_ceil(helpers::TILE_WIDTH));
        let num_tiles = scratch.num_tiles;
        let max_isects = scratch.isect_capacity;

        let focal = camera.focal(img_size);
        let camera_pos = camera.position;
        let viewmat = glam::Mat4::from(camera.w2c()).transpose().to_cols_array();
        let sh_per_ch = self.sh_coeffs.shape[1] as u32;

        // Frame-local clones of the growable buffers: if they are reallocated
        // after the counters readback below, this frame still finishes on the
        // old buffers and the larger ones take effect next frame.
        let tile_ids = scratch.tile_ids.clone();
        let gaussian_ids = scratch.gaussian_ids.clone();
        // viewmat stays a per-frame 64-byte upload — measured negligible,
        // not worth scalar-arg plumbing.
        let viewmat = GpuTensor::from(client, [16], viewmat);

        zero_buffers::launch::<WgpuRuntime>(
            client,
            cube_count_1d(client, (2 + 2 * num_tiles) as u32, helpers::TILE_SIZE),
            CubeDim::new_1d(helpers::TILE_SIZE),
            scratch.counters.as_buffer_arg(),
            scratch.tile_ranges.as_buffer_arg(),
        );

        crate::project::project_splats::launch::<WgpuRuntime>(
            client,
            cube_count_1d(client, total as u32, helpers::TILE_SIZE),
            CubeDim::new_1d(helpers::TILE_SIZE),
            viewmat.as_buffer_arg(),
            helpers::Vec2FLaunch::new(focal.x, focal.y),
            helpers::Vec3FLaunch::new(camera_pos.x, camera_pos.y, camera_pos.z),
            self.attributes.as_buffer_arg(),
            self.sh_coeffs.as_buffer_arg(),
            sh_per_ch,
            helpers::Vec2FLaunch::new(tile_bounds.x as f32, tile_bounds.y as f32),
            helpers::Vec2FLaunch::new(img_size.x as f32, img_size.y as f32),
            scratch.depth_order.as_buffer_arg(),
            scratch.depth_keys.as_buffer_arg(),
            scratch.projected.as_buffer_arg(),
            scratch.counters.as_buffer_arg(),
            max_isects as u32,
            tile_ids.as_buffer_arg(),
            gaussian_ids.as_buffer_arg(),
        );

        let [num_isects_raw, num_visible] = scratch.counters.read_pair().await;
        let num_isects = num_isects_raw.min(max_isects as u32);
        if num_isects_raw as usize >= max_isects {
            // Saturated: this frame is truncated to the clamped count; grow so
            // the next frame fits. Reallocating now is safe — everything below
            // runs on the frame-local clones of the old buffers. Once the
            // ceiling is reached the computed capacity equals the current one
            // and no realloc (or warn) happens.
            let new_cap = (num_isects_raw as usize)
                .next_power_of_two()
                .min(INTERSECTS_UPPER_BOUND);
            if new_cap > scratch.isect_capacity {
                log::warn!(
                    "intersection capacity {max_isects} reached ({num_isects_raw} emitted); growing to {}",
                    new_cap
                );
                scratch.isect_capacity = new_cap;
                scratch.tile_ids = GpuTensor::empty(client, [new_cap]);
                scratch.gaussian_ids = GpuTensor::empty(client, [new_cap]);
            }
        }
        let (inv_perm, depth_order) = radix_argsort(
            scratch.depth_keys.clone(),
            scratch.depth_order.clone(),
            num_visible,
            DEPTH_KEY_BITS,
        );

        invert_permutation::launch::<WgpuRuntime>(
            client,
            cube_count_1d(client, num_visible, helpers::TILE_SIZE),
            CubeDim::new_1d(helpers::TILE_SIZE),
            depth_order.as_buffer_arg(),
            inv_perm.as_buffer_arg(),
            num_visible,
        );

        remap_global_ids::launch::<WgpuRuntime>(
            client,
            cube_count_1d(client, num_isects, helpers::TILE_SIZE),
            CubeDim::new_1d(helpers::TILE_SIZE),
            gaussian_ids.as_buffer_arg(),
            inv_perm.as_buffer_arg(),
            num_isects,
        );

        // Two radix sorts equivalent to the paper's single composite key sort — the sort itself is not stable within equal keys, which is harmless for alpha blending.
        let gid_bits = bits_for(num_visible);
        let (gaussian_ids, tile_ids) = radix_argsort(gaussian_ids, tile_ids, num_isects, gid_bits);
        let tile_bits = bits_for(num_tiles as u32);
        let (tile_ids, gaussian_ids) = radix_argsort(tile_ids, gaussian_ids, num_isects, tile_bits);

        build_tile_ranges::launch::<WgpuRuntime>(
            client,
            cube_count_1d(client, num_isects, helpers::TILE_SIZE),
            CubeDim::new_1d(helpers::TILE_SIZE),
            tile_ids.as_buffer_arg(),
            scratch.tile_ranges.as_buffer_arg(),
            num_isects,
        );

        let row_stride = scratch.bitmap.shape[1] as u32;
        rasterize_kernel::launch::<WgpuRuntime>(
            client,
            CubeCount::new_2d(tile_bounds.x, tile_bounds.y),
            CubeDim::new_2d(helpers::TILE_WIDTH, helpers::TILE_WIDTH),
            img_size.x,
            img_size.y,
            row_stride,
            gaussian_ids.as_buffer_arg(),
            scratch.tile_ranges.as_buffer_arg(),
            scratch.projected.as_buffer_arg(),
            depth_order.as_buffer_arg(),
            scratch.bitmap.as_buffer_arg(),
        );
        scratch.bitmap.clone()
    }
}

#[cube(launch)]
fn invert_permutation(perm: &mut [u32], inv: &mut [u32], n: u32) {
    if ABSOLUTE_POS_X < n {
        inv[perm[ABSOLUTE_POS_X as usize] as usize] = ABSOLUTE_POS_X;
    }
}

#[cube(launch)]
fn remap_global_ids(gids: &mut [u32], inv_perm: &[u32], n: u32) {
    if ABSOLUTE_POS_X < n {
        gids[ABSOLUTE_POS_X as usize] = inv_perm[gids[ABSOLUTE_POS_X as usize] as usize];
    }
}

#[cube(launch)]
fn zero_buffers(counters: &mut [u32], tile_ranges: &mut [u32]) {
    let idx = ABSOLUTE_POS_X as usize;
    if idx < 2 {
        counters[idx] = 0;
    } else if idx < 2 + tile_ranges.len() {
        tile_ranges[idx - 2] = 0;
    }
}

#[cube(launch)]
fn build_tile_ranges(ids: &[u32], ranges: &mut [u32], num_isects: u32) {
    if ABSOLUTE_POS_X < num_isects {
        let cur = ids[ABSOLUTE_POS_X as usize];
        if ABSOLUTE_POS_X == 0 || ids[(ABSOLUTE_POS_X - 1) as usize] != cur {
            ranges[cur as usize * 2] = ABSOLUTE_POS_X;
        }
        if ABSOLUTE_POS_X == num_isects - 1 || ids[(ABSOLUTE_POS_X + 1) as usize] != cur {
            ranges[cur as usize * 2 + 1] = ABSOLUTE_POS_X + 1;
        }
    }
}

#[cube]
fn gaussian_power(conic: helpers::Vec3F, dx: f32, dy: f32) -> f32 {
    0.5f32 * (conic.x * dx * dx + conic.z * dy * dy) + conic.y * dx * dy
}

#[cube(launch)]
fn rasterize_kernel(
    img_size_x: u32,
    img_size_y: u32,
    row_stride: u32,
    gaussian_ids_by_tile: &[u32],
    tile_ranges: &[u32],
    projected: &[f32],
    depth_order: &[u32],
    bitmap: &mut [u32],
) {
    let px = ABSOLUTE_POS_X;
    let py = ABSOLUTE_POS_Y;
    let tile_id = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    if px < img_size_x && py < img_size_y {
        let pixel_x = px as f32 + 0.5f32;
        let pixel_y = py as f32 + 0.5f32;
        let range_start = tile_ranges[tile_id as usize * 2];
        let range_end = tile_ranges[tile_id as usize * 2 + 1];

        let mut transmittance = 1.0f32;
        let mut pix_r = 0.0;
        let mut pix_g = 0.0;
        let mut pix_b = 0.0;

        for i in range_start..range_end {
            let depth_idx = gaussian_ids_by_tile[i as usize];
            // Projected layout: [mean2d_x, mean2d_y, conic_x, conic_y, conic_z, r, g, b, opacity]
            let base = (depth_order[depth_idx as usize] * 9u32) as usize;

            let mean_x = projected[base];
            let mean_y = projected[base + 1];
            let conic = helpers::Vec3F {
                x: projected[base + 2],
                y: projected[base + 3],
                z: projected[base + 4],
            };
            let color_r = projected[base + 5];
            let color_g = projected[base + 6];
            let color_b = projected[base + 7];
            let color_a = projected[base + 8];

            let power = gaussian_power(conic, mean_x - pixel_x, mean_y - pixel_y);
            let alpha = (color_a * (-power).exp()).min(0.999);

            if alpha >= 1.0f32 / u8::MAX as f32 {
                let vis = alpha * transmittance;
                pix_r += color_r * vis;
                pix_g += color_g * vis;
                pix_b += color_b * vis;
                transmittance *= 1.0f32 - alpha;
                // Remaining weight < 1 LSB of the final 8-bit channels.
                if transmittance < 1.0f32 / 255.0f32 {
                    break;
                }
            }
        }

        let r = helpers::quantize_u8(pix_r);
        let g = helpers::quantize_u8(pix_g);
        let b = helpers::quantize_u8(pix_b);
        let a = helpers::quantize_u8(1.0f32 - transmittance);
        bitmap[(px + py * row_stride) as usize] = r | (g << 8u32) | (b << 16u32) | (a << 24u32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::GpuTensor;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    const SENTINEL: u32 = 0xDEAD_BEEF;

    // Golden from current rasterizer; regenerate only with documented behavior change.
    const GOLDEN_CENTER: [i32; 3] = [127, 127, 127];

    #[test]
    fn test_render_stacked_opaque_splats() {
        let client = WgpuRuntime::client(&WgpuDevice::default());
        // Five opaque splats stacked in depth at the same XY, nearest at z = 1.0.
        let mut attributes = Vec::new();
        for i in 0..5u32 {
            let z = 1.0 + i as f32 * 0.5;
            attributes.extend_from_slice(&[0.0, 0.0, z, 1.0, 0.0, 0.0, 0.0, -2.0, -2.0, -2.0, 8.0]);
        }
        let sh = vec![0.0; 5 * 3];
        let splats = Splats::new(attributes, sh, &client);
        let camera = crate::camera::Camera {
            fov: glam::Vec2::splat(0.8),
            position: glam::Vec3::ZERO,
            rotation: glam::Quat::IDENTITY,
        };
        let bitmap = pollster::block_on(splats.render(&camera, glam::uvec2(32, 32)));
        let px: Vec<u32> = bitmap.read_vec();
        let stride = bitmap.shape[1] as usize;
        let center = px[16 * stride + 16];
        let [r, g, b, a] = [
            (center & 0xFF) as i32,
            ((center >> 8) & 0xFF) as i32,
            ((center >> 16) & 0xFF) as i32,
            ((center >> 24) & 0xFF) as i32,
        ];
        eprintln!("center pixel: r={r} g={g} b={b} a={a}");
        let corner = px[0];
        eprintln!("corner pixel: {corner:#010x}");
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
    fn test_render_with_matches_render() {
        let client = WgpuRuntime::client(&WgpuDevice::default());
        let attributes: Vec<f32> = (0..50 * 11)
            .map(|i| (i as f32 * 0.13).sin() * 0.5)
            .collect();
        let sh = vec![0.1f32; 50 * 3];
        let splats = Splats::new(attributes, sh, &client);
        let camera = crate::camera::Camera {
            fov: glam::Vec2::splat(0.8),
            position: glam::Vec3::new(0.0, -1.0, 0.0),
            rotation: glam::Quat::IDENTITY,
        };
        let a = pollster::block_on(splats.render(&camera, glam::uvec2(64, 64))).read_vec::<u32>();
        let mut scratch = RenderScratch::new(&client, 50, glam::uvec2(64, 64));
        let b = pollster::block_on(splats.render_with(&mut scratch, &camera, glam::uvec2(64, 64)))
            .read_vec::<u32>();
        // second frame through the same scratch — the steady-state path
        let c = pollster::block_on(splats.render_with(&mut scratch, &camera, glam::uvec2(64, 64)))
            .read_vec::<u32>();
        assert_eq!(a, b);
        assert_eq!(b, c);
    }

    #[test]
    fn test_scratch_stops_allocating_after_first_frame() {
        let client = WgpuRuntime::client(&WgpuDevice::default());
        let attributes: Vec<f32> = (0..100 * 11)
            .map(|i| (i as f32 * 0.07).sin() * 0.4)
            .collect();
        let splats = Splats::new(attributes, vec![0.0; 100 * 3], &client);
        let camera = crate::camera::Camera {
            fov: glam::Vec2::splat(0.8),
            position: glam::Vec3::new(0.0, -1.2, 0.0),
            rotation: glam::Quat::IDENTITY,
        };
        let mut scratch = RenderScratch::new(&client, 100, glam::uvec2(64, 64));
        pollster::block_on(splats.render_with(&mut scratch, &camera, glam::uvec2(64, 64)));
        let m1 = client.memory_usage().unwrap().bytes_in_use;
        pollster::block_on(splats.render_with(&mut scratch, &camera, glam::uvec2(64, 64)));
        let m2 = client.memory_usage().unwrap().bytes_in_use;
        assert!(
            m2 <= m1 + 65_536,
            "steady-state frame allocated {} bytes",
            m2.saturating_sub(m1)
        );
    }

    #[test]
    fn test_tile_ranges_zeroed_for_empty_tiles() {
        let client = WgpuRuntime::client(&WgpuDevice::default());
        let num_tiles = 8usize;
        // Sorted tile ids: tiles 0, 1, 2, 5 have intersections; 3, 4, 6, 7 are empty.
        let ids: Vec<u32> = vec![0, 0, 1, 2, 2, 2, 5, 5];
        let ids_t = GpuTensor::from(&client, [ids.len()], &ids[..]);
        let ranges = GpuTensor::from(&client, [num_tiles * 2], &vec![SENTINEL; num_tiles * 2][..]);
        let counters = GpuTensor::from(&client, [2], &[SENTINEL, SENTINEL][..]);

        zero_buffers::launch::<WgpuRuntime>(
            &client,
            cube_count_1d(&client, (2 + 2 * num_tiles) as u32, helpers::TILE_SIZE),
            CubeDim::new_1d(helpers::TILE_SIZE),
            counters.as_buffer_arg(),
            ranges.as_buffer_arg(),
        );
        build_tile_ranges::launch::<WgpuRuntime>(
            &client,
            cube_count_1d(&client, ids.len() as u32, helpers::TILE_SIZE),
            CubeDim::new_1d(helpers::TILE_SIZE),
            ids_t.as_buffer_arg(),
            ranges.as_buffer_arg(),
            ids.len() as u32,
        );

        let r: Vec<u32> = ranges.read_vec();
        assert_eq!(&r[0..2], &[0, 2], "tile 0 range");
        assert_eq!(&r[2..4], &[2, 3], "tile 1 range");
        assert_eq!(&r[4..6], &[3, 6], "tile 2 range");
        assert_eq!(&r[10..12], &[6, 8], "tile 5 range");
        for t in [3usize, 4, 6, 7] {
            assert_eq!(&r[t * 2..t * 2 + 2], &[0, 0], "tile {t} must be zeroed");
        }
        let c: Vec<u32> = counters.read_vec();
        assert_eq!(c, vec![0, 0]);
    }
}
