use crate::camera::Camera;
use crate::helpers;
use crate::scan::{ScanScratch, exclusive_scan_gather};
use crate::sort::{RadixScratch, bits_for, radix_argsort_with};
use crate::tensor::GpuTensor;
use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

// Safety cap: 2 * max_tiles_per_dim * max_splats
const INTERSECTS_UPPER_BOUND: usize = 2 * 512 * 65535;

// z ∈ (0.1, 1e4) keeps the top 4 mantissa bits of the f32 bit pattern constant;
// dropping 8 low mantissa bits (relative granularity 2^-15) saves two radix
// passes vs full keys. Ties within the band blend in arbitrary order — the
// same treatment equal keys already get.
const DEPTH_KEY_BITS: u32 = 24;

/// Web builds compile the rasterizer without the shared-memory early exit —
/// browsers reject the break-on-shared-count it needs (see rasterize_kernel).
const EARLY_EXIT: bool = !cfg!(target_arch = "wasm32");

/// Monotonic depth key for `z > 0` (float order == depth order): project.rs
/// packs keys with this, render.rs sorts with `DEPTH_KEY_BITS`.
#[cube]
pub(crate) fn depth_key(z: f32) -> u32 {
    z.to_bits() >> (32 - DEPTH_KEY_BITS)
}

/// Host-side splat payload produced by the PLY/SOG parsers, uploaded once by
/// `Splats::new`. Both arrays are FIELD-MAJOR (plane-per-field) so the
/// projection kernel's per-thread reads coalesce across a warp:
/// attribute plane k of splat i is `attributes[k * n + i]` with plane order
/// `x, y, z, qw, qx, qy, qz, sx, sy, sz, opacity`; SH coefficient k channel c
/// of splat i is `sh_coeffs[(k * 3 + c) * n + i]`.
#[derive(Debug, Clone, Default)]
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
#[derive(Debug)]
pub struct RenderScratch {
    depth_order: GpuTensor,
    depth_keys: GpuTensor,
    projected: GpuTensor,
    counters: GpuTensor,
    tile_counts: GpuTensor,
    tile_bbox: GpuTensor,
    tile_ids: GpuTensor,
    gaussian_ids: GpuTensor,
    tile_ranges: GpuTensor,
    bitmap: GpuTensor,
    // Ping-pong scratch for the two sorts. The tile sort must not share the
    // depth sort's: an odd pass count returns buffers aliasing the scratch.
    sort_depth: RadixScratch,
    sort_tile: RadixScratch,
    scan: ScanScratch,
    total: usize,
    img_size: glam::UVec2,
    tile_bounds: glam::UVec2,
    isect_capacity: usize,
}

impl RenderScratch {
    pub fn new(client: &ComputeClient<WgpuRuntime>, total: usize, img_size: glam::UVec2) -> Self {
        let tile_bounds = img_size.map(|c| c.div_ceil(helpers::TILE_WIDTH));
        let num_tiles = (tile_bounds.x * tile_bounds.y) as usize;
        // Power of two: the growth check is one `next_power_of_two()` compare.
        let isect_capacity = (num_tiles.saturating_mul(total).min(1 << 22)).next_power_of_two();
        // 256-byte rows: wgpu buffer→texture copies align rows to
        // COPY_BYTES_PER_ROW_ALIGNMENT; texture.rs copies with this stride.
        let row_stride = (img_size.x * 4).next_multiple_of(256) / 4;
        Self {
            depth_order: GpuTensor::empty(client, [total]),
            depth_keys: GpuTensor::empty(client, [total]),
            projected: GpuTensor::empty(client, [total, 9]),
            counters: GpuTensor::empty(client, [2]),
            tile_counts: GpuTensor::empty(client, [total]),
            tile_bbox: GpuTensor::empty(client, [total, 2]),
            tile_ids: GpuTensor::empty(client, [isect_capacity]),
            gaussian_ids: GpuTensor::empty(client, [isect_capacity]),
            tile_ranges: GpuTensor::empty(client, [num_tiles * 2]),
            bitmap: GpuTensor::empty(client, [img_size.y as usize, row_stride as usize]),
            sort_depth: RadixScratch::new(client, total),
            sort_tile: RadixScratch::new(client, isect_capacity),
            scan: ScanScratch::new(client, total),
            total,
            img_size,
            tile_bounds,
            isect_capacity,
        }
    }

    fn matches(&self, total: usize, img_size: glam::UVec2) -> bool {
        self.total == total && self.img_size == img_size
    }

    /// Grow the isect buffers when `raw` exceeded capacity this frame. This
    /// frame stays truncated at the old capacity (frame-local clones); at the
    /// ceiling the computed capacity equals the current one: no realloc, no warn.
    fn grow_isects(&mut self, client: &ComputeClient<WgpuRuntime>, raw: u32) {
        let new_cap = (raw as usize)
            .next_power_of_two()
            .min(INTERSECTS_UPPER_BOUND);
        if new_cap > self.isect_capacity {
            log::warn!(
                "intersection capacity {} reached ({raw} emitted); growing to {new_cap}",
                self.isect_capacity
            );
            self.isect_capacity = new_cap;
            self.tile_ids = GpuTensor::empty(client, [new_cap]);
            self.gaussian_ids = GpuTensor::empty(client, [new_cap]);
            self.sort_tile = RadixScratch::new(client, new_cap);
        }
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
        for i in 0..n {
            let p = glam::vec3(attributes[i], attributes[n + i], attributes[2 * n + i]);
            min = min.min(p);
            max = max.max(p);
        }

        Self {
            attributes: GpuTensor::from(client, [n, 11], attributes),
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
        let tile_bounds = scratch.tile_bounds;
        let num_tiles = (tile_bounds.x * tile_bounds.y) as usize;
        let max_isects = scratch.isect_capacity;
        let cube_dim = CubeDim::new_1d(helpers::TILE_SIZE);

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
            calculate_cube_count_elemwise(client, 2 + 2 * num_tiles, cube_dim),
            cube_dim,
            scratch.counters.as_buffer_arg(),
            scratch.tile_ranges.as_buffer_arg(),
        );

        crate::project::project_splats::launch::<WgpuRuntime>(
            client,
            calculate_cube_count_elemwise(client, total, cube_dim),
            cube_dim,
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
            scratch.tile_counts.as_buffer_arg(),
            scratch.tile_bbox.as_buffer_arg(),
        );

        let [num_isects_raw, num_visible] = scratch.counters.read_pair().await;
        let num_isects = num_isects_raw.min(max_isects as u32);
        scratch.grow_isects(client, num_isects_raw);
        let (_sorted_keys, depth_order) = radix_argsort_with(
            scratch.depth_keys.clone(),
            scratch.depth_order.clone(),
            num_visible,
            DEPTH_KEY_BITS,
            &scratch.sort_depth,
        );

        // Emission offsets in depth order: offsets[rank] = number of
        // intersections emitted by all nearer splats. The map kernel then
        // writes each splat's tile intersections contiguously, so the isect
        // array arrives pre-sorted by depth rank. depth_keys is dead here —
        // the depth sort discarded its output keys — so it doubles as the
        // scan's offsets scratch (no dedicated buffer).
        exclusive_scan_gather(
            depth_order.clone(),
            scratch.tile_counts.clone(),
            scratch.depth_keys.clone(),
            num_visible,
            &scratch.scan,
        );

        map_isects::launch::<WgpuRuntime>(
            client,
            calculate_cube_count_elemwise(
                client,
                num_visible as usize,
                CubeDim::new_1d(helpers::TILE_SIZE),
            ),
            CubeDim::new_1d(helpers::TILE_SIZE),
            depth_order.as_buffer_arg(),
            scratch.tile_bbox.as_buffer_arg(),
            scratch.depth_keys.as_buffer_arg(),
            tile_bounds.x,
            max_isects as u32,
            tile_ids.as_buffer_arg(),
            gaussian_ids.as_buffer_arg(),
            num_visible,
        );

        // Isects are emitted in depth order, so this single stable sort by
        // tile id yields tile-major, depth-minor order — no separate gid sort
        // or composite key needed.
        let tile_bits = bits_for(num_tiles as u32);
        let (tile_ids, gaussian_ids) = radix_argsort_with(
            tile_ids,
            gaussian_ids,
            num_isects,
            tile_bits,
            &scratch.sort_tile,
        );

        build_tile_ranges::launch::<WgpuRuntime>(
            client,
            calculate_cube_count_elemwise(
                client,
                num_isects as usize,
                CubeDim::new_1d(helpers::TILE_SIZE),
            ),
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
            scratch.bitmap.as_buffer_arg(),
            // Browser shader validation forbids the shared-memory early exit
            // (see the kernel); native keeps it.
            EARLY_EXIT,
        );
        scratch.bitmap.clone()
    }
}

/// Emit each visible splat's tile intersections at its prefix-sum offset.
/// Ranks are dispatched in ascending depth order and each splat's tiles are
/// walked in ascending tile id, so the output is sorted by (rank, tile id);
/// the stable tile sort that follows only has to move tiles together. The
/// payload is the splat id — inert for ordering, but it lets the rasterizer
/// index `projected` directly instead of re-translating through depth_order.
#[cube(launch)]
fn map_isects(
    depth_order: &[u32],
    tile_bbox: &[u32],
    offsets: &[u32],
    tiles_per_row: u32,
    max_isects: u32,
    tile_ids: &mut [u32],
    gaussian_ids: &mut [u32],
    num_visible: u32,
) {
    let rank = ABSOLUTE_POS_X;
    if rank >= num_visible {
        terminate!();
    }
    let vis = depth_order[rank as usize];
    let lo = tile_bbox[vis as usize * 2];
    let hi = tile_bbox[vis as usize * 2 + 1];
    let min_x = lo & 0xFFFFu32;
    let min_y = lo >> 16u32;
    let max_x = hi & 0xFFFFu32;
    let max_y = hi >> 16u32;

    let mut slot = offsets[rank as usize];
    for ty in min_y..max_y {
        for tx in min_x..max_x {
            // Past capacity this frame is truncated (the buffers grow for the
            // next frame); the offset walk must continue regardless so all
            // ranks stay consistent.
            if slot < max_isects {
                tile_ids[slot as usize] = tx + ty * tiles_per_row;
                gaussian_ids[slot as usize] = vis;
            }
            slot += 1;
        }
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
    bitmap: &mut [u32],
    #[comptime] early_exit: bool,
) {
    let px = ABSOLUTE_POS_X;
    let py = ABSOLUTE_POS_Y;
    let tile_id = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    let in_bounds = px < img_size_x && py < img_size_y;

    let range_start = tile_ranges[tile_id as usize * 2];
    let range_end = tile_ranges[tile_id as usize * 2 + 1];

    // Stage one workgroup-sized chunk of isects in shared memory: each isect's
    // 36-byte row is fetched from global memory once per tile instead of once
    // per pixel. All threads run the chunk loop uniformly so sync_cube never
    // diverges; converged threads just stop accumulating.
    //
    // The whole-tile early exit skips the remaining chunks once every pixel is
    // saturated. It counts finishers in a shared atomic and `break`s the chunk
    // loop on the count — which the WGSL uniformity analysis rejects, because
    // it treats every load from workgroup memory as non-uniform, making the
    // break (and with it the next iteration's sync_cube) divergent. Browsers
    // refuse to compile the shader, so web builds specialize with early_exit
    // off and pay only per-thread `done` latching; native keeps the exit.
    let mut stage = Shared::<[f32]>::new_slice(helpers::TILE_SIZE as usize * 9);
    let done_count = Shared::<[Atomic<u32>]>::new_slice(1usize);
    if early_exit {
        if UNIT_POS == 0 {
            done_count[0usize].store(0u32);
        }
        if !in_bounds {
            done_count[0usize].fetch_add(1u32);
        }
    }

    let mut transmittance = 1.0f32;
    let mut pix_r = 0.0;
    let mut pix_g = 0.0;
    let mut pix_b = 0.0;
    let mut done = !in_bounds;

    let pixel_x = px as f32 + 0.5f32;
    let pixel_y = py as f32 + 0.5f32;

    let num_chunks = (range_end - range_start).div_ceil(helpers::TILE_SIZE);
    for c in 0..num_chunks {
        let chunk = range_start + c * helpers::TILE_SIZE;
        let idx = chunk + UNIT_POS;
        sync_cube();
        if idx < range_end {
            // Projected layout: [mean2d_x, mean2d_y, conic_x, conic_y, conic_z, r, g, b, opacity]
            let src = (gaussian_ids_by_tile[idx as usize] * 9u32) as usize;
            let dst = UNIT_POS as usize * 9;
            for k in 0..9 {
                stage[dst + k] = projected[src + k];
            }
        }
        sync_cube();

        // Whole tile converged: every remaining chunk would be a no-op.
        if early_exit && done_count[0usize].load() == helpers::TILE_SIZE {
            break;
        }

        let n_in_chunk = (range_end - chunk).min(helpers::TILE_SIZE);
        for j in 0..n_in_chunk {
            if !done {
                let base = j as usize * 9;
                let mean_x = stage[base];
                let mean_y = stage[base + 1];
                let conic = helpers::Vec3F {
                    x: stage[base + 2],
                    y: stage[base + 3],
                    z: stage[base + 4],
                };
                let color_r = stage[base + 5];
                let color_g = stage[base + 6];
                let color_b = stage[base + 7];
                let color_a = stage[base + 8];

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
                        done = true;
                        if early_exit {
                            done_count[0usize].fetch_add(1u32);
                        }
                    }
                }
            }
        }
    }

    if in_bounds {
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
        let _gpu = crate::tensor::GPU_TEST_LOCK.lock().unwrap();
        let client = WgpuRuntime::client(&WgpuDevice::default());
        // Five opaque splats stacked in depth at the same XY, nearest at z = 1.0.
        // Field-major attribute planes: [x | y | z | qw | qx | qy | qz | sx | sy | sz | opacity].
        let n = 5usize;
        let mut attributes = vec![0f32; n * 11];
        for i in 0..n {
            attributes[2 * n + i] = 1.0 + i as f32 * 0.5; // z
            attributes[3 * n + i] = 1.0; // qw
            attributes[7 * n + i] = -2.0; // sx
            attributes[8 * n + i] = -2.0; // sy
            attributes[9 * n + i] = -2.0; // sz
            attributes[10 * n + i] = 8.0; // opacity
        }
        let sh = vec![0.0; 5 * 3];
        let splats = Splats::new(attributes, sh, &client);
        let camera = crate::camera::Camera {
            fov: glam::Vec2::splat(0.8),
            position: glam::Vec3::ZERO,
            rotation: glam::Quat::IDENTITY,
        };
        let mut scratch = RenderScratch::new(&client, 5, glam::uvec2(32, 32));
        let bitmap =
            pollster::block_on(splats.render_with(&mut scratch, &camera, glam::uvec2(32, 32)));
        let px: Vec<u32> = bitmap.read_vec();
        let stride = bitmap.shape[1] as usize;
        let center = px[16 * stride + 16];
        let [r, g, b, a] = [
            (center & 0xFF) as i32,
            ((center >> 8) & 0xFF) as i32,
            ((center >> 16) & 0xFF) as i32,
            ((center >> 24) & 0xFF) as i32,
        ];
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
    fn test_render_reuses_scratch() {
        let _gpu = crate::tensor::GPU_TEST_LOCK.lock().unwrap();
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
        let mut scratch = RenderScratch::new(&client, 50, glam::uvec2(64, 64));
        let b = pollster::block_on(splats.render_with(&mut scratch, &camera, glam::uvec2(64, 64)))
            .read_vec::<u32>();
        let c = pollster::block_on(splats.render_with(&mut scratch, &camera, glam::uvec2(64, 64)))
            .read_vec::<u32>();
        assert_eq!(b, c);
    }

    #[test]
    fn test_scratch_stops_allocating_after_first_frame() {
        let _gpu = crate::tensor::GPU_TEST_LOCK.lock().unwrap();
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

    #[test]
    fn test_tile_ranges_zeroed_for_empty_tiles() {
        let _gpu = crate::tensor::GPU_TEST_LOCK.lock().unwrap();
        let client = WgpuRuntime::client(&WgpuDevice::default());
        let num_tiles = 8usize;
        // Sorted tile ids: tiles 0, 1, 2, 5 have intersections; 3, 4, 6, 7 are empty.
        let ids: Vec<u32> = vec![0, 0, 1, 2, 2, 2, 5, 5];
        let ids_t = GpuTensor::from(&client, [ids.len()], &ids[..]);
        let ranges = GpuTensor::from(&client, [num_tiles * 2], &vec![SENTINEL; num_tiles * 2][..]);
        let counters = GpuTensor::from(&client, [2], &[SENTINEL, SENTINEL][..]);

        zero_buffers::launch::<WgpuRuntime>(
            &client,
            calculate_cube_count_elemwise(
                &client,
                2 + 2 * num_tiles,
                CubeDim::new_1d(helpers::TILE_SIZE),
            ),
            CubeDim::new_1d(helpers::TILE_SIZE),
            counters.as_buffer_arg(),
            ranges.as_buffer_arg(),
        );
        build_tile_ranges::launch::<WgpuRuntime>(
            &client,
            calculate_cube_count_elemwise(&client, ids.len(), CubeDim::new_1d(helpers::TILE_SIZE)),
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
