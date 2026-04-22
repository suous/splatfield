use crate::camera::Camera;
use crate::helpers;
use crate::sort::radix_argsort;
use crate::tensor::{F32, GpuTensor, U32, create_tensor_from_data, cube_count_1d, empty_tensor};
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

// Safety cap: 2 * max_tiles_per_dim * max_splats
const INTERSECTS_UPPER_BOUND: u32 = 2 * 512 * 65535;

#[derive(Debug, Clone)]
pub struct Splats {
    pub attributes: GpuTensor,
    pub sh_coeffs: GpuTensor,
    pub bounds: (glam::Vec3, glam::Vec3),
}

impl Splats {
    pub fn new(
        attributes: &[f32],
        sh_coeffs: &[f32],
        device: &WgpuDevice,
        bounds: (glam::Vec3, glam::Vec3),
    ) -> Self {
        let n = attributes.len() / 11;
        let n_coeffs = sh_coeffs.len() / n;
        Self {
            attributes: create_tensor_from_data([n, 11], device, F32, attributes),
            sh_coeffs: create_tensor_from_data([n, n_coeffs / 3, 3], device, F32, sh_coeffs),
            bounds,
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
    pub fn render(&self, camera: &Camera, img_size: glam::UVec2) -> GpuTensor {
        let client = &self.attributes.client;
        let device = &self.attributes.device;
        let total = self.attributes.shape[0];
        let tile_bounds = img_size.map(|c| c.div_ceil(helpers::TILE_WIDTH));
        let max_isects = (tile_bounds.x * tile_bounds.y)
            .saturating_mul(total as u32)
            .min(INTERSECTS_UPPER_BOUND) as usize;

        let focal = camera.focal(img_size);
        let pixel_center = img_size.as_vec2() * 0.5;
        let camera_pos = camera.position;
        let viewmat = glam::Mat4::from(camera.w2c()).transpose().to_cols_array();
        let sh_per_ch = self.sh_coeffs.shape[1] as u32;

        let depth_order = empty_tensor([total], device, U32);
        let depth_keys = empty_tensor([total], device, U32);
        let projected = empty_tensor([total, 9], device, F32);
        let isect_counter = create_tensor_from_data([1], device, U32, &[0u32]);
        let tile_ids = empty_tensor([max_isects], device, U32);
        let gaussian_ids = empty_tensor([max_isects], device, U32);
        let viewmat = create_tensor_from_data([16], device, F32, &viewmat);

        crate::project::project_splats::launch::<WgpuRuntime>(
            client,
            cube_count_1d(client, total as u32, helpers::TILE_SIZE),
            CubeDim::new_1d(helpers::TILE_SIZE),
            viewmat.as_array_arg(),
            helpers::Vec2FLaunch::new(focal.x, focal.y),
            helpers::Vec2FLaunch::new(pixel_center.x, pixel_center.y),
            helpers::Vec3FLaunch::new(camera_pos.x, camera_pos.y, camera_pos.z),
            total as u32,
            self.attributes.as_array_arg(),
            self.sh_coeffs.as_array_arg(),
            sh_per_ch,
            helpers::Vec2FLaunch::new(tile_bounds.x as f32, tile_bounds.y as f32),
            depth_order.as_array_arg(),
            depth_keys.as_array_arg(),
            projected.as_array_arg(),
            isect_counter.as_array_arg(),
            tile_ids.as_array_arg(),
            gaussian_ids.as_array_arg(),
        );

        let num_isects = isect_counter.read_u32_at(0);
        let (inv_perm, depth_order) = radix_argsort(depth_keys, depth_order, total as u32, 32);

        invert_permutation::launch::<WgpuRuntime>(
            client,
            cube_count_1d(client, total as u32, helpers::TILE_SIZE),
            CubeDim::new_1d(helpers::TILE_SIZE),
            depth_order.as_array_arg(),
            inv_perm.as_array_arg(),
            total as u32,
        );

        remap_global_ids::launch::<WgpuRuntime>(
            client,
            cube_count_1d(client, num_isects, helpers::TILE_SIZE),
            CubeDim::new_1d(helpers::TILE_SIZE),
            gaussian_ids.as_array_arg(),
            inv_perm.as_array_arg(),
            num_isects,
        );

        // Two stable radix sorts equivalent to the paper's single composite key
        // (tile_id << depth_bits) | depth_id: first sort by depth, then by tile.
        let gid_bits = u32::BITS - (total as u32).leading_zeros().max(1);
        let (gaussian_ids, tile_ids) = radix_argsort(gaussian_ids, tile_ids, num_isects, gid_bits);
        let tile_bits = u32::BITS - (tile_bounds.x * tile_bounds.y).leading_zeros();
        let (tile_ids, gaussian_ids) = radix_argsort(tile_ids, gaussian_ids, num_isects, tile_bits);

        let tile_ranges = empty_tensor([(tile_bounds.x * tile_bounds.y) as usize, 2], device, U32);
        build_tile_ranges::launch::<WgpuRuntime>(
            client,
            cube_count_1d(client, num_isects, helpers::TILE_SIZE),
            CubeDim::new_1d(helpers::TILE_SIZE),
            tile_ids.as_array_arg(),
            tile_ranges.as_array_arg(),
            num_isects,
        );

        let row_stride = (img_size.x * 4).next_multiple_of(256) / 4;
        let bitmap = empty_tensor([img_size.y as usize, row_stride as usize], device, U32);
        rasterize_kernel::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(tile_bounds.x * tile_bounds.y, 1, 1),
            CubeDim::new_1d(helpers::TILE_SIZE),
            img_size.x,
            img_size.y,
            row_stride,
            gaussian_ids.as_array_arg(),
            tile_ranges.as_array_arg(),
            projected.as_array_arg(),
            depth_order.as_array_arg(),
            bitmap.as_array_arg(),
        );
        bitmap
    }
}

#[cube(launch)]
fn invert_permutation(perm: &mut Array<u32>, inv: &mut Array<u32>, n: u32) {
    if ABSOLUTE_POS_X < n {
        inv[perm[ABSOLUTE_POS_X as usize] as usize] = ABSOLUTE_POS_X;
    }
}

#[cube(launch)]
fn remap_global_ids(gids: &mut Array<u32>, inv_perm: &Array<u32>, n: u32) {
    if ABSOLUTE_POS_X < n {
        gids[ABSOLUTE_POS_X as usize] = inv_perm[gids[ABSOLUTE_POS_X as usize] as usize];
    }
}

#[cube(launch)]
fn build_tile_ranges(ids: &Array<u32>, ranges: &mut Array<u32>, num_isects: u32) {
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
pub fn gaussian_power(conic: helpers::Vec3F, dx: f32, dy: f32) -> f32 {
    0.5 * (conic.x * dx * dx + conic.z * dy * dy) + conic.y * dx * dy
}

#[cube(launch)]
fn rasterize_kernel(
    img_size_x: u32,
    img_size_y: u32,
    row_stride: u32,
    gaussian_ids_by_tile: &Array<u32>,
    tile_ranges: &Array<u32>,
    projected: &Array<f32>,
    depth_order: &Array<u32>,
    bitmap: &mut Array<u32>,
) {
    let (px, py, tile_id) = helpers::map_1d_to_2d(ABSOLUTE_POS_X, img_size_x / helpers::TILE_WIDTH);
    if px < img_size_x && py < img_size_y {
        let pixel_x = px as f32 + 0.5f32;
        let pixel_y = py as f32 + 0.5f32;
        let range_start = tile_ranges[tile_id as usize * 2];
        let range_end = tile_ranges[tile_id as usize * 2 + 1];

        let mut transmittance = 1.0;
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

            if alpha >= 1.0 / 255.0 {
                let vis = alpha * transmittance;
                pix_r += color_r * vis;
                pix_g += color_g * vis;
                pix_b += color_b * vis;
                transmittance *= 1.0 - alpha;
            }
        }

        let r = helpers::quantize_u8(pix_r);
        let g = helpers::quantize_u8(pix_g);
        let b = helpers::quantize_u8(pix_b);
        let a = helpers::quantize_u8(1.0f32 - transmittance);
        bitmap[(px + py * row_stride) as usize] = r | (g << 8u32) | (b << 16u32) | (a << 24u32);
    }
}
