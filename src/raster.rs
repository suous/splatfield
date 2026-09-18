//! Raster-stage kernels: intersection emission and front-to-back tile blending.

use crate::layout::{self, PROJ_FLOATS};
use cubecl::prelude::*;

/// Emit each visible splat's tile intersections at its prefix-sum offset.
/// Ranks are dispatched in ascending depth order and each splat's tiles are
/// walked in ascending tile id, so the output is sorted by (rank, tile id);
/// the stable tile sort that follows only has to move tiles together. The
/// payload is the splat id — inert for ordering, but it lets the rasterizer
/// index `projected` directly instead of re-translating through depth_order.
#[cube(launch)]
pub(crate) fn map_isects(
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

/// Index of the first isect in the sorted ids whose tile is >= `target` —
/// this tile's run start; searching for `tile + 1` gives its end.
#[cube]
pub(crate) fn lower_bound(ids: &[u32], n: u32, target: u32) -> u32 {
    let mut lo = 0u32;
    let mut hi = n;
    while lo < hi {
        let mid = (lo + hi) / 2u32;
        if ids[mid as usize] < target {
            lo = mid + 1u32;
        } else {
            hi = mid;
        }
    }
    lo
}

#[cube]
fn gaussian_power(conic: layout::Vec3F, dx: f32, dy: f32) -> f32 {
    0.5f32 * (conic.x * dx * dx + conic.z * dy * dy) + conic.y * dx * dy
}

#[cube]
fn quantize_u8(v: f32) -> u32 {
    (v * 255.0).clamp(0.0, 255.0) as u32
}

#[cube(launch)]
pub(crate) fn rasterize_kernel(
    img_size_x: u32,
    img_size_y: u32,
    row_stride: u32,
    num_isects: u32,
    tile_ids: &[u32],
    gaussian_ids_by_tile: &[u32],
    projected: &[f32],
    bitmap: &mut [u32],
) {
    let px = ABSOLUTE_POS_X;
    let py = ABSOLUTE_POS_Y;
    let tile_id = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    let in_bounds = px < img_size_x && py < img_size_y;

    // Each tile locates its run in the sorted isects with two lower-bound
    // searches — no separate range-building pass; empty tiles fall out as
    // start == end.
    let range_start = lower_bound(tile_ids, num_isects, tile_id);
    let range_end = lower_bound(tile_ids, num_isects, tile_id + 1u32);

    // Stage one workgroup-sized chunk of isects in shared memory: each 36-byte
    // row is fetched from global memory once per tile instead of once per
    // pixel, and every thread runs the chunk loop uniformly.
    //
    // The whole-tile early exit skips the remaining chunks once every pixel is
    // saturated. The count read is `workgroup_uniform_load_atomic` because its
    // barrier doubles as the staging fence, and only a uniform load keeps the
    // `break` convergent: a plain load would put the next iteration's
    // sync_cube in non-uniform control flow, which WGSL validators reject.
    let mut stage = Shared::<[f32]>::new_slice(layout::TILE_SIZE as usize * PROJ_FLOATS);
    let done_count = Shared::<[Atomic<u32>]>::new_slice(1usize);
    if UNIT_POS == 0 {
        done_count[0usize].store(0u32);
    }
    // The init store must be visible before finishers fetch_add: shared
    // atomics are ordered across threads only by a barrier.
    sync_cube();
    if !in_bounds {
        done_count[0usize].fetch_add(1u32);
    }

    let mut transmittance = 1.0f32;
    let mut pix_r = 0.0;
    let mut pix_g = 0.0;
    let mut pix_b = 0.0;
    let mut done = !in_bounds;

    let pixel_x = px as f32 + 0.5f32;
    let pixel_y = py as f32 + 0.5f32;

    let num_chunks = (range_end - range_start).div_ceil(layout::TILE_SIZE);
    for c in 0..num_chunks {
        let chunk = range_start + c * layout::TILE_SIZE;
        let idx = chunk + UNIT_POS;
        sync_cube();
        if idx < range_end {
            let src = (gaussian_ids_by_tile[idx as usize] * PROJ_FLOATS as u32) as usize;
            let dst = UNIT_POS as usize * PROJ_FLOATS;
            for k in 0..PROJ_FLOATS {
                stage[dst + k] = projected[src + k];
            }
        }
        // Whole tile converged: every remaining chunk would be a no-op. The
        // uniform load's barrier doubles as the staging-write fence for the
        // reads below, so this replaces the second sync_cube.
        if workgroup_uniform_load_atomic(&done_count[0usize]) == layout::TILE_SIZE {
            break;
        }

        let n_in_chunk = (range_end - chunk).min(layout::TILE_SIZE);
        for j in 0..n_in_chunk {
            if !done {
                let base = j as usize * PROJ_FLOATS;
                let mean_x = stage[base];
                let mean_y = stage[base + 1];
                let conic = layout::Vec3F {
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
                        done_count[0usize].fetch_add(1u32);
                    }
                }
            }
        }
    }

    if in_bounds {
        let r = quantize_u8(pix_r);
        let g = quantize_u8(pix_g);
        let b = quantize_u8(pix_b);
        let a = quantize_u8(1.0f32 - transmittance);
        bitmap[(px + py * row_stride) as usize] = r | (g << 8u32) | (b << 16u32) | (a << 24u32);
    }
}
