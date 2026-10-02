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
pub(crate) fn gaussian_power(conic: layout::Vec3F, dx: f32, dy: f32) -> f32 {
    0.5f32 * (conic.x * dx * dx + conic.z * dy * dy) + conic.y * dx * dy
}

#[cube]
fn quantize_u8(v: f32) -> u32 {
    (v * 255.0).clamp(0.0, 255.0) as u32
}

/// Finalize the tile pipeline: walk each pixel's depth-sorted splat run.
/// The RGB mode blends per-pixel colors into `bitmap`; the ε modes
/// (`rgb = false`) accumulate each splat's rendering weight w = α·T
/// instead — total responsibility into `resp`, or in evidence mode split
/// by the mask bit into foreground/background pseudo-counts.
///
/// Each pixel locates its tile's run with two lower-bound searches and
/// walks it with direct global reads. An earlier version staged each
/// 36-byte row through workgroup shared memory with a uniform-load-gated
/// early exit; that variant silently produces an all-zero bitmap on
/// WebKit's WebGPU (iOS), so the simpler direct-read form is load-bearing
/// for WebKit compatibility — do not re-introduce the shared staging
/// without an on-device test.
#[cube(launch)]
pub(crate) fn rasterize_kernel(
    img_size_x: u32,
    img_size_y: u32,
    row_stride: u32,
    num_isects: u32,
    scale: f32,
    tile_ids: &[u32],
    gaussian_ids_by_tile: &[u32],
    projected: &[f32],
    #[comptime] rgb: bool,
    #[comptime] evidence: bool,
    mask: &[u32],
    mask_words_per_row: u32,
    bitmap: &mut [u32],
    resp: &mut [Atomic<u32>],
    fg: &mut [Atomic<u32>],
    bg: &mut [Atomic<u32>],
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

    let pixel_x = px as f32 + 0.5f32;
    let pixel_y = py as f32 + 0.5f32;

    // Loop-invariant per pixel; comptime-dead outside evidence mode,
    // `in_bounds` guards edge tiles.
    let mut inside = 0u32;
    if evidence && in_bounds {
        inside = (mask[(px / layout::MASK_BITS + py * mask_words_per_row) as usize]
            >> (px % layout::MASK_BITS))
            & 1u32;
    }

    let mut transmittance = 1.0f32;
    let mut pix_r = 0.0;
    let mut pix_g = 0.0;
    let mut pix_b = 0.0;
    // The saturation cutoff is semantic, not an optimization: the ε modes
    // accumulate w = α·T into per-splat counters, and post-saturation
    // contributions — though each below one LSB — add up to real counter
    // deltas at the ×scale rounding. Once T < 1/255 the pixel discards
    // every remaining splat, exactly like the shared-staging variant did.
    let mut done = false;

    let num_isects_tile = range_end - range_start;
    for j in 0..num_isects_tile {
        if !done {
            let idx = range_start + j;
            let src = (gaussian_ids_by_tile[idx as usize] * PROJ_FLOATS as u32) as usize;
            let mean_x = projected[src];
            let mean_y = projected[src + 1];
            let conic = layout::Vec3F {
                x: projected[src + 2],
                y: projected[src + 3],
                z: projected[src + 4],
            };
            let color_a = projected[src + 8];

            let power = gaussian_power(conic, mean_x - pixel_x, mean_y - pixel_y);
            let alpha = (color_a * (-power).exp()).min(0.999);

            if alpha >= 1.0f32 / u8::MAX as f32 {
                let vis = alpha * transmittance;
                if rgb {
                    let color_r = projected[src + 5];
                    let color_g = projected[src + 6];
                    let color_b = projected[src + 7];
                    pix_r += color_r * vis;
                    pix_g += color_g * vis;
                    pix_b += color_b * vis;
                } else {
                    // w ≤ 1, so the fixed-point product always fits u32.
                    let splat = gaussian_ids_by_tile[idx as usize] as usize;
                    let q = (vis * scale) as u32;
                    if evidence {
                        if inside == 1u32 {
                            fg[splat].fetch_add(q);
                        } else {
                            bg[splat].fetch_add(q);
                        }
                    } else {
                        resp[splat].fetch_add(q);
                    }
                }
                transmittance *= 1.0f32 - alpha;
                // Remaining weight < 1 LSB of the final 8-bit channels.
                if transmittance < 1.0f32 / 255.0f32 {
                    done = true;
                }
            }
        }
    }

    if rgb && in_bounds {
        let r = quantize_u8(pix_r);
        let g = quantize_u8(pix_g);
        let b = quantize_u8(pix_b);
        let a = quantize_u8(1.0f32 - transmittance);
        bitmap[(px + py * row_stride) as usize] = r | (g << 8u32) | (b << 16u32) | (a << 24u32);
    }
}
