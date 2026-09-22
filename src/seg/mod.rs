//! B3-Seg: camera-free, training-free 3DGS segmentation via analytic EIG and
//! Beta–Bernoulli Bayesian updates (Kamata, Munro & Homma).
//!
//! The renderer's RGB path blends per-pixel colors; segmentation needs the
//! per-Gaussian view of the same math: how much each Gaussian contributes to
//! the image, ε_i(v) = Σ_p α_i(p)·T_i(p) — its "rendering responsibility".
//! The raster finalize kernel computes ε in its comptime non-RGB modes over
//! the shared tile pipeline (see `raster::rasterize_kernel`); this module
//! adds the fixed-point accumulators, the mask packing, and the Bayesian
//! state machinery the paper builds on it.

pub mod active;
pub mod beta;
pub(crate) mod paint;
pub mod prompted;
pub(crate) mod views;

use crate::camera::Camera;
use crate::render::{Finalize, RenderScratch, Splats, TileIsects};
use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;
use splat_sort::tensor::GpuTensor;

/// Fractional fixed-point bits for per-Gaussian accumulators rendered at
/// `img_size`. A Gaussian's Σ_p α·T ≤ pixel count, so u32 with this scale
/// cannot overflow: 2^32 / 2^bits = pixels.next_power_of_two() ≥ Σ —
/// strictly, because the kernel caps α at 0.999 (`raster.rs`), which keeps
/// Σ under the bound even at power-of-two sizes. Don't spend a bit to
/// "widen" that margin: it shifts every fixed-point word for nothing.
fn fractional_bits(img_size: glam::UVec2) -> u32 {
    let pixels = img_size.x as u64 * img_size.y as u64;
    32u32.saturating_sub(pixels.next_power_of_two().trailing_zeros())
}

/// One PCIe readback of a u32 fixed-point buffer, converted to f32.
#[cfg(test)]
fn read_scaled(t: &GpuTensor, scale: f32) -> Vec<f32> {
    t.read_vec::<u32>()
        .into_iter()
        .map(|q| q as f32 / scale)
        .collect()
}

/// Fixed-point ε accumulators for one resolution: the total responsibility
/// `bits` (candidate-scoring mode) and the mask-split foreground/background
/// pseudo-counts `fg`/`bg` (evidence mode). ε_i is u32 with
/// `fractional_bits(img_size)` fractional bits — WGSL atomics are
/// integer-only, and this scale keeps exact adds while never overflowing.
/// One struct because both modes share the scale and lifetime; the mode
/// distinction lives in the render path's `Finalize` enum, not in the type.
pub struct Accumulators {
    pub(crate) bits: GpuTensor,
    pub(crate) fg: GpuTensor,
    pub(crate) bg: GpuTensor,
    /// Fixed-point scale, owned by the render paths: set from the actual
    /// render resolution before any kernel or readback touches it.
    pub(crate) scale: f32,
}

impl Accumulators {
    pub fn new(client: &ComputeClient<WgpuRuntime>, total: usize) -> Self {
        Self {
            bits: GpuTensor::empty(client, [total]),
            fg: GpuTensor::empty(client, [total]),
            bg: GpuTensor::empty(client, [total]),
            scale: 1.0,
        }
    }

    /// ε_i in f32: one PCIe readback of N u32s plus a divide.
    #[cfg(test)]
    fn read_f32(&self) -> Vec<f32> {
        read_scaled(&self.bits, self.scale)
    }

    /// (e_{i,1}, e_{i,0}) in f32.
    #[cfg(test)]
    fn read_evidence_f32(&self) -> (Vec<f32>, Vec<f32>) {
        (
            read_scaled(&self.fg, self.scale),
            read_scaled(&self.bg, self.scale),
        )
    }
}

#[cube(launch)]
fn zero_u32(buf: &mut [Atomic<u32>]) {
    let i = ABSOLUTE_POS_X;
    if i < buf.len() as u32 {
        buf[i as usize].store(0u32);
    }
}

/// Clear a u32 accumulator buffer in one 256-thread elementwise pass.
fn zero_buf(client: &ComputeClient<WgpuRuntime>, buf: &GpuTensor) {
    zero_u32::launch::<WgpuRuntime>(
        client,
        calculate_cube_count_elemwise(client, buf.shape[0], CubeDim::new_1d(256)),
        CubeDim::new_1d(256),
        buf.as_buffer_arg(),
    );
}

/// A binary 2D mask, bit-packed row-major: word w of row r holds pixels
/// 32w..32w+31 of r, LSB-first. The oracle-facing format.
pub(crate) struct Mask {
    bits: GpuTensor,
    words_per_row: u32,
    /// Foreground pixels (`bytes` entries that were nonzero) — counted by
    /// the same walk that packs `bits`, so callers never re-scan the bytes.
    pub(crate) fg_pixels: usize,
}

impl Mask {
    /// Pack a CPU-side mask (`bytes`: row-major, nonzero = inside) for a
    /// `size` frame.
    pub(crate) fn from_bytes(
        client: &ComputeClient<WgpuRuntime>,
        size: glam::UVec2,
        bytes: &[u8],
    ) -> Self {
        assert_eq!(
            bytes.len(),
            (size.x * size.y) as usize,
            "mask bytes must be one per pixel"
        );
        let words_per_row = size.x.div_ceil(32);
        let mut bits = vec![0u32; (words_per_row * size.y) as usize];
        let mut fg_pixels = 0usize;
        for (p, &b) in bytes.iter().enumerate() {
            if b != 0 {
                let (y, x) = ((p as u32) / size.x, (p as u32) % size.x);
                bits[(y * words_per_row + x / 32) as usize] |= 1 << (x % 32);
                fg_pixels += 1;
            }
        }
        Self {
            bits: GpuTensor::from(client, [bits.len()], bits),
            words_per_row,
            fg_pixels,
        }
    }
}

impl Splats {
    /// Accumulate every Gaussian's rendering responsibility ε_i for `camera`
    /// into `acc` (cleared first). Shares the projection/sort pipeline with
    /// the RGB path; only the finalize mode differs.
    pub fn render_responsibility(
        &self,
        scratch: &mut RenderScratch,
        acc: &mut Accumulators,
        camera: &Camera,
        img_size: glam::UVec2,
    ) {
        let isects = self.prepare_isects(scratch, camera, img_size);
        self.accumulate_prepared(scratch, &isects, img_size, acc, None);
    }

    /// ε finalize over intersections already prepared for this exact camera
    /// and resolution (see `Splats::prepare_isects`): the same kernel over
    /// the same projected/isect data, so the fixed-point results are
    /// bit-identical to re-preparing. `scratch` is only read — its
    /// `projected` rows must still hold that camera's projection. The scale
    /// tracks the render resolution, which can change while the splat count
    /// (and so the buffers) stays the same. `None` fills `acc.bits` with the
    /// total responsibility; `Some(mask)` splits the weights by the mask
    /// into the fg/bg pseudo-counts.
    fn accumulate_prepared(
        &self,
        scratch: &RenderScratch,
        isects: &TileIsects,
        img_size: glam::UVec2,
        acc: &mut Accumulators,
        mask: Option<&Mask>,
    ) {
        acc.scale = (1u64 << fractional_bits(img_size)) as f32;
        let client = &self.attributes.client;
        match mask {
            None => {
                zero_buf(client, &acc.bits);
                self.finalize(
                    scratch,
                    isects,
                    img_size,
                    Finalize::Total {
                        scale: acc.scale,
                        resp: &acc.bits,
                    },
                );
            }
            Some(mask) => {
                zero_buf(client, &acc.fg);
                zero_buf(client, &acc.bg);
                self.finalize(
                    scratch,
                    isects,
                    img_size,
                    Finalize::Evidence {
                        scale: acc.scale,
                        mask: &mask.bits,
                        words_per_row: mask.words_per_row,
                        fg: &acc.fg,
                        bg: &acc.bg,
                    },
                );
            }
        }
    }
}

#[cfg(test)]
mod tests;
