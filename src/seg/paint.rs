//! GUI-facing paint ops over the shared scene: the posterior-mean heatmap,
//! foreground recoloring, and the DC snapshot/restore pair the app uses for
//! undo and scene deletion. Nothing here participates in the Bayesian loop —
//! the consumers are the viewer's UI paths, not `seg::active`.

use crate::camera::Camera;
use crate::layout;
use crate::render::{Finalize, RenderScratch, Splats};
use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use splat_sort::tensor::GpuTensor;

impl Splats {
    /// Render the scene with each Gaussian painted by its posterior mean
    /// m = a/(a+b) as grayscale — black = certain background, white = certain
    /// foreground — the paper's "render mean image of Beta dist.". Shares
    /// the pipeline with `render_with_async` but overwrites the projected
    /// color rows between projection and blending. A `posterior` sized for a
    /// different splat count (model swapped mid-run) falls back to the
    /// normal color render rather than painting out of bounds. Async because
    /// the pipeline syncs on the counters readback, and cubecl's blocking
    /// reads poll once and panic on wasm.
    pub async fn render_posterior_async(
        &self,
        scratch: &mut RenderScratch,
        a: &GpuTensor,
        b: &GpuTensor,
        camera: &Camera,
        img_size: glam::UVec2,
    ) -> GpuTensor {
        if a.shape[0] != self.attributes.shape[0] || b.shape[0] != self.attributes.shape[0] {
            return self.render_with_async(scratch, camera, img_size).await;
        }
        let client = &self.attributes.client;
        let isects = self.prepare_isects_async(scratch, camera, img_size).await;

        paint_posterior::launch(
            client,
            calculate_cube_count_elemwise(client, a.shape[0], CubeDim::new_1d(256)),
            CubeDim::new_1d(256),
            scratch.projected.as_buffer_arg(),
            a.as_buffer_arg(),
            b.as_buffer_arg(),
        );

        self.finalize(scratch, &isects, img_size, Finalize::Rgb);
        scratch.bitmap.clone()
    }

    /// Blocking [`Self::render_posterior_async`]: drives the readback future
    /// on this thread, which is what cubecl's own sync reads do internally.
    /// Test-only — the app drives the async body directly, so this survives
    /// as the pin harness's sync seam.
    #[cfg(test)]
    pub(crate) fn render_posterior(
        &self,
        scratch: &mut RenderScratch,
        a: &GpuTensor,
        b: &GpuTensor,
        camera: &Camera,
        img_size: glam::UVec2,
    ) -> GpuTensor {
        cubecl::future::block_on(self.render_posterior_async(scratch, a, b, camera, img_size))
    }

    /// Overwrite the SH DC planes of foreground splats (a_i > b_i) with
    /// `color`, in place on the GPU.
    pub fn tint(&self, a: &GpuTensor, b: &GpuTensor, color: [f32; 3]) {
        let client = &self.attributes.client;
        let total = a.shape[0];
        tint_colors::launch(
            client,
            calculate_cube_count_elemwise(client, total, CubeDim::new_1d(256)),
            CubeDim::new_1d(256),
            self.sh_coeffs.as_buffer_arg(),
            a.as_buffer_arg(),
            b.as_buffer_arg(),
            color[0],
            color[1],
            color[2],
        );
    }

    /// Snapshot the three DC color planes (layout [3, total], field-major)
    /// with a GPU-side copy of the sh buffer's first 3·total floats — no
    /// readback of the full coefficient buffer.
    pub fn save_colors(&self) -> GpuTensor {
        let saved = GpuTensor::empty(&self.attributes.client, [3, self.attributes.shape[0]]);
        self.copy_dc(&self.sh_coeffs, &saved);
        saved
    }

    /// Write a previously saved snapshot back into the DC planes.
    pub fn restore_colors(&self, saved: &GpuTensor) {
        self.copy_dc(saved, &self.sh_coeffs);
    }

    /// The DC snapshot/restore launch shared by [`Splats::save_colors`] and
    /// [`Splats::restore_colors`].
    fn copy_dc(&self, src: &GpuTensor, dst: &GpuTensor) {
        let client = &self.attributes.client;
        let total = 3 * self.attributes.shape[0];
        copy_f32::launch(
            client,
            calculate_cube_count_elemwise(client, total, CubeDim::new_1d(256)),
            CubeDim::new_1d(256),
            src.as_buffer_arg(),
            dst.as_buffer_arg(),
        );
    }
}

/// Overwrite each projected splat's RGB with grayscale posterior mean m.
/// No +0.5 display offset: the composited pixel equals Σ m_i·ω_i·T_i, the
/// paper's soft prior image.
#[cube(launch)]
fn paint_posterior(projected: &mut [f32], a: &[f32], b: &[f32]) {
    let i = ABSOLUTE_POS_X as usize;
    if i < a.len() {
        let m = a[i] / (a[i] + b[i]);
        let base = i * layout::PROJ_FLOATS + 5;
        projected[base] = m;
        projected[base + 1] = m;
        projected[base + 2] = m;
    }
}

/// Overwrite splat i's DC channels with (r, g, b) wherever a_i > b_i. The sh
/// buffer is field-major: DC channel c of splat i sits at flat c·total + i.
#[cube(launch)]
fn tint_colors(sh: &mut [f32], a: &[f32], b: &[f32], r: f32, g: f32, b_: f32) {
    let i = ABSOLUTE_POS_X;
    let total = a.len() as u32;
    if i < total && a[i as usize] > b[i as usize] {
        sh[i as usize] = r;
        sh[(total + i) as usize] = g;
        sh[(2 * total + i) as usize] = b_;
    }
}

/// Element-wise f32 copy `src` → `dst`, bounded by the shorter buffer: the
/// DC snapshot/restore primitive (the sh buffer and the [3, total] snapshot
/// coincide on their first 3·total floats; the bound never overruns the sh
/// buffer's rest planes).
#[cube(launch)]
fn copy_f32(src: &[f32], dst: &mut [f32]) {
    let i = ABSOLUTE_POS_X as usize;
    if i < src.len() && i < dst.len() {
        dst[i] = src[i];
    }
}
