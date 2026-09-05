//! Diagnostic: launch every render-pipeline kernel once so cubecl-wgpu's
//! trace log dumps the exact WGSL each one compiles to.
//!
//! ```sh
//! RUST_LOG=cubecl_wgpu=trace cargo run --example dump_wgsl 2> wgsl.log
//! ```
use cubecl::Runtime;
use splatfield::camera::Camera;
use splatfield::render::{RenderScratch, Splats, sample_opaque_attributes};

fn main() {
    env_logger::init();
    let client = cubecl::wgpu::WgpuRuntime::client(&Default::default());
    println!("backend: {:?}", cubecl::Runtime::name(&client));

    let n = 5usize;
    let attributes = sample_opaque_attributes(n, |i| 1.0 + i as f32 * 0.5);
    let splats = Splats::new(attributes, vec![0.0; n * 3], &client);
    let camera = Camera::default();
    let mut scratch = RenderScratch::new(&client, n, glam::uvec2(32, 32));
    let bitmap = pollster::block_on(splats.render_with(&mut scratch, &camera, glam::uvec2(32, 32)));
    let px: Vec<u32> = bitmap.read_vec();
    // Rows are padded to COPY_BYTES_PER_ROW_ALIGNMENT — index via the
    // bitmap's real row stride, not the 32px image width.
    let stride = bitmap.shape[1];
    println!(
        "rendered {} pixels, center: {:#x}",
        px.len(),
        px[16 * stride + 16]
    );
}
