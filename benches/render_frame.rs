use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use cubecl::Runtime;
use splatfield::camera::Camera;
use splatfield::render::{RenderScratch, Splats};
use splatfield::sog;
use std::time::{Duration, Instant};

fn bench_frame(c: &mut Criterion) {
    let Some((splats, n)) = load_bear() else {
        eprintln!("skipping: data/bear.3d71a266_sh2.sog not found");
        return;
    };
    let client = splats.attributes.client.clone();
    let mut group = c.benchmark_group("render/frame");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(5));

    for &(w, h) in &[(1280u32, 720u32), (1920, 1080)] {
        let camera = fixed_camera(&splats);
        let mut scratch = RenderScratch::new(&client, n, glam::uvec2(w, h));
        group.throughput(Throughput::Elements((w * h) as u64));
        group.bench_function(format!("{w}x{h}"), |b| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    // render_with syncs internally via the counters readback,
                    // so no extra per-iteration sync is needed.
                    black_box(pollster::block_on(splats.render_with(
                        &mut scratch,
                        &camera,
                        glam::uvec2(w, h),
                    )));
                }
                start.elapsed()
            })
        });
    }
    group.finish();
}

fn load_bear() -> Option<(Splats, usize)> {
    let path = std::path::Path::new("data/bear.3d71a266_sh2.sog");
    if !path.exists() {
        return None;
    }
    let client = cubecl::wgpu::WgpuRuntime::client(&Default::default());
    let splats = sog::load_sog(std::fs::File::open(path).ok()?, &client).ok()?;
    let n = splats.attributes.shape[0];
    Some((splats, n))
}

/// Centered, pulled back along -Y, pitched -90° around X (`Camera::frame_bounds`), fov 0.8.
fn fixed_camera(splats: &Splats) -> Camera {
    let mut camera = Camera {
        fov: glam::Vec2::splat(0.8),
        position: glam::Vec3::ZERO,
        rotation: glam::Quat::IDENTITY,
    };
    camera.frame_bounds(splats.bounds);
    camera
}

criterion_group!(benches, bench_frame);
criterion_main!(benches);
