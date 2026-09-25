use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use splatfield::camera::Camera;
use splatfield::render::{RenderScratch, Splats};
use std::hint::black_box;
use std::time::Duration;

fn bench_frame(c: &mut Criterion) {
    let Some(splats) = load_bear() else {
        eprintln!("skipping: data/bear.3d71a266_sh2.sog not found");
        return;
    };
    let client = splats.attributes.client.clone();
    let n = splats.attributes.shape[0];
    let mut group = c.benchmark_group("render/frame");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(5));

    for &(w, h) in &[(1280u32, 720u32), (1920, 1080)] {
        let mut camera = Camera::default();
        camera.frame_bounds(splats.bounds);
        let mut scratch = RenderScratch::new(&client, n, glam::uvec2(w, h));
        group.throughput(Throughput::Elements((w * h) as u64));
        group.bench_function(format!("{w}x{h}"), |b| {
            // render_with syncs internally via the counters readback, so
            // criterion's default loop needs no extra per-iteration sync.
            b.iter(|| black_box(splats.render_with(&mut scratch, &camera, glam::uvec2(w, h))));
        });
    }
    group.finish();
}

/// The segmentation finalize path: per-candidate responsibility renders and
/// EIG scoring — the kernels the RGB bench never touches.
fn bench_seg_finalize(c: &mut Criterion) {
    let Some(splats) = load_bear() else {
        eprintln!("skipping: data/bear.3d71a266_sh2.sog not found");
        return;
    };
    let client = splats.attributes.client.clone();
    let n = splats.attributes.shape[0];
    let mut group = c.benchmark_group("seg/finalize");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(5));

    let (w, h) = (1920u32, 1080u32);
    let mut camera = Camera::default();
    camera.frame_bounds(splats.bounds);
    let mut scratch = RenderScratch::new(&client, n, glam::uvec2(w, h));
    let mut acc = splatfield::seg::Accumulators::new(&client, n);
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("responsibility/1920x1080", |b| {
        b.iter(|| splats.render_responsibility(&mut scratch, &mut acc, &camera, glam::uvec2(w, h)))
    });

    // One responsibility render outside the timed loop; EIG scoring consumes
    // it per iteration — one candidate-scoring round's shape.
    let state = splatfield::seg::beta::BetaState::new_uniform(&client, n);
    splats.render_responsibility(&mut scratch, &mut acc, &camera, glam::uvec2(w, h));
    group.bench_function("eig/1920x1080", |b| b.iter(|| black_box(state.eig(&acc))));
    group.finish();
}

/// The full active loop end to end — project→sort→EIG scoring, oracle call,
/// evidence lift, posterior update — on a synthetic scene (the shared seg
/// fixtures, scaled up for a GPU-meaningful load), so it runs on any
/// checkout with no model or asset dependency.
fn bench_seg_round(c: &mut Criterion) {
    use splatfield::render::{jitter, sample_opaque_attributes, target};
    use splatfield::seg::active::{Config, Segmenter};

    let client = cubecl::Device::default().client();
    let n = 20_000;
    let splats = std::sync::Arc::new(
        splatfield::render::CpuSplats {
            attributes: sample_opaque_attributes(n, |i| {
                glam::vec3(1.2, 0.0, 0.0) + 0.4 * jitter(i)
            }),
            sh_coeffs: vec![0.0; n * 3],
        }
        .upload(&client),
    );
    let mut camera = Camera::default();
    camera.frame_bounds(splats.bounds);
    let mut oracle = target(glam::vec3(1.2, 0.0, 0.0), 0.30);

    let rounds = 3usize;
    let mut group = c.benchmark_group("seg/round");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(4));
    group.throughput(Throughput::Elements(n as u64 * rounds as u64));
    group.bench_function("e2e_3rounds_256p_12cand", |b| {
        b.iter_batched(
            || {
                Segmenter::new(
                    splats.clone(),
                    Config {
                        resolution: 256,
                        candidates: 12,
                        iterations: rounds,
                    },
                    camera,
                )
                .unwrap()
            },
            |mut segmenter| black_box(segmenter.run_with(&mut oracle, |_, _| true).unwrap()),
            criterion::BatchSize::PerIteration,
        )
    });
    group.finish();
}

fn load_bear() -> Option<Splats> {
    let path = std::path::Path::new("data/bear.3d71a266_sh2.sog");
    if !path.exists() {
        return None;
    }
    let client = cubecl::Device::default().client();
    Some(
        splatfield::load_scene(
            path.extension().unwrap_or_default(),
            std::fs::File::open(path).ok()?,
        )
        .ok()?
        .upload(&client),
    )
}

criterion_group!(benches, bench_frame, bench_seg_finalize, bench_seg_round);
criterion_main!(benches);
