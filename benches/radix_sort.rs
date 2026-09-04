use criterion::measurement::WallTime;
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
};
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::wgpu::WgpuRuntime;
use rand::RngExt;
use splatfield::sort::{RadixScratch, radix_argsort_with};
use splatfield::tensor::GpuTensor;
use std::hint::black_box;
use std::time::Duration;

const SIZES: &[usize] = &[
    1_000, 10_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000,
];

fn make_data(n: usize, dist: &str) -> Vec<u32> {
    let mut rng = rand::rng();
    match dist {
        "random" => (0..n).map(|_| rng.random::<u32>()).collect(),
        "sequential" => (0..n as u32).collect(),
        "reverse" => (0..n).map(|i| n as u32 - 1 - i as u32).collect(),
        _ => panic!("unknown distribution: {dist}"),
    }
}

/// Keys (distribution `dist`), identity values, a sync flag, and scratch —
/// the setup every sort bench shares.
fn sort_fixture(
    client: &ComputeClient<WgpuRuntime>,
    n: usize,
    dist: &str,
) -> (GpuTensor, GpuTensor, GpuTensor, RadixScratch) {
    let keys = GpuTensor::from(client, [n], make_data(n, dist));
    let vals = GpuTensor::from(client, [n], (0..n as u32).collect::<Vec<_>>());
    let flag = GpuTensor::from(client, [2], &[0u32, 0][..]);
    (keys, vals, flag, RadixScratch::new(client, n))
}

fn bench_group<'a>(c: &'a mut Criterion, name: &str) -> BenchmarkGroup<'a, WallTime> {
    let mut group = c.benchmark_group(name);
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(5));
    group
}

fn bench_sort(
    group: &mut BenchmarkGroup<WallTime>,
    id: BenchmarkId,
    keys: &GpuTensor,
    vals: &GpuTensor,
    flag: &GpuTensor,
    scratch: &RadixScratch,
    bits: u32,
) {
    let n = keys.shape[0];
    group.throughput(Throughput::Elements(n as u64));
    // Each iteration feeds the previous sort's output back in, so scratch
    // must alternate: after an odd pass count the output aliases the scratch
    // dst buffers, and with exclusive-memory-only bindings a buffer can't be
    // src (read-only) and out (read-write) in the same dispatch.
    let scratch_b = RadixScratch::new(&keys.client, n);
    group.bench_with_input(id, &(), |b, _| {
        b.iter_custom(|iters| {
            let mut k = keys.clone();
            let mut v = vals.clone();
            let start = std::time::Instant::now();
            for i in 0..iters {
                let s = if i % 2 == 0 { scratch } else { &scratch_b };
                let (nk, nv) = radix_argsort_with(black_box(&k), black_box(&v), n as u32, bits, s);
                k = nk;
                v = nv;
            }
            black_box(flag.read_vec::<u32>()); // one sync per measurement
            start.elapsed()
        });
    });
}

fn bench_size_sweep(c: &mut Criterion) {
    let client = cubecl::wgpu::WgpuRuntime::client(&Default::default());
    let mut group = bench_group(c, "radix_argsort/kernel");

    for &n in SIZES {
        let (keys, vals, flag, scratch) = sort_fixture(&client, n, "random");
        bench_sort(
            &mut group,
            BenchmarkId::from_parameter(n),
            &keys,
            &vals,
            &flag,
            &scratch,
            32,
        );
    }
    group.finish();
}

fn bench_bits_sweep(c: &mut Criterion) {
    let client = cubecl::wgpu::WgpuRuntime::client(&Default::default());
    let mut group = bench_group(c, "radix_argsort/bits");

    let n: usize = 1_000_000;
    let (keys, vals, flag, scratch) = sort_fixture(&client, n, "random");

    for &bits in &[4u32, 8, 12, 16, 20, 24, 28, 32] {
        bench_sort(
            &mut group,
            BenchmarkId::from_parameter(bits),
            &keys,
            &vals,
            &flag,
            &scratch,
            bits,
        );
    }
    group.finish();
}

fn bench_distribution(c: &mut Criterion) {
    let client = cubecl::wgpu::WgpuRuntime::client(&Default::default());
    let mut group = bench_group(c, "radix_argsort/distribution");

    let n: usize = 1_000_000;

    for &dist in &["random", "sequential", "reverse"] {
        let (keys, vals, flag, scratch) = sort_fixture(&client, n, dist);
        bench_sort(
            &mut group,
            BenchmarkId::from_parameter(dist),
            &keys,
            &vals,
            &flag,
            &scratch,
            32,
        );
    }
    group.finish();
}

fn bench_end_to_end(c: &mut Criterion) {
    let client = cubecl::wgpu::WgpuRuntime::client(&Default::default());
    let mut group = bench_group(c, "radix_argsort/end_to_end");

    for &n in SIZES {
        let keys_data = make_data(n, "random");
        let vals_data: Vec<u32> = (0..n as u32).collect();
        let scratch = RadixScratch::new(&client, n);

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let k = GpuTensor::from(&client, [n], &keys_data[..]);
                let v = GpuTensor::from(&client, [n], &vals_data[..]);
                let (sorted_k, sorted_v) =
                    radix_argsort_with(black_box(&k), black_box(&v), n as u32, 32, &scratch);
                sorted_k.read_vec::<u32>();
                sorted_v.read_vec::<u32>();
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_size_sweep,
    bench_bits_sweep,
    bench_distribution,
    bench_end_to_end
);
criterion_main!(benches);
