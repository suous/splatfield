use criterion::measurement::WallTime;
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
};
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::wgpu::WgpuRuntime;
use rand::RngExt;
use splat_sort::sort::RadixScratch;
use splat_sort::tensor::GpuTensor;
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
        // Depth-like keys: positive f32 bit patterns in a bounded z range,
        // so the high digits are near-constant like the render pipeline's.
        "depth" => (0..n)
            .map(|_| (rng.random::<f32>() * 999.9 + 0.1).to_bits())
            .collect(),
        _ => panic!("unknown distribution: {dist}"),
    }
}

/// Keys (distribution `dist`), identity values, and scratch — the setup every
/// sort bench shares.
fn sort_fixture(
    client: &ComputeClient<WgpuRuntime>,
    n: usize,
    dist: &str,
) -> (GpuTensor, GpuTensor, RadixScratch) {
    let keys = GpuTensor::from(client, [n], make_data(n, dist));
    let vals = GpuTensor::from(client, [n], (0..n as u32).collect::<Vec<_>>());
    (keys, vals, RadixScratch::new(client, n))
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
    scratch: &RadixScratch,
    bits: u32,
    write_keys: bool,
) {
    let n = keys.shape[0];
    group.throughput(Throughput::Elements(n as u64));
    let flag = GpuTensor::from(&keys.client, [2], &[0u32, 0][..]);
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
                let (nk, nv) = s.argsort(black_box(&k), black_box(&v), n as u32, bits, write_keys);
                k = nk;
                v = nv;
            }
            black_box(flag.read_vec::<u32>()); // one sync per measurement
            start.elapsed()
        });
    });
}

/// One criterion group over a shared client; each case is
/// (benchmark id, element count, key distribution, key bits).
fn bench_sweep(
    c: &mut Criterion,
    name: &str,
    cases: Vec<(BenchmarkId, usize, &'static str, u32)>,
    write_keys: bool,
) {
    let client = cubecl::wgpu::WgpuRuntime::client(&Default::default());
    let mut group = bench_group(c, name);
    for (id, n, dist, bits) in &cases {
        let (keys, vals, scratch) = sort_fixture(&client, *n, dist);
        bench_sort(
            &mut group,
            id.clone(),
            &keys,
            &vals,
            &scratch,
            *bits,
            write_keys,
        );
    }
    group.finish();
}

fn bench_size_sweep(c: &mut Criterion) {
    bench_sweep(
        c,
        "radix_argsort/kernel",
        SIZES
            .iter()
            .map(|&n| (BenchmarkId::from_parameter(n), n, "random", 32))
            .collect(),
        true,
    );
}

fn bench_bits_sweep(c: &mut Criterion) {
    bench_sweep(
        c,
        "radix_argsort/bits",
        [4u32, 8, 12, 16, 20, 24, 28, 32]
            .map(|bits| (BenchmarkId::from_parameter(bits), 1_000_000, "random", bits))
            .to_vec(),
        true,
    );
}

fn bench_distribution(c: &mut Criterion) {
    bench_sweep(
        c,
        "radix_argsort/distribution",
        ["random", "sequential", "reverse"]
            .map(|d| (BenchmarkId::from_parameter(d), 1_000_000, d, 32))
            .to_vec(),
        true,
    );
}

/// Values-only mode at depth-like keys — the render pipeline's depth sort
/// (write_keys=false), whose per-pass traffic the key-writing sweeps miss.
fn bench_values_only(c: &mut Criterion) {
    bench_sweep(
        c,
        "radix_argsort/values_only",
        [1_000_000usize, 5_000_000]
            .map(|n| (BenchmarkId::from_parameter(n), n, "depth", 32))
            .to_vec(),
        false,
    );
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
                    scratch.argsort(black_box(&k), black_box(&v), n as u32, 32, true);
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
    bench_values_only,
    bench_end_to_end
);
criterion_main!(benches);
