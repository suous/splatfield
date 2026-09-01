use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use cubecl::Runtime;
use rand::RngExt;
use splatfield::sort::radix_argsort;
use splatfield::tensor::GpuTensor;
use std::time::Duration;

fn make_data(n: usize, dist: &str) -> Vec<u32> {
    let mut rng = rand::rng();
    match dist {
        "random" => (0..n).map(|_| rng.random::<u32>()).collect(),
        "sequential" => (0..n as u32).collect(),
        "reverse" => (0..n).map(|i| n as u32 - 1 - i as u32).collect(),
        _ => panic!("unknown distribution: {dist}"),
    }
}

fn bench_size_sweep(c: &mut Criterion) {
    let client = cubecl::wgpu::WgpuRuntime::client(&Default::default());
    let mut group = c.benchmark_group("radix_argsort/kernel");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(5));

    for &n in &[
        1_000, 10_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000,
    ] {
        let keys_data = make_data(n, "random");
        let vals_data: Vec<u32> = (0..n as u32).collect();
        let keys = GpuTensor::from(&client, [n], &keys_data[..]);
        let vals = GpuTensor::from(&client, [n], &vals_data[..]);
        let flag = GpuTensor::from(&client, [2], &[0u32, 0][..]);

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter_custom(|iters| {
                let mut k = keys.clone();
                let mut v = vals.clone();
                let start = std::time::Instant::now();
                for _ in 0..iters {
                    let (nk, nv) = radix_argsort(black_box(k), black_box(v), n as u32, 32);
                    k = nk;
                    v = nv;
                }
                black_box(flag.read_vec::<u32>()); // one sync per measurement
                start.elapsed()
            });
        });
    }
    group.finish();
}

fn bench_bits_sweep(c: &mut Criterion) {
    let client = cubecl::wgpu::WgpuRuntime::client(&Default::default());
    let mut group = c.benchmark_group("radix_argsort/bits");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(5));

    let n: usize = 1_000_000;
    let keys_data = make_data(n, "random");
    let vals_data: Vec<u32> = (0..n as u32).collect();
    let keys = GpuTensor::from(&client, [n], &keys_data[..]);
    let vals = GpuTensor::from(&client, [n], &vals_data[..]);
    let flag = GpuTensor::from(&client, [2], &[0u32, 0][..]);

    for &bits in &[4u32, 8, 12, 16, 20, 24, 28, 32] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(bits), &bits, |b, _| {
            b.iter_custom(|iters| {
                let mut k = keys.clone();
                let mut v = vals.clone();
                let start = std::time::Instant::now();
                for _ in 0..iters {
                    let (nk, nv) = radix_argsort(black_box(k), black_box(v), n as u32, bits);
                    k = nk;
                    v = nv;
                }
                black_box(flag.read_vec::<u32>()); // one sync per measurement
                start.elapsed()
            });
        });
    }
    group.finish();
}

fn bench_distribution(c: &mut Criterion) {
    let client = cubecl::wgpu::WgpuRuntime::client(&Default::default());
    let mut group = c.benchmark_group("radix_argsort/distribution");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(5));

    let n: usize = 1_000_000;

    for &dist in &["random", "sequential", "reverse"] {
        let keys_data = make_data(n, dist);
        let vals_data: Vec<u32> = (0..n as u32).collect();
        let keys = GpuTensor::from(&client, [n], &keys_data[..]);
        let vals = GpuTensor::from(&client, [n], &vals_data[..]);
        let flag = GpuTensor::from(&client, [2], &[0u32, 0][..]);

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(dist), &dist, |b, _| {
            b.iter_custom(|iters| {
                let mut k = keys.clone();
                let mut v = vals.clone();
                let start = std::time::Instant::now();
                for _ in 0..iters {
                    let (nk, nv) = radix_argsort(black_box(k), black_box(v), n as u32, 32);
                    k = nk;
                    v = nv;
                }
                black_box(flag.read_vec::<u32>()); // one sync per measurement
                start.elapsed()
            });
        });
    }
    group.finish();
}

fn bench_end_to_end(c: &mut Criterion) {
    let client = cubecl::wgpu::WgpuRuntime::client(&Default::default());
    let mut group = c.benchmark_group("radix_argsort/end_to_end");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(5));

    for &n in &[
        1_000, 10_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000,
    ] {
        let keys_data = make_data(n, "random");
        let vals_data: Vec<u32> = (0..n as u32).collect();

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let k = GpuTensor::from(&client, [n], &keys_data[..]);
                let v = GpuTensor::from(&client, [n], &vals_data[..]);
                let (sorted_k, sorted_v) = radix_argsort(black_box(k), black_box(v), n as u32, 32);
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
