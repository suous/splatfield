use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use cubecl::Runtime;
use rand::RngExt;
use splatfield::sort::radix_argsort;
use splatfield::tensor::GpuTensor;
use std::time::Duration;

fn readback_u32(tensor: GpuTensor) -> Vec<u32> {
    let bytes = tensor.client.read_one_unchecked(tensor.handle);
    bytemuck::cast_slice::<u8, u32>(&bytes).to_vec()
}

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
    let device: cubecl::wgpu::WgpuDevice = Default::default();
    let client = cubecl::wgpu::WgpuRuntime::client(&device);

    let mut group = c.benchmark_group("radix_argsort/size");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(5));

    for &n in &[
        1_000, 10_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000,
    ] {
        let keys_inp: Vec<u32> = (0..n).map(|_| rand::rng().random::<u32>()).collect();
        let vals_inp: Vec<u32> = (0..n as u32).collect();

        let keys_data = keys_inp.clone();
        let vals_data = vals_inp.clone();

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let k = GpuTensor::from(&client, [n], &keys_data);
                let v = GpuTensor::from(&client, [n], &vals_data);
                let (sorted_k, sorted_v) = radix_argsort(black_box(k), black_box(v), n as u32, 32);
                readback_u32(sorted_k);
                readback_u32(sorted_v);
            });
        });
    }
    group.finish();
}

fn bench_bits_sweep(c: &mut Criterion) {
    let device: cubecl::wgpu::WgpuDevice = Default::default();
    let client = cubecl::wgpu::WgpuRuntime::client(&device);

    let mut group = c.benchmark_group("radix_argsort/bits");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(5));

    let n: usize = 1_000_000;
    let keys_data: Vec<u32> = (0..n).map(|_| rand::rng().random::<u32>()).collect();
    let vals_data: Vec<u32> = (0..n as u32).collect();

    for &bits in &[4u32, 8, 12, 16, 20, 24, 28, 32] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(bits), &bits, |b, &bits| {
            b.iter(|| {
                let k = GpuTensor::from(&client, [n], &keys_data);
                let v = GpuTensor::from(&client, [n], &vals_data);
                let (sorted_k, sorted_v) =
                    radix_argsort(black_box(k), black_box(v), n as u32, bits);
                readback_u32(sorted_k);
                readback_u32(sorted_v);
            });
        });
    }
    group.finish();
}

fn bench_distribution(c: &mut Criterion) {
    let device: cubecl::wgpu::WgpuDevice = Default::default();
    let client = cubecl::wgpu::WgpuRuntime::client(&device);

    let mut group = c.benchmark_group("radix_argsort/distribution");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(5));

    let n: usize = 1_000_000;

    for &dist in &["random", "sequential", "reverse"] {
        let keys_data = make_data(n, dist);
        let vals_data: Vec<u32> = (0..n as u32).collect();

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(dist),
            &keys_data,
            |b, keys_data| {
                b.iter(|| {
                    let k = GpuTensor::from(&client, [n], keys_data);
                    let v = GpuTensor::from(&client, [n], &vals_data);
                    let (sorted_k, sorted_v) =
                        radix_argsort(black_box(k), black_box(v), n as u32, 32);
                    readback_u32(sorted_k);
                    readback_u32(sorted_v);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_size_sweep,
    bench_bits_sweep,
    bench_distribution
);
criterion_main!(benches);
