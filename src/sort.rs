//! GPU radix sort.
//!
//! References:
//! - <https://github.com/ArthurBrussee/brush/blob/main/crates/brush-sort/src/lib.rs>
use crate::tensor::{GpuTensor, cube_count_1d};
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

const SORT_WG: u32 = 32;
const SORT_BINS: u32 = 16;
const ELEMS_PER_THREAD: u32 = 32;
const SORT_BLOCK: u32 = SORT_WG * ELEMS_PER_THREAD;

// Shared memory fallback for WASM — cubecl's built-in plane ops generate
// subgroup ops which aren't available on WebGPU. Native uses the built-in.
#[cfg(target_arch = "wasm32")]
#[cube]
fn plane_exclusive_sum(value: u32) -> u32 {
    let mut lds = SharedMemory::<u32>::new(SORT_WG as usize);
    lds[UNIT_POS as usize] = value;
    sync_cube();

    let mut sum = 0u32;
    for i in 0u32..UNIT_POS {
        sum += lds[i as usize];
    }

    lds[UNIT_POS as usize] = sum;
    sync_cube();

    sum
}

#[cube(launch)]
fn count_kernel(
    num_wgs: u32,
    shift: u32,
    num_keys: u32,
    src: &Array<u32>,
    counts: &mut Array<u32>,
) {
    // Workgroup-shared histogram: each key is read exactly once, then atomically
    // bucketed into one of SORT_BINS counters — replacing a per-bin loop that
    // re-read every key SORT_BINS times.
    let histogram = SharedMemory::<Atomic<u32>>::new(SORT_BINS as usize);
    if UNIT_POS < SORT_BINS {
        histogram[UNIT_POS as usize].store(0u32);
    }
    sync_cube();

    // The grid may be spread over Y/Z when the X count exceeds the hardware
    // limit (cubecl's CubeCountSelection), so flatten the cube position
    // instead of addressing work by CUBE_POS_X alone.
    let wg = CUBE_POS_X + CUBE_COUNT_X * (CUBE_POS_Y + CUBE_COUNT_Y * CUBE_POS_Z);
    if wg < num_wgs {
        let base = SORT_BLOCK * wg + UNIT_POS;
        for e in 0..ELEMS_PER_THREAD {
            let idx = base + e * SORT_WG;
            if idx < num_keys {
                let bin = (src[idx as usize] >> shift) & 0xf;
                histogram[bin as usize].fetch_add(1u32);
            }
        }
    }
    sync_cube();

    if UNIT_POS < SORT_BINS && wg < num_wgs {
        counts[(UNIT_POS * num_wgs + wg) as usize] = histogram[UNIT_POS as usize].load();
    }
}

#[cube(launch)]
fn prefix_kernel(num_keys: u32, counts: &mut Array<u32>) {
    let num_wgs = num_keys.div_ceil(SORT_BLOCK);

    let mut bin_total = 0u32;
    if UNIT_POS < SORT_BINS {
        let offset = UNIT_POS * num_wgs;
        for wg in 0..num_wgs {
            bin_total += counts[(offset + wg) as usize];
        }
    }

    let global = plane_exclusive_sum(bin_total);

    if UNIT_POS < SORT_BINS {
        let offset = UNIT_POS * num_wgs;
        let mut prefix = global;
        for wg in 0..num_wgs {
            let count = counts[(offset + wg) as usize];
            counts[(offset + wg) as usize] = prefix;
            prefix += count;
        }
    }
}

#[cube(launch)]
fn scatter_kernel(
    num_wgs: u32,
    shift: u32,
    num_keys: u32,
    src: &Array<u32>,
    values: &Array<u32>,
    counts: &Array<u32>,
    out: &mut Array<u32>,
    out_values: &mut Array<u32>,
) {
    // See count_kernel: the cube position must be linearized to survive
    // CubeCountSelection spreading the grid over Y/Z.
    let wg = CUBE_POS_X + CUBE_COUNT_X * (CUBE_POS_Y + CUBE_COUNT_Y * CUBE_POS_Z);
    if wg >= num_wgs {
        terminate!();
    }

    let mut bin_offsets = SharedMemory::<u32>::new(SORT_BINS as usize);
    let histogram = SharedMemory::<Atomic<u32>>::new(SORT_BINS as usize);
    if UNIT_POS < SORT_BINS {
        bin_offsets[UNIT_POS as usize] = counts[(UNIT_POS * num_wgs + wg) as usize];
        histogram[UNIT_POS as usize].store(0u32);
    }
    sync_cube();

    let base = SORT_BLOCK * wg + UNIT_POS;
    for e in 0..ELEMS_PER_THREAD {
        let idx = base + e * SORT_WG;
        if idx < num_keys {
            let key = src[idx as usize];
            let val = values[idx as usize];
            let bin = (key >> shift) & 0xf;
            let rank = histogram[bin as usize].fetch_add(1u32);
            let pos = bin_offsets[bin as usize] + rank;
            out[pos as usize] = key;
            out_values[pos as usize] = val;
        }
    }
}

/// Bits needed to represent any value in `0..max_exclusive`.
pub(crate) fn bits_for(max_exclusive: u32) -> u32 {
    u32::BITS - max_exclusive.saturating_sub(1).leading_zeros()
}

pub fn radix_argsort(
    keys: GpuTensor,
    vals: GpuTensor,
    n: u32,
    bits: u32,
) -> (GpuTensor, GpuTensor) {
    if n <= 1 || bits == 0 {
        return (keys, vals);
    }
    let client = keys.client.clone();
    // The launched grid may exceed the hardware X limit and get spread over
    // Y/Z (CubeCountSelection), so kernels linearize the workgroup id.
    let num_wgs = n.div_ceil(SORT_BLOCK);
    let cube_count = cube_count_1d(&client, n, SORT_BLOCK);
    let cube_dim = CubeDim::new_1d(SORT_WG);

    let count_buf = GpuTensor::empty(&client, [(num_wgs * SORT_BINS) as usize]);
    let mut dst_keys = GpuTensor::empty(&client, [n as usize]);
    let mut dst_vals = GpuTensor::empty(&client, [n as usize]);

    let mut cur_keys = keys;
    let mut cur_vals = vals;

    for shift in (0..bits).step_by(4) {
        count_kernel::launch::<WgpuRuntime>(
            &client,
            cube_count.clone(),
            cube_dim,
            num_wgs,
            shift,
            n,
            cur_keys.as_array_arg(),
            count_buf.as_array_arg(),
        );

        prefix_kernel::launch::<WgpuRuntime>(
            &client,
            CubeCount::new_single(),
            cube_dim,
            n,
            count_buf.as_array_arg(),
        );

        scatter_kernel::launch::<WgpuRuntime>(
            &client,
            cube_count.clone(),
            cube_dim,
            num_wgs,
            shift,
            n,
            cur_keys.as_array_arg(),
            cur_vals.as_array_arg(),
            count_buf.as_array_arg(),
            dst_keys.as_array_arg(),
            dst_vals.as_array_arg(),
        );

        std::mem::swap(&mut cur_keys, &mut dst_keys);
        std::mem::swap(&mut cur_vals, &mut dst_vals);
    }
    (cur_keys, cur_vals)
}

#[cfg(test)]
mod radix_sort_tests {
    use super::*;
    use cubecl::client::ComputeClient;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    use rand::RngExt;

    fn argsort<T: Ord>(data: &[T]) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..data.len()).collect();
        indices.sort_by_key(|&i| &data[i]);
        indices
    }

    fn assert_argsort(client: &ComputeClient<WgpuRuntime>, keys_inp: &[u32], values_inp: &[u32]) {
        let keys = GpuTensor::from(client, [keys_inp.len()], keys_inp);
        let values = GpuTensor::from(client, [values_inp.len()], values_inp);
        let (ret_keys, ret_values) = radix_argsort(keys, values, keys_inp.len() as u32, 32);

        let ret_keys: Vec<u32> = ret_keys.read_vec();
        let ret_values: Vec<u32> = ret_values.read_vec();

        let inds = argsort(keys_inp);
        let ref_keys: Vec<u32> = inds.iter().map(|&i| keys_inp[i]).collect();
        let ref_values: Vec<u32> = inds.iter().map(|&i| values_inp[i]).collect();

        assert_eq!(ret_keys, ref_keys);
        assert_eq!(ret_values, ref_values);
    }

    fn assert_argsort_bits(
        client: &ComputeClient<WgpuRuntime>,
        keys_inp: &[u32],
        values_inp: &[u32],
        bits: u32,
    ) {
        let keys = GpuTensor::from(client, [keys_inp.len()], keys_inp);
        let values = GpuTensor::from(client, [values_inp.len()], values_inp);
        let (ret_keys, ret_values) = radix_argsort(keys, values, keys_inp.len() as u32, bits);
        let ret_keys: Vec<u32> = ret_keys.read_vec();
        let ret_values: Vec<u32> = ret_values.read_vec();

        assert_eq!(ret_keys.len(), keys_inp.len());
        assert_eq!(ret_values.len(), keys_inp.len());

        // The GPU radix sort is not stable within equal keys, so assert sorted
        // order and key/value pairing instead of an exact stable reference.
        for i in 1..keys_inp.len() {
            assert!(
                ret_keys[i - 1] <= ret_keys[i],
                "Keys not sorted at index {i}: {} > {}",
                ret_keys[i - 1],
                ret_keys[i]
            );
        }

        for i in 0..keys_inp.len() {
            let sorted_key = ret_keys[i];
            let original_idx = ret_values[i] as usize;
            assert_eq!(
                keys_inp[original_idx], sorted_key,
                "Value at index {i} points to wrong original index"
            );
        }
    }

    #[test]
    fn test_bits_for() {
        assert_eq!(bits_for(0), 0);
        assert_eq!(bits_for(1), 0);
        assert_eq!(bits_for(2), 1);
        assert_eq!(bits_for(3), 2);
        assert_eq!(bits_for(255), 8);
        assert_eq!(bits_for(256), 8);
        assert_eq!(bits_for(257), 9);
        assert_eq!(bits_for(65536), 16);
        assert_eq!(bits_for(u32::MAX), 32);
    }

    #[test]
    fn test_sorting_partial_bits() {
        let client = WgpuRuntime::client(&WgpuDevice::default());
        for max in [2u32, 64, 1000, 65536] {
            let bits = bits_for(max);
            let mut rng = rand::rng();
            let keys_inp: Vec<u32> = (0..5000).map(|_| rng.random_range(0..max)).collect();
            let values_inp: Vec<u32> = (0..5000).map(|i| i as u32).collect();
            assert_argsort_bits(&client, &keys_inp, &values_inp, bits);
        }
    }

    #[test]
    fn test_sorting() {
        let client = WgpuRuntime::client(&WgpuDevice::default());
        for i in 0..128u32 {
            let keys_inp = [
                5 + i * 4,
                i,
                6,
                123,
                74657,
                123,
                999,
                2u32.pow(24) + 123,
                6,
                7,
                8,
                0,
                i * 2,
                16 + i,
                128 * i,
            ];

            let values_inp: Vec<_> = keys_inp.iter().copied().map(|x| x * 2 + 5).collect();

            assert_argsort(&client, &keys_inp, &values_inp);
        }
    }

    #[test]
    fn test_sorting_big() {
        let client = WgpuRuntime::client(&WgpuDevice::default());
        let mut rng = rand::rng();
        let mut keys_inp = Vec::new();
        for i in 0..10000u32 {
            let start = rng.random_range(i..i + 150);
            let end = rng.random_range(start..start + 250);

            for j in start..end {
                if rng.random::<f32>() < 0.5 {
                    keys_inp.push(j);
                }
            }
        }

        let values_inp: Vec<_> = keys_inp.iter().map(|&x| x * 2 + 5).collect();
        assert_argsort(&client, &keys_inp, &values_inp);
    }

    #[test]
    fn test_sorting_large() {
        const NUM_ELEMENTS: usize = 500_000;

        let client = WgpuRuntime::client(&WgpuDevice::default());
        let mut rng = rand::rng();

        let keys_inp: Vec<u32> = (0..NUM_ELEMENTS)
            .map(|_| rng.random_range(0..1_000_000))
            .collect();
        let values_inp: Vec<u32> = (0..NUM_ELEMENTS).map(|i| i as u32).collect();

        let keys = GpuTensor::from(&client, [NUM_ELEMENTS], &keys_inp[..]);
        let values = GpuTensor::from(&client, [NUM_ELEMENTS], &values_inp[..]);
        let (ret_keys, ret_values) = radix_argsort(keys, values, NUM_ELEMENTS as u32, 32);

        let ret_keys: Vec<u32> = ret_keys.read_vec();
        let ret_values: Vec<u32> = ret_values.read_vec();

        assert_eq!(ret_keys.len(), NUM_ELEMENTS);
        assert_eq!(ret_values.len(), NUM_ELEMENTS);

        for i in 1..NUM_ELEMENTS {
            assert!(
                ret_keys[i - 1] <= ret_keys[i],
                "Keys not sorted at index {i}: {} > {}",
                ret_keys[i - 1],
                ret_keys[i]
            );
        }

        let check_indices = [0, 1000, 10_000, 100_000, 249_999];
        for &idx in &check_indices {
            let sorted_key = ret_keys[idx];
            let original_idx = ret_values[idx] as usize;
            assert_eq!(
                keys_inp[original_idx], sorted_key,
                "Value at index {idx} points to wrong original index"
            );
        }
    }

    #[test]
    fn test_scratch_sized_by_sorted_length() {
        let client = WgpuRuntime::client(&WgpuDevice::default());
        let cap = 1_000_000usize;
        let n = 1_000usize;
        let keys_inp: Vec<u32> = (0..cap)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761))
            .collect();
        let vals_inp: Vec<u32> = (0..cap).map(|i| i as u32).collect();
        let keys = GpuTensor::from(&client, [cap], &keys_inp[..]);
        let vals = GpuTensor::from(&client, [cap], &vals_inp[..]);

        let before = client.memory_usage().unwrap().bytes_in_use;
        let (sorted_keys, sorted_vals) = radix_argsort(keys, vals, n as u32, 32);
        let out: Vec<u32> = sorted_keys.read_vec(); // forces completion before measuring
        let after = client.memory_usage().unwrap().bytes_in_use;
        assert_eq!(out.len(), cap); // tensor shape unchanged; only first n are sorted

        // Scratch (dst pair + counts) must track n, not the tensor capacity.
        assert!(
            after - before < 256 * 1024,
            "scratch allocation {} bytes exceeds n-sized budget",
            after - before
        );
        let sv: Vec<u32> = sorted_vals.read_vec();
        let mut sorted = keys_inp[..n].to_vec();
        sorted.sort_unstable();
        assert_eq!(&out[..n], &sorted[..]);
        for (i, &idx) in sv[..n].iter().enumerate() {
            assert_eq!(
                keys_inp[idx as usize], out[i],
                "Value at index {i} points at the wrong key"
            );
        }
    }
}
