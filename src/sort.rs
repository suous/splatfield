//! GPU radix sort.
//!
//! References:
//! - <https://github.com/ArthurBrussee/brush/blob/main/crates/brush-sort/src/lib.rs>
use crate::tensor::{GpuTensor, U32, cube_count_1d, empty_tensor};
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

const SORT_WG: u32 = 32;
const SORT_BINS: u32 = 16;
const ELEMS_PER_THREAD: u32 = 32;
const SORT_BLOCK: u32 = SORT_WG * ELEMS_PER_THREAD;

// Custom plane functions using shared memory — cubecl's built-in plane_sum/
// plane_exclusive_sum generate subgroup ops which aren't available on WASM.
#[cube]
fn plane_sum(value: u32) -> u32 {
    let mut lds = SharedMemory::<u32>::new(SORT_WG as usize);
    lds[UNIT_POS as usize] = value;
    sync_cube();

    let mut stride = 16u32;
    for _ in 0..5 {
        if UNIT_POS < stride {
            lds[UNIT_POS as usize] += lds[(UNIT_POS + stride) as usize];
        }
        stride >>= 1;
        sync_cube();
    }

    lds[0]
}

#[cube]
fn plane_exclusive_sum(value: u32) -> u32 {
    let mut lds = SharedMemory::<u32>::new(SORT_WG as usize);
    lds[UNIT_POS as usize] = value;
    sync_cube();

    let mut sum = 0u32;
    for i in 0u32..UNIT_POS {
        sum += lds[i as usize];
    }
    sync_cube();

    lds[UNIT_POS as usize] = sum;
    sync_cube();

    sum
}

#[cube(launch)]
fn count_kernel(shift: u32, num_keys: u32, src: &Array<u32>, counts: &mut Array<u32>) {
    let num_wgs = num_keys.div_ceil(SORT_BLOCK);
    let base = SORT_BLOCK * CUBE_POS_X + UNIT_POS;

    if CUBE_POS_X < num_wgs {
        for bin in 0..SORT_BINS {
            let mut local_count = 0u32;
            for e in 0..ELEMS_PER_THREAD {
                let idx = base + e * SORT_WG;
                if idx < num_keys && (src[idx as usize] >> shift) & 0xf == bin {
                    local_count += 1;
                }
            }
            let total = plane_sum(local_count);
            if UNIT_POS == 0 {
                counts[(bin * num_wgs + CUBE_POS_X) as usize] = total;
            }
        }
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
    shift: u32,
    num_keys: u32,
    src: &Array<u32>,
    values: &Array<u32>,
    counts: &Array<u32>,
    out: &mut Array<u32>,
    out_values: &mut Array<u32>,
) {
    let num_wgs = num_keys.div_ceil(SORT_BLOCK);

    let mut bin_offsets = SharedMemory::<u32>::new(SORT_BINS as usize);
    let histogram = SharedMemory::<Atomic<u32>>::new(SORT_BINS as usize);
    let base = SORT_BLOCK * CUBE_POS_X + UNIT_POS;

    if CUBE_POS_X < num_wgs {
        if UNIT_POS < SORT_BINS {
            bin_offsets[UNIT_POS as usize] = counts[(UNIT_POS * num_wgs + CUBE_POS_X) as usize];
            histogram[UNIT_POS as usize].store(0u32);
        }
        sync_cube();

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
}

pub fn radix_argsort(
    keys: GpuTensor,
    vals: GpuTensor,
    n: u32,
    bits: u32,
) -> (GpuTensor, GpuTensor) {
    let client = keys.client.clone();
    let device = keys.device.clone();
    let max_n = keys.shape[0] as u32;
    let max_wgs = max_n.div_ceil(SORT_BLOCK);

    let num_wgs = cube_count_1d(&client, n, SORT_BLOCK);
    let cube_dim = CubeDim::new_1d(SORT_WG);

    let mut cur_keys = keys;
    let mut cur_vals = vals;
    let count_buf = empty_tensor([(max_wgs as usize) * SORT_BINS as usize], &device, U32);
    let mut dst_keys = empty_tensor([max_n as usize], &device, cur_keys.dtype);
    let mut dst_vals = empty_tensor([max_n as usize], &device, cur_vals.dtype);

    for pass in 0..bits.div_ceil(4) {
        let shift = pass * 4;

        count_kernel::launch::<WgpuRuntime>(
            &client,
            num_wgs.clone(),
            cube_dim,
            shift,
            n,
            cur_keys.as_array_arg(),
            count_buf.as_array_arg(),
        );

        prefix_kernel::launch::<WgpuRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            cube_dim,
            n,
            count_buf.as_array_arg(),
        );

        scatter_kernel::launch::<WgpuRuntime>(
            &client,
            num_wgs.clone(),
            cube_dim,
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
    use crate::tensor::create_tensor_from_data;
    use cubecl::wgpu::WgpuDevice;
    use rand::RngExt;

    fn tensor_to_vec<T: bytemuck::Pod>(tensor: GpuTensor) -> Vec<T> {
        let bytes = tensor.client.read_one_unchecked(tensor.handle);
        bytemuck::cast_slice::<u8, T>(&bytes).to_vec()
    }

    fn argsort<T: Ord>(data: &[T]) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..data.len()).collect();
        indices.sort_unstable_by_key(|&i| &data[i]);
        indices
    }

    #[test]
    fn test_sorting() {
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

            let device: WgpuDevice = Default::default();
            let keys = create_tensor_from_data([keys_inp.len()], &device, U32, &keys_inp);
            let values = create_tensor_from_data([values_inp.len()], &device, U32, &values_inp);
            let (ret_keys, ret_values) = radix_argsort(keys, values, keys_inp.len() as u32, 32);

            let ret_keys = tensor_to_vec::<u32>(ret_keys);
            let ret_values = tensor_to_vec::<u32>(ret_values);

            let inds = argsort(&keys_inp);

            let ref_keys: Vec<u32> = inds.iter().map(|&i| keys_inp[i]).collect();
            let ref_values: Vec<u32> = inds.iter().map(|&i| values_inp[i]).collect();

            for (((key, val), ref_key), ref_val) in ret_keys
                .iter()
                .zip(ret_values.iter())
                .zip(ref_keys)
                .zip(ref_values)
            {
                assert_eq!(*key, ref_key);
                assert_eq!(*val, ref_val);
            }
        }
    }

    #[test]
    fn test_sorting_big() {
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

        let device: WgpuDevice = Default::default();
        let keys = create_tensor_from_data([keys_inp.len()], &device, U32, &keys_inp);
        let values = create_tensor_from_data([values_inp.len()], &device, U32, &values_inp);
        let (ret_keys, ret_values) = radix_argsort(keys, values, keys_inp.len() as u32, 32);

        let ret_keys = tensor_to_vec::<u32>(ret_keys);
        let ret_values = tensor_to_vec::<u32>(ret_values);

        let inds = argsort(&keys_inp);
        let ref_keys: Vec<u32> = inds.iter().map(|&i| keys_inp[i]).collect();
        let ref_values: Vec<u32> = inds.iter().map(|&i| values_inp[i]).collect();

        for (((key, val), ref_key), ref_val) in ret_keys
            .iter()
            .zip(ret_values.iter())
            .zip(ref_keys)
            .zip(ref_values)
        {
            assert_eq!(*key, ref_key);
            assert_eq!(*val, ref_val);
        }
    }

    #[test]
    fn test_sorting_large() {
        const NUM_ELEMENTS: usize = 500_000;

        let mut rng = rand::rng();

        let keys_inp: Vec<u32> = (0..NUM_ELEMENTS)
            .map(|_| rng.random_range(0..1_000_000))
            .collect();
        let values_inp: Vec<u32> = (0..NUM_ELEMENTS).map(|i| i as u32).collect();

        let device: WgpuDevice = Default::default();
        let keys = create_tensor_from_data([NUM_ELEMENTS], &device, U32, &keys_inp);
        let values = create_tensor_from_data([NUM_ELEMENTS], &device, U32, &values_inp);
        let (ret_keys, ret_values) = radix_argsort(keys, values, NUM_ELEMENTS as u32, 32);

        let ret_keys = tensor_to_vec::<u32>(ret_keys);
        let ret_values = tensor_to_vec::<u32>(ret_values);

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
}
