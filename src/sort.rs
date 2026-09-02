//! GPU radix sort.
//!
//! References:
//! - <https://github.com/ArthurBrussee/brush/blob/main/crates/brush-sort/src/lib.rs>
use crate::tensor::GpuTensor;
use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

const SORT_WG: u32 = 128;
const SORT_BINS: u32 = 16;
const ELEMS_PER_THREAD: u32 = 8;
const SORT_BLOCK: u32 = SORT_WG * ELEMS_PER_THREAD;
// scatter scan: SORT_WG lanes are scanned by SORT_BINS x SCAN_GROUPS threads
// (group sums -> group offsets -> apply), shortening the serial chain.
const SCAN_GROUPS: u32 = 8;
const SCAN_CHUNK: u32 = SORT_WG / SCAN_GROUPS;

// Shared memory fallback for WASM — cubecl's built-in plane ops generate
// subgroup ops which aren't available on WebGPU. Native uses the built-in.
#[cfg(target_arch = "wasm32")]
#[cube]
fn plane_exclusive_sum(value: u32) -> u32 {
    let mut lds = Shared::<[u32]>::new_slice(SORT_WG as usize);
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
fn count_kernel(num_wgs: u32, shift: u32, num_keys: u32, src: &[u32], counts: &mut [u32]) {
    // Workgroup-shared histogram: each key is read exactly once, then atomically
    // bucketed into one of SORT_BINS counters — replacing a per-bin loop that
    // re-read every key SORT_BINS times.
    let histogram = Shared::<[Atomic<u32>]>::new_slice(SORT_BINS as usize);
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
fn prefix_kernel(num_keys: u32, counts: &mut [u32]) {
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

/// Parallel variant of prefix_kernel for large grids: each bin's
/// per-workgroup counts are scanned by SCAN_PARTS threads (chunk totals ->
/// chunk offsets -> apply), shortening the serial chain 32x.
const SCAN_PARTS: u32 = 32;
const PREFIX_WG: u32 = SORT_BINS * SCAN_PARTS;

#[cube(launch)]
fn prefix_kernel_parallel(num_keys: u32, counts: &mut [u32]) {
    let num_wgs = num_keys.div_ceil(SORT_BLOCK);
    let bin = UNIT_POS / SCAN_PARTS;
    let part = UNIT_POS % SCAN_PARTS;
    let mut partials = Shared::<[u32]>::new_slice(PREFIX_WG as usize);
    let mut totals = Shared::<[u32]>::new_slice(SORT_BINS as usize);

    let chunk = num_wgs.div_ceil(SCAN_PARTS);
    let lo = part * chunk;
    let hi = (lo + chunk).min(num_wgs);

    let mut sum = 0u32;
    for w in lo..hi {
        sum += counts[(bin * num_wgs + w) as usize];
    }
    partials[UNIT_POS as usize] = sum;
    if UNIT_POS < SORT_BINS {
        totals[UNIT_POS as usize] = 0u32;
    }
    sync_cube();

    // Per-bin totals, then exclusive bin offsets (single thread, 16 bins).
    if UNIT_POS < SORT_BINS {
        let mut total = 0u32;
        for p in 0..SCAN_PARTS {
            total += partials[(UNIT_POS * SCAN_PARTS + p) as usize];
        }
        totals[UNIT_POS as usize] = total;
    }
    sync_cube();
    if UNIT_POS == 0 {
        let mut running = 0u32;
        for b in 0..SORT_BINS {
            let t = totals[b as usize];
            totals[b as usize] = running;
            running += t;
        }
    }
    sync_cube();

    // Exclusive scan of chunk totals within each bin, offset by the bin base.
    if UNIT_POS < SORT_BINS {
        let mut running = totals[UNIT_POS as usize];
        for p in 0..SCAN_PARTS {
            let cell = (UNIT_POS * SCAN_PARTS + p) as usize;
            let c = partials[cell];
            partials[cell] = running;
            running += c;
        }
    }
    sync_cube();

    // Apply: rewrite each per-workgroup count as its global exclusive offset.
    let mut running = partials[UNIT_POS as usize];
    for w in lo..hi {
        let idx = (bin * num_wgs + w) as usize;
        let c = counts[idx];
        counts[idx] = running;
        running += c;
    }
}

#[cube(launch)]
fn scatter_kernel(
    num_wgs: u32,
    shift: u32,
    num_keys: u32,
    src: &[u32],
    values: &[u32],
    counts: &[u32],
    out: &mut [u32],
    out_values: &mut [u32],
) {
    // See count_kernel: the cube position must be linearized to survive
    // CubeCountSelection spreading the grid over Y/Z.
    let wg = CUBE_POS_X + CUBE_COUNT_X * (CUBE_POS_Y + CUBE_COUNT_Y * CUBE_POS_Z);
    if wg >= num_wgs {
        terminate!();
    }

    // Stable block scatter: each thread owns a contiguous ELEMS_PER_THREAD
    // chunk and a PRIVATE column of the [bin][lane] histogram, so ranking
    // needs no atomics and equal keys keep their relative order. The tile
    // sort of the render pipeline relies on this stability to preserve depth
    // order within a tile.
    let mut hist = Shared::<[u32]>::new_slice((SORT_BINS * SORT_WG) as usize);
    for i in 0..SORT_BINS {
        hist[(UNIT_POS + i * SORT_WG) as usize] = 0u32;
    }
    sync_cube();

    let base = SORT_BLOCK * wg + UNIT_POS * ELEMS_PER_THREAD;
    for e in 0..ELEMS_PER_THREAD {
        let idx = base + e;
        if idx < num_keys {
            let bin = (src[idx as usize] >> shift) & 0xf;
            hist[(bin * SORT_WG + UNIT_POS) as usize] += 1u32;
        }
    }
    sync_cube();

    // One thread per (bin, lane-group): group sums of the private lane
    // counts. Lane order matches the chunk layout (lane-major), which is what
    // makes the scatter stable across the block.
    let mut partials = Shared::<[u32]>::new_slice((SORT_BINS * SCAN_GROUPS) as usize);
    let bin = UNIT_POS % SORT_BINS;
    let group = UNIT_POS / SORT_BINS;
    let lane0 = group * SCAN_CHUNK;
    let mut sum = 0u32;
    for l in lane0..lane0 + SCAN_CHUNK {
        sum += hist[(bin * SORT_WG + l) as usize];
    }
    partials[UNIT_POS as usize] = sum;
    sync_cube();

    // One thread per bin: group sums -> global exclusive offsets.
    if UNIT_POS < SORT_BINS {
        let mut running = counts[(UNIT_POS * num_wgs + wg) as usize];
        for g in 0..SCAN_GROUPS {
            let cell = (g * SORT_BINS + UNIT_POS) as usize;
            let c = partials[cell];
            partials[cell] = running;
            running += c;
        }
    }
    sync_cube();

    // Apply the offsets to each lane's private count.
    let mut running = partials[UNIT_POS as usize];
    for l in lane0..lane0 + SCAN_CHUNK {
        let cell = (bin * SORT_WG + l) as usize;
        let c = hist[cell];
        hist[cell] = running;
        running += c;
    }
    sync_cube();

    for e in 0..ELEMS_PER_THREAD {
        let idx = base + e;
        if idx < num_keys {
            let key = src[idx as usize];
            let val = values[idx as usize];
            let bin = (key >> shift) & 0xf;
            let cell = (bin * SORT_WG + UNIT_POS) as usize;
            let pos = hist[cell];
            hist[cell] = pos + 1u32;
            out[pos as usize] = key;
            out_values[pos as usize] = val;
        }
    }
}

/// Bits needed to represent any value in `0..max_exclusive`.
pub(crate) fn bits_for(max_exclusive: u32) -> u32 {
    u32::BITS - max_exclusive.saturating_sub(1).leading_zeros()
}

/// Reusable ping-pong buffers for [`radix_argsort_with`], sized for a maximum
/// element count so repeated sorts allocate nothing. Two sorts whose output
/// feeds the next sort's input must use *separate* scratch instances: after an
/// odd pass count the returned tensors alias the scratch.
#[derive(Debug)]
pub struct RadixScratch {
    count_buf: GpuTensor,
    dst_keys: GpuTensor,
    dst_vals: GpuTensor,
}

impl RadixScratch {
    pub fn new(client: &ComputeClient<WgpuRuntime>, max_elems: usize) -> Self {
        let max_wgs = (max_elems as u32).div_ceil(SORT_BLOCK);
        Self {
            count_buf: GpuTensor::empty(client, [(max_wgs * SORT_BINS) as usize]),
            dst_keys: GpuTensor::empty(client, [max_elems]),
            dst_vals: GpuTensor::empty(client, [max_elems]),
        }
    }
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
    let scratch = RadixScratch::new(&keys.client, n as usize);
    radix_argsort_with(keys, vals, n, bits, &scratch)
}

pub fn radix_argsort_with(
    keys: GpuTensor,
    vals: GpuTensor,
    n: u32,
    bits: u32,
    scratch: &RadixScratch,
) -> (GpuTensor, GpuTensor) {
    if n <= 1 || bits == 0 {
        return (keys, vals);
    }
    let client = keys.client.clone();
    debug_assert!(
        (n as usize) <= scratch.dst_keys.shape[0],
        "radix scratch undersized for {n} elements"
    );
    // The launched grid may exceed the hardware X limit and get spread over
    // Y/Z (CubeCountSelection), so kernels linearize the workgroup id.
    let num_wgs = n.div_ceil(SORT_BLOCK);
    let cube_count =
        calculate_cube_count_elemwise(&client, n as usize, CubeDim::new_1d(SORT_BLOCK));
    let cube_dim = CubeDim::new_1d(SORT_WG);

    let count_buf = &scratch.count_buf;
    let mut dst_keys = scratch.dst_keys.clone();
    let mut dst_vals = scratch.dst_vals.clone();

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
            cur_keys.as_buffer_arg(),
            count_buf.as_buffer_arg(),
        );

        // Serial prefix is cheaper for tiny grids; the parallel variant
        // wins once num_wgs outgrows one scan chunk.
        if num_wgs < SCAN_PARTS {
            prefix_kernel::launch::<WgpuRuntime>(
                &client,
                CubeCount::new_single(),
                cube_dim,
                n,
                count_buf.as_buffer_arg(),
            );
        } else {
            prefix_kernel_parallel::launch::<WgpuRuntime>(
                &client,
                CubeCount::new_single(),
                CubeDim::new_1d(PREFIX_WG),
                n,
                count_buf.as_buffer_arg(),
            );
        }

        scatter_kernel::launch::<WgpuRuntime>(
            &client,
            cube_count.clone(),
            cube_dim,
            num_wgs,
            shift,
            n,
            cur_keys.as_buffer_arg(),
            cur_vals.as_buffer_arg(),
            count_buf.as_buffer_arg(),
            dst_keys.as_buffer_arg(),
            dst_vals.as_buffer_arg(),
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

        let values_inp: Vec<u32> = (0..keys_inp.len()).map(|i| i as u32).collect();
        assert_argsort_bits(&client, &keys_inp, &values_inp, 32);
    }

    #[test]
    fn test_sorting_large() {
        const NUM_ELEMENTS: usize = 500_000;

        let _gpu = crate::tensor::GPU_TEST_LOCK.lock().unwrap();
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

        // Warm every buffer class the measurement window will touch (sort
        // scratch, readback staging), then quiesce: other tests' deferred
        // frees and this client's page setup must not leak into the delta.
        {
            let (sk, sv) = radix_argsort(keys.clone(), vals.clone(), n as u32, 32);
            let _ = sk.read_vec::<u32>();
            let _ = sv.read_vec::<u32>();
        }
        pollster::block_on(client.sync()).unwrap();

        let before = client.memory_usage().unwrap().bytes_in_use;
        let (sorted_keys, sorted_vals) = radix_argsort(keys, vals, n as u32, 32);
        let out: Vec<u32> = sorted_keys.read_vec(); // forces completion before measuring
        let after = client.memory_usage().unwrap().bytes_in_use;
        // An even pass count ends on the original (capacity-sized) tensors;
        // an odd count ends on the n-sized ping-pong scratch. Either is a
        // valid result — only the first n entries are read downstream.
        assert!(out.len() == cap || out.len() == n);

        // Scratch (dst pair + counts) must track n, not the tensor capacity.
        // cubecl 0.11 may reclaim pages between the two readings, so measure
        // growth only — a decrease still passes the budget property.
        let grown = after.saturating_sub(before);
        assert!(
            grown < 256 * 1024,
            "scratch allocation {} bytes exceeds n-sized budget",
            grown
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
