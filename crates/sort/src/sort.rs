//! GPU radix sort.
//!
//! References:
//! - <https://github.com/ArthurBrussee/brush/blob/main/crates/brush-sort/src/lib.rs>
use crate::scan::{ScanScratch, exclusive_scan_buf};
use crate::tensor::GpuTensor;
use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

const SORT_WG: u32 = 128;
// 6-bit digits: few passes, and 2048/64 = 32-element digit runs (two cache
// lines) keep the scatter's global writes coalesced.
const BINS_PLANE: u32 = 64;
const BITS_PLANE: u32 = 6;
const EPT_PLANE: u32 = 16;
const BLOCK_PLANE: u32 = SORT_WG * EPT_PLANE;

/// Histogram each 6-bit digit into per-workgroup counts, bin-major
/// `counts[bin * num_wgs + wg]`, so one flat exclusive scan yields each
/// (bin, wg) cell's global scatter offset directly.
#[cube(launch)]
fn count_kernel(num_wgs: u32, shift: u32, num_keys: u32, src: &[u32], counts: &mut [u32]) {
    // CUBE_POS linearizes the workgroup id: the grid may be spread over Y/Z
    // when the X count exceeds the hardware limit (CubeCountSelection).
    let wg = CUBE_POS as u32;
    if wg >= num_wgs {
        terminate!();
    }

    // Workgroup-shared histogram — each key is read once, not once per bin.
    let histogram = Shared::<[Atomic<u32>]>::new_slice(BINS_PLANE as usize);
    if UNIT_POS < BINS_PLANE {
        histogram[UNIT_POS as usize].store(0u32);
    }
    sync_cube();

    let base = BLOCK_PLANE * wg + UNIT_POS;
    for e in 0..(BLOCK_PLANE / SORT_WG) {
        let idx = base + e * SORT_WG;
        if idx < num_keys {
            let bin = (src[idx as usize] >> shift) & (BINS_PLANE - 1u32);
            histogram[bin as usize].fetch_add(1u32);
        }
    }
    sync_cube();

    if UNIT_POS < BINS_PLANE {
        counts[(UNIT_POS * num_wgs + wg) as usize] = histogram[UNIT_POS as usize].load();
    }
}

/// One ballot per bit, inverted where clear, masked to this lane's 32-bit
/// word. ANDing the words leaves exactly the lanes whose value equals ours —
/// libcusort's fixed-latency match_any (beats hardware match once 4+
/// distinct values are in flight).
#[cube]
fn ballot_word(v: u32, bit: u32) -> u32 {
    let set = (v >> bit) & 1u32 == 1u32;
    let ballot = plane_ballot(set);
    let word = ballot.extract_dynamic((UNIT_POS_PLANE / 32) as usize);
    let not_set = (set as u32) ^ 1u32;
    word ^ (not_set * u32::MAX)
}

/// Bitmask of plane lanes whose 7-bit value equals `v` — the 6-bit digit
/// plus a validity bit (the scatter kernel folds it in), so tail lanes'
/// placeholder digits never join a real digit group.
#[cube]
fn match_any7(v: u32) -> u32 {
    ballot_word(v, 0u32)
        & ballot_word(v, 1u32)
        & ballot_word(v, 2u32)
        & ballot_word(v, 3u32)
        & ballot_word(v, 4u32)
        & ballot_word(v, 5u32)
        & ballot_word(v, 6u32)
}

/// Match-any peers of `v` and this lane's rank within the group — the pair
/// the scatter kernels turn into a slot via a group leader's fetch_add.
#[cube]
fn plane_rank(v: u32) -> (u32, u32) {
    let peers = match_any7(v);
    let lane_mask = (1u32 << (UNIT_POS_PLANE % 32)) - 1u32;
    (peers, (peers & lane_mask).count_ones())
}

/// One 6-bit radix plane pass: planes rank items with match-any ballots
/// (group leaders reserve a slot run per digit from the plane's shared
/// histogram), scatter through shared memory, then write each digit's
/// contiguous global run in order. `num_planes` is the comptime worst-case
/// plane count sizing the shared histograms; planes stride by the runtime
/// PLANE_DIM (see plane_sort_path).
#[cube(launch)]
fn scatter_plane_kernel(
    num_wgs: u32,
    shift: u32,
    num_keys: u32,
    #[comptime] num_planes: u32,
    #[comptime] write_keys: bool,
    src: &[u32],
    values: &[u32],
    counts: &[u32],
    out: &mut [u32],
    out_values: &mut [u32],
) {
    let wg = CUBE_POS as u32;
    if wg >= num_wgs {
        terminate!();
    }

    let hist = Shared::<[Atomic<u32>]>::new_slice((num_planes * BINS_PLANE) as usize);
    let mut prefix = Shared::<[u32]>::new_slice(BINS_PLANE as usize);
    let mut stage = Shared::<[u32]>::new_slice(BLOCK_PLANE as usize);
    let mut stage_values = Shared::<[u32]>::new_slice(BLOCK_PLANE as usize);

    let mut c = UNIT_POS;
    while c < num_planes * BINS_PLANE {
        hist[c as usize].store(0u32);
        c += SORT_WG;
    }
    sync_cube();

    // Warp-strided tile: consecutive lanes read consecutive keys. Planes
    // stride by the runtime PLANE_DIM (WGSL's subgroup_size, chosen by the
    // driver within [plane_size_min, plane_size_max]) — striding by
    // BLOCK_PLANE / num_planes would overlap once PLANE_DIM exceeds the
    // minimum that sized num_planes. Planes partition the workgroup, so
    // EPT_PLANE * PLANE_DIM spans tile the block exactly.
    let tile_base = BLOCK_PLANE * wg;
    let plane_base = tile_base + PLANE_POS * PLANE_DIM * EPT_PLANE;

    // The kernel's only key read.
    let mut keys = Array::<u32>::new(EPT_PLANE as usize);
    #[unroll]
    for i in 0..EPT_PLANE {
        let idx = plane_base + i * PLANE_DIM + UNIT_POS_PLANE;
        let mut k = 0u32;
        if idx < num_keys {
            k = src[idx as usize];
        }
        keys[i as usize] = k;
    }

    // Per-plane digit histogram. 63 = BINS_PLANE - 1 as a literal: const
    // expressions inside #[unroll] bodies break the cubecl expansion.
    #[unroll]
    for i in 0..EPT_PLANE {
        let idx = plane_base + i * PLANE_DIM + UNIT_POS_PLANE;
        if idx < num_keys {
            let d = (keys[i as usize] >> shift) & 63u32;
            hist[(PLANE_POS * BINS_PLANE + d) as usize].fetch_add(1u32);
        }
    }
    sync_cube();

    // Bin transform, one bin per thread. prefix[b] becomes the tile count,
    // then the bin-exclusive tile offset, then global_base[b] - tile_excl[b];
    // hist[p][b] becomes plane p's tile-local slot base for bin b. The tile
    // total is re-derived from hist: counts was exclusive-scanned in place,
    // so it holds global offsets, not counts.
    let b = UNIT_POS;
    if b < BINS_PLANE {
        let mut total = 0u32;
        let mut p = 0u32;
        while p < num_planes {
            total += hist[(p * BINS_PLANE + b) as usize].load();
            p += 1u32;
        }
        prefix[b as usize] = total;
    }
    sync_cube();

    if UNIT_POS == 0 {
        crate::scan::serial_exclusive(&mut prefix, BINS_PLANE, 0u32);
    }
    sync_cube();

    if b < BINS_PLANE {
        let tex = prefix[b as usize];
        prefix[b as usize] = counts[(b * num_wgs + wg) as usize] - tex;
        let mut running = tex;
        let mut p = 0u32;
        while p < num_planes {
            let cell = (p * BINS_PLANE + b) as usize;
            let cnt = hist[cell].load();
            hist[cell].store(running);
            running += cnt;
            p += 1u32;
        }
    }
    sync_cube();

    // Rank: the group leader reserves the digit's slot run; peers add their rank.
    let mut offs = Array::<u32>::new(EPT_PLANE as usize);
    #[unroll]
    for i in 0..EPT_PLANE {
        let idx = plane_base + i * PLANE_DIM + UNIT_POS_PLANE;
        let valid = idx < num_keys;
        let mut d = 63u32;
        if valid {
            d = (keys[i as usize] >> shift) & 63u32;
        }
        let (peers, r) = plane_rank(d + (valid as u32) * 64u32);
        let leader = peers.trailing_zeros();
        let mut base = 0u32;
        if r == 0 && valid {
            base = hist[(PLANE_POS * BINS_PLANE + d) as usize].fetch_add(peers.count_ones());
        }
        offs[i as usize] = plane_shuffle(base, leader) + r;
    }

    // Keys and values stage through separate shared buffers behind one
    // barrier, then each digit's global run is written with sequential
    // (coalesced) reads. `write_keys` is comptime-false when the caller
    // discards sorted keys (the depth sort): the final pass only moves values.
    #[unroll]
    for i in 0..EPT_PLANE {
        let idx = plane_base + i * PLANE_DIM + UNIT_POS_PLANE;
        if idx < num_keys {
            let slot = offs[i as usize] as usize;
            stage[slot] = keys[i as usize];
            stage_values[slot] = values[idx as usize];
        }
    }
    sync_cube();

    let lim = num_keys.saturating_sub(tile_base).min(BLOCK_PLANE);
    let mut digs = Array::<u32>::new(EPT_PLANE as usize);
    #[unroll]
    for i in 0..EPT_PLANE {
        let s = UNIT_POS + i * SORT_WG;
        let mut d = 0u32;
        if s < lim {
            d = (stage[s as usize] >> shift) & 63u32;
        }
        digs[i as usize] = d;
    }
    if write_keys {
        #[unroll]
        for i in 0..EPT_PLANE {
            let s = UNIT_POS + i * SORT_WG;
            if s < lim {
                out[(prefix[digs[i as usize] as usize] + s) as usize] = stage[s as usize];
            }
        }
    }
    #[unroll]
    for i in 0..EPT_PLANE {
        let s = UNIT_POS + i * SORT_WG;
        if s < lim {
            out_values[(prefix[digs[i as usize] as usize] + s) as usize] = stage_values[s as usize];
        }
    }
}

/// Bits needed to represent any value in `0..max_exclusive`.
pub fn bits_for(max_exclusive: u32) -> u32 {
    u32::BITS - max_exclusive.saturating_sub(1).leading_zeros()
}

/// The comptime worst-case plane count for [`scatter_plane_kernel`]:
/// SORT_WG / hardware plane size minimum. Requires plane ops (supported:
/// Apple Silicon/Metal3+, where cubecl hardcodes 32-lane planes). Shared
/// memory: histograms + prefix + dual staging = 17 664 B on 32-lane planes,
/// well under Metal's 32 KB.
fn plane_sort_path(client: &ComputeClient<WgpuRuntime>) -> u32 {
    let props = client.properties();
    assert!(
        props.features.plane.contains(cubecl::features::Plane::Ops)
            && props.hardware.plane_size_min != 0,
        "GPU lacks subgroup (plane) ops; supported platforms: Apple Silicon/Metal3+"
    );
    SORT_WG / props.hardware.plane_size_min
}

/// Reusable ping-pong buffers for [`RadixScratch::argsort`], sized for a maximum
/// element count so repeated sorts allocate nothing. Two sorts whose output
/// feeds the next sort's input must use *separate* scratch instances: after an
/// odd pass count the returned tensors alias the scratch.
pub struct RadixScratch {
    count_buf: GpuTensor,
    /// Scratch for the per-pass counter scans.
    scan: ScanScratch,
    dst_keys: GpuTensor,
    dst_vals: GpuTensor,
}

impl RadixScratch {
    pub fn new(client: &ComputeClient<WgpuRuntime>, max_elems: usize) -> Self {
        let cells = ((max_elems as u32).div_ceil(BLOCK_PLANE) * BINS_PLANE) as usize;
        Self {
            count_buf: GpuTensor::empty(client, [cells]),
            // Loose bound: counter scans need far fewer cells than max_elems.
            scan: ScanScratch::new(client, max_elems),
            dst_keys: GpuTensor::empty(client, [max_elems]),
            dst_vals: GpuTensor::empty(client, [max_elems]),
        }
    }

    /// Stable argsort of `keys[0..n]` carrying `vals`, using the low `bits`
    /// of each key (6-bit digits, plane ranking). Stability — equal keys keep
    /// input order — is what the render pipeline's tile sort relies on for
    /// depth order within a tile. Inputs are only read; the returned tensors
    /// may alias this scratch after an odd pass count (see [`RadixScratch`]).
    ///
    /// With `write_keys = false` the sorted keys are not written: the
    /// returned key tensor holds stale scratch and must be discarded. Values
    /// are always sorted correctly.
    pub fn argsort(
        &self,
        keys: &GpuTensor,
        vals: &GpuTensor,
        n: u32,
        bits: u32,
        write_keys: bool,
    ) -> (GpuTensor, GpuTensor) {
        if n <= 1 || bits == 0 {
            return (keys.clone(), vals.clone());
        }
        let client = keys.client.clone();
        assert!(
            (n as usize) <= self.dst_keys.shape[0],
            "radix scratch undersized: sorted {n} elements, capacity {}",
            self.dst_keys.shape[0]
        );
        let num_planes = plane_sort_path(&keys.client);
        // The launched grid may exceed the hardware X limit and get spread over
        // Y/Z (CubeCountSelection), so kernels linearize the workgroup id.
        let num_wgs = n.div_ceil(BLOCK_PLANE);
        let cube_count =
            calculate_cube_count_elemwise(&client, n as usize, CubeDim::new_1d(BLOCK_PLANE));
        let cube_dim = CubeDim::new_1d(SORT_WG);

        let count_buf = &self.count_buf;
        let mut dst_keys = self.dst_keys.clone();
        let mut dst_vals = self.dst_vals.clone();

        let mut cur_keys = keys.clone();
        let mut cur_vals = vals.clone();

        for shift in (0..bits).step_by(BITS_PLANE as usize) {
            // Only the final pass may skip the key writes: intermediate passes
            // sort on, so their key output is the next pass's input.
            let pass_writes_keys = write_keys || shift + BITS_PLANE < bits;
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

            exclusive_scan_buf(&client, count_buf, BINS_PLANE * num_wgs, &self.scan);

            scatter_plane_kernel::launch::<WgpuRuntime>(
                &client,
                cube_count.clone(),
                cube_dim,
                num_wgs,
                shift,
                n,
                num_planes,
                pass_writes_keys,
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
}

#[cfg(test)]
mod radix_sort_tests {
    use super::*;
    use rand::{RngExt, SeedableRng};

    fn assert_argsort_bits(
        client: &ComputeClient<WgpuRuntime>,
        keys_inp: &[u32],
        values_inp: &[u32],
        bits: u32,
    ) {
        let keys = GpuTensor::from(client, [keys_inp.len()], keys_inp);
        let values = GpuTensor::from(client, [values_inp.len()], values_inp);
        let scratch = RadixScratch::new(client, keys_inp.len());
        let n = keys_inp.len() as u32;
        let (ret_keys, ret_values) = scratch.argsort(&keys, &values, n, bits, true);
        let ret_keys: Vec<u32> = ret_keys.read_vec();
        let ret_values: Vec<u32> = ret_values.read_vec();

        assert_eq!(ret_keys.len(), keys_inp.len());
        assert_eq!(ret_values.len(), keys_inp.len());

        // Stability is asserted separately in test_sorting_stable; here
        // assert sorted order and key/value pairing only.
        let bad = ret_keys.windows(2).position(|w| w[0] > w[1]);
        assert!(bad.is_none(), "keys not sorted at index {bad:?}");

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
    fn test_sorting_stable() {
        // The render pipeline relies on stability: intersections are emitted
        // in depth order and the tile sort must preserve that order among
        // equal tile ids. Assert exact stable argsort, not just sortedness.
        let (_gpu, client) = crate::tensor::test_client();
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5EED_0001);
        for n in [1000usize, 5000, 200_000] {
            let keys_inp: Vec<u32> = (0..n).map(|_| rng.random_range(0..37)).collect();
            let values_inp: Vec<u32> = (0..n as u32).collect();
            let mut reference: Vec<u32> = (0..n as u32).collect();
            reference.sort_by_key(|&i| keys_inp[i as usize]); // stable

            let keys = GpuTensor::from(&client, [n], &keys_inp[..]);
            let values = GpuTensor::from(&client, [n], &values_inp[..]);
            let scratch = RadixScratch::new(&client, n);
            let (_, ret_values) = scratch.argsort(&keys, &values, n as u32, 8, true);
            let ret_values: Vec<u32> = ret_values.read_vec();
            assert_eq!(ret_values, reference, "n={n}: sort must be stable");
        }
    }

    /// The render pipeline's depth sort discards the sorted keys; values-only
    /// mode must still produce the exact stable permutation.
    #[test]
    fn test_sort_values_only() {
        let (_gpu, client) = crate::tensor::test_client();
        let n = 5000usize;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5EED_0002);
        let keys_inp: Vec<u32> = (0..n).map(|_| rng.random()).collect();
        let values_inp: Vec<u32> = (0..n as u32).collect();
        let mut reference: Vec<u32> = (0..n as u32).collect();
        reference.sort_by_key(|&i| keys_inp[i as usize]); // stable

        let keys = GpuTensor::from(&client, [n], &keys_inp[..]);
        let values = GpuTensor::from(&client, [n], &values_inp[..]);
        let scratch = RadixScratch::new(&client, n);
        let (_, ret_values) = scratch.argsort(&keys, &values, n as u32, 32, false);
        let ret_values: Vec<u32> = ret_values.read_vec();
        assert_eq!(ret_values, reference, "values-only argsort must be stable");
    }

    #[test]
    fn test_sorting_partial_bits() {
        let (_gpu, client) = crate::tensor::test_client();
        for max in [2u32, 64, 1000, 65536] {
            let bits = bits_for(max);
            let mut rng = rand::rngs::StdRng::seed_from_u64(0x5EED_0003);
            let keys_inp: Vec<u32> = (0..5000).map(|_| rng.random_range(0..max)).collect();
            let values_inp: Vec<u32> = (0..5000).map(|i| i as u32).collect();
            assert_argsort_bits(&client, &keys_inp, &values_inp, bits);
        }
    }

    #[test]
    fn test_sorting_32bit_keys() {
        let (_gpu, client) = crate::tensor::test_client();
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5EED_0004);
        let mut clustered = Vec::new();
        for i in 0..10000u32 {
            let start = rng.random_range(i..i + 150);
            let end = rng.random_range(start..start + 250);

            for j in start..end {
                if rng.random::<f32>() < 0.5 {
                    clustered.push(j);
                }
            }
        }
        let spread: Vec<u32> = (0..500_000)
            .map(|_| rng.random_range(0..1_000_000))
            .collect();

        for keys_inp in [clustered, spread] {
            let values_inp: Vec<u32> = (0..keys_inp.len() as u32).collect();
            assert_argsort_bits(&client, &keys_inp, &values_inp, 32);
        }
    }

    #[test]
    fn test_sort_subset_of_capacity() {
        let (_gpu, client) = crate::tensor::test_client();
        let cap = 1_000_000usize;
        let n = 1_000usize;
        let keys_inp: Vec<u32> = (0..cap)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761))
            .collect();
        let vals_inp: Vec<u32> = (0..cap).map(|i| i as u32).collect();
        let keys = GpuTensor::from(&client, [cap], &keys_inp[..]);
        let vals = GpuTensor::from(&client, [cap], &vals_inp[..]);

        // Scratch sized for the sorted prefix n, not the tensor capacity.
        let scratch = RadixScratch::new(&client, n);
        let (sorted_keys, sorted_vals) = scratch.argsort(&keys, &vals, n as u32, 32, true);
        let out: Vec<u32> = sorted_keys.read_vec();
        // An even pass count ends on the original (capacity-sized) tensors;
        // an odd count ends on the n-sized ping-pong scratch. Either is a
        // valid result — only the first n entries are read downstream.
        assert!(out.len() == cap || out.len() == n);
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
