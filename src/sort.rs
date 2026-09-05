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
// Fallback path (no plane ops): 4-bit digits, per-lane private histograms.
const BITS_PER_PASS_FALLBACK: u32 = 4;
const SORT_BINS: u32 = 1 << BITS_PER_PASS_FALLBACK;
const ELEMS_PER_THREAD: u32 = 8;
const SORT_BLOCK: u32 = SORT_WG * ELEMS_PER_THREAD;
// scatter scan: SORT_WG lanes are scanned by SORT_BINS x SCAN_GROUPS threads
// (group sums -> group offsets -> apply), shortening the serial chain.
const SCAN_GROUPS: u32 = 8;
const SCAN_CHUNK: u32 = SORT_WG / SCAN_GROUPS;
// Plane path: 6-bit digits cut the pass count by a third while digit runs
// stay wide enough (2048/64 = 32 elements, two cache lines) that the
// scatter's global writes coalesce; ranking uses one match-any ballot group
// per item instead of a per-lane histogram column.
pub(crate) const BINS_PLANE: u32 = 64;
const BITS_PLANE: u32 = 6;
const EPT_PLANE: u32 = 16;
const BLOCK_PLANE: u32 = SORT_WG * EPT_PLANE;

/// Histogram each `bins`-bit digit into per-workgroup counts, BIN-MAJOR:
/// `counts[bin * num_wgs + wg]`. This layout lets one flat exclusive scan
/// (scan::exclusive_scan_buf) produce each (bin, wg) cell's global scatter
/// offset directly — lower bins' totals plus this bin's lower workgroups.
#[cube(launch)]
fn count_kernel(
    num_wgs: u32,
    shift: u32,
    num_keys: u32,
    #[comptime] bins: u32,
    #[comptime] block: u32,
    src: &[u32],
    counts: &mut [u32],
) {
    // CUBE_POS linearizes the workgroup id: the grid may be spread over Y/Z
    // when the X count exceeds the hardware limit (CubeCountSelection).
    let wg = CUBE_POS as u32;
    if wg >= num_wgs {
        terminate!();
    }

    // Workgroup-shared histogram: each key is read exactly once, then atomically
    // bucketed — replacing a per-bin loop that re-read every key `bins` times.
    let histogram = Shared::<[Atomic<u32>]>::new_slice(bins as usize);
    if UNIT_POS < bins {
        histogram[UNIT_POS as usize].store(0u32);
    }
    sync_cube();

    let base = block * wg + UNIT_POS;
    for e in 0..(block / SORT_WG) {
        let idx = base + e * SORT_WG;
        if idx < num_keys {
            let bin = (src[idx as usize] >> shift) & (bins - 1u32);
            histogram[bin as usize].fetch_add(1u32);
        }
    }
    sync_cube();

    // bins is comptime and below SORT_WG: one bin per thread.
    if UNIT_POS < bins {
        counts[(UNIT_POS * num_wgs + wg) as usize] = histogram[UNIT_POS as usize].load();
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
    let wg = CUBE_POS as u32;
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
            let bin = (src[idx as usize] >> shift) & (SORT_BINS - 1u32);
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
            let bin = (key >> shift) & (SORT_BINS - 1u32);
            let cell = (bin * SORT_WG + UNIT_POS) as usize;
            let pos = hist[cell];
            hist[cell] = pos + 1u32;
            out[pos as usize] = key;
            out_values[pos as usize] = val;
        }
    }
}

/// Ballot of "bit `bit` of `v` is set" over this thread's plane, inverted
/// when the bit is clear, restricted to the 32-bit word holding this lane.
/// ANDing one word per bit leaves exactly the lanes whose value equals ours
/// (libcusort's fixed-latency match_any — one ballot per bit beats hardware
/// match once 4+ distinct values are in flight).
#[cube]
fn ballot_word(v: u32, bit: u32) -> u32 {
    let set = (v >> bit) & 1u32 == 1u32;
    let ballot = plane_ballot(set);
    let word = ballot.extract_dynamic((UNIT_POS_PLANE / 32) as usize);
    let not_set = (set as u32) ^ 1u32;
    word ^ (not_set * u32::MAX)
}

/// Bitmask of the plane lanes whose 7-bit value equals `v`: the 6-bit digit
/// plus a caller-supplied validity bit (the scatter kernel folds it in) so
/// tail lanes' placeholder digits never join a real digit group. ANDing one
/// ballot per bit leaves exactly the lanes equal in all seven.
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

/// One 6-bit radix plane pass. Planes rank their items with
/// match-any ballots (group leaders reserve a run of slots per digit from the
/// plane's shared histogram), scatter through shared memory, and write each
/// digit's contiguous global run in order. `num_planes` is the comptime
/// worst-case plane count (SORT_WG / hardware plane size minimum) sizing the
/// shared histograms; planes stride by the runtime PLANE_DIM, which the
/// driver may dispatch wider than that minimum.
#[cube(launch)]
fn scatter_plane_kernel(
    num_wgs: u32,
    shift: u32,
    num_keys: u32,
    #[comptime] num_planes: u32,
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
    let plane = PLANE_POS;
    let plane_dim = PLANE_DIM;
    let lane = UNIT_POS_PLANE;

    let hist = Shared::<[Atomic<u32>]>::new_slice((num_planes * BINS_PLANE) as usize);
    let mut prefix = Shared::<[u32]>::new_slice(BINS_PLANE as usize);
    let mut stage = Shared::<[u32]>::new_slice(BLOCK_PLANE as usize);

    let mut c = UNIT_POS;
    while c < num_planes * BINS_PLANE {
        hist[c as usize].store(0u32);
        c += SORT_WG;
    }
    sync_cube();

    // Warp-strided tile: consecutive lanes read consecutive keys. Planes
    // stride by the runtime plane size — WGSL's subgroup_size, which a driver
    // may pick anywhere in [plane_size_min, plane_size_max] — so striding by
    // BLOCK_PLANE / num_planes would overlap once PLANE_DIM exceeds the
    // minimum that sized num_planes. Planes partition the workgroup, so
    // EPT_PLANE * plane_dim spans always tile the block exactly.
    let tile_base = BLOCK_PLANE * wg;
    let plane_base = tile_base + plane * plane_dim * EPT_PLANE;

    // The kernel's only key read.
    let mut keys = Array::<u32>::new(EPT_PLANE as usize);
    #[unroll]
    for i in 0..EPT_PLANE {
        let idx = plane_base + i * plane_dim + lane;
        let mut k = 0u32;
        if idx < num_keys {
            k = src[idx as usize];
        }
        keys[i as usize] = k;
    }

    // Per-plane digit histogram, one aggregated fetch_add per match group.
    // 63 = BINS_PLANE - 1: literals here, not the const — const expressions
    // inside #[unroll] bodies break the cubecl expansion.
    #[unroll]
    for i in 0..EPT_PLANE {
        let idx = plane_base + i * plane_dim + lane;
        if idx < num_keys {
            let d = (keys[i as usize] >> shift) & 63u32;
            hist[(plane * BINS_PLANE + d) as usize].fetch_add(1u32);
        }
    }
    sync_cube();

    // Bin transform, one bin per thread. prefix[b] holds the tile count,
    // then the bin-exclusive tile offset, then global_base[b] - tile_excl[b];
    // hist[p][b] becomes the tile-local slot base of plane p for bin b.
    // The tile total is re-derived from hist: counts was already
    // exclusive-scanned in place, so it holds global offsets, not counts.
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

    // Rank (see plane_rank): group leaders reserve one slot run per digit,
    // peers add their rank within the group.
    let mut offs = Array::<u32>::new(EPT_PLANE as usize);
    #[unroll]
    for i in 0..EPT_PLANE {
        let idx = plane_base + i * plane_dim + lane;
        let valid = idx < num_keys;
        let mut d = 63u32;
        if valid {
            d = (keys[i as usize] >> shift) & 63u32;
        }
        let (peers, r) = plane_rank(d + (valid as u32) * 64u32);
        let leader = peers.trailing_zeros();
        let mut base = 0u32;
        if r == 0 && valid {
            base = hist[(plane * BINS_PLANE + d) as usize].fetch_add(peers.count_ones());
        }
        offs[i as usize] = plane_shuffle(base, leader) + r;
    }

    // Scatter keys through shared, then write each digit's global run with
    // sequential reads — coalesced where the direct register scatter isn't.
    #[unroll]
    for i in 0..EPT_PLANE {
        let idx = plane_base + i * plane_dim + lane;
        if idx < num_keys {
            stage[offs[i as usize] as usize] = keys[i as usize];
        }
    }
    sync_cube();

    let lim = (num_keys - tile_base).min(BLOCK_PLANE);
    let mut digs = Array::<u32>::new(EPT_PLANE as usize);
    #[unroll]
    for i in 0..EPT_PLANE {
        let s = UNIT_POS + i * SORT_WG;
        let mut d = 0u32;
        if s < lim {
            let k = stage[s as usize];
            d = (k >> shift) & 63u32;
            out[(prefix[d as usize] + s) as usize] = k;
        }
        digs[i as usize] = d;
    }
    sync_cube();
    #[unroll]
    for i in 0..EPT_PLANE {
        let idx = plane_base + i * plane_dim + lane;
        if idx < num_keys {
            stage[offs[i as usize] as usize] = values[idx as usize];
        }
    }
    sync_cube();
    #[unroll]
    for i in 0..EPT_PLANE {
        let s = UNIT_POS + i * SORT_WG;
        if s < lim {
            let cell = (prefix[digs[i as usize] as usize] + s) as usize;
            out_values[cell] = stage[s as usize];
        }
    }
}

/// Bits needed to represent any value in `0..max_exclusive`.
pub(crate) fn bits_for(max_exclusive: u32) -> u32 {
    u32::BITS - max_exclusive.saturating_sub(1).leading_zeros()
}

/// Whether the 6-bit plane path can run on this device, and its comptime
/// worst-case plane count. The ballot ranking reads a single 32-bit word per
/// lane, so planes beyond 32 lanes fall back; shared memory must fit the
/// per-plane histograms, the bin prefixes, and the staging buffer.
fn plane_sort_path(client: &ComputeClient<WgpuRuntime>) -> Option<u32> {
    // Browsers are excluded regardless of what the adapter reports: cubecl's
    // WGSL backend emits `subgroupBallot`/`subgroupShuffle` without the
    // `enable subgroups;` directive, which every conforming validator rejects.
    #[cfg(target_arch = "wasm32")]
    {
        let _ = client;
        return None;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let props = client.properties();
        if !props.features.plane.contains(cubecl::features::Plane::Ops) {
            return None;
        }
        let hw = &props.hardware;
        if hw.plane_size_min == 0 || hw.plane_size_max > 32 {
            return None;
        }
        let num_planes = SORT_WG / hw.plane_size_min;
        let shared = (num_planes * BINS_PLANE + BINS_PLANE + BLOCK_PLANE) * 4;
        if shared as usize > hw.max_shared_memory_size {
            return None;
        }
        Some(num_planes)
    }
}

/// Reusable ping-pong buffers for [`radix_argsort_with`], sized for a maximum
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
        // The plane path's 64 bins over 2K-element blocks always need the
        // most counter cells (4-bit: 16 bins over 1K blocks).
        let cells = ((max_elems as u32).div_ceil(BLOCK_PLANE) * BINS_PLANE) as usize;
        Self {
            count_buf: GpuTensor::empty(client, [cells]),
            // Loose bound: counter scans need far fewer cells than max_elems.
            scan: ScanScratch::new(client, max_elems),
            dst_keys: GpuTensor::empty(client, [max_elems]),
            dst_vals: GpuTensor::empty(client, [max_elems]),
        }
    }
}

fn radix_argsort_path(
    keys: &GpuTensor,
    vals: &GpuTensor,
    n: u32,
    bits: u32,
    planes: Option<u32>,
    scratch: &RadixScratch,
) -> (GpuTensor, GpuTensor) {
    let client = keys.client.clone();
    debug_assert!(
        (n as usize) <= scratch.dst_keys.shape[0],
        "radix scratch undersized for {n} elements"
    );
    // The launched grid may exceed the hardware X limit and get spread over
    // Y/Z (CubeCountSelection), so kernels linearize the workgroup id.
    let (block, bins, bits_per_pass) = match planes {
        Some(_) => (BLOCK_PLANE, BINS_PLANE, BITS_PLANE),
        None => (SORT_BLOCK, SORT_BINS, BITS_PER_PASS_FALLBACK),
    };
    let num_wgs = n.div_ceil(block);
    let cube_count = calculate_cube_count_elemwise(&client, n as usize, CubeDim::new_1d(block));
    let cube_dim = CubeDim::new_1d(SORT_WG);

    let count_buf = &scratch.count_buf;
    let mut dst_keys = scratch.dst_keys.clone();
    let mut dst_vals = scratch.dst_vals.clone();

    let mut cur_keys = keys.clone();
    let mut cur_vals = vals.clone();

    for shift in (0..bits).step_by(bits_per_pass as usize) {
        count_kernel::launch::<WgpuRuntime>(
            &client,
            cube_count.clone(),
            cube_dim,
            num_wgs,
            shift,
            n,
            bins,
            block,
            cur_keys.as_buffer_arg(),
            count_buf.as_buffer_arg(),
        );

        exclusive_scan_buf(&client, count_buf, bins * num_wgs, &scratch.scan);

        match planes {
            Some(num_planes) => scatter_plane_kernel::launch::<WgpuRuntime>(
                &client,
                cube_count.clone(),
                cube_dim,
                num_wgs,
                shift,
                n,
                num_planes,
                cur_keys.as_buffer_arg(),
                cur_vals.as_buffer_arg(),
                count_buf.as_buffer_arg(),
                dst_keys.as_buffer_arg(),
                dst_vals.as_buffer_arg(),
            ),
            None => scatter_kernel::launch::<WgpuRuntime>(
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
            ),
        };

        std::mem::swap(&mut cur_keys, &mut dst_keys);
        std::mem::swap(&mut cur_vals, &mut dst_vals);
    }
    (cur_keys, cur_vals)
}

/// Stable argsort of `keys[0..n]` carrying `vals`, restricted to the low
/// `bits` of each key. Sorts with 6-bit digits and warp-aggregated ranking
/// when the device exposes plane ops, falling back to the 4-bit per-lane
/// histogram path otherwise. The inputs are only read; the returned tensors
/// may alias the scratch after an odd pass count (see [`RadixScratch`]).
pub fn radix_argsort_with(
    keys: &GpuTensor,
    vals: &GpuTensor,
    n: u32,
    bits: u32,
    scratch: &RadixScratch,
) -> (GpuTensor, GpuTensor) {
    if n <= 1 || bits == 0 {
        return (keys.clone(), vals.clone());
    }
    // The plane path's per-element ranking costs ~25% more than the 4-bit
    // per-lane histogram, so it only pays when it drops enough passes.
    let planes = plane_sort_path(&keys.client)
        .filter(|_| bits.div_ceil(BITS_PLANE) * 4 <= bits.div_ceil(4) * 3);
    radix_argsort_path(keys, vals, n, bits, planes, scratch)
}

#[cfg(test)]
mod radix_sort_tests {
    use super::*;
    use rand::RngExt;

    /// Canary for the plane primitives the 6-bit path builds on: emulated
    /// match-any, group-leader slot reservation, and leader broadcast.
    #[cube(launch)]
    fn probe_plane_kernel(
        digits: &[u32],
        ranks: &mut [u32],
        peers_out: &mut [u32],
        plane_dims: &mut [u32],
    ) {
        let d = digits[UNIT_POS as usize];
        let (peers, rank) = plane_rank(d + 256u32);
        peers_out[UNIT_POS as usize] = peers;

        let hist = Shared::<[Atomic<u32>]>::new_slice(128usize);
        hist[UNIT_POS as usize].store(16u32 * (PLANE_POS + 1u32));
        sync_cube();

        let leader_lane = peers.trailing_zeros();
        let mut base = 0u32;
        if rank == 0 {
            base = hist[(PLANE_POS * 32u32 + d) as usize].fetch_add(peers.count_ones());
        }
        ranks[UNIT_POS as usize] = plane_shuffle(base, leader_lane) + rank;

        if UNIT_POS == 0 {
            plane_dims[0] = PLANE_DIM;
        }
    }

    /// Both dispatch paths: the 4-bit fallback always, plus the plane path
    /// when the device supports it.
    fn all_paths(client: &ComputeClient<WgpuRuntime>) -> Vec<Option<u32>> {
        let mut paths = vec![None];
        if let Some(planes) = plane_sort_path(client) {
            paths.push(Some(planes));
        }
        paths
    }

    #[test]
    fn test_plane_probe() {
        let (_gpu, client) = crate::tensor::test_client();
        if !client
            .properties()
            .features
            .plane
            .contains(cubecl::features::Plane::Ops)
        {
            eprintln!("skipping: no plane ops");
            return;
        }
        let digits: Vec<u32> = (0..128u32).map(|i| i % 7).collect();
        let digits_t = GpuTensor::from(&client, [128], &digits[..]);
        let ranks = GpuTensor::empty(&client, [128]);
        let peers = GpuTensor::empty(&client, [128]);
        let plane_dims = GpuTensor::empty(&client, [1]);
        probe_plane_kernel::launch::<WgpuRuntime>(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(128),
            digits_t.as_buffer_arg(),
            ranks.as_buffer_arg(),
            peers.as_buffer_arg(),
            plane_dims.as_buffer_arg(),
        );
        let ranks: Vec<u32> = ranks.read_vec();
        let peers: Vec<u32> = peers.read_vec();
        let plane_dim = plane_dims.read_vec::<u32>()[0] as usize;
        for l in 0..128usize {
            let p = l / plane_dim;
            let lane = l % plane_dim;
            // Same-digit lanes must see exactly each other as peers.
            let expect_peers = (0..plane_dim)
                .filter(|&l2| digits[p * plane_dim + l2] == digits[l])
                .fold(0u32, |m, l2| m | (1 << l2));
            assert_eq!(
                peers[l] & (u32::MAX >> (32 - plane_dim)),
                expect_peers,
                "lane {l}: peer mask"
            );
            // hist starts at 16*(plane+1): same-digit lanes get consecutive
            // slots in lane order.
            let rank = (0..lane)
                .filter(|&l2| digits[p * plane_dim + l2] == digits[l])
                .count() as u32;
            assert_eq!(ranks[l], 16 * (p as u32 + 1) + rank, "lane {l}: offset");
        }
    }

    fn assert_argsort_bits(
        client: &ComputeClient<WgpuRuntime>,
        keys_inp: &[u32],
        values_inp: &[u32],
        bits: u32,
    ) {
        let paths = all_paths(client);
        for planes in &paths {
            let keys = GpuTensor::from(client, [keys_inp.len()], keys_inp);
            let values = GpuTensor::from(client, [values_inp.len()], values_inp);
            let scratch = RadixScratch::new(client, keys_inp.len());
            let n = keys_inp.len() as u32;
            let (ret_keys, ret_values) =
                radix_argsort_path(&keys, &values, n, bits, *planes, &scratch);
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
        // Both dispatch paths are exercised so the fallback stays correct on
        // plane-capable machines too.
        let (_gpu, client) = crate::tensor::test_client();
        let paths = all_paths(&client);
        let mut rng = rand::rng();
        for n in [1000usize, 5000, 200_000] {
            let keys_inp: Vec<u32> = (0..n).map(|_| rng.random_range(0..37)).collect();
            let values_inp: Vec<u32> = (0..n as u32).collect();
            let mut reference: Vec<u32> = (0..n as u32).collect();
            reference.sort_by_key(|&i| keys_inp[i as usize]); // stable

            for planes in &paths {
                let keys = GpuTensor::from(&client, [n], &keys_inp[..]);
                let values = GpuTensor::from(&client, [n], &values_inp[..]);
                let scratch = RadixScratch::new(&client, n);
                let (_, ret_values) =
                    radix_argsort_path(&keys, &values, n as u32, 8, *planes, &scratch);
                let ret_values: Vec<u32> = ret_values.read_vec();
                assert_eq!(ret_values, reference, "n={n}: sort must be stable");
            }
        }
    }

    #[test]
    fn test_sorting_partial_bits() {
        let (_gpu, client) = crate::tensor::test_client();
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
        let (_gpu, client) = crate::tensor::test_client();
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
        let (_gpu, client) = crate::tensor::test_client();
        let mut rng = rand::rng();
        let keys_inp: Vec<u32> = (0..500_000)
            .map(|_| rng.random_range(0..1_000_000))
            .collect();
        let values_inp: Vec<u32> = (0..500_000).map(|i| i as u32).collect();
        assert_argsort_bits(&client, &keys_inp, &values_inp, 32);
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
        let (sorted_keys, sorted_vals) = radix_argsort_with(&keys, &vals, n as u32, 32, &scratch);
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
