//! GPU exclusive prefix sum with a fused gather.
//!
//! The render pipeline uses this to turn per-splat tile counts into
//! intersection emission offsets: after the depth sort, counts are gathered
//! into depth order (`counts[gids[i]]`) and exclusively scanned, so the map
//! kernel can emit each splat's intersections contiguously in depth order.
//! A stable tile sort then yields tile-major, depth-minor order with no
//! separate per-isect depth sort.
use crate::tensor::GpuTensor;
use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

const SCAN_WG: u32 = 256;
const SCAN_EPT: u32 = 4;
const SCAN_BLOCK: u32 = SCAN_WG * SCAN_EPT;

// Measured crossover, not a guess: below this, one launch beats the
// hierarchical three (64K cells ≈ 2M keys at the radix sort's n/32 cells);
// above it, the single workgroup loses ~12% of a frame.
const SERIAL_SCAN_CELLS: u32 = 1 << 16;

/// Exclusive-scan `len` cells of `a` in place, starting from `seed`.
/// Single-threaded: callers gate it on `UNIT_POS == 0` + `sync_cube`.
#[cube]
pub(crate) fn serial_exclusive(a: &mut [u32], len: u32, seed: u32) {
    let mut running = seed;
    for t in 0..len {
        let c = a[t as usize];
        a[t as usize] = running;
        running += c;
    }
}

/// Thread 0 collapses a block's per-thread partials into `block_sums[wg]`.
#[cube]
fn block_total(partials: &[u32], block_sums: &mut [u32], wg: u32) {
    if UNIT_POS == 0 {
        let mut total = 0u32;
        for t in 0..SCAN_WG {
            total += partials[t as usize];
        }
        block_sums[wg as usize] = total;
    }
}

/// Per-block totals for the in-place scan, sized for a maximum element count
/// so repeated scans allocate nothing.
pub struct ScanScratch {
    totals: GpuTensor,
}

impl ScanScratch {
    pub fn new(client: &ComputeClient<WgpuRuntime>, max_elems: usize) -> Self {
        let max_blocks = (max_elems as u32).div_ceil(SCAN_BLOCK);
        Self {
            totals: GpuTensor::empty(client, [max_blocks as usize]),
        }
    }
}

/// Sums each block's cells. With `gather`, cell `idx` contributes
/// `counts[gids[idx]]` and the raw count lands in `offsets[idx]` (see
/// [`apply_block_offsets`]); without it, `counts` is read in place.
/// `gather` is comptime: cubecl compiles one kernel per call site.
#[cube(launch)]
fn block_sums_kernel(
    n: u32,
    #[comptime] gather: bool,
    gids: &[u32],
    counts: &[u32],
    offsets: &mut [u32],
    block_sums: &mut [u32],
) {
    let wg = CUBE_POS as u32;
    if wg >= n.div_ceil(SCAN_BLOCK) {
        terminate!();
    }

    let mut partials = Shared::<[u32]>::new_slice(SCAN_WG as usize);
    let base = SCAN_BLOCK * wg + UNIT_POS * SCAN_EPT;
    let mut sum = 0u32;
    for e in 0..SCAN_EPT {
        let idx = base + e;
        if idx < n {
            if gather {
                let v = counts[gids[idx as usize] as usize];
                offsets[idx as usize] = v;
                sum += v;
            } else {
                sum += counts[idx as usize];
            }
        }
    }
    partials[UNIT_POS as usize] = sum;
    sync_cube();
    block_total(&partials, block_sums, wg);
}

/// Single-workgroup exclusive scan of the first `num_blocks` cells of `buf`,
/// in place. A serial chunk per thread scales fine: 4M elements need only
/// 4096 cells. Also serves the radix sort, whose bin-major counters flatten
/// to one array (see count_kernel in sort.rs).
#[cube(launch)]
fn scan_serial(num_blocks: u32, block_sums: &mut [u32]) {
    let mut partials = Shared::<[u32]>::new_slice(SCAN_WG as usize);

    let chunk = num_blocks.div_ceil(SCAN_WG);
    let lo = UNIT_POS * chunk;
    let hi = (lo + chunk).min(num_blocks);

    let mut sum = 0u32;
    for i in lo..hi {
        sum += block_sums[i as usize];
    }
    partials[UNIT_POS as usize] = sum;
    sync_cube();

    if UNIT_POS == 0 {
        serial_exclusive(&mut partials, SCAN_WG, 0u32);
    }
    sync_cube();

    let mut running = partials[UNIT_POS as usize];
    for i in lo..hi {
        let c = block_sums[i as usize];
        block_sums[i as usize] = running;
        running += c;
    }
}

/// Convert gathered counts in `offsets` to exclusive offsets, in place —
/// race-free: per-thread ranges are disjoint, and each thread reads only
/// indices it later overwrites.
#[cube(launch)]
fn apply_block_offsets(n: u32, block_sums: &[u32], offsets: &mut [u32]) {
    let wg = CUBE_POS as u32;
    if wg >= n.div_ceil(SCAN_BLOCK) {
        terminate!();
    }

    let mut thread_sums = Shared::<[u32]>::new_slice(SCAN_WG as usize);
    let base = SCAN_BLOCK * wg + UNIT_POS * SCAN_EPT;
    let mut sum = 0u32;
    for e in 0..SCAN_EPT {
        let idx = base + e;
        if idx < n {
            sum += offsets[idx as usize];
        }
    }
    thread_sums[UNIT_POS as usize] = sum;
    sync_cube();

    // `block_sums` is already exclusive; thread 0 extends it to per-thread bases.
    if UNIT_POS == 0 {
        serial_exclusive(&mut thread_sums, SCAN_WG, block_sums[wg as usize]);
    }
    sync_cube();

    let mut running = thread_sums[UNIT_POS as usize];
    for e in 0..SCAN_EPT {
        let idx = base + e;
        if idx < n {
            let v = offsets[idx as usize];
            offsets[idx as usize] = running;
            running += v;
        }
    }
}

/// Single-workgroup serial scan launch — one dispatch.
fn launch_serial_scan(client: &ComputeClient<WgpuRuntime>, n: u32, buf: &GpuTensor) {
    scan_serial::launch::<WgpuRuntime>(
        client,
        CubeCount::new_single(),
        CubeDim::new_1d(SCAN_WG),
        n,
        buf.as_buffer_arg(),
    );
}

/// Exclusive-scan the first `n` cells of `buf` in place — the radix sort's
/// per-pass counter scan. Past `SERIAL_SCAN_CELLS`, the hierarchical path
/// takes over.
pub(crate) fn exclusive_scan_buf(
    client: &ComputeClient<WgpuRuntime>,
    buf: &GpuTensor,
    n: u32,
    scratch: &ScanScratch,
) {
    if n <= SERIAL_SCAN_CELLS {
        launch_serial_scan(client, n, buf);
        return;
    }
    let cube_dim = CubeDim::new_1d(SCAN_WG);
    let cube_count = calculate_cube_count_elemwise(client, n as usize, CubeDim::new_1d(SCAN_BLOCK));
    // The comptime `gather = false` expansion writes nothing through `offsets`,
    // so the dead slot aliases the `counts` input (buffer arg 1: gids, counts,
    // …) instead of needing a real sink buffer — cubecl binds an alias to the
    // input's resource, it registers no extra wgpu binding.
    block_sums_kernel::launch::<WgpuRuntime>(
        client,
        cube_count,
        cube_dim,
        n,
        false,
        buf.as_buffer_arg(),
        buf.as_buffer_arg(),
        BufferArg::alias(1, n as usize),
        scratch.totals.as_buffer_arg(),
    );
    scan_and_apply(client, n, scratch, buf);
}

/// Exclusive scan of `counts[gids[i]]` into `offsets[i]`, for `i in 0..n`.
/// `gids` is typically the depth permutation, making `offsets` the per-rank
/// emission base for the map kernel. `offsets` is scratch during the call:
/// it transiently holds the gathered counts.
pub fn exclusive_scan_gather(
    gids: &GpuTensor,
    counts: &GpuTensor,
    offsets: &GpuTensor,
    n: u32,
    scratch: &ScanScratch,
) {
    if n == 0 {
        return;
    }
    let client = gids.client.clone();
    assert!(
        (n as usize) <= offsets.shape[0],
        "offsets undersized: scanned {n} elements, capacity {}",
        offsets.shape[0]
    );
    let cube_dim = CubeDim::new_1d(SCAN_WG);
    let cube_count =
        calculate_cube_count_elemwise(&client, n as usize, CubeDim::new_1d(SCAN_BLOCK));
    block_sums_kernel::launch::<WgpuRuntime>(
        &client,
        cube_count,
        cube_dim,
        n,
        true,
        gids.as_buffer_arg(),
        counts.as_buffer_arg(),
        offsets.as_buffer_arg(),
        scratch.totals.as_buffer_arg(),
    );

    if n <= SERIAL_SCAN_CELLS {
        launch_serial_scan(&client, n, offsets);
        return;
    }
    scan_and_apply(&client, n, scratch, offsets);
}

/// Shared hierarchical tail: exclusive-scan the per-block totals, then add
/// each block's offset into its cells.
fn scan_and_apply(
    client: &ComputeClient<WgpuRuntime>,
    n: u32,
    scratch: &ScanScratch,
    buf: &GpuTensor,
) {
    let cube_dim = CubeDim::new_1d(SCAN_WG);
    let cube_count = calculate_cube_count_elemwise(client, n as usize, CubeDim::new_1d(SCAN_BLOCK));
    launch_serial_scan(client, n.div_ceil(SCAN_BLOCK), &scratch.totals);
    apply_block_offsets::launch::<WgpuRuntime>(
        client,
        cube_count,
        cube_dim,
        n,
        scratch.totals.as_buffer_arg(),
        buf.as_buffer_arg(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngExt;

    #[test]
    fn test_scan_gather_matches_cpu() {
        let (_gpu, client) = crate::tensor::test_client();
        let mut rng = rand::rng();
        for n in [1usize, 5, 1024, 1025, 100_000] {
            let counts: Vec<u32> = (0..n).map(|_| rng.random_range(0..20)).collect();
            // Random permutation as the gather indices.
            let mut gids: Vec<u32> = (0..n as u32).collect();
            for i in (1..n).rev() {
                let j = rng.random_range(0..=i);
                gids.swap(i, j);
            }
            let gids_t = GpuTensor::from(&client, [n], &gids[..]);
            let counts_t = GpuTensor::from(&client, [n], &counts[..]);
            let offsets_t = GpuTensor::empty(&client, [n]);
            let scratch = ScanScratch::new(&client, n);
            exclusive_scan_gather(&gids_t, &counts_t, &offsets_t, n as u32, &scratch);

            let offsets: Vec<u32> = offsets_t.read_vec();
            let mut running = 0u32;
            for i in 0..n {
                assert_eq!(
                    offsets[i], running,
                    "n={n}: wrong exclusive offset at rank {i}"
                );
                running += counts[gids[i] as usize];
            }
        }
    }

    #[test]
    fn test_scan_buf_hierarchical_matches_cpu() {
        let (_gpu, client) = crate::tensor::test_client();
        // 1025 stays on the serial branch, 70_000 crosses SERIAL_SCAN_CELLS
        // into the hierarchical one with a non-whole number of blocks.
        for n in [1025, 70_000] {
            let counts: Vec<u32> = (0..n as u32).map(|i| i % 4).collect();
            let buf_t = GpuTensor::from(&client, [n], &counts[..]);
            let scratch = ScanScratch::new(&client, n);
            exclusive_scan_buf(&client, &buf_t, n as u32, &scratch);

            let scanned: Vec<u32> = buf_t.read_vec();
            let mut running = 0u32;
            for i in 0..n {
                assert_eq!(scanned[i], running, "n={n}: wrong exclusive offset at {i}");
                running += counts[i];
            }
        }
    }

    #[test]
    fn test_scan_zero() {
        let (_gpu, client) = crate::tensor::test_client();
        let gids = GpuTensor::from(&client, [1], &[0u32][..]);
        let counts = GpuTensor::from(&client, [1], &[7u32][..]);
        let offsets = GpuTensor::from(&client, [1], &[0xDEAD_BEEFu32][..]);
        let scratch = ScanScratch::new(&client, 1);
        exclusive_scan_gather(&gids, &counts, &offsets, 0, &scratch);
        let out: Vec<u32> = offsets.read_vec();
        assert_eq!(out, vec![0xDEAD_BEEF], "n=0 must not touch the output");
    }
}
