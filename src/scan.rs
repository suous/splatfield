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

// One launch beats the hierarchical path's three while the serial chunk stays
// cheap: 64K cells is ~2M keys at the radix sort's widest plane (n/32 cells).
// Above that the single workgroup loses — measured at ~12% of a frame with the
// hierarchical path bypassed — so the threshold is real, not conservative.
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

/// Per-block partial sums for both scan entry points, sized for a maximum
/// element count so repeated scans allocate nothing.
pub struct ScanScratch {
    block_sums: GpuTensor,
}

impl ScanScratch {
    pub fn new(client: &ComputeClient<WgpuRuntime>, max_elems: usize) -> Self {
        let max_blocks = (max_elems as u32).div_ceil(SCAN_BLOCK);
        Self {
            block_sums: GpuTensor::empty(client, [max_blocks as usize]),
        }
    }
}

/// Gather counts into depth order (written into `offsets` as raw counts) and
/// reduce each block to `block_sums`.
#[cube(launch)]
fn gather_block_sums(
    n: u32,
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
            let v = counts[gids[idx as usize] as usize];
            offsets[idx as usize] = v;
            sum += v;
        }
    }
    partials[UNIT_POS as usize] = sum;
    sync_cube();
    block_total(&partials, block_sums, wg);
}

/// Single-workgroup exclusive scan of the first `num_blocks` cells of `buf`,
/// in place — a serial chunk per thread is plenty (even 4M elements need
/// only 4096 cells). Also serves the radix sort, whose bin-major counters
/// flatten to one addressable array (see count_kernel in sort.rs).
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

/// Reduce each `SCAN_BLOCK`-cell block of `buf` to a per-block total. The
/// gather-free counterpart of [`gather_block_sums`].
#[cube(launch)]
fn reduce_block_sums(n: u32, buf: &[u32], block_sums: &mut [u32]) {
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
            sum += buf[idx as usize];
        }
    }
    partials[UNIT_POS as usize] = sum;
    sync_cube();
    block_total(&partials, block_sums, wg);
}

/// Convert the gathered counts in `offsets` to exclusive offsets in place:
/// each thread only reads indices it later overwrites, and per-thread ranges
/// are disjoint.
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

    // `block_sums` is already exclusively scanned; thread 0 extends the scan
    // to per-thread bases.
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

/// Single-workgroup serial scan launch — one dispatch, the shape that makes
/// the serial path win below `SERIAL_SCAN_CELLS`.
fn launch_serial_scan(client: &ComputeClient<WgpuRuntime>, n: u32, buf: &GpuTensor) {
    scan_serial::launch::<WgpuRuntime>(
        client,
        CubeCount::new_single(),
        CubeDim::new_1d(SCAN_WG),
        n,
        buf.as_buffer_arg(),
    );
}

/// Exclusive-scan the first `n` cells of `buf` in place. Exposed for the
/// radix sort's per-pass counter scan; cell counts past `SERIAL_SCAN_CELLS`
/// take the hierarchical path.
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
    reduce_block_sums::launch::<WgpuRuntime>(
        client,
        cube_count,
        cube_dim,
        n,
        buf.as_buffer_arg(),
        scratch.block_sums.as_buffer_arg(),
    );
    scan_and_apply(client, n, scratch, buf);
}

/// Exclusive scan of `counts[gids[i]]` into `offsets[i]`, for `i in 0..n`.
/// `gids` is typically the depth permutation (rank -> source index), making
/// `offsets` the per-rank emission base for the map kernel. `offsets` is
/// scratch for the duration of the call: it transiently holds gathered counts.
pub(crate) fn exclusive_scan_gather(
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
    debug_assert!(
        (n as usize) <= offsets.shape[0],
        "offsets undersized for {n} elements"
    );
    let cube_dim = CubeDim::new_1d(SCAN_WG);
    let cube_count =
        calculate_cube_count_elemwise(&client, n as usize, CubeDim::new_1d(SCAN_BLOCK));
    gather_block_sums::launch::<WgpuRuntime>(
        &client,
        cube_count,
        cube_dim,
        n,
        gids.as_buffer_arg(),
        counts.as_buffer_arg(),
        offsets.as_buffer_arg(),
        scratch.block_sums.as_buffer_arg(),
    );

    // Serial path: one launch instead of three (see SERIAL_SCAN_CELLS).
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
    launch_serial_scan(client, n.div_ceil(SCAN_BLOCK), &scratch.block_sums);
    apply_block_offsets::launch::<WgpuRuntime>(
        client,
        cube_count,
        cube_dim,
        n,
        scratch.block_sums.as_buffer_arg(),
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
