//! Beta–Bernoulli belief state and the analytic EIG of a candidate view.
//!
//! Every Gaussian i carries a Beta(a_i, b_i) posterior over its object
//! membership probability p_i (a_i: accumulated evidence for, b_i: against).
//! A 2D mask observation later adds pseudo-counts to (a_i, b_i); this module
//! holds, scores, and updates the state.
//!
//! The view score is the paper's analytic expected information gain: pretend
//! the future mask matched the current belief, i.e. Gaussian i (posterior
//! mean m_i) that carries rendering responsibility ε_i would gain
//! ẽ_1 = m_i·ε_i foreground and ẽ_0 = (1−m_i)·ε_i background pseudo-counts.
//! The predicted entropy drop is
//!
//!   ΔH_i = H(a_i, b_i) − H(a_i + ẽ_1, b_i + ẽ_0)
//!
//! and EIG(v) = Σ_i ΔH_i. No segmentation model is involved, which is what
//! makes evaluating ~20 candidate views affordable.
//!
//! H itself is never computed on the GPU: it is read from `entropy_table`
//! — a log-spaced grid computed once per process in f64 — with bilinear
//! interpolation in (ln a, ln b) (`table_entropy`). Evaluating H in f32
//! runs into a numerical stability wall — its terms cancel catastrophically
//! once the counts grow — and shoring f32 up in-kernel takes reams of
//! asymptotic special-casing that buries the math; the f64 table keeps the
//! values exact where they are computed and the GPU side down to a few
//! lines of interpolation. The table's two failure modes are bounded and
//! pinned: interpolation stays under 1e-3 nats on the whole grid, and
//! queries past the last grid point clamp to the edge cell, making ΔH
//! exactly 0 for splats saturated past the table — the same treatment the
//! eps = 0 skip gives invisible ones. In-kernel f64 is not an option:
//! Metal exposes no fp64.
//!
//! The f64 table build is pure std CPU code, so it runs on wasm as-is; the
//! kernel launches are launch-only on both targets. Only the EIG readback
//! is target-split: `eig` (host) and `eig_async` (wasm) share the launch —
//! cubecl's blocking reads poll once and panic on wasm.

use super::Accumulators;
use crate::layout::ELEM_WG;
use cubecl::{calculate_cube_count_elemwise, prelude::*};
use splat_sort::tensor::GpuTensor;
use std::sync::OnceLock;

const EIG_WG: u32 = 256;
const EIG_EPT: u32 = 4;
const EIG_BLOCK: u32 = EIG_WG * EIG_EPT;

/// Entropy table geometry: `GRID_PTS` log-spaced samples per axis over
/// [1, 1024], uniform in (ln a, ln b) with step [`LN_STEP`]. The log grid
/// is densest where H's log-space curvature peaks (small counts), so
/// bilinear interpolation holds under 1e-3 nats across the whole grid.
const GRID_PTS: u32 = 445;
/// ln(1024)/(GRID_PTS−1) as f32. A literal because f64::ln is not
/// const-callable; `test_table_grid_geometry_matches_b_max` pins it.
const LN_STEP: f32 = 0.015_611_423;

/// ln Γ(x), Lanczos g = 7, in f64 — the single reference for the H table
/// and the parity tests. No reflection branch: Beta parameters only grow
/// from 1, so x ≥ 1 always.
fn lgamma(x: f64) -> f64 {
    let z = x - 1.0;
    let c = [
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let mut series = 0.999_999_999_999_809_9;
    for (k, ck) in c.iter().enumerate() {
        series += ck / (z + k as f64 + 1.0);
    }
    let t = z + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + series.ln()
}

/// ψ(x): recurrence up the ladder into the asymptotic regime, then
/// ln x − 1/(2x) − 1/(12x²) + 1/(120x⁴) − 1/(252x⁶).
fn digamma(mut x: f64) -> f64 {
    let mut r = 0.0;
    while x < 6.0 {
        r -= 1.0 / x;
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    r + x.ln() - 0.5 * inv - inv2 / 12.0 + inv2 * inv2 / 120.0 - inv2 * inv2 * inv2 / 252.0
}

/// Differential entropy of Beta(a, b) in f64 — the table's generator and
/// the tests' oracle. The textbook form is safe here: at the largest
/// tabulated counts (a+b ≈ 2·10³) the cancelling terms are ±1.5e4, and f64
/// carries ~1e-12 of rounding where f32 would lose the digits.
fn beta_entropy(a: f64, b: f64) -> f64 {
    let s = a + b;
    lgamma(a) + lgamma(b) - lgamma(s) - (a - 1.0) * digamma(a) - (b - 1.0) * digamma(b)
        + (s - 2.0) * digamma(s)
}

/// The H grid: row i is a = exp(i·[`LN_STEP`]), column j likewise for b —
/// `beta_entropy` over the outer product of one log-spaced axis.
fn build_entropy_table() -> Vec<f32> {
    let n = GRID_PTS as usize;
    let axis: Vec<f64> = (0..n)
        .map(|i| ((i as f32 * LN_STEP) as f64).exp())
        .collect();
    let mut table = Vec::with_capacity(n * n);
    for &a in &axis {
        for &b in &axis {
            table.push(beta_entropy(a, b) as f32);
        }
    }
    table
}

static ENTROPY_TABLE: OnceLock<Vec<f32>> = OnceLock::new();

/// The shared H table — one f64 build per process, uploaded by every
/// `BetaState` and read by both the EIG kernel and the parity tests.
fn entropy_table() -> &'static [f32] {
    ENTROPY_TABLE.get_or_init(build_entropy_table)
}

/// H(a, b) from `entropy_table`: bilinear over the log-spaced grid.
/// Past the last grid point the query clamps to the edge cells, so a
/// saturated splat gets ΔH exactly 0 — the eps = 0 skip's mirror for
/// saturated ones. a, b < 1 never occurs (the prior starts at 1 and counts
/// only grow), but the clamps make OOB indices impossible regardless.
#[cube]
fn table_entropy(a: f32, b: f32, table: &[f32]) -> f32 {
    let xa = a.ln() / LN_STEP;
    let xb = b.ln() / LN_STEP;
    let ia = xa.floor().max(0f32).min((GRID_PTS - 2) as f32);
    let ib = xb.floor().max(0f32).min((GRID_PTS - 2) as f32);
    let ta = (xa - ia).clamp(0f32, 1f32);
    let tb = (xb - ib).clamp(0f32, 1f32);
    let row = (ia as u32 * GRID_PTS + ib as u32) as usize;
    let h00 = table[row];
    let h10 = table[row + GRID_PTS as usize];
    let h01 = table[row + 1];
    let h11 = table[row + GRID_PTS as usize + 1];
    h00 * (1f32 - ta) * (1f32 - tb)
        + h10 * ta * (1f32 - tb)
        + h01 * (1f32 - ta) * tb
        + h11 * ta * tb
}

/// Per-Gaussian ΔH, reduced per workgroup into `partials`. The paper proves
/// ΔH ≥ 0 (adaptive monotonicity), but FMA contraction makes the two
/// entropy evaluations carry different rounding, so the difference is
/// clamped at zero; the residual noise is the 5e-5-nat table interpolation
/// error.
#[cube(launch)]
fn entropy_delta_kernel(
    n: u32,
    scale: f32,
    a: &[f32],
    b: &[f32],
    table: &[f32],
    resp_bits: &[u32],
    partials: &mut [f32],
) {
    let wg = CUBE_POS as u32;
    if wg >= n.div_ceil(EIG_BLOCK) {
        terminate!();
    }

    let mut sum = 0f32;
    #[unroll]
    for e in 0..EIG_EPT {
        let i = wg * EIG_BLOCK + UNIT_POS + e * EIG_WG;
        let mut d = 0f32;
        if i < n {
            let eps = resp_bits[i as usize] as f32 / scale;
            // eps = 0 leaves (a, b) untouched, so ΔH is exactly zero —
            // skip both evaluations (the common invisible-splat case).
            if eps > 0f32 {
                // Posterior scale form, algebraically equal to
                // a + m·ε / b + (1−m)·ε but without the (1 − m)
                // cancellation when one count dominates: the background
                // increment keeps full relative precision at any a:b ratio.
                let k = 1f32 + eps / (a[i as usize] + b[i as usize]);
                let a2 = a[i as usize] * k;
                let b2 = b[i as usize] * k;
                d = (table_entropy(a[i as usize], b[i as usize], table)
                    - table_entropy(a2, b2, table))
                .max(0f32);
            }
        }
        sum += d;
    }

    let mut reduce = Shared::<[f32]>::new_slice(EIG_WG as usize);
    reduce[UNIT_POS as usize] = sum;
    sync_cube();
    if UNIT_POS == 0 {
        let mut total = 0f32;
        for t in 0..EIG_WG {
            total += reduce[t as usize];
        }
        partials[wg as usize] = total;
    }
}

/// Conjugate Beta–Bernoulli update in place: (a, b) ← (a + e₁, b + e₀),
/// with the pseudo-counts converted back from the evidence's fixed point.
#[cube(launch)]
fn beta_update_kernel(scale: f32, fg_bits: &[u32], bg_bits: &[u32], a: &mut [f32], b: &mut [f32]) {
    let i = ABSOLUTE_POS_X;
    if i < a.len() as u32 {
        a[i as usize] += fg_bits[i as usize] as f32 / scale;
        b[i as usize] += bg_bits[i as usize] as f32 / scale;
    }
}

/// Per-Gaussian Beta posteriors over object membership.
pub struct BetaState {
    pub a: GpuTensor,
    pub b: GpuTensor,
    /// The `entropy_table` grid uploaded for the kernels.
    table: GpuTensor,
    /// Workgroup partial sums of [`BetaState::eig`]'s kernel — one f32 per
    /// EIG workgroup, allocated once for the state's life and CPU-summed
    /// into each total.
    partials: GpuTensor,
}

impl BetaState {
    /// The paper's uninformative prior: a_i = b_i = 1 for every Gaussian, so
    /// every posterior mean starts at 0.5 and the MAP decision a > b starts
    /// uncommitted.
    pub fn new_uniform(client: &Client, total: usize) -> Self {
        let table = entropy_table();
        Self {
            a: GpuTensor::from(client, [total], vec![1f32; total]),
            b: GpuTensor::from(client, [total], vec![1f32; total]),
            table: GpuTensor::from(client, [table.len()], table),
            partials: GpuTensor::empty(client, [total.div_ceil(EIG_BLOCK as usize).max(1)]),
        }
    }

    /// Launch the ΔH reduction for one candidate's responsibility map.
    /// Shared by the [`BetaState::eig`]/[`BetaState::eig_async`] twins —
    /// the readback is the only difference between them, so the launch (the
    /// math) lives here once. Returns the workgroup count the `partials`
    /// buffer holds.
    fn launch_eig(&self, acc: &Accumulators) -> u32 {
        let client = &self.a.client;
        let n = self.a.shape[0];
        // Each workgroup covers EIG_BLOCK elements (EPT per thread); size the
        // launch for exactly num_wgs workgroups, not num_wgs*EIG_EPT.
        let num_wgs = n.div_ceil(EIG_BLOCK as usize) as u32;
        let count = calculate_cube_count_elemwise(
            client,
            num_wgs as usize * EIG_WG as usize,
            CubeDim::new_1d(EIG_WG),
        );
        entropy_delta_kernel::launch(
            client,
            count,
            CubeDim::new_1d(EIG_WG),
            n as u32,
            acc.scale,
            self.a.as_buffer_arg(),
            self.b.as_buffer_arg(),
            self.table.as_buffer_arg(),
            acc.bits.as_buffer_arg(),
            self.partials.as_buffer_arg(),
        );
        num_wgs
    }

    /// Analytic EIG for one candidate's responsibility map: returns Σ_i ΔH_i
    /// — the blocking twin of [`BetaState::eig_async`], host-only because
    /// cubecl's blocking reads poll once and panic on wasm.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn eig(&self, acc: &Accumulators) -> f32 {
        let num_wgs = self.launch_eig(acc);
        // One f32 per workgroup: a few KB over PCIe, summed on the CPU.
        let partials: Vec<f32> = self.partials.read_vec();
        partials[..num_wgs as usize].iter().sum()
    }

    /// Async twin of [`BetaState::eig`] — same kernel via `launch_eig`,
    /// async readback (cubecl's blocking reads poll once and panic on wasm).
    pub async fn eig_async(&self, acc: &Accumulators) -> f32 {
        let num_wgs = self.launch_eig(acc);
        // One f32 per workgroup: a few KB over PCIe, summed on the CPU.
        let partials: Vec<f32> = self.partials.read_vec_async().await;
        partials[..num_wgs as usize].iter().sum()
    }

    /// Absorb one observation: the selected view's mask evidence as
    /// foreground/background pseudo-counts (conjugate to the Beta — no
    /// likelihood computation, just accumulation).
    pub(crate) fn update(&self, acc: &Accumulators) {
        let client = &self.a.client;
        let n = self.a.shape[0];
        beta_update_kernel::launch(
            client,
            calculate_cube_count_elemwise(client, n, CubeDim::new_1d(ELEM_WG)),
            CubeDim::new_1d(ELEM_WG),
            acc.scale,
            acc.fg.as_buffer_arg(),
            acc.bg.as_buffer_arg(),
            self.a.as_buffer_arg(),
            self.b.as_buffer_arg(),
        );
    }
}

/// MAP foreground labels from Beta posteriors: foreground iff `a > b`.
/// Ties are background. Segmentation and cut_object shared this predicate
/// inline; this is the one definition. The views' localize applies the same
/// tie rule inline — a change here must move all three. Callers index the
/// result by splat, so a length mismatch is a bug — fail loud.
pub fn map_labels(a: &[f32], b: &[f32]) -> Vec<bool> {
    assert_eq!(a.len(), b.len(), "posterior length mismatch");
    a.iter().zip(b).map(|(&a, &b)| a > b).collect()
}

/// BetaState from CPU slices — the seg tests' synthetic posterior, shared by
/// the beta and localize test suites.
#[cfg(test)]
pub(crate) fn state(client: &Client, a: &[f32], b: &[f32]) -> BetaState {
    let s = BetaState::new_uniform(client, a.len());
    s.a.write(a);
    s.b.write(b);
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng};

    /// map_labels is the single MAP decision: strict a > b, ties background.
    #[test]
    fn test_map_labels_ties_are_background() {
        assert_eq!(
            map_labels(&[1.0, 2.0, 0.5], &[1.0, 1.0, 0.5]),
            [false, true, false]
        );
    }

    /// Table edge coordinate: a_max = exp((GRID_PTS−1)·LN_STEP) ≈ 1024.
    fn a_max() -> f32 {
        (((GRID_PTS - 1) as f32 * LN_STEP) as f64).exp() as f32
    }

    /// Interpolation band vs the f64 oracle; measured worst 5.1e-5 nats,
    /// ~20× margin. Pinned by `test_entropy_table_band_vs_f64`.
    const INT_BAND: f32 = 1e-3;

    /// Test-only probe exposing the production table lookup.
    #[cube(launch)]
    fn probe_entropy(table: &[f32], a: &[f32], b: &[f32], out: &mut [f32]) {
        let i = ABSOLUTE_POS_X as usize;
        if i < a.len() {
            out[i] = table_entropy(a[i], b[i], table);
        }
    }

    fn delta_ref(a: f64, b: f64, eps: f64) -> f64 {
        // Same scale form as the kernel — see entropy_delta_kernel.
        let k = 1.0 + eps / (a + b);
        beta_entropy(a, b) - beta_entropy(a * k, b * k)
    }

    /// Accumulator with per-Gaussian `eps` quantized at `scale` in `bits`.
    fn eps_acc(client: &Client, eps: &[f32], scale: f32) -> Accumulators {
        let bits: Vec<u32> = eps.iter().map(|&e| (e * scale) as u32).collect();
        Accumulators {
            bits: GpuTensor::from(client, [eps.len()], &bits[..]),
            fg: GpuTensor::empty(client, [eps.len()]),
            bg: GpuTensor::empty(client, [eps.len()]),
            scale,
        }
    }

    /// H(a_i, b_i) per pair, through the probe kernel.
    fn probe(client: &Client, a: &[f32], b: &[f32]) -> Vec<f32> {
        let table = GpuTensor::from(client, [entropy_table().len()], entropy_table());
        let a_t = GpuTensor::from(client, [a.len()], a);
        let b_t = GpuTensor::from(client, [b.len()], b);
        let out = GpuTensor::empty(client, [a.len()]);
        probe_entropy::launch(
            client,
            calculate_cube_count_elemwise(client, a.len(), CubeDim::new_1d(64)),
            CubeDim::new_1d(64),
            table.as_buffer_arg(),
            a_t.as_buffer_arg(),
            b_t.as_buffer_arg(),
            out.as_buffer_arg(),
        );
        out.read_vec()
    }

    /// LN_STEP is a literal (f64::ln is not const-callable); this pins its
    /// drift: the grid must span [1, 1024] as documented.
    #[test]
    fn test_table_grid_geometry_matches_b_max() {
        let want = (1024f64).ln() / (GRID_PTS - 1) as f64;
        // One f32 ulp of rounding on top of the f64 comparison.
        assert!(
            (LN_STEP as f64 - want).abs() < 2e-9,
            "LN_STEP {} drifted from ln(1024)/{} = {want}",
            LN_STEP,
            GRID_PTS - 1
        );
        let amax = a_max() as f64;
        assert!(
            (amax - 1024.0).abs() < 0.5,
            "table must end at a_max ≈ 1024, got {amax}"
        );
    }

    /// The interpolation golden band: H from the GPU table must track the
    /// f64 oracle within [`INT_BAND`] nats over the whole grid domain —
    /// random log-uniform pairs for the interior, the edge lattice for the
    /// high-curvature corners.
    #[test]
    fn test_entropy_table_band_vs_f64() {
        let (_gpu, client) = crate::gpu_testing::test_client();
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5EED_0007);
        let u_max = (GRID_PTS - 1) as f32 * LN_STEP;
        let mut a: Vec<f32> = (0..2048)
            .map(|_| (rng.random::<f32>() * u_max).exp())
            .collect();
        let mut b: Vec<f32> = (0..2048)
            .map(|_| (rng.random::<f32>() * u_max).exp())
            .collect();
        let edge = [1.0f32, 1.05, 1.5, 4.0, 21.0, 300.0, a_max()];
        for &x in &edge {
            for &y in &edge {
                a.push(x);
                b.push(y);
            }
        }
        let got = probe(&client, &a, &b);
        let worst = got
            .iter()
            .zip(&a)
            .zip(&b)
            .map(|((&g, &x), &y)| (g - beta_entropy(x as f64, y as f64) as f32).abs())
            .fold(0f32, f32::max);
        assert!(
            worst < INT_BAND,
            "interpolation band exceeded: {worst:.2e} nats"
        );
    }

    #[test]
    fn test_beta_entropy_matches_reference() {
        let (_gpu, client) = crate::gpu_testing::test_client();
        // H(1,1) = 0 exactly; the rest sit in the high-curvature region.
        let cases = [
            (1.0f32, 1.0f32), // uniform on (0,1): H = 0
            (1.5, 1.5),
            (2.0, 5.0),
            (9.0, 1.0),
            (30.0, 30.0),
            (100.0, 3.0),
        ];
        let (a, b): (Vec<f32>, Vec<f32>) = cases.iter().copied().unzip();
        let got = probe(&client, &a, &b);
        for (k, &hi) in got.iter().enumerate() {
            let (ai, bi) = cases[k];
            let want = beta_entropy(ai as f64, bi as f64) as f32;
            assert!(
                (hi - want).abs() < INT_BAND,
                "H({ai},{bi}): gpu {hi} vs ref {want}"
            );
        }
        assert!(got[0].abs() < 1e-6, "H(Beta(1,1)) must be 0");
    }

    /// Beyond the grid every saturated splat clamps to the edge cell, so
    /// its ΔH is 0 up to f32 ln/exp noise — deep saturation can never
    /// manufacture a spurious gain. Edge cells stay exact vs the oracle.
    #[test]
    fn test_beta_entropy_clamps_beyond_table() {
        let (_gpu, client) = crate::gpu_testing::test_client();
        let amax = a_max();
        let got = probe(
            &client,
            &[5.0e5, 1.0e7, amax, 1.0, 1.0],
            &[5.0e5, 1.0e7, amax, amax, 1.0e6],
        );
        // Queries far past the edge clamp hard (t = 1 exactly) and land
        // bit-equal on the corner cell.
        assert_eq!(
            got[0], got[1],
            "saturated pairs must clamp to the same corner value"
        );
        let corner = beta_entropy(amax as f64, amax as f64) as f32;
        for k in [0, 2] {
            // k = 2 rides the f32 ln/exp roundtrip at t ≈ 0.9999 — a hair
            // off the exact corner, still the same cell.
            assert!(
                (got[k] - corner).abs() < 1e-5,
                "clamped pair must equal H(a_max, a_max): {} vs {corner}",
                got[k]
            );
        }
        let edge = beta_entropy(1.0, amax as f64) as f32;
        for k in [3, 4] {
            assert!(
                (got[k] - edge).abs() < 1e-5,
                "(1, ≫a_max) must clamp to H(1, a_max): {} vs {edge}",
                got[k]
            );
        }
    }

    /// A synthetic posterior with known totals: EIG must match the f64
    /// mirror (the tolerance folds the 5e-5-nat per-ΔH interpolation error,
    /// random sign across splats, into a relative budget), and a fully
    /// visible scene must score above a half-visible one.
    #[test]
    fn test_eig_matches_reference_and_orders() {
        let (_gpu, client) = crate::gpu_testing::test_client();
        let n = 1000usize;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5EED_0006);
        let a: Vec<f32> = (0..n).map(|_| 1.0 + rng.random::<f32>() * 20.0).collect();
        let b: Vec<f32> = (0..n).map(|_| 1.0 + rng.random::<f32>() * 20.0).collect();
        // Half the Gaussians visible with weight ≤ 0.5, the rest invisible.
        let eps: Vec<f32> = (0..n)
            .map(|i| {
                if i % 2 == 0 {
                    rng.random::<f32>() * 0.5
                } else {
                    0.0
                }
            })
            .collect();

        let state = state(&client, &a, &b);

        let scale = 4096.0f32;
        let acc = eps_acc(&client, &eps, scale);

        let eig = state.eig(&acc);

        let want: f64 = (0..n)
            .map(|i| delta_ref(a[i] as f64, b[i] as f64, eps[i] as f64))
            .sum();
        assert!(
            (eig as f64 - want).abs() < 5e-3 * want.abs().max(1.0),
            "EIG gpu {eig} vs ref {want}"
        );

        // Same state, every Gaussian twice as visible: strictly more gain.
        let eps2: Vec<f32> = eps.iter().map(|&e| e * 2.0).collect();
        let acc2 = eps_acc(&client, &eps2, scale);
        let eig2 = state.eig(&acc2);
        assert!(
            eig2 > eig,
            "more visibility must gain more: {eig2} vs {eig}"
        );
        assert!(eig > 0.0);
    }

    /// ΔH is nonnegative — the paper's adaptive monotonicity — and shrinks
    /// as the posterior concentrates (diminishing returns). Each state holds
    /// one Gaussian, so its EIG is exactly that splat's ΔH.
    #[test]
    fn test_eig_monotone_and_diminishing() {
        let (_gpu, client) = crate::gpu_testing::test_client();
        // Fresh uniform, half-decided, nearly decided; all shown the same
        // responsibility.
        let states = [(1.0f32, 1.0f32), (3.0, 1.0), (50.0, 1.0)];

        let scale = 4096.0;
        let acc = eps_acc(&client, &[0.7], scale);
        let deltas: Vec<f32> = states
            .iter()
            .map(|&(a, b)| state(&client, &[a], &[b]).eig(&acc))
            .collect();

        for &d in &deltas {
            assert!(d >= 0.0, "ΔH must be nonnegative, got {d}");
        }
        assert!(deltas[0] > deltas[1]);
        assert!(deltas[1] > deltas[2]);
    }

    /// Zero visibility must yield no gain beyond f32 noise: a' == a makes the
    /// two entropy evaluations mathematically identical.
    #[test]
    fn test_eig_zero_for_invisible() {
        let (_gpu, client) = crate::gpu_testing::test_client();
        let state = BetaState::new_uniform(&client, 512);
        let acc = eps_acc(&client, &[0.0; 512], 4096.0);
        let eig = state.eig(&acc);
        assert!(
            eig.abs() < 1e-4,
            "invisible candidate must have ~zero EIG, got {eig}"
        );
    }

    /// Late-round EIG sanity: once a dominant splat saturates (a+b ≈ 10⁶),
    /// a view touching it must gain ~nothing while a view over a fresh
    /// posterior keeps its full gain. Saturation past the table clamps to
    /// ΔH = 0 exactly; the fresh splat's ΔH holds the interpolation band.
    #[test]
    fn test_eig_ranking_survives_dominant_splats() {
        let (_gpu, client) = crate::gpu_testing::test_client();
        // Splats 0–1 saturated, 2–3 fresh uniform.
        let a = [5.0e5f32, 5.0e5, 1.0, 1.0];
        let b = [5.0e5f32, 5.0e5, 1.0, 1.0];
        let state = state(&client, &a, &b);
        let scale = 4096.0f32;

        // View A touches only the saturated splat: eps = 0 skip elsewhere,
        // clamped ΔH = 0 here.
        let eig_a = state.eig(&eps_acc(&client, &[0.5, 0.0, 0.0, 0.0], scale));
        assert!(eig_a < 1e-2, "view A must score ~zero EIG, got {eig_a}");

        // View B touches only a fresh splat: true ΔH₂ = H(1,1) − H(1.25,1.25).
        let want_fresh = (beta_entropy(1.0, 1.0) - beta_entropy(1.25, 1.25)) as f32;
        let eig_b = state.eig(&eps_acc(&client, &[0.0, 0.0, 0.5, 0.0], scale));
        assert!(
            (eig_b - want_fresh).abs() < 5e-3,
            "fresh splat ΔH {eig_b} vs ref {want_fresh}"
        );
        assert!(
            eig_b > eig_a,
            "fresh-posterior view must outrank the saturated one: {eig_b} vs {eig_a}"
        );
    }
}
