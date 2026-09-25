//! The active segmentation loop (B3-Seg Algorithm 1).
//!
//! Per iteration: pick a view (round 0 observes from the camera the caller
//! hands in — the user's viewport in the app, the paper's one bootstrap
//! view; afterwards the highest-EIG candidate, over the object once it is
//! foreground and over the whole scene while nothing is foreground yet),
//! pay the one expensive oracle call there, lift the 2D mask to
//! per-Gaussian pseudo-counts, update the Beta posterior. The candidate
//! sphere re-centers on the object every round, so the loop needs no
//! reconstruction cameras. The final 3D mask is {i : a_i > b_i}.
//!
//! The text prompt is baked into the oracle at construction. A run ends
//! early on an empty mask only while nothing is localized yet: an empty
//! mask carries no foreground evidence, so with the object still unseen
//! the remaining rounds would only spend oracle calls. Once a mask has
//! found foreground, all remaining rounds run — empty ones still fold
//! background evidence into the posterior.

use std::sync::Arc;

use anyhow::{Context, ensure};
use core::future::Future;

use super::beta::BetaState;
#[cfg(test)]
use super::beta::map_labels;
use super::views::{self, ObjectLocalization};
use super::{Accumulators, Mask};
use crate::camera::Camera;
use crate::render::{Finalize, RenderScratch, Splats};
use glam::UVec2;
use splat_sort::tensor::GpuTensor;

/// Loop parameters: the square oracle render resolution, the candidate views
/// per iteration (N_cand), and the active iterations (T).
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Square render resolution for candidate scoring and oracle views.
    pub resolution: u32,
    /// Candidate views per iteration (N_cand).
    pub candidates: usize,
    /// Active iterations (T).
    pub iterations: usize,
}

/// One oracle observation, as the loop took it.
pub struct Iteration {
    pub camera: Camera,
    /// Analytic EIG the selected view scored (0 for the round-0 camera,
    /// which is picked blind).
    pub eig: f32,
    /// Foreground pixels in the mask — 0 means this round's evidence is
    /// all-background; if nothing is localized yet, `run_with` ends the
    /// run on such a round.
    pub fg_pixels: usize,
}

/// GPU loop state, sized once for (splat count × resolution).
pub struct Segmenter {
    splats: Arc<Splats>,
    pub state: BetaState,
    cfg: Config,
    scratch: RenderScratch,
    acc: Accumulators,
    /// The round-0 observation view (the paper's one bootstrap view; the
    /// app passes the user's viewport).
    camera: Camera,
    /// False until the first observation lands: round 0 observes blind,
    /// later rounds may survey.
    bootstrapped: bool,
}

impl Segmenter {
    /// Sizes every buffer from `splats` once. The `Arc` is owned for the
    /// segmenter's whole life and the splat count is fixed at construction,
    /// so the accumulators and scratch can never mismatch the count mid-run.
    /// The config is validated before any GPU allocation.
    pub fn new(splats: Arc<Splats>, cfg: Config, camera: Camera) -> anyhow::Result<Self> {
        ensure!(cfg.iterations >= 1, "iterations must be >= 1");
        ensure!(cfg.resolution >= 16, "resolution must be >= 16");
        ensure!(
            cfg.candidates >= 1,
            "candidates must be >= 1: zero would panic select_view_async's cameras[0] index"
        );
        let client = &splats.attributes.client;
        let n = splats.attributes.shape[0];
        let res = UVec2::splat(cfg.resolution);
        Ok(Self {
            state: BetaState::new_uniform(client, n),
            scratch: RenderScratch::new(client, n, res),
            acc: Accumulators::new(client, n),
            splats,
            cfg,
            camera,
            bootstrapped: false,
        })
    }

    /// Score all candidate views around the current localization and return
    /// the best. The current posterior never changes during scoring, so the
    /// candidates compete purely on predicted entropy reduction — rendered
    /// through the async responsibility/EIG paths.
    async fn select_view_async(&mut self, localization: &ObjectLocalization) -> (Camera, f32) {
        let cameras = views::candidates(localization, self.cfg.candidates);
        let mut best = (cameras[0], f32::NEG_INFINITY);
        for camera in &cameras {
            self.splats
                .render_responsibility_async(
                    &mut self.scratch,
                    &mut self.acc,
                    camera,
                    UVec2::splat(self.cfg.resolution),
                )
                .await;
            let eig = self.state.eig_async(&self.acc).await;
            if eig > best.1 {
                best = (*camera, eig);
            }
        }
        best
    }

    /// One observation round: round 0 from the bootstrap camera, then
    /// EIG-best candidates. Awaits sit at the readbacks (`localize_async`,
    /// `bitmap_to_rgb_async`) and the oracle call, which is an async sensor
    /// here. `oracle` borrows the rendered `rgb` slice only to hand it to
    /// the sensor; the sensor's future must therefore consume what it needs
    /// (e.g. copy the bytes) before its first await.
    async fn step_async<O, Fut>(&mut self, oracle: &mut O) -> anyhow::Result<Iteration>
    where
        O: FnMut(&[u8], UVec2, &Camera) -> Fut,
        Fut: Future<Output = anyhow::Result<Vec<u8>>>,
    {
        let res = UVec2::splat(self.cfg.resolution);

        let (camera, eig) = match views::localize_async(&self.splats, &self.state).await {
            Some(loc) => self.select_view_async(&loc).await,
            // Round 0 observes blind from the caller's camera; the uniform
            // posterior has nothing foreground to localize.
            None if !self.bootstrapped => (self.camera, 0.0),
            // Fallback while nothing is foreground: the whole scene as the
            // "object", so the EIG survey can look somewhere new instead of
            // replaying a view that taught nothing.
            None => {
                let (min, max) = self.splats.bounds;
                self.select_view_async(&ObjectLocalization {
                    center: (min + max) * 0.5,
                    radius: (max - min).max_element() * 0.5,
                })
                .await
            }
        };
        self.bootstrapped = true;

        // Render the selected view and hand it to the oracle — the one
        // expensive call per iteration. This round's evidence render is the
        // same camera and resolution, so it reuses the render's
        // intersections instead of re-running the project→sort pipeline.
        // The isects alias `scratch`'s sort buffers (valid until its next
        // `prepare_isects`) and `scratch.projected` must still hold this
        // frame's projection.
        let isects = self
            .splats
            .prepare_isects_async(&mut self.scratch, &camera, res)
            .await;
        self.splats
            .finalize(&self.scratch, &isects, res, Finalize::Rgb);
        let view = bitmap_to_rgb_async(&self.scratch.bitmap, self.cfg.resolution).await;
        let mask_bytes = oracle(&view, res, &camera).await.context("oracle failed")?;
        let mask = Mask::from_bytes(&self.state.a.client, res, &mask_bytes);

        // Lift 2D evidence to 3D pseudo-counts and fold into the posterior —
        // over the render's own intersections, so the fixed-point results are
        // bit-identical to re-preparing the same camera.
        self.splats
            .accumulate_prepared(&self.scratch, &isects, res, &mut self.acc, Some(&mask));
        self.state.update(&self.acc);

        Ok(Iteration {
            camera,
            eig,
            fg_pixels: mask.fg_pixels,
        })
    }

    /// Blocking [`Self::run_with_async`] for host callers: drives the loop
    /// future on this thread, which is what cubecl's own sync reads do
    /// internally. The oracle closure wraps each call in
    /// `std::future::ready`, so the async body's future-must-not-borrow-
    /// `rgb` rule holds trivially.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn run_with(
        &mut self,
        mut oracle: impl FnMut(&[u8], UVec2, &Camera) -> anyhow::Result<Vec<u8>>,
        on_round: impl FnMut(&Iteration, &BetaState) -> bool,
    ) -> anyhow::Result<Vec<Iteration>> {
        cubecl::future::block_on(self.run_with_async(
            move |rgb: &[u8], size: UVec2, camera: &Camera| {
                std::future::ready(oracle(rgb, size, camera))
            },
            on_round,
        ))
    }

    /// `on_round` sees each iteration as it lands, together with the
    /// current posterior, so an interactive caller publishes progress
    /// without re-implementing the loop's stop rules. Returning `false`
    /// stops the run after the round just delivered: the returned rounds and
    /// the segmenter's posterior are the state so far — a cancelled run is a
    /// valid partial result.
    ///
    /// `oracle` is the expensive 2D semantic sensor as a mutable closure:
    /// `rgb` is RGB8, row-major, `size` pixels, rendered from `camera`; the
    /// return is one byte per pixel, nonzero = foreground. Any black box
    /// with this signature plugs in — the text prompt rides in the capture.
    ///
    /// The loop body — the one implementation both entry points run: the
    /// host `run_with` is its `block_on` shim, so the host GPU tests pin
    /// exactly this body and the wasm app executes it directly (invariant
    /// row 45). `oracle` is generic over a future-returning `FnMut` because
    /// the wasm sensor awaits a worker round trip; its future must not
    /// borrow the `rgb` argument (copy before the first await) — that keeps
    /// `Fut` a single type across the higher-ranked calls, which is what
    /// lets this be a plain generic instead of a boxed dyn callback.
    pub async fn run_with_async<O, Fut>(
        &mut self,
        mut oracle: O,
        mut on_round: impl FnMut(&Iteration, &BetaState) -> bool,
    ) -> anyhow::Result<Vec<Iteration>>
    where
        O: FnMut(&[u8], UVec2, &Camera) -> Fut,
        Fut: Future<Output = anyhow::Result<Vec<u8>>>,
    {
        let mut any_fg = false;
        let mut rounds = Vec::with_capacity(self.cfg.iterations);
        for _ in 0..self.cfg.iterations {
            let round = self.step_async(&mut oracle).await?;
            any_fg |= round.fg_pixels > 0;
            let stop = !on_round(&round, &self.state);
            let missed = round.fg_pixels == 0;
            rounds.push(round);
            if stop {
                break;
            }
            if missed && !any_fg {
                break;
            }
        }
        Ok(rounds)
    }

    /// The 3D segmentation so far: {i : a_i > b_i} (the MAP label under the
    /// symmetric prior). Test-only twin of [`Segmenter::posteriors_async`]
    /// (the wasm loop labels via `posteriors_async` + `map_labels` directly);
    /// this survives as the blocking-readback pin, same shape as
    /// [`bitmap_to_rgb`].
    #[cfg(test)]
    fn segmentation(&self) -> Vec<bool> {
        let (a, b) = self.posteriors();
        map_labels(&a, &b)
    }

    /// The current posterior counts, blocking readback (test-only twin of
    /// [`Segmenter::posteriors_async`]).
    #[cfg(test)]
    fn posteriors(&self) -> (Vec<f32>, Vec<f32>) {
        (self.state.a.read_vec(), self.state.b.read_vec())
    }

    /// Async twin of the test-only blocking `posteriors` readback — the
    /// terminal a/b readback the wasm loop's tint step labels with
    /// `map_labels`.
    pub async fn posteriors_async(&self) -> (Vec<f32>, Vec<f32>) {
        (
            self.state.a.read_vec_async().await,
            self.state.b.read_vec_async().await,
        )
    }
}

/// Unpack the RGBA8 bitmap (packed u32 words, rows padded to the scratch
/// stride) into tightly-packed RGB bytes for oracle consumption — the
/// models take RGB, so the alpha byte is dropped while un-striding.
/// `cast_slice` reinterprets natively — byte order matches `to_le_bytes` on
/// every little-endian host wgpu supports. Test-only twin of
/// [`bitmap_to_rgb_async`] (the loop runs the async readback via `run_with`'s
/// `block_on` shim; this survives as the bit-identity pin for the shared
/// `unpack_rgb`).
#[cfg(test)]
pub(crate) fn bitmap_to_rgb(bitmap: &GpuTensor, width: u32) -> Vec<u8> {
    let stride = bitmap.shape[1];
    let height = bitmap.shape[0];
    let words: Vec<u32> = bitmap.read_vec();
    unpack_rgb(&words, stride, height, width)
}

/// Async twin of [`bitmap_to_rgb`] — same unpack over the async readback
/// (cubecl's blocking reads poll once and panic on wasm).
pub(crate) async fn bitmap_to_rgb_async(bitmap: &GpuTensor, width: u32) -> Vec<u8> {
    let stride = bitmap.shape[1];
    let height = bitmap.shape[0];
    let words: Vec<u32> = bitmap.read_vec_async().await;
    unpack_rgb(&words, stride, height, width)
}

/// The pure row-unpack both `bitmap_to_rgb` twins share — keeping it here
/// once is what stops the sync and async bodies from drifting.
fn unpack_rgb(words: &[u32], stride: usize, height: usize, width: u32) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(height * width as usize * 3);
    for row in 0..height {
        for px in bytemuck::cast_slice::<u32, [u8; 4]>(&words[row * stride..][..width as usize]) {
            rgb.extend_from_slice(&px[..3]);
        }
    }
    rgb
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::{sphere_oracle, target};
    use crate::seg::views::{PER_CLUSTER, cluster_scene};

    /// The row-unpack must skip stride padding and drop alpha: a [2, 5]
    /// bitmap with 3 valid pixels per row yields exactly the 18 real RGB
    /// bytes (every 4th byte is alpha), padding dropped.
    #[test]
    fn test_bitmap_to_rgb_handles_row_padding() {
        let (_gpu, client) = crate::gpu_testing::test_client();
        let bitmap = GpuTensor::from(
            &client,
            [2, 5],
            [
                0x0403_0201u32,
                0x0807_0605,
                0x0C0B_0A09,
                0xDEAD_BEEF,
                0xDEAD_BEEF,
                0x1413_1211,
                0x1817_1615,
                0x1C1B_1A19,
                0xDEAD_BEEF,
                0xDEAD_BEEF,
            ],
        );
        let rgb = bitmap_to_rgb(&bitmap, 3);
        assert_eq!(
            rgb,
            (1u8..=12)
                .chain(17..=28)
                .filter(|b| b % 4 != 0)
                .collect::<Vec<u8>>()
        );
    }

    /// Shared fixture with the standard candidate count of 12.
    fn harness(
        iterations: usize,
    ) -> (
        std::sync::MutexGuard<'static, ()>,
        Segmenter,
        Vec<(glam::Vec3, f32)>,
    ) {
        let (gpu, client) = crate::gpu_testing::test_client();
        let (attributes, centers) = cluster_scene();
        let splats = Arc::new(crate::render::opaque_splats(&client, attributes));
        let mut camera = Camera::default();
        camera.frame_bounds(splats.bounds);
        let cfg = Config {
            resolution: 256,
            candidates: 12,
            iterations,
        };
        (gpu, Segmenter::new(splats, cfg, camera).unwrap(), centers)
    }

    /// Config validation fails at the constructor, before any GPU work: zero
    /// candidates would panic `select_view_async`'s `cameras[0]` index.
    #[test]
    fn test_segmenter_rejects_invalid_config() {
        let (_gpu, client) = crate::gpu_testing::test_client();
        let splats = std::sync::Arc::new(crate::render::opaque_splats(
            &client,
            crate::render::sample_opaque_attributes(1, |_| glam::Vec3::ZERO),
        ));
        let base = Config {
            resolution: 256,
            candidates: 12,
            iterations: 2,
        };
        for cfg in [
            Config {
                candidates: 0,
                ..base
            },
            Config {
                resolution: 8,
                ..base
            },
            Config {
                iterations: 0,
                ..base
            },
        ] {
            assert!(
                Segmenter::new(splats.clone(), cfg, Camera::default()).is_err(),
                "{cfg:?}"
            );
        }
    }

    /// `on_round` returning false stops the run at the round boundary: exactly
    /// the delivered rounds come back and the posterior-so-far is a real
    /// partial result, not the uniform prior.
    #[test]
    fn test_run_with_stops_when_on_round_returns_false() {
        let (_gpu, mut segmenter, centers) = harness(6);
        let mut oracle = target(centers[0].0, centers[0].1);
        let mut seen = 0usize;
        let rounds = segmenter
            .run_with(&mut oracle, |_, _| {
                seen += 1;
                seen < 3
            })
            .unwrap();
        assert_eq!(rounds.len(), 3);
        let (a, b) = segmenter.posteriors();
        assert!(
            a.iter()
                .zip(&b)
                .any(|(&a, &b)| (a - 1.0).abs() > 1e-3 || (b - 1.0).abs() > 1e-3),
            "posterior must have moved off uniform"
        );
    }

    /// The full loop must isolate the target cluster: after T observations
    /// with a perfect oracle, the MAP mask matches the ground truth.
    #[test]
    fn test_active_loop_segments_target_cluster() {
        let (_gpu, mut segmenter, centers) = harness(6);
        let n = 3 * PER_CLUSTER;
        let mut oracle = target(centers[0].0, centers[0].1);
        let rounds = segmenter.run_with(&mut oracle, |_, _| true).unwrap();
        assert_eq!(rounds.len(), 6);

        // The round-0 view picks blind; every later round must
        // beat noise and report a positive EIG for its chosen view.
        for r in &rounds[1..] {
            assert!(r.eig > 0.0, "selected view must have positive EIG");
        }

        let labels = segmenter.segmentation();
        let correct = labels
            .iter()
            .enumerate()
            .filter(|&(i, &l)| l == (i < 40))
            .count();
        // Not 100%: cluster-edge splats stick their Gaussian tails past the
        // oracle's ideal object mask and steadily collect counter-evidence —
        // the object-centric limitation the paper itself reports (its mIoU
        // on real scenes is 84.5). A perfect oracle on clean clusters should
        // still clear 90% comfortably.
        assert!(
            correct >= (n * 90) / 100,
            "MAP mask must match ground truth for ≥90% of splats ({correct}/{n})"
        );

        // Posterior means should sit well apart, not mush around 0.5. Edge
        // splats legitimately catch counter-evidence where their Gaussian
        // tails stick out past the oracle's sphere mask, and equal-depth
        // ties blend in schedule-dependent order (see project.rs), which can
        // nudge the EIG ranking between equally valid runs — so the
        // threshold leaves room for that variance.
        let (pa, pb) = segmenter.posteriors();
        let means: Vec<f32> = pa.iter().zip(&pb).map(|(&a, &b)| a / (a + b)).collect();
        let fg_avg: f32 = means[..40].iter().sum::<f32>() / 40.0;
        let bg_max = means[40..].iter().cloned().fold(0.0f32, f32::max);
        assert!(fg_avg > 0.7, "fg posterior average {fg_avg}");
        // No non-target splat may cross the decision boundary at all.
        assert!(bg_max < 0.5, "bg posterior max {bg_max}");
    }

    /// The candidate ranking prefers views that see the (uncertain) object:
    /// after the first observation, the highest-EIG view must look at the
    /// object's neighborhood rather than away from it.
    #[test]
    fn test_selected_views_face_the_object() {
        let (_gpu, mut segmenter, centers) = harness(3);
        let mut oracle = target(centers[0].0, centers[0].1);
        let rounds = segmenter.run_with(&mut oracle, |_, _| true).unwrap();
        for round in &rounds[1..] {
            let to_object = (centers[0].0 - round.camera.position).normalize();
            let forward = (round.camera.rotation * glam::Vec3::Z).normalize();
            assert!(
                forward.dot(to_object) > 0.0,
                "selected camera must face the object hemisphere"
            );
        }
    }

    /// The round-0 view is a constructor argument: a text prompt observes
    /// from the camera the caller hands in (the app passes the user's
    /// viewport), and the loop honors it exactly.
    #[test]
    fn test_initial_camera_is_honored_for_text() {
        let (_gpu, client) = crate::gpu_testing::test_client();
        let (attributes, centers) = cluster_scene();
        let splats = Arc::new(crate::render::opaque_splats(&client, attributes));
        // look_at's minimal arc from +Z to +Z is the identity: this pins that
        // the caller's camera is honored and that look_at is the arc.
        let cam = Camera::look_at(centers[0].0 - glam::Vec3::Z * 3.0, centers[0].0);
        let cfg = Config {
            resolution: 256,
            candidates: 12,
            iterations: 6,
        };
        let mut segmenter = Segmenter::new(splats, cfg, cam).unwrap();
        let mut oracle = target(centers[0].0, centers[0].1);
        let rounds = segmenter.run_with(&mut oracle, |_, _| true).unwrap();
        assert_eq!(rounds[0].camera.position, cam.position);
        assert_eq!(rounds[0].camera.rotation, cam.rotation);
        assert_eq!(rounds.len(), 6);
    }

    /// A text prompt whose oracle returns nothing also stops after one
    /// round: an empty mask teaches nothing wherever it is observed, so
    /// the remaining rounds would spend their oracle calls for nothing.
    #[test]
    fn test_text_prompt_stops_when_oracle_is_blind() {
        let (_gpu, mut segmenter, _centers) = harness(6);
        let mut oracle = sphere_oracle(Vec::new());
        let rounds = segmenter.run_with(&mut oracle, |_, _| true).unwrap();
        assert_eq!(rounds.len(), 1);
        assert_eq!(rounds[0].fg_pixels, 0);
    }

    /// A view that teaches nothing must not be replayed: a one-pixel mask
    /// fires the oracle (the loop keeps going) but tips no splat past
    /// a > b, so after round 0 the EIG survey widens to the whole scene
    /// and the observed camera changes.
    #[test]
    fn test_unproductive_view_is_not_replayed() {
        // The one-pixel-mask oracle: fires (fg_pixels > 0) but tips no
        // splat past a > b.
        let mut oracle = |_rgb: &[u8], size: UVec2, _camera: &Camera| -> anyhow::Result<Vec<u8>> {
            let mut mask = vec![0u8; (size.x * size.y) as usize];
            mask[0] = 1;
            Ok(mask)
        };
        let (_gpu, mut segmenter, _centers) = harness(3);
        let rounds = segmenter.run_with(&mut oracle, |_, _| true).unwrap();
        assert_eq!(rounds.len(), 3);
        assert_eq!(rounds[0].fg_pixels, 1, "the sliver fires the oracle");
        assert_ne!(
            rounds[1].camera.position, rounds[0].camera.position,
            "round 1 must survey the scene, not replay the bootstrap view"
        );
        assert!(rounds[1].eig > 0.0, "the survey must score its candidates");
    }
}
