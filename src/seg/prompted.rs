//! Text-prompted 2D segmentation front-end (ONNX Runtime, via the `gsam`
//! crate).
//!
//! B3-Seg's Bayesian update (`seg::beta`) consumes a binary foreground mask
//! per observed view. [`run_text`]'s sensor closure produces that mask from a
//! natural-language target description: GroundingDINO localizes the
//! described object in a (rendered) view, the best-confidence detection box
//! becomes a SAM2 box prompt, and the decoded instance mask comes back at
//! source resolution.
//!
//! Deferred from the paper (Sec 3.4): CLIP re-ranks the per-box candidates
//! instead of taking the detector's best box, and conditions SAM2 on the
//! current posterior as a soft logit prior. The exported SAM2 decoder has
//! no mask-prior input, so the prior has no input path here. Multi-instance
//! prompts ("chairs") segment one instance — the detector's best.
//!
//! Models are HuggingFace transformers ONNX exports in the platform app
//! cache (`gsam::release_dir`); first use downloads them — see `fetch`. The
//! fetch-before-oracle-load ordering lives in [`run_text`]; that glue is
//! exercised only end-to-end (asset-gated) — the loop itself is pinned by
//! the synthetic `sphere_oracle` tests.

use gsam::{Detector, Sam2};

use super::active::Segmenter;
use crate::camera::Camera;
use glam::UVec2;

/// The text-prompted pipeline — uncalled on any target since the worker
/// took over; kept as the pinned native-oracle parity reference with the
/// host fetch front-end. Fetch the oracle models (the first call
/// downloads them), bake the prompt into a detector+SAM2 chain, run
/// `cfg.iterations` active rounds. Returns the segmenter — label via
/// `posteriors_async` + `map_labels`, as the wasm loop does; `state` holds
/// the posterior buffers. `on_round` returning false stops the run (see
/// `Segmenter::run_with`).
pub fn run_text(
    splats: std::sync::Arc<crate::render::Splats>,
    prompt: &str,
    cfg: super::active::Config,
    camera: crate::camera::Camera,
    on_progress: &mut dyn FnMut(&str),
    on_round: &mut dyn FnMut(&super::active::Iteration, &super::beta::BetaState) -> bool,
) -> anyhow::Result<super::active::Segmenter> {
    crate::fetch::ensure_models(on_progress)?;
    // The sensor closure bakes the prompt in — `Detector::load` tokenizes
    // and fixes it, so one target description serves the whole run.
    // Session creation is tens of seconds of silent ONNX work; announce it.
    on_progress("loading GroundingDINO (detector)…");
    let mut detector = Detector::load(prompt)?;
    on_progress("loading SAM2…");
    let mut sam = Sam2::load()?;
    // GroundingDINO localizes the target in the view and the best-confidence
    // box prompts one SAM2 decode; nothing found means an all-background mask
    // and no decode — the loop's stop rule reads the zero count, so the two
    // cases need not be distinguished.
    let oracle = move |rgb: &[u8], size: UVec2, _camera: &Camera| {
        let detections = detector.detect(rgb, size.x, size.y)?;
        let Some(best) = detections.first() else {
            return Ok(vec![0; (size.x * size.y) as usize]);
        };
        sam.segment(rgb, size.x, size.y, &best.xyxy)
    };
    let mut seg = Segmenter::new(splats, cfg, camera)?;
    seg.run_with(oracle, on_round)?;
    Ok(seg)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The loose golden foreground-fraction band (±~10%) the bus-asset
    /// smoke test asserts against — guards against gross regressions, not
    /// exact numerics.
    const BUS_GOLDEN: std::ops::RangeInclusive<f64> = 0.09..=0.115;

    /// Shared asset gate for the oracle smoke tests: true = go. Absent
    /// models/assets fail loud unless the env var restores the skip.
    fn gate_or_skip(cached: &[String]) -> bool {
        if cached.iter().all(|f| std::path::Path::new(f).exists()) {
            return true;
        }
        if std::env::var_os("SPLATFIELD_ALLOW_MISSING_ASSETS").is_none() {
            panic!(
                "missing model/asset files {cached:?} — set SPLATFIELD_ALLOW_MISSING_ASSETS=1 to skip"
            );
        }
        eprintln!("skipping: models/asset not cached");
        false
    }

    /// A real SAM2 decode with an explicit box prompt against the bus
    /// asset — the decode stage the text chain feeds detector boxes into.
    /// Foreground fraction must land in [`BUS_GOLDEN`]. Fails loud when the
    /// cached models/asset are absent (the env var restores the skip).
    #[test]
    fn oracle_smoke_on_bus() {
        let asset = "data/bus.jpg";
        let cached: Vec<String> = [
            gsam::sam_file("vision_encoder"),
            gsam::sam_file("prompt_encoder_mask_decoder"),
            gsam::grounding_file(),
        ]
        .iter()
        .filter_map(|p| p.as_deref().ok())
        .map(|p| p.display().to_string())
        .chain(std::iter::once(asset.to_string()))
        .collect();
        if !gate_or_skip(&cached) {
            return;
        }

        let mut sam = Sam2::load().unwrap();
        eprintln!("sam2 loaded");
        let image = image::ImageReader::open(asset)
            .unwrap()
            .with_guessed_format()
            .unwrap()
            .decode()
            .unwrap()
            .to_rgb8();
        let (w, h) = (image.width(), image.height());
        let rgb = image.as_raw();
        let total = (w * h) as usize;
        eprintln!("image {w}x{h}, running sam2 segment");
        let mask = sam
            .segment(rgb, w, h, &[0.0, 0.0, w as f32 * 0.4, h as f32])
            .unwrap();
        eprintln!("segment done");
        let frac = mask.iter().filter(|&&b| b == 255).count() as f64 / total as f64;
        eprintln!("box prompt (left 40%): {:.2}%", frac * 100.0);
        assert!(
            BUS_GOLDEN.contains(&frac),
            "golden band: {:.2}%",
            frac * 100.0
        );
    }

    /// The text chain end to end against the cached models and the bear
    /// scene — the test leg the module doc promises for [`run_text`]: the
    /// fetch-before-load ordering, the detector→SAM2 sensor closure, and one
    /// active round through `run_with`. Structural assertions only (a round
    /// ran, the posterior is splat-sized); decode numerics are pinned by
    /// [`oracle_smoke_on_bus`], the loop math by the synthetic
    /// `sphere_oracle` tests. Fails loud when the models/asset are absent
    /// (the env var restores the skip).
    #[test]
    fn run_text_smoke_on_bear() {
        let scene = "data/bear.3d71a266.sog";
        let cached: Vec<String> = [
            gsam::grounding_file(),
            gsam::grounding_tokenizer(),
            gsam::sam_file("vision_encoder"),
            gsam::sam_file("prompt_encoder_mask_decoder"),
        ]
        .iter()
        .filter_map(|p| p.as_deref().ok())
        .map(|p| p.display().to_string())
        .chain(std::iter::once(scene.to_string()))
        .collect();
        if !gate_or_skip(&cached) {
            return;
        }

        let rounds = std::cell::Cell::new(0usize);
        // The loop renders on the GPU: take the serialized test client so
        // this run orders against the other GPU tests.
        let (_lock, client) = crate::gpu_testing::test_client();
        let splats = std::sync::Arc::new(
            crate::load_scene(
                std::path::Path::new(scene).extension().unwrap_or_default(),
                std::fs::File::open(scene).unwrap(),
            )
            .unwrap()
            .upload(&client),
        );
        let n = splats.attributes.shape[0];
        let mut camera = crate::camera::Camera::default();
        camera.frame_bounds(splats.bounds);
        let seg = run_text(
            splats,
            "bear",
            crate::seg::active::Config {
                resolution: 128,
                candidates: 3,
                iterations: 1,
            },
            camera,
            &mut |stage| eprintln!("{stage}"),
            &mut |_round, _state| {
                rounds.set(rounds.get() + 1);
                false
            },
        )
        .unwrap();
        // The terminal posterior readback is splat-sized — the wasm loop's
        // posteriors_async + map_labels spelling.
        let (a, _b) = cubecl::future::block_on(seg.posteriors_async());
        assert_eq!(a.len(), n);
        assert_eq!(rounds.get(), 1);
    }
}
