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

/// The text-prompted pipeline both front-ends run: fetch the oracle models
/// (the first call downloads them), bake the prompt into a detector+SAM2
/// chain, run `cfg.iterations` active rounds. Returns the segmenter —
/// `segmentation()` for labels, `state` for the posterior buffers.
/// `on_round` returning false stops the run (see `Segmenter::run_with`).
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
        let boxes = detector.detect(rgb, size.x, size.y)?;
        let Some(best) = boxes.first() else {
            return Ok(vec![0; (size.x * size.y) as usize]);
        };
        sam.segment(rgb, size.x, size.y, best)
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
        if cached.iter().any(|f| !std::path::Path::new(f).exists()) {
            if std::env::var_os("SPLATFIELD_ALLOW_MISSING_ASSETS").is_none() {
                panic!(
                    "missing model/asset files {cached:?} — set SPLATFIELD_ALLOW_MISSING_ASSETS=1 to skip"
                );
            }
            eprintln!("skipping: models/asset not cached");
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
}
