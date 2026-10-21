// Same contract as the root crate: `pub` only for the exported API
// (ModelStore, REQUIRED_FILES, and on wasm the Detector/Sam2 sessions);
// everything else pub(crate).
#![deny(unreachable_pub)]

// Web inference runs the graphs through onnxruntime-web (`ortweb`, driven
// over js_sys). There is no native backend: the workspace targets wasm32
// only — host builds exist for the test suite, which exercises the
// platform-neutral halves (decode logic, ModelStore, the pin tables).
mod dino;
pub mod ortweb;
mod preprocess;
mod sam2;

pub use dino::Detection;
#[cfg(target_arch = "wasm32")]
pub use dino::Detector;
#[cfg(target_arch = "wasm32")]
pub use sam2::Sam2;

/// Every file the release zip must provide, keyed relative to the release
/// leaf (the zip wraps everything in one top-level `models/` directory;
/// store keys drop that prefix). GroundingDINO graph + tokenizer, then SAM2
/// encoder/decoder graphs each with its external-weights sibling. The order
/// is the pairing contract: `splatfield::fetch::FILE_SHA256` pins these
/// paths positionally (pinned by tests on both halves).
pub const REQUIRED_FILES: [&str; 6] = [
    "grounding_dino_tiny/onnx/model_q4f16.onnx",
    "grounding_dino_tiny/tokenizer.json",
    "sam2_tiny/onnx/vision_encoder_q4f16.onnx",
    "sam2_tiny/onnx/vision_encoder_q4f16.onnx_data",
    "sam2_tiny/onnx/prompt_encoder_mask_decoder_q4f16.onnx",
    "sam2_tiny/onnx/prompt_encoder_mask_decoder_q4f16.onnx_data",
];

/// In-memory stand-in for the on-disk model cache: the verified files of one
/// release, keyed by cache-relative path ([`REQUIRED_FILES`]). It lives only
/// as long as the page — the durable copy is the OPFS cache
/// (`splatfield::opfs`).
#[derive(Debug, Default)]
pub struct ModelStore {
    files: std::collections::HashMap<&'static str, Vec<u8>>,
}

impl ModelStore {
    /// Install one release file. Keys outside [`REQUIRED_FILES`] are
    /// rejected: the store models exactly the pinned release, nothing else —
    /// a stray key would silently pass `is_complete` later while never
    /// reaching a session.
    pub fn insert(&mut self, path: &'static str, bytes: Vec<u8>) -> anyhow::Result<()> {
        anyhow::ensure!(REQUIRED_FILES.contains(&path), "unknown model file: {path}");
        self.files.insert(path, bytes);
        Ok(())
    }

    pub fn get(&self, path: &str) -> anyhow::Result<&Vec<u8>> {
        self.files
            .get(path)
            .ok_or_else(|| anyhow::anyhow!("model file absent from store: {path}"))
    }

    /// Release files not (yet) in the store, in [`REQUIRED_FILES`] order.
    pub fn missing(&self) -> Vec<&'static str> {
        REQUIRED_FILES
            .iter()
            .filter(|f| !self.files.contains_key(*f))
            .copied()
            .collect()
    }

    pub fn is_complete(&self) -> bool {
        self.missing().is_empty()
    }
}

/// The graph-input count both backends' loads ensure before positional
/// dtype indexing — a short graph must error, not index blind (a wasm
/// abort). `what` names the graph; `>=` keeps appended-input exports loadable.
#[cfg(target_arch = "wasm32")]
pub(crate) fn ensure_graph_inputs(declared: usize, want: usize, what: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        declared >= want,
        "{what} graph declares {declared} inputs, expected at least {want}"
    );
    Ok(())
}

/// Index of the greatest element; ties go to the last maximum, NaN never
/// wins. Both backends' postprocessors share the selection rule so
/// detections can't drift between platforms.
#[cfg(any(target_arch = "wasm32", test))]
pub(crate) fn argmax(xs: &[f32]) -> usize {
    let mut best = 0;
    let mut max = f32::NEG_INFINITY;
    for (i, &x) in xs.iter().enumerate() {
        if x >= max {
            max = x;
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::{ModelStore, REQUIRED_FILES, argmax};

    /// Pins the postprocessors' shared argmax contract: ties resolve
    /// last-wins and NaNs never win.
    #[test]
    fn test_argmax_ties_last_wins_and_skips_nan() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 2);
        assert_eq!(argmax(&[f32::NAN, 1.0, 0.5]), 1);
        assert_eq!(argmax(&[0.5, f32::NAN, 1.0]), 2);
        assert_eq!(argmax(&[1.0, f32::NAN]), 0);
        assert_eq!(argmax(&[-0.5]), 0);
        assert_eq!(argmax(&[f32::NAN; 3]), 0);
    }

    #[test]
    fn missing_lists_required_files_in_order() {
        let mut store = ModelStore::default();
        store
            .insert(REQUIRED_FILES[1], b"tokenizer".to_vec())
            .unwrap();
        assert_eq!(
            store.missing(),
            vec![
                REQUIRED_FILES[0],
                REQUIRED_FILES[2],
                REQUIRED_FILES[3],
                REQUIRED_FILES[4],
                REQUIRED_FILES[5]
            ]
        );
        assert!(!store.is_complete());
    }

    #[test]
    fn insert_rejects_paths_outside_the_release() {
        let mut store = ModelStore::default();
        assert!(store.insert("models/evil.onnx", b"x".to_vec()).is_err());
        assert!(store.insert("../../etc/passwd", b"x".to_vec()).is_err());
        assert_eq!(store.missing(), REQUIRED_FILES.to_vec());
    }

    #[test]
    fn complete_store_has_no_missing_and_get_works() {
        let mut store = ModelStore::default();
        for f in REQUIRED_FILES {
            store.insert(f, vec![0u8; 4]).unwrap();
        }
        assert!(store.is_complete());
        assert_eq!(store.missing(), Vec::<&'static str>::new());
        assert_eq!(store.get(REQUIRED_FILES[0]).unwrap().len(), 4);
        assert!(store.get("nope").is_err());
    }
}
