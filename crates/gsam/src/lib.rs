// Same contract as the root crate: `pub` only for the exported API
// (Detector, Sam2, the model-path helpers); everything else pub(crate).
#![deny(unreachable_pub)]

// The ort session layer is native-only: ort cannot target wasm32. The web
// build runs the same graphs through onnxruntime-web instead (`ortweb`,
// driven over js_sys) — everything above the session layer (ModelStore,
// tokenizer, prompt tensors, pre/post-processing) is cfg-dual and shared.
mod dino;
pub mod ortweb;
mod preprocess;
mod sam2;
#[cfg(not(target_arch = "wasm32"))]
mod session;

#[cfg(not(target_arch = "wasm32"))]
use anyhow::{Result, bail};
#[cfg(not(target_arch = "wasm32"))]
use half::{f16, slice::HalfFloatSliceExt, vec::HalfFloatVecExt};
#[cfg(not(target_arch = "wasm32"))]
use ort::{
    session::Session,
    tensor::TensorElementType as TE,
    value::{DynValue, Tensor, ValueType},
};
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;

pub use dino::{Detection, Detector};
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
/// release, keyed by cache-relative path ([`REQUIRED_FILES`]). On the web it
/// lives only as long as the page — the durable copy is the OPFS cache
/// (`splatfield::opfs`); on the host it holds what fetch extracted.
#[derive(Debug, Default)]
pub struct ModelStore {
    files: std::collections::HashMap<&'static str, Vec<u8>>,
}

impl ModelStore {
    pub fn new() -> Self {
        Self::default()
    }

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

/// Index of the greatest element; ties go to the last maximum, NaN never
/// wins. Both backends' postprocessors share the selection rule so
/// detections can't drift between platforms.
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

/// App model cache root (`{cache}/splatfield/models`, per the platform
/// convention), overridable with `SPLATFIELD_MODELS_DIR`.
#[cfg(not(target_arch = "wasm32"))]
pub fn models_dir() -> anyhow::Result<PathBuf> {
    if let Some(dir) = std::env::var_os("SPLATFIELD_MODELS_DIR") {
        return Ok(PathBuf::from(dir));
    }
    dirs::cache_dir()
        .map(|dir| dir.join("splatfield/models"))
        .ok_or_else(|| anyhow::anyhow!("no user cache directory on this platform"))
}

/// The model release identity: the fetch URL's release tag (splatfield::fetch)
/// AND the cache leaf directory under `models_dir()`. One const so a new
/// release moves URL and cache together and the previous leaf survives
/// for instant revert.
#[cfg(not(target_arch = "wasm32"))]
pub const RELEASE_TAG: &str = "models-v1";

/// The directory this release's models live in — always the versioned
/// leaf, never the root: a leaf pinned to the root (e.g. by adopting a
/// legacy flat layout in place) would make every future release a
/// silent no-op, its files matching the presence check by name forever.
#[cfg(not(target_arch = "wasm32"))]
pub fn release_dir() -> anyhow::Result<PathBuf> {
    Ok(models_dir()?.join(RELEASE_TAG))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn grounding_file() -> anyhow::Result<PathBuf> {
    Ok(release_dir()?.join("grounding_dino_tiny/onnx/model_q4f16.onnx"))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn sam_file(stem: &str) -> anyhow::Result<PathBuf> {
    Ok(release_dir()?.join(format!("sam2_tiny/onnx/{stem}_q4f16.onnx")))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn grounding_tokenizer() -> anyhow::Result<PathBuf> {
    Ok(release_dir()?.join("grounding_dino_tiny/tokenizer.json"))
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn input_ty(t: &ValueType) -> Result<TE> {
    match t {
        ValueType::Tensor { ty, .. } => Ok(*ty),
        t => bail!("unsupported value type {t:?}"),
    }
}

/// Element types the session's graph inputs declare, in input order.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn input_dtypes(session: &Session) -> Result<Vec<TE>> {
    session
        .inputs()
        .iter()
        .map(|i| input_ty(i.dtype()))
        .collect()
}

/// The encoders' shared input: squash-resize RGB to the graph's fixed net
/// size, ImageNet-normalize to CHW, and wrap in the session's element type.
/// The f16 graphs (both releases are q4f16) normalize straight into the
/// tensor buffer — the f32 CHW plane never exists.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn encoder_input(
    session: &Session,
    rgb: &[u8],
    (w, h): (u32, u32),
    net: u32,
) -> Result<DynValue> {
    let want = input_ty(session.inputs()[0].dtype())?;
    // Normalized f32 has no meaningful i64 cast — fail loud, not with a
    // cast tensor (parity: preprocess::encoder_input, test-pinned there).
    if want == TE::Int64 {
        bail!("pixel_values as int64 is not a valid encoder input");
    }
    let canvas = preprocess::resize_rgb8(rgb, (w, h), (net, net));
    let shape = [1, 3, net as i64, net as i64];
    if want == TE::Float16 {
        let buf = preprocess::normalize_chw::<f16>(&canvas, f16::from_f32);
        return Ok(Tensor::from_array((shape.to_vec(), buf))?.into_dyn());
    }
    make_input(&shape, preprocess::normalize_chw(&canvas, |x| x), want)
}

/// Build an input tensor in the element type the session declares.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn make_input(shape: &[i64], data: Vec<f32>, want: TE) -> Result<DynValue> {
    Ok(match want {
        TE::Float32 => Tensor::from_array((shape.to_vec(), data))?.into_dyn(),
        TE::Float16 => {
            let buf: Vec<f16> = Vec::from_f32_slice(&data);
            Tensor::from_array((shape.to_vec(), buf))?.into_dyn()
        }
        TE::Int64 => Tensor::from_array((
            shape.to_vec(),
            data.iter().map(|&x| x as i64).collect::<Vec<_>>(),
        ))?
        .into_dyn(),
        t => bail!("unsupported input dtype {t:?}"),
    })
}

/// Extract a tensor as f32 (f16 widened), returning (dims, data).
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn extract_f32(v: &DynValue) -> Result<(Vec<i64>, Vec<f32>)> {
    extract_f32_with(v, |_, len| Ok(0..len))
}

/// [`extract_f32`] with the flat range picked from the dims/len before any
/// widening: a [.., C, H, W] output whose one used channel is `chan` (SAM2's
/// argmax-IoU mask) widens len/C elements instead of all of them. Host twin
/// of ortweb's `into_f32_slice`; the fallible-range contract is its docs.
#[cfg(not(target_arch = "wasm32"))]
fn extract_f32_with(
    v: &DynValue,
    take: impl FnOnce(&[i64], usize) -> Result<std::ops::Range<usize>>,
) -> Result<(Vec<i64>, Vec<f32>)> {
    match v.dtype() {
        ValueType::Tensor {
            ty: TE::Float32, ..
        } => {
            let (shape, data) = v.try_extract_tensor::<f32>()?;
            let range = take(shape, data.len())?;
            Ok((shape.to_vec(), data[range].to_vec()))
        }
        ValueType::Tensor {
            ty: TE::Float16, ..
        } => {
            let (shape, data) = v.try_extract_tensor::<f16>()?;
            let range = take(shape, data.len())?;
            Ok((shape.to_vec(), data[range].to_f32_vec()))
        }
        t => bail!("unsupported output dtype {t:?}"),
    }
}

// Host-only: `RELEASE_TAG`/`release_dir` are host-gated (download-side);
// the wasm build ships no release dir to resolve.
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::{ModelStore, RELEASE_TAG, REQUIRED_FILES, argmax, release_dir};

    /// The release leaf is ALWAYS the versioned directory under the
    /// models root — resolving a legacy flat root in place would pin the
    /// leaf to the root: its files pass the presence check by name
    /// forever, unverified, and every future release becomes a silent
    /// no-op.
    #[test]
    fn test_release_dir_is_versioned_leaf() {
        let dir = release_dir().unwrap();
        assert_eq!(dir.file_name().unwrap(), RELEASE_TAG);
    }

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
        let mut store = ModelStore::new();
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
        let mut store = ModelStore::new();
        assert!(store.insert("models/evil.onnx", b"x".to_vec()).is_err());
        assert!(store.insert("../../etc/passwd", b"x".to_vec()).is_err());
        assert_eq!(store.missing(), REQUIRED_FILES.to_vec());
    }

    #[test]
    fn complete_store_has_no_missing_and_get_works() {
        let mut store = ModelStore::new();
        for f in REQUIRED_FILES {
            store.insert(f, vec![0u8; 4]).unwrap();
        }
        assert!(store.is_complete());
        assert_eq!(store.missing(), Vec::<&'static str>::new());
        assert_eq!(store.get(REQUIRED_FILES[0]).unwrap().len(), 4);
        assert!(store.get("nope").is_err());
    }
}
