// Same contract as the root crate: `pub` only for the exported API
// (Detector, Sam2, the model-path helpers); everything else pub(crate).
#![deny(unreachable_pub)]

mod dino;
mod preprocess;
mod sam2;
mod session;

use anyhow::{Result, bail};
use half::{f16, vec::HalfFloatVecExt};
use ort::{
    session::Session,
    tensor::TensorElementType as TE,
    value::{DynValue, Tensor, ValueType},
};
use std::path::PathBuf;

pub use dino::Detector;
pub use sam2::Sam2;

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
pub const RELEASE_TAG: &str = "models-v1";

/// The directory this release's models live in — always the versioned
/// leaf, never the root: a leaf pinned to the root (e.g. by adopting a
/// legacy flat layout in place) would make every future release a
/// silent no-op, its files matching the presence check by name forever.
pub fn release_dir() -> anyhow::Result<PathBuf> {
    Ok(models_dir()?.join(RELEASE_TAG))
}

pub fn grounding_file() -> anyhow::Result<PathBuf> {
    Ok(release_dir()?.join("grounding_dino_tiny/onnx/model_q4f16.onnx"))
}

pub fn sam_file(stem: &str) -> anyhow::Result<PathBuf> {
    Ok(release_dir()?.join(format!("sam2_tiny/onnx/{stem}_q4f16.onnx")))
}

pub fn grounding_tokenizer() -> anyhow::Result<PathBuf> {
    Ok(release_dir()?.join("grounding_dino_tiny/tokenizer.json"))
}

pub(crate) fn input_ty(t: &ValueType) -> Result<TE> {
    match t {
        ValueType::Tensor { ty, .. } => Ok(*ty),
        t => bail!("unsupported value type {t:?}"),
    }
}

/// Element types the session's graph inputs declare, in input order.
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
pub(crate) fn encoder_input(
    session: &Session,
    rgb: &[u8],
    (w, h): (u32, u32),
    net: u32,
) -> Result<DynValue> {
    let want = input_ty(session.inputs()[0].dtype())?;
    let canvas = preprocess::resize_rgb8(rgb, (w, h), (net, net));
    let shape = [1, 3, net as i64, net as i64];
    if want == TE::Float16 {
        let buf = preprocess::normalize_chw::<f16>(&canvas, f16::from_f32);
        return Ok(Tensor::from_array((shape.to_vec(), buf))?.into_dyn());
    }
    make_input(&shape, preprocess::normalize_chw(&canvas, |x| x), want)
}

/// Build an input tensor in the element type the session declares.
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
pub(crate) fn extract_f32(v: &DynValue) -> Result<(Vec<i64>, Vec<f32>)> {
    extract_f32_with(v, |_, len| 0..len)
}

/// [`extract_f32`] with the flat range picked from the dims/len before any
/// widening: a [.., C, H, W] output whose one used channel is `chan` (SAM2's
/// argmax-IoU mask) widens len/C elements instead of all of them.
fn extract_f32_with(
    v: &DynValue,
    take: impl FnOnce(&[i64], usize) -> std::ops::Range<usize>,
) -> Result<(Vec<i64>, Vec<f32>)> {
    match v.dtype() {
        ValueType::Tensor {
            ty: TE::Float32, ..
        } => {
            let (shape, data) = v.try_extract_tensor::<f32>()?;
            let range = take(shape, data.len());
            Ok((shape.to_vec(), data[range].to_vec()))
        }
        ValueType::Tensor {
            ty: TE::Float16, ..
        } => {
            let (shape, data) = v.try_extract_tensor::<f16>()?;
            let range = take(shape, data.len());
            Ok((
                shape.to_vec(),
                data[range].iter().map(|&h| h.to_f32()).collect(),
            ))
        }
        t => bail!("unsupported output dtype {t:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::{RELEASE_TAG, argmax, release_dir};

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
}
