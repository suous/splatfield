//! SAM2 box-prompted segmentation: image → encoder → decoder → mask bytes.
//!
//! Logic preserved from the desktop oracle (commit 1a06ff6): squash to
//! 1024², encoder feed → 3 outputs (image embed + two high-res feats),
//! decoder feed = "no point" ([0,0]/[-1]) + box scaled to the NET frame +
//! the 3 encoder outputs, argmax-IoU channel, bilinear resize to source,
//! 0.5 threshold → 255/0 bytes.
//!
//! The session layer is the only backend split: host builds two `ort` CPU
//! sessions from the release cache and re-runs the encoder per segment;
//! wasm builds `ortweb::Session`s from the [`crate::ModelStore`] and caches
//! the encoder outputs between decodes. The box scaling and mask bytes are
//! shared verbatim.

use anyhow::Result;

#[cfg(target_arch = "wasm32")]
use crate::{ModelStore, argmax, ortweb, preprocess};
#[cfg(not(target_arch = "wasm32"))]
use crate::{
    argmax, encoder_input, extract_f32, extract_f32_with, input_dtypes, make_input, sam_file,
    session,
};
#[cfg(not(target_arch = "wasm32"))]
use ort::session::{Session, SessionInputValue, builder::GraphOptimizationLevel};
#[cfg(not(target_arch = "wasm32"))]
use ort::tensor::TensorElementType as TE;

/// Fixed input size; the aspect ratio is squashed, and box prompts are
/// pixel coords in this frame.
pub(crate) const NET: u32 = 1024;

/// Store keys (REQUIRED_FILES entries) for the graphs, plus — deliberately
/// distinct — the bare external-weights location strings the ONNX protos
/// reference at session-create time. `ModelStore::get` takes the keyed form
/// only; passing the proto-ref form is the "absent from store" failure mode.
#[cfg(target_arch = "wasm32")]
const VISION_ONNX: &str = "sam2_tiny/onnx/vision_encoder_q4f16.onnx";
#[cfg(target_arch = "wasm32")]
const VISION_DATA_KEY: &str = "sam2_tiny/onnx/vision_encoder_q4f16.onnx_data";
#[cfg(target_arch = "wasm32")]
const VISION_DATA_REF: &str = "vision_encoder_q4f16.onnx_data";
#[cfg(target_arch = "wasm32")]
const DECODER_ONNX: &str = "sam2_tiny/onnx/prompt_encoder_mask_decoder_q4f16.onnx";
#[cfg(target_arch = "wasm32")]
const DECODER_DATA_KEY: &str = "sam2_tiny/onnx/prompt_encoder_mask_decoder_q4f16.onnx_data";
#[cfg(target_arch = "wasm32")]
const DECODER_DATA_REF: &str = "prompt_encoder_mask_decoder_q4f16.onnx_data";

/// Scale a source-frame xyxy box into the squash-mapped NET frame.
pub(crate) fn scale_box(box_prompt: [f32; 4], width: u32, height: u32) -> [f32; 4] {
    let (sx, sy) = (NET as f32 / width as f32, NET as f32 / height as f32);
    [
        box_prompt[0] * sx,
        box_prompt[1] * sy,
        box_prompt[2] * sx,
        box_prompt[3] * sy,
    ]
}

/// The decoder-output tail both backends share: the mask tensor's
/// argmax-IoU channel range. The checks run BEFORE the range is used —
/// they guard it: a rank or count mismatch, or a zero channel count,
/// must error rather than slice or divide blind. mh/mw live at
/// dims[3]/[4], so anything but [1, 1, C, mh, mw] is a mismatch.
fn iou_channel(ious: &[f32], shape: &[i64], len: usize) -> Result<std::ops::Range<usize>> {
    anyhow::ensure!(
        shape.len() == 5,
        "SAM2 mask dims {shape:?} not [1, 1, C, mh, mw]"
    );
    anyhow::ensure!(
        shape[2] > 0 && ious.len() == shape[2] as usize,
        "iou shape mismatch"
    );
    let px = len / shape[2] as usize;
    let chan = argmax(ious);
    Ok(chan * px..(chan + 1) * px)
}

/// Widen the argmax-IoU channel's `mask_h × mask_w` logits, bilinear-resize
/// to source dims, threshold at 0.5 → 255/0 mask bytes (row-major).
///
/// Source and destination sizes are (width, height) pairs. The shipped
/// export's mask plane is square (256²), which is the only reason a
/// transposed call is invisible: a non-square mask export would stretch
/// the mask axes into each other if the pair order drifts back.
pub(crate) fn mask_bytes(
    logits: &[f32],
    mask_w: usize,
    mask_h: usize,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let src = image::ImageBuffer::<image::Luma<f32>, &[f32]>::from_raw(
        mask_w as u32,
        mask_h as u32,
        logits,
    )
    .expect("logits hold mask_w*mask_h samples");
    let up = image::imageops::resize(&src, width, height, image::imageops::FilterType::Triangle)
        .into_raw();
    up.iter().map(|&x| if x > 0.5 { 255 } else { 0 }).collect()
}

/// The host backend: two ort CPU sessions, encoder + decoder per segment.
#[cfg(not(target_arch = "wasm32"))]
pub struct Sam2 {
    encoder: Session,
    decoder: Session,
    /// Decoder input dtypes, resolved once at load (session metadata never
    /// changes). The `points`/`labels` tensors themselves stay per-call in
    /// `segment` — cloning an ort `DynValue` may deep-copy the buffer.
    decoder_dtypes: Vec<TE>,
}

#[cfg(not(target_arch = "wasm32"))]
impl Sam2 {
    pub fn load() -> Result<Self> {
        let decoder = session::build(
            &sam_file("prompt_encoder_mask_decoder")?,
            GraphOptimizationLevel::Level3,
        )?;
        let decoder_dtypes = input_dtypes(&decoder)?;
        Ok(Self {
            encoder: session::build(&sam_file("vision_encoder")?, GraphOptimizationLevel::Level3)?,
            decoder,
            decoder_dtypes,
        })
    }

    pub fn segment(
        &mut self,
        rgb: &[u8],
        width: u32,
        height: u32,
        box_prompt: &[f32; 4],
    ) -> Result<Vec<u8>> {
        let image = encoder_input(&self.encoder, rgb, (width, height), NET)?;

        let coords = scale_box(*box_prompt, width, height);
        let points = make_input(&[1, 1, 1, 2], vec![0.0, 0.0], self.decoder_dtypes[0])?;
        let labels = make_input(&[1, 1, 1], vec![-1.0], self.decoder_dtypes[1])?;
        let boxes_in = make_input(&[1, 1, 4], coords.to_vec(), self.decoder_dtypes[2])?;
        let embeddings = self.encoder.run([SessionInputValue::from(&image)])?;
        let decoded = self.decoder.run([
            SessionInputValue::from(&points),
            SessionInputValue::from(&labels),
            SessionInputValue::from(&boxes_in),
            SessionInputValue::from(&embeddings[0]),
            SessionInputValue::from(&embeddings[1]),
            SessionInputValue::from(&embeddings[2]),
        ])?;
        // ious first: only the argmax-IoU mask channel is used, so the
        // [1, 1, C, mh, mw] output widens one mh·mw channel, not all C.
        let (_, ious) = extract_f32(&decoded[0])?;
        let (mask_dims, soft) =
            extract_f32_with(&decoded[1], |shape, len| iou_channel(&ious, shape, len))?;
        let (mh, mw) = (mask_dims[3] as u32, mask_dims[4] as u32);
        Ok(mask_bytes(&soft, mw as usize, mh as usize, width, height))
    }
}

/// The wasm backend: the encoder runs once per image and its outputs are
/// cached; each box decodes against them.
#[cfg(target_arch = "wasm32")]
pub struct Sam2 {
    encoder: ortweb::Session,
    decoder: ortweb::Session,
    /// Decoder input dtypes, resolved once at load (session metadata never
    /// changes). The `points`/`labels`/`boxes` tensors stay per-call — `run`
    /// borrows them, nothing is retained by the session.
    decoder_dtypes: Vec<ortweb::DType>,
    /// pixel_values element type for the encoder, resolved once at load.
    pixel_dtype: ortweb::DType,
    /// Cached encoder outputs + the frame they were computed for. `decode`
    /// reuses them until the next `encode` and refuses a mismatched frame:
    /// box coords against the wrong image's embeddings fail loud, not
    /// silently wrong.
    image: Option<([ortweb::OrtTensor; 3], (u32, u32))>,
}

#[cfg(target_arch = "wasm32")]
impl Sam2 {
    /// One EP for both sessions — the worker probes once and every session
    /// follows it. ORT #27291 constrains mixing EPs WITHIN a session's
    /// graph, which this never does — each session keeps exactly one (see
    /// [`ortweb::Ep`]).
    pub async fn load(store: &ModelStore, ep: ortweb::Ep) -> Result<Self> {
        // Sessions load sequentially — same constraint as the WebSAM app:
        // no Promise.all over session creates. Store bytes are borrowed —
        // the bridge copies every byte into a fresh Uint8Array at create
        // time, so nothing here needs ownership.
        let decoder = ortweb::Session::load(
            store.get(DECODER_ONNX)?,
            &[(DECODER_DATA_REF, store.get(DECODER_DATA_KEY)?)],
            ep,
        )
        .await?;
        let decoder_dtypes = decoder.input_dtypes().await?;
        anyhow::ensure!(
            decoder_dtypes.len() >= 3,
            "SAM2 decoder declares {} inputs, expected at least 3",
            decoder_dtypes.len()
        );
        let encoder = ortweb::Session::load(
            store.get(VISION_ONNX)?,
            &[(VISION_DATA_REF, store.get(VISION_DATA_KEY)?)],
            ep,
        )
        .await?;
        let encoder_dtypes = encoder.input_dtypes().await?;
        Ok(Self {
            pixel_dtype: *encoder_dtypes
                .first()
                .ok_or_else(|| anyhow::anyhow!("SAM2 encoder declares no inputs"))?,
            encoder,
            decoder,
            decoder_dtypes,
            image: None,
        })
    }

    /// Run the encoder for one image and cache its outputs — the expensive
    /// half; `decode` reuses them until the next `encode`.
    pub async fn encode(&mut self, rgb: &[u8], width: u32, height: u32) -> Result<()> {
        let input = preprocess::encoder_input(rgb, width, height, NET, self.pixel_dtype)?;
        let outputs = self.encoder.run(&[&input]).await?;
        let embeds: [ortweb::OrtTensor; 3] = ortweb::fixed_outputs::<3>(outputs, "SAM2 encoder")?;
        self.image = Some((embeds, (width, height)));
        Ok(())
    }

    /// Decode one box against the cached encoder outputs.
    pub async fn decode(
        &mut self,
        box_prompt: [f32; 4],
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>> {
        let (embeds, encoded) = self
            .image
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("decode called before encode"))?;
        anyhow::ensure!(
            *encoded == (width, height),
            "decode frame {width}x{height} doesn't match the encoded {}x{}",
            encoded.0,
            encoded.1
        );
        let coords = scale_box(box_prompt, width, height);
        let points = ortweb::make_input(&[1, 1, 1, 2], vec![0.0, 0.0], self.decoder_dtypes[0])?;
        let labels = ortweb::make_input(&[1, 1, 1], vec![-1.0], self.decoder_dtypes[1])?;
        let boxes_in = ortweb::make_input(&[1, 1, 4], coords.to_vec(), self.decoder_dtypes[2])?;
        let outputs = self
            .decoder
            .run(&[
                &points, &labels, &boxes_in, &embeds[0], &embeds[1], &embeds[2],
            ])
            .await?;
        // Graph order: [iou_scores, pred_masks, object_score_logits]. The
        // object-presence head stays unconsumed — this pipeline reads the
        // argmax-IoU mask channel only.
        let [ious_t, masks_t, _object_score] = ortweb::fixed_outputs::<3>(outputs, "SAM2 decoder")?;
        // ious first: only the argmax-IoU mask channel is used, so the
        // [1, 1, C, mh, mw] output widens one mh·mw channel, not all C.
        let (_, ious) = ious_t.into_f32()?;
        let (mask_dims, soft) =
            masks_t.into_f32_slice(|shape, len| iou_channel(&ious, shape, len))?;
        let (mh, mw) = (mask_dims[3] as u32, mask_dims[4] as u32);
        Ok(mask_bytes(&soft, mw as usize, mh as usize, width, height))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The count check guards the range it produces: a mismatch must Err,
    /// never slice or divide blind.
    #[test]
    fn iou_channel_checks_count_before_slicing() {
        let shape = [1i64, 1, 3, 16, 16];
        // argmax = 1 → the second channel's 256-element plane.
        assert_eq!(
            iou_channel(&[0.1, 0.9, 0.5], &shape, 3 * 256).unwrap(),
            256..512
        );
        let err = iou_channel(&[0.1, 0.9, 0.5, 0.3], &shape, 3 * 256)
            .unwrap_err()
            .to_string();
        assert!(err.contains("iou shape mismatch"), "{err}");
        // The rank gate folds in here: mh/mw live at dims[3]/[4].
        let err = iou_channel(&[0.9], &[1, 1, 1, 16], 16)
            .unwrap_err()
            .to_string();
        assert!(err.contains("not [1, 1, C, mh, mw]"), "{err}");
        // Zero channels passes the count check (0 == 0) but must not
        // reach the `len / channels` division.
        let err = iou_channel(&[], &[1, 1, 0, 16, 16], 0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("iou shape mismatch"), "{err}");
    }

    /// The store lookups in `load` must address REQUIRED_FILES entries; the
    /// bare proto-ref names must never. A rename in either table fails here
    /// instead of at first browser session create.
    #[test]
    #[cfg(target_arch = "wasm32")]
    fn store_keys_are_required_files_entries() {
        for key in [VISION_ONNX, VISION_DATA_KEY, DECODER_ONNX, DECODER_DATA_KEY] {
            assert!(
                crate::REQUIRED_FILES.contains(&key),
                "{key} is not a REQUIRED_FILES entry"
            );
        }
        for (key, reference) in [
            (VISION_DATA_KEY, VISION_DATA_REF),
            (DECODER_DATA_KEY, DECODER_DATA_REF),
        ] {
            assert_ne!(key, reference);
            assert!(key.ends_with(reference), "{key} must end with {reference}");
        }
    }

    /// 1024/512 and 1024/256 are exact powers of two, so the scaled coords
    /// are exact f32 — no epsilon slack could hide a swapped axis.
    #[test]
    fn scale_box_maps_xyxy_into_the_net_frame() {
        assert_eq!(
            scale_box([10.0, 20.0, 30.0, 40.0], 512, 256),
            [20.0, 80.0, 60.0, 160.0]
        );
        // Aspect-destroying squash: the full-frame box's far corner lands
        // on the far corner of the NET square; the origin stays at 0.
        assert_eq!(
            scale_box([0.0, 0.0, 512.0, 256.0], 512, 256),
            [0.0, 0.0, 1024.0, 1024.0]
        );
    }

    #[test]
    fn scale_box_fractional_scales_stay_put() {
        let got = scale_box([100.0, 50.0, 300.0, 250.0], 640, 320);
        for (g, want) in got.iter().zip([160.0, 160.0, 480.0, 800.0]) {
            assert!((g - want).abs() < 1e-2, "{got:?} vs [160, 160, 480, 800]");
        }
    }

    /// Identity-size logits threshold per cell. The comparison is strict,
    /// so 0.5 itself is off, and the byte order pins row-major layout
    /// ([255, 0] then [255, 0] rows — column-major would read
    /// [255, 255, 0, 0]).
    #[test]
    fn mask_bytes_thresholds_strictly_and_stays_row_major() {
        assert_eq!(
            mask_bytes(&[2.0, -2.0, 0.6, 0.4], 2, 2, 2, 2),
            vec![255, 0, 255, 0]
        );
        assert_eq!(
            mask_bytes(&[0.5, 0.51, -0.5, 4.0], 2, 2, 2, 2),
            vec![0, 255, 0, 255]
        );
    }

    #[test]
    fn mask_bytes_upscales_constant_logits_to_the_source_frame() {
        assert_eq!(mask_bytes(&[0.75; 4], 2, 2, 5, 3), vec![255; 15]);
        assert_eq!(mask_bytes(&[-1.0; 4], 2, 2, 5, 3), vec![0; 15]);
    }

    /// Triangle keeps a constant plane constant through a non-square
    /// down/up-resize (6×4 → 3×8), so the mask stays uniformly foreground.
    #[test]
    fn mask_bytes_resize_preserves_constant() {
        assert_eq!(mask_bytes(&[0.75f32; 6 * 4], 6, 4, 3, 8), vec![255; 3 * 8]);
    }

    /// A non-square source pins the (width, height) pair order: reading the
    /// six logits as 3-wide instead of 2-wide (a mask_w/mask_h swap) samples
    /// different source rows and cannot reproduce this output.
    #[test]
    fn mask_bytes_source_pair_is_width_height() {
        assert_eq!(
            mask_bytes(&[1.0, -1.0, 1.0, -1.0, 1.0, -1.0], 2, 3, 2, 3),
            vec![255, 0, 255, 0, 255, 0]
        );
    }
}
