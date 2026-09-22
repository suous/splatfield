use crate::{
    argmax, encoder_input, extract_f32, extract_f32_with, input_dtypes, make_input, preprocess,
    sam_file, session,
};
use anyhow::Result;
use ort::session::{Session, SessionInputValue, builder::GraphOptimizationLevel};
use ort::tensor::TensorElementType as TE;

/// Fixed input size; the aspect ratio is squashed, and prompts are pixel coords in this frame.
const NET: u32 = 1024;

pub struct Sam2 {
    encoder: Session,
    decoder: Session,
    /// Decoder input dtypes, resolved once at load (session metadata never
    /// changes). The `points`/`labels` tensors themselves stay per-call in
    /// `segment` — cloning an ort `DynValue` may deep-copy the buffer.
    decoder_dtypes: Vec<TE>,
}

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

        let (sx, sy) = (NET as f32 / width as f32, NET as f32 / height as f32);
        let coords = [
            box_prompt[0] * sx,
            box_prompt[1] * sy,
            box_prompt[2] * sx,
            box_prompt[3] * sy,
        ];
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
        // [1, C, 1, mh, mw] output widens one mh·mw channel, not all C.
        let (_, ious) = extract_f32(&decoded[0])?;
        let chan = argmax(&ious);
        let (mask_dims, soft) = extract_f32_with(&decoded[1], |shape, len| {
            let px = len / shape[2] as usize;
            chan * px..(chan + 1) * px
        })?;

        let (mh, mw) = (mask_dims[3] as u32, mask_dims[4] as u32);
        anyhow::ensure!(ious.len() == mask_dims[2] as usize, "iou shape mismatch");
        let up = preprocess::resize_bilinear_f32(&soft, (mh, mw), (width, height));
        Ok(up.iter().map(|&x| if x > 0.5 { 255 } else { 0 }).collect())
    }
}
