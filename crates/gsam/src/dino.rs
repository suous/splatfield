use crate::{
    argmax, encoder_input, extract_f32, grounding_file, grounding_tokenizer, input_dtypes,
    make_input, session,
};
use anyhow::Result;
use ort::session::{Session, SessionInputValue, builder::GraphOptimizationLevel};
use ort::value::DynValue;
use tokenizers::Tokenizer;

const NET: u32 = 800;
const BOX_CONF: f32 = 0.35;

pub struct Detector {
    session: Session,
    /// Content-token flags (false for [CLS]/[SEP] and the label's
    /// terminating period).
    content: Vec<bool>,
    /// The four prompt-constant inputs - token ids, type ids, attention and
    /// pixel masks - in session input order, built once against the graph's
    /// declared dtypes.
    consts: [DynValue; 4],
}

impl Detector {
    pub fn load(prompt: &str) -> Result<Self> {
        let label = prompt.trim().to_ascii_lowercase();
        anyhow::ensure!(!label.is_empty(), "at least one text label is required");
        let prompt = format!("{label}.");
        // tokenizers errors are Box<dyn Error + Send + Sync>, not Sized, so
        // they need an explicit lift into anyhow.
        let tokenizer =
            Tokenizer::from_file(grounding_tokenizer()?).map_err(|e| anyhow::anyhow!("{e}"))?;
        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let ids = encoding.get_ids().to_vec();
        // Token strings, not id literals: the content flags derive from what
        // actually encoded, so vocab drift cannot silently reclassify a
        // special token as class-bearing.
        let content: Vec<bool> = encoding
            .get_tokens()
            .iter()
            .map(|t| !matches!(t.as_str(), "[CLS]" | "[SEP]" | "."))
            .collect();
        let session = session::build(&grounding_file()?, GraphOptimizationLevel::Level1)?;
        let dtypes = input_dtypes(&session)?;
        let ntok = ids.len() as i64;
        let consts = [
            make_input(
                &[1, ntok],
                ids.iter().map(|&i| i as f32).collect(),
                dtypes[1],
            )?,
            make_input(&[1, ntok], vec![0f32; ntok as usize], dtypes[2])?,
            make_input(&[1, ntok], vec![1f32; ntok as usize], dtypes[3])?,
            make_input(
                &[1, NET as i64, NET as i64],
                vec![1f32; (NET * NET) as usize],
                dtypes[4],
            )?,
        ];
        Ok(Self {
            session,
            content,
            consts,
        })
    }

    pub fn detect(&mut self, rgb: &[u8], width: u32, height: u32) -> Result<Vec<[f32; 4]>> {
        let pixel_values = encoder_input(&self.session, rgb, (width, height), NET)?;
        let (logits_dims, logits, boxes) = {
            let [ids, types, attention, pixels] = &self.consts;
            let outputs = self.session.run([
                SessionInputValue::from(&pixel_values),
                SessionInputValue::from(ids),
                SessionInputValue::from(types),
                SessionInputValue::from(attention),
                SessionInputValue::from(pixels),
            ])?;
            let (logits_dims, logits) = extract_f32(&outputs[0])?;
            let (_, boxes) = extract_f32(&outputs[1])?;
            (logits_dims, logits, boxes)
        };
        Ok(self.postprocess(
            &logits,
            &boxes,
            logits_dims[1] as usize,
            self.content.len(),
            width,
            height,
        ))
    }

    /// Argmax over the unpadded token positions (sigmoid applied once for
    /// the confidence gate); `boxes` are normalized cxcywh scaled straight
    /// to source dims (squash cancels). Survivors come back best-confidence
    /// first.
    fn postprocess(
        &self,
        logits: &[f32],
        boxes: &[f32],
        nq: usize,
        ntok: usize,
        width: u32,
        height: u32,
    ) -> Vec<[f32; 4]> {
        let mut hits: Vec<(f32, [f32; 4])> = Vec::new();
        for q in 0..nq {
            let row = &logits[q * ntok..][..ntok];
            let class_id = argmax(row);
            let conf = 1.0 / (1.0 + (-row[class_id]).exp());
            if conf < BOX_CONF || !self.content[class_id] {
                continue;
            }
            let b = &boxes[q * 4..][..4];
            let (cx, cy, w, h) = (
                b[0] * width as f32,
                b[1] * height as f32,
                b[2] * width as f32,
                b[3] * height as f32,
            );
            let x = (cx - w / 2.0).clamp(0.0, width as f32);
            let y = (cy - h / 2.0).clamp(0.0, height as f32);
            hits.push((conf, [x, y, x + w, y + h]));
        }
        hits.sort_by(|a, b| b.0.total_cmp(&a.0));
        hits.into_iter().map(|(_, b)| b).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grounding_tokenizer;

    /// The tokenizer behind [`Detector`]'s prompt encoding, with the same
    /// loud-fail-on-missing-asset contract as the model smoke tests.
    fn tokenizer() -> Option<Tokenizer> {
        let p = grounding_tokenizer().ok()?;
        if !p.exists() {
            if std::env::var_os("SPLATFIELD_ALLOW_MISSING_ASSETS").is_none() {
                panic!(
                    "missing {} - these tests are the only tokenizer coverage; set SPLATFIELD_ALLOW_MISSING_ASSETS=1 to skip",
                    p.display()
                );
            }
            eprintln!("skipping: grounding-dino/tokenizer.json not cached");
            return None;
        }
        Some(Tokenizer::from_file(&p).unwrap())
    }

    /// Ids hand-verified against the bert-base-uncased vocab: person=2711,
    /// teddy=11389, bear=4562.
    #[test]
    fn test_wordpiece_ids() {
        let Some(t) = tokenizer() else {
            eprintln!("skipping: grounding-dino/tokenizer.json not cached");
            return;
        };
        assert_eq!(
            t.encode("person", true).unwrap().get_ids().to_vec(),
            vec![101, 2711, 102]
        );
        assert_eq!(
            t.encode("teddy bear", true).unwrap().get_ids().to_vec(),
            vec![101, 11389, 4562, 102]
        );
        assert_eq!(
            t.encode("teddy-bear", true).unwrap().get_ids().to_vec(),
            vec![101, 11389, 1011, 4562, 102]
        );
        // The prompt path appends a period (Detector::load); its token must
        // survive verbatim so the content flags can classify it.
        let enc = t.encode("person.", true).unwrap();
        assert_eq!(enc.get_tokens(), ["[CLS]", "person", ".", "[SEP]"]);
    }

    /// Accents strip before lookup, case folds, punctuation splits.
    #[test]
    fn test_wordpiece_normalization() {
        let Some(t) = tokenizer() else {
            eprintln!("skipping: grounding-dino/tokenizer.json not cached");
            return;
        };
        assert_eq!(
            t.encode("café", true).unwrap().get_ids().to_vec(),
            vec![101, 7668, 102]
        );
        assert_eq!(
            t.encode("Bus!", true).unwrap().get_ids().to_vec(),
            vec![101, 3902, 999, 102]
        );
        // A char absent from the vocab even after lowercasing (nothing
        // matches, not even one piece) -> whole-word [UNK]; 'ß' (1096)
        // tokenizes.
        assert_eq!(
            t.encode("\u{1F600}", true).unwrap().get_ids().to_vec(),
            vec![101, 100, 102]
        );
        assert_eq!(
            t.encode("ß", true).unwrap().get_ids().to_vec(),
            vec![101, 1096, 102]
        );
    }
}
