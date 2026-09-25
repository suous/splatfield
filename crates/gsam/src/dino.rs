//! GroundingDINO detection: prompt → token tensors → boxes.
//!
//! Desktop-oracle flow (commit 1a06ff6) against the onnx-community export:
//! squash-resize to 800², ImageNet normalize, 4 prompt-constant inputs,
//! per-query argmax over content token positions (the export pads the text
//! axis to max_text_len=256 — the row stride is the padded vocab, not the
//! prompt's token count) with a 0.35 sigmoid confidence gate, normalized
//! cxcywh → pixel xyxy. Best-confidence detection first.
//!
//! The session layer is the only backend split: host loads an `ort` CPU
//! session from the release cache, wasm builds an `ortweb::Session` from the
//! [`crate::ModelStore`] — the decode below is shared verbatim.

use anyhow::Result;
use tokenizers::Tokenizer;

#[cfg(target_arch = "wasm32")]
use crate::{ModelStore, REQUIRED_FILES, ortweb, preprocess};
#[cfg(not(target_arch = "wasm32"))]
use crate::{
    encoder_input, extract_f32, grounding_file, grounding_tokenizer, input_dtypes, make_input,
    session,
};
#[cfg(not(target_arch = "wasm32"))]
use ort::session::{Session, SessionInputValue, builder::GraphOptimizationLevel};
#[cfg(not(target_arch = "wasm32"))]
use ort::value::DynValue;

/// Fixed input size; the aspect ratio is squashed.
pub(crate) const NET: u32 = 800;
/// Confidence gate on the sigmoid of the best token logit.
pub(crate) const BOX_CONF: f32 = 0.35;

/// Store keys this detector reads, indexed into [`REQUIRED_FILES`] so a
/// release-manifest rename can't drift. The dino graph is self-contained —
/// no external-weights sibling (unlike the SAM2 graphs).
#[cfg(target_arch = "wasm32")]
const DINO_ONNX: &str = REQUIRED_FILES[0];
#[cfg(target_arch = "wasm32")]
const DINO_TOKENIZER: &str = REQUIRED_FILES[1];

/// One detection that survived the confidence + content-token gate.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub conf: f32,
    /// Pixel-space xyxy in the source image frame.
    pub xyxy: [f32; 4],
}

/// Non-content token = no class evidence: specials ([CLS]/[SEP]) and the
/// label's terminating period. Derived from the encoded token *strings*, so
/// vocab drift can't reclassify a special token as class-bearing.
pub(crate) fn content_flags(tokens: &[String]) -> Vec<bool> {
    tokens
        .iter()
        .map(|t| !matches!(t.as_str(), "[CLS]" | "[SEP]" | "."))
        .collect()
}

/// Sigmoid — the logit → confidence map used by the gate.
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Per query row of token logits: argmax over content token positions,
/// sigmoid confidence, gate on `BOX_CONF`, then normalized cxcywh → pixel
/// xyxy (clamped into the frame). Survivors sorted best-conf first.
///
/// `logits` is `[nq × vocab]` — the export pads the text axis to
/// max_text_len (256), so `vocab` is the real row stride while only the
/// first `ntok = content.len()` positions carry this prompt's tokens.
/// `boxes_cxcywh` is `[nq × 4]` normalized, `content` is `[ntok]`. Argmax
/// is taken over content positions only: a stray high [CLS]/[SEP]/period
/// logit must not veto (or fabricate) a detection.
#[allow(clippy::too_many_arguments)] // a pure decode over one tensor's parts
pub(crate) fn decode_detections(
    logits: &[f32],
    boxes_cxcywh: &[f32],
    nq: usize,
    vocab: usize,
    ntok: usize,
    content: &[bool],
    width: u32,
    height: u32,
) -> Vec<Detection> {
    let (w, h) = (width as f32, height as f32);
    let mut hits: Vec<Detection> = Vec::new();
    for q in 0..nq {
        let row = &logits[q * vocab..][..vocab];
        // NaN never wins the scan (a `>` against it is always false), so
        // all-pad rows fall through to the gate and are dropped.
        let mut best_logit = f32::NEG_INFINITY;
        for t in 0..ntok.min(vocab) {
            if content[t] && row[t] > best_logit {
                best_logit = row[t];
            }
        }
        let conf = sigmoid(best_logit);
        if conf < BOX_CONF {
            continue;
        }
        let b = &boxes_cxcywh[q * 4..][..4];
        let (cx, cy, bw, bh) = (b[0] * w, b[1] * h, b[2] * w, b[3] * h);
        // Only the origin clamps — the desktop contract leaves an
        // overhanging right/bottom edge as-is.
        let x = (cx - bw / 2.0).clamp(0.0, w);
        let y = (cy - bh / 2.0).clamp(0.0, h);
        hits.push(Detection {
            conf,
            xyxy: [x, y, x + bw, y + bh],
        });
    }
    hits.sort_by(|a, b| b.conf.total_cmp(&a.conf));
    hits
}

/// The shared prompt-encode chain (both backends' `Detector::load`s call
/// this after building their different-source tokenizer): trim +
/// lowercase + the trailing "." — the terminating `.` must survive
/// verbatim so [`content_flags`] can classify it as non-content; a
/// re-spelled prompt silently shifts every token id the graph sees —
/// then tokenize and derive the flags from the encoded token strings.
fn encode_prompt(tokenizer: &Tokenizer, prompt: &str) -> Result<(Vec<u32>, Vec<bool>)> {
    let label = prompt.trim().to_ascii_lowercase();
    anyhow::ensure!(!label.is_empty(), "at least one text label is required");
    let prompt = format!("{label}.");
    // tokenizers errors are Box<dyn Error + Send + Sync>, not Sized, so
    // they need an explicit lift into anyhow.
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let ids = encoding.get_ids().to_vec();
    // Token strings, not id literals: the content flags derive from what
    // actually encoded, so vocab drift cannot silently reclassify a
    // special token as class-bearing.
    let content = content_flags(encoding.get_tokens());
    Ok((ids, content))
}

/// The shared decode tail: pull `nq`/`vocab` off the logits dims (the
/// export pads the text axis to 256 — the stride must stay `dims[2]`,
/// not the prompt's token count) and run [`decode_detections`]. The
/// call sites do their own backend-specific tensor extraction first;
/// this starts where the two converge on plain slices.
fn decode_outputs(
    logits_dims: &[i64],
    logits: &[f32],
    boxes_cxcywh: &[f32],
    content: &[bool],
    width: u32,
    height: u32,
) -> Result<Vec<Detection>> {
    let nq = *logits_dims
        .get(1)
        .ok_or_else(|| anyhow::anyhow!("logits dims {logits_dims:?} lack a query axis"))?
        as usize;
    let vocab = *logits_dims
        .get(2)
        .ok_or_else(|| anyhow::anyhow!("logits dims {logits_dims:?} lack a token axis"))?
        as usize;
    Ok(decode_detections(
        logits,
        boxes_cxcywh,
        nq,
        vocab,
        content.len(),
        content,
        width,
        height,
    ))
}

/// The four prompt-constant fills in graph input order after pixel_values
/// (ids as f32, type-0s, attention-1s, all-ones NET² mask) — shared by both
/// cfg arms, so the platforms' consts cannot drift.
fn prompt_fills(ids: &[u32]) -> [Vec<f32>; 4] {
    let ntok = ids.len();
    [
        ids.iter().map(|&i| i as f32).collect(),
        vec![0.0; ntok],
        vec![1.0; ntok],
        vec![1.0; (NET * NET) as usize],
    ]
}

/// The host backend: one ort CPU session plus the prompt-constant tensors.
#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(not(target_arch = "wasm32"))]
impl Detector {
    pub fn load(prompt: &str) -> Result<Self> {
        // tokenizers errors are Box<dyn Error + Send + Sync>, not Sized, so
        // they need an explicit lift into anyhow.
        let tokenizer =
            Tokenizer::from_file(grounding_tokenizer()?).map_err(|e| anyhow::anyhow!("{e}"))?;
        let (ids, content) = encode_prompt(&tokenizer, prompt)?;
        let session = session::build(&grounding_file()?, GraphOptimizationLevel::Level1)?;
        let dtypes = input_dtypes(&session)?;
        let ntok = ids.len() as i64;
        let [ids_t, types_t, attn_t, mask_t] = prompt_fills(&ids);
        let consts = [
            make_input(&[1, ntok], ids_t, dtypes[1])?,
            make_input(&[1, ntok], types_t, dtypes[2])?,
            make_input(&[1, ntok], attn_t, dtypes[3])?,
            make_input(&[1, NET as i64, NET as i64], mask_t, dtypes[4])?,
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
        Ok(
            decode_outputs(&logits_dims, &logits, &boxes, &self.content, width, height)?
                .into_iter()
                .map(|d| d.xyxy)
                .collect(),
        )
    }
}

/// The wasm backend: same prompt tensors over an onnxruntime-web session.
#[cfg(target_arch = "wasm32")]
pub struct Detector {
    session: ortweb::Session,
    /// Content-token flags (false for [CLS]/[SEP] and the label's
    /// terminating period).
    content: Vec<bool>,
    /// The four prompt-constant inputs — ids, type ids, attention, pixel
    /// masks — in session input order, built once at the graph's declared
    /// dtypes (positions 1..5; position 0 is pixel_values, built per call).
    consts: [ortweb::OrtTensor; 4],
    /// pixel_values element type, resolved once at load.
    pixel_dtype: ortweb::DType,
}

#[cfg(target_arch = "wasm32")]
impl Detector {
    /// Tokenize the prompt (lowercased, `.` appended, special tokens
    /// flagged), build the session and the 4 prompt-constant tensors against
    /// the graph's declared dtypes. `ep` is this session's execution
    /// provider — the worker probes once, before any detector create, and
    /// passes it here (see [`ortweb::Ep`]).
    pub async fn load(store: &ModelStore, prompt: &str, ep: ortweb::Ep) -> Result<Self> {
        // tokenizers errors are Box<dyn Error + Send + Sync>, not Sized, so
        // they need an explicit lift into anyhow.
        let json = std::str::from_utf8(store.get(DINO_TOKENIZER)?)?;
        let tokenizer = Tokenizer::from_bytes(json).map_err(|e| anyhow::anyhow!("{e}"))?;
        let (ids, content) = encode_prompt(&tokenizer, prompt)?;
        let session = ortweb::Session::load(store.get(DINO_ONNX)?, &[], ep).await?;
        let dtypes = session.input_dtypes().await?;
        anyhow::ensure!(
            dtypes.len() >= 5,
            "grounding-dino graph declares {} inputs, expected 5",
            dtypes.len()
        );
        let ntok = ids.len() as i64;
        let [ids_t, types_t, attn_t, mask_t] = prompt_fills(&ids);
        let consts = [
            ortweb::make_input(&[1, ntok], ids_t, dtypes[1])?,
            ortweb::make_input(&[1, ntok], types_t, dtypes[2])?,
            ortweb::make_input(&[1, ntok], attn_t, dtypes[3])?,
            ortweb::make_input(&[1, NET as i64, NET as i64], mask_t, dtypes[4])?,
        ];
        Ok(Self {
            session,
            content,
            consts,
            pixel_dtype: dtypes[0],
        })
    }

    /// Detect the prompt's object; detections come back best-confidence
    /// first, empty when nothing passed the gate.
    pub async fn detect(&mut self, rgb: &[u8], width: u32, height: u32) -> Result<Vec<Detection>> {
        let pixel_values = preprocess::encoder_input(rgb, width, height, NET, self.pixel_dtype)?;
        let [c0, c1, c2, c3] = &self.consts;
        let outputs = self.session.run(&[&pixel_values, c0, c1, c2, c3]).await?;
        let [logits_t, boxes_t] = ortweb::fixed_outputs::<2>(outputs, "grounding-dino")?;
        let (logits_dims, logits) = logits_t.into_f32()?;
        let (_, boxes) = boxes_t.into_f32()?;
        decode_outputs(&logits_dims, &logits, &boxes, &self.content, width, height)
    }
}

// Host-only: `grounding_tokenizer` (and the tokenizer it loads) exist on the
// host build — wasm has no assets dir to point at.
#[cfg(all(test, not(target_arch = "wasm32")))]
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

    /// The prompt-constant fills in graph input order: ids as f32, type ids
    /// 0, attention 1, all-ones NET² pixel mask — both backends' consts
    /// build from this one spelling.
    #[test]
    fn prompt_fills_match_graph_input_order() {
        let [ids_t, types_t, attn_t, mask_t] = prompt_fills(&[101, 2711]);
        assert_eq!(ids_t, vec![101.0, 2711.0]);
        assert_eq!(types_t, vec![0.0, 0.0]);
        assert_eq!(attn_t, vec![1.0, 1.0]);
        assert_eq!(mask_t.len(), (NET * NET) as usize);
        assert!(mask_t.iter().all(|&x| x == 1.0));
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
        // ...and the flags the detector derives from those strings.
        assert_eq!(
            content_flags(enc.get_tokens()),
            vec![false, true, false, false]
        );
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

    #[test]
    fn sigmoid_zero_is_half_monotonic_and_saturates() {
        assert_eq!(sigmoid(0.0), 0.5);
        let mut prev = sigmoid(-30.0);
        assert!(prev > 0.0 && prev < 1e-6);
        for x in [-20.0f32, -5.0, -1.0, 0.0, 1.0, 5.0, 20.0, 30.0] {
            let s = sigmoid(x);
            assert!(s >= prev, "sigmoid not monotonic at {x}");
            prev = s;
        }
        assert!((sigmoid(30.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn content_flags_mark_only_class_bearing_tokens() {
        assert_eq!(
            content_flags(&["[CLS]".into(), "person".into(), ".".into(), "[SEP]".into()]),
            vec![false, true, false, false]
        );
        assert_eq!(
            content_flags(&["teddy".into(), "bear".into(), ".".into()]),
            vec![true, true, false]
        );
        assert!(content_flags(&[]).is_empty());
    }

    /// Content vs non-content and the 0.35 gate: argmax runs over CONTENT
    /// positions only (a stray high [CLS] logit must not veto a detection),
    /// q2 is content but under the 0.35 gate (dropped), q0/q1/q3 survive.
    /// All box math exact f32.
    #[test]
    fn decode_keeps_content_above_gate_and_drops_the_rest() {
        let content = [true, false];
        let logits = [
            2.0, 5.0, // q0: content token 0 = 2.0 (the 5.0 non-content is ignored)
            2.0, -1.0, // q1: content token 0, conf = sigmoid(2)
            -0.7, -1.0, // q2: content token 0, conf = sigmoid(-0.7) < 0.35
            4.0, 0.0, // q3: content token 0, conf = sigmoid(4)
        ];
        let boxes = [
            0.0, 0.0, 0.0, 0.0, //
            0.5, 0.5, 0.5, 0.5, //
            0.0, 0.0, 0.0, 0.0, //
            // 1.25 and 0.8×frame are exact in f32, keeping the box math bit-exact.
            1.25, -0.2, 0.8, 0.4,
        ];
        let got = decode_detections(&logits, &boxes, 4, 2, 2, &content, 100, 50);
        assert_eq!(got.len(), 3, "q0/q1/q3 survive, q2 gated: {got:?}");
        // q3: x=125-40=85 stays, y=-10-10 clamps to 0; x2=x+w=165 runs past
        // the frame — the desktop contract clamps only the box origin.
        assert_eq!(
            got[0],
            Detection {
                conf: sigmoid(4.0),
                xyxy: [85.0, 0.0, 165.0, 20.0]
            }
        );
        // q0 and q1 tie at sigmoid(2.0); the stable sort keeps input order.
        assert_eq!(
            got[1],
            Detection {
                conf: sigmoid(2.0),
                xyxy: [0.0, 0.0, 0.0, 0.0]
            }
        );
        assert_eq!(
            got[2],
            Detection {
                conf: sigmoid(2.0),
                xyxy: [25.0, 12.5, 75.0, 37.5]
            }
        );
    }

    /// A fully off-frame box keeps its size: only the origin clamps.
    #[test]
    fn decode_clamps_off_frame_box_origin_into_frame() {
        let got = decode_detections(&[4.0], &[-1.0, -1.0, 0.1, 0.1], 1, 1, 1, &[true], 100, 100);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].xyxy, [0.0, 0.0, 10.0, 10.0]);
    }

    /// The onnx-community export pads logits to [nq × 256] while the prompt
    /// fills only the first ntok columns; decoding at ntok stride reads
    /// cross-query garbage (the "bear" regression: a huge +6.9 [CLS] logit
    /// at the row-0 argmax vetoed a real 0.80 detection). The stride must
    /// be the padded vocab, pads (nonfinite) never win, and a high
    /// non-content logit must not veto.
    #[test]
    fn decode_strides_by_padded_vocab_ignoring_pads_and_noncontent() {
        let content = [false, true, false, false]; // "bear." → only token 1
        let mut logits = vec![f32::NAN; 2 * 256];
        logits[0] = 8.0; // q0 [CLS]: higher than the content token — must be ignored
        logits[1] = 4.0; // q0 "bear" — the real detection
        logits[256 + 1] = 3.0; // q1 "bear" also above the gate
        let boxes = [0.1, 0.1, 0.5, 0.5, 0.2, 0.2, 0.5, 0.5];
        let got = decode_detections(&logits, &boxes, 2, 256, 4, &content, 100, 100);
        assert_eq!(got.len(), 2, "both queries pass the gate: {got:?}");
        assert_eq!(got[0].conf, sigmoid(4.0));
        assert_eq!(got[1].conf, sigmoid(3.0));
    }
}
