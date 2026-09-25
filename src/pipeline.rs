//! Worker<->main-thread message protocol for the segmentation pipeline.
//!
//! The worker owns the whole model pipeline: [`crate::pipeline::Request::EnsureModels`]
//! downloads/verifies the pinned release and loads the SAM2 sessions, and
//! the per-image [`crate::pipeline::Request::Segment`] runs detect → encode → decode. Frames
//! cross postMessage as bincode bytes in a `Uint8Array` — the same
//! encode/decode the host tests exercise, so the wire format is pinned
//! where it can be run.

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

/// Main thread -> worker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Request {
    /// Fetch, verify, and install the pinned model release.
    EnsureModels,
    /// Segment one rgb8 image by a text prompt. Appended at the END — the
    /// wire contract is declaration order (see the pin tests below).
    Segment {
        rgb: Vec<u8>,
        width: u32,
        height: u32,
        prompt: String,
    },
}

/// Worker -> main thread.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Response {
    /// Fractional release-download progress.
    DownloadProgress(f64),
    /// Named pipeline stage transition.
    Stage(String),
    /// The pinned release is installed and its sessions load; timings in
    /// ms. `provenance` carries the per-model execution-provider plan
    /// summary the sessions were loaded under, e.g. "dino:webgpu
    /// enc:webgpu dec:wasm".
    ModelsReady {
        fetch_ms: f64,
        sam2_ms: f64,
        provenance: String,
    },
    ModelsFailed(String),
    /// One segmented frame; `mask` is w*h threshold bytes (255/0), row-major.
    /// `encode_ms` is measured on every request — the worker re-encodes per
    /// segment (Sam2 itself keeps the embeddings between that encode and
    /// the decode). Empty detection (the prompt matched nothing) is a DONE
    /// with an all-zero w*h mask, confidence 0, box [0; 4], and both
    /// `encode_ms` and `decode_ms` exactly 0.0 — no encode/decode ran on
    /// that path, and 0.0 is impossible for a real one, so the consumer can
    /// distinguish it without a mask scan. It mirrors the native oracle:
    /// "nothing found" feeds the loop an all-background mask (its stop rule
    /// reads the zero count) instead of an error.
    SegmentDone {
        box_px: [f32; 4],
        confidence: f32,
        mask: Vec<u8>,
        encode_ms: f64,
        detect_ms: f64,
        decode_ms: f64,
    },
    SegmentFailed(String),
}

pub fn encode(msg: &impl Serialize) -> Result<Vec<u8>> {
    Ok(bincode::serialize(msg)?)
}

pub fn decode_request(bytes: &[u8]) -> Result<Request> {
    let req = bincode::deserialize(bytes)?;
    ensure_valid(&req)?;
    Ok(req)
}

pub fn decode_response(bytes: &[u8]) -> Result<Response> {
    Ok(bincode::deserialize(bytes)?)
}

/// bincode deserializes every `Vec` with a read-as-u64 length; a corrupt or
/// hostile frame can declare gigabytes before any byte is inspected. The
/// decode site rejects absurd segment frames up front — real frames are
/// whole messages, never chunked streams. (u32² fits u64; only the ×3 needs
/// the saturating form.) The rgb-len cross-check then bounds both sides at
/// once: the declared prefix must equal the bytes actually shipped.
const MAX_SEGMENT_PIXELS: u64 = 64 * 1024 * 1024; // 8192² — any real frame passes, no absurd allocation

fn ensure_valid(req: &Request) -> Result<()> {
    if let Request::Segment {
        rgb, width, height, ..
    } = req
    {
        ensure!(
            *width > 0 && *height > 0,
            "segment frame dims must be nonzero, got {width}x{height}"
        );
        let pixels = *width as u64 * *height as u64;
        ensure!(
            pixels <= MAX_SEGMENT_PIXELS,
            "segment frame {width}x{height} exceeds the {MAX_SEGMENT_PIXELS}-pixel ceiling"
        );
        let want = pixels.saturating_mul(3);
        ensure!(
            rgb.len() as u64 == want,
            "segment frame {}x{} wants {} rgb bytes, got {}",
            width,
            height,
            want,
            rgb.len()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_variant_round_trips() {
        let requests = [
            Request::EnsureModels,
            Request::Segment {
                rgb: vec![1, 2, 3, 4, 5, 6],
                width: 1,
                height: 2,
                prompt: "bus.".into(),
            },
        ];
        for req in requests {
            let bytes = encode(&req).unwrap();
            assert_eq!(decode_request(&bytes).unwrap(), req);
        }

        let responses = [
            Response::DownloadProgress(0.5),
            Response::Stage("caching models…".into()),
            Response::ModelsReady {
                fetch_ms: 12.5,
                sam2_ms: 340.25,
                provenance: "dino:webgpu enc:webgpu dec:wasm".into(),
            },
            Response::ModelsFailed("sha256 mismatch".into()),
            Response::SegmentDone {
                box_px: [10.0, 20.0, 30.0, 40.0],
                confidence: 0.87,
                mask: vec![255, 0, 255],
                encode_ms: 1.0,
                detect_ms: 2.0,
                decode_ms: 3.0,
            },
            Response::SegmentFailed("no box".into()),
        ];
        for resp in responses {
            let bytes = encode(&resp).unwrap();
            assert_eq!(decode_response(&bytes).unwrap(), resp);
        }
    }

    /// The wire contract is declaration order: an enum variant serializes as
    /// its u32 LE declaration index, Vec/String as u64 LE length + bytes,
    /// u32/f32/f64 LE. Inserting a variant mid-enum silently renumbers every
    /// later discriminant, so the layout stays pinned here — a new variant
    /// joins at the END of its enum or this pin breaks on purpose. A future
    /// hand-rolled JS encoder of these frames inherits this exact layout.
    #[test]
    fn segment_wire_prefix_is_stable() {
        let bytes = encode(&Request::Segment {
            rgb: vec![0xAB, 0xCD, 0xEF],
            width: 1,
            height: 2,
            prompt: "hi".into(),
        })
        .unwrap();
        let want: Vec<u8> = vec![
            0x01, 0x00, 0x00, 0x00, // variant 1 = Segment, u32 LE
            0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // rgb len 3, u64 LE
            0xAB, 0xCD, 0xEF, // rgb bytes
            0x01, 0x00, 0x00, 0x00, // width 1, u32 LE
            0x02, 0x00, 0x00, 0x00, // height 2, u32 LE
            0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // prompt len 2, u64 LE
            b'h', b'i', // prompt utf8
        ];
        assert_eq!(bytes, want, "Request::Segment wire format drifted");
    }

    /// Pin the SegmentDone layout byte-for-byte: variant u32, box 4xf32,
    /// confidence f32, mask u64-length + bytes, then the three f64 timings
    /// in declaration order. The response gained no variant in step 3, so
    /// SegmentDone stays discriminant 4.
    #[test]
    fn segment_done_wire_layout_is_stable() {
        let resp = Response::SegmentDone {
            box_px: [1.5, 2.5, 3.5, 4.5],
            confidence: 0.5,
            mask: vec![0xFF, 0x00],
            encode_ms: 1.0,
            detect_ms: 2.0,
            decode_ms: 3.0,
        };
        let b = encode(&resp).unwrap();
        assert_eq!(&b[0..4], &[4, 0, 0, 0]); // variant 4 = SegmentDone
        assert_eq!(&b[4..8], &1.5f32.to_le_bytes());
        assert_eq!(&b[8..12], &2.5f32.to_le_bytes());
        assert_eq!(&b[12..16], &3.5f32.to_le_bytes());
        assert_eq!(&b[16..20], &4.5f32.to_le_bytes());
        assert_eq!(&b[20..24], &0.5f32.to_le_bytes());
        assert_eq!(&b[24..32], &[2, 0, 0, 0, 0, 0, 0, 0]); // mask len
        assert_eq!(&b[32..34], &[0xFF, 0x00]);
        assert_eq!(&b[34..42], &1.0f64.to_le_bytes());
        assert_eq!(&b[42..50], &2.0f64.to_le_bytes());
        assert_eq!(&b[50..58], &3.0f64.to_le_bytes());
    }

    /// The empty-detection path (prompt matched nothing) is a DONE, not a
    /// failure — the B3-Seg stop rule reads the mask's zero count, so the
    /// wire must carry an all-zero w*h mask with confidence 0, a zero box,
    /// and encode/decode_ms = 0.0 (no encode/decode ran). Round-trips like
    /// any other frame.
    #[test]
    fn empty_detection_done_round_trips_with_zero_mask() {
        let resp = Response::SegmentDone {
            box_px: [0.0; 4],
            confidence: 0.0,
            mask: vec![0; 6], // 3x2 frame
            detect_ms: 4.25,
            encode_ms: 0.0,
            decode_ms: 0.0,
        };
        let round = decode_response(&encode(&resp).unwrap()).unwrap();
        let Response::SegmentDone {
            box_px,
            confidence,
            mask,
            detect_ms,
            encode_ms,
            decode_ms,
        } = round
        else {
            panic!("round trip changed the variant");
        };
        assert_eq!(box_px, [0.0; 4]);
        assert_eq!(confidence, 0.0);
        assert_eq!(mask.len(), 6);
        assert!(mask.iter().all(|&b| b == 0));
        assert_eq!((detect_ms, encode_ms, decode_ms), (4.25, 0.0, 0.0));
    }

    #[test]
    fn segment_rgb_length_must_match_the_frame() {
        let ok = Request::Segment {
            rgb: vec![0; 6],
            width: 1,
            height: 2,
            prompt: "x".into(),
        };
        assert!(ensure_valid(&ok).is_ok());
        // Zero dims are nonsense, not an empty frame.
        let zero = Request::Segment {
            rgb: vec![],
            width: 0,
            height: 0,
            prompt: "x".into(),
        };
        assert!(ensure_valid(&zero).is_err());
        // A frame whose rgb byte count disagrees with its dims is rejected
        // at decode time, before the pipeline can act on it.
        let bad = Request::Segment {
            rgb: vec![0; 3],
            width: 2,
            height: 2,
            prompt: "x".into(),
        };
        assert!(ensure_valid(&bad).is_err());
        // A lying length prefix cannot wrap the u64 cross-check.
        let huge = Request::Segment {
            rgb: vec![0; 3],
            width: u32::MAX,
            height: u32::MAX,
            prompt: "x".into(),
        };
        assert!(ensure_valid(&huge).is_err());
        // And the pixel ceiling stops a self-consistent absurd frame —
        // every byte present, still gigabytes the worker must not run.
        let giant = Request::Segment {
            rgb: vec![0; 3 * 8192 * 8192 + 3],
            width: 8192,
            height: 8192 + 1,
            prompt: "x".into(),
        };
        assert!(ensure_valid(&giant).is_err());
    }

    #[test]
    fn garbage_bytes_fail_loud() {
        assert!(decode_request(&[0xff; 16]).is_err());
        assert!(decode_request(&[]).is_err());
        assert!(decode_response(&[]).is_err());
    }
}
