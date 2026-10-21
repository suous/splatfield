//! Settings persistence (src/settings.rs): the two sidebar controls —
//! iteration count and heatmap preview — survive a page refresh in the
//! page's localStorage under one key. Parsing is pure and total (host
//! testable); the I/O half is wasm-only and best-effort, because a
//! preference must never break the UI: private-mode browsers deny
//! localStorage, and the host tombstone never draws the panel that would
//! read it.

use serde::{Deserialize, Serialize};

/// The two persisted sidebar settings — the same pair [`crate::worker::SegUi`]
/// holds, kept as one plain record so the whole thing serializes in one
/// line and the save-on-change compares it in one `!=`.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct Settings {
    pub(crate) iters: usize,
    pub(crate) heatmap: bool,
}

impl Default for Settings {
    /// The shipped defaults — `SegUi::default`'s pair (20 iters, heatmap
    /// on): anything unreadable must land exactly where a fresh page does.
    fn default() -> Self {
        Self {
            iters: 20,
            heatmap: true,
        }
    }
}

/// The one localStorage key: the two settings move together.
const KEY: &str = "splatfield.settings";

/// The iteration-count window, spelled once for the validator below and the
/// UI's DragValue (`ui.rs` ranges its DragValue on this same const) so the
/// two sides cannot drift.
pub(crate) const ITERS_RANGE: std::ops::RangeInclusive<usize> = 1..=20;

/// Decode a stored record. Total: undecodable JSON, a missing field, or an
/// `iters` outside [`ITERS_RANGE`] all yield the defaults.
///
/// HAZARD: a hand-edited or stale localStorage record must not bypass
/// `Segmenter::new`'s validated config (docs/06_invariants.md, the
/// "validates iterations ≥ 1 … before any GPU allocation" row) — this clamp
/// IS the guard, and it falls back WHOLESALE (not per-field) so a tampered
/// `iters` can't smuggle in a stale `heatmap` either.
fn parse(raw: &str) -> Settings {
    serde_json::from_str::<Settings>(raw)
        .ok()
        .filter(|s| ITERS_RANGE.contains(&s.iters))
        .unwrap_or_default()
}

/// Best-effort read at startup: no window, denied storage, a failed read,
/// or an absent key all land on the defaults (parse re-checks the range —
/// see its hazard note).
#[cfg(target_arch = "wasm32")]
pub(crate) fn load() -> Settings {
    eframe::web::storage::local_storage_get(KEY).map_or_default(|raw| parse(&raw))
}

/// Best-effort write on change: eframe logs and drops storage errors, so
/// denied storage can't break the UI over a preference.
#[cfg(target_arch = "wasm32")]
pub(crate) fn save(s: &Settings) {
    if let Ok(json) = serde_json::to_string(s) {
        eframe::web::storage::local_storage_set(KEY, &json);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_record_round_trips() {
        let s = Settings {
            iters: 7,
            heatmap: false,
        };
        assert_eq!(parse(&serde_json::to_string(&s).unwrap()), s);
        // The boundary values the UI allows survive too.
        let s = Settings {
            iters: 1,
            heatmap: true,
        };
        assert_eq!(parse(&serde_json::to_string(&s).unwrap()), s);
        let s = Settings {
            iters: 20,
            heatmap: false,
        };
        assert_eq!(parse(&serde_json::to_string(&s).unwrap()), s);
    }

    #[test]
    fn out_of_range_iters_falls_back_wholesale() {
        // Below and above the 1..=20 window: the whole record resets — a
        // tampered iters must not carry a stale heatmap through (the
        // wholesale fallback is the pinned shape, not a per-field clamp).
        for raw in [
            r#"{"iters":0,"heatmap":false}"#,
            r#"{"iters":9999,"heatmap":false}"#,
        ] {
            assert_eq!(parse(raw), Settings::default(), "{raw}");
        }
    }

    #[test]
    fn garbage_missing_and_defaults_agree() {
        // Undecodable, empty (the absent-key stand-in on host), and a
        // record missing a field all land on the defaults.
        assert_eq!(parse("not json"), Settings::default());
        assert_eq!(parse(""), Settings::default());
        assert_eq!(parse(r#"{"heatmap":false}"#), Settings::default());
        // And the defaults are SegUi::default's shipped pair.
        assert_eq!(
            Settings::default(),
            Settings {
                iters: 20,
                heatmap: true,
            }
        );
    }
}
