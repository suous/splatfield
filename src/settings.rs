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
    match serde_json::from_str::<Settings>(raw) {
        Ok(s) if ITERS_RANGE.contains(&s.iters) => s,
        _ => Settings::default(),
    }
}

/// Best-effort read at startup: no window, denied storage, a failed read,
/// or an absent key all land on the defaults (parse re-checks the range —
/// see its hazard note).
#[cfg(target_arch = "wasm32")]
pub(crate) fn load() -> Settings {
    local_storage()
        .and_then(|s| s.get_item(KEY).ok().flatten())
        .map(|raw| parse(&raw))
        .unwrap_or_default()
}

/// Best-effort write on change: errors ignored — private-mode browsers must
/// not break the UI over a preference. Encode can't fail for two plain
/// fields, but the guard keeps the write honest anyway.
#[cfg(target_arch = "wasm32")]
pub(crate) fn save(s: &Settings) {
    if let (Some(storage), Ok(json)) = (local_storage(), serde_json::to_string(s)) {
        let _ = storage.set_item(KEY, &json);
    }
}

/// `window().local_storage()`, flattened: `None` on a no-window build or a
/// storage the browser refuses to hand out.
#[cfg(target_arch = "wasm32")]
fn local_storage() -> Option<web_sys::Storage> {
    web_sys::window().and_then(|w| w.local_storage().ok().flatten())
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
