#![deny(unreachable_pub)]

pub mod camera;
/// First-run model fetch (src/fetch.rs): the host half downloads the release
/// zip into the gsam cache; the wasm half streams it, verifies the pins, and
/// hands a [`gsam::ModelStore`] to the OPFS cache.
pub mod fetch;
pub mod layout;
/// OPFS cache of the verified release (src/opfs.rs): persist the files once
/// after a pinned download, then load straight from disk on later page
/// loads. Manifest + checks are pure and host-tested; the I/O half is
/// wasm-only and best-effort — any cache problem reads as a miss.
pub mod opfs;
/// Worker<->main-thread protocol: request/response enums + bincode wire
/// codec, host-tested (src/pipeline.rs).
pub mod pipeline;
mod ply;
mod project;
mod raster;
pub mod render;
pub mod seg;
pub mod sog;
pub mod texture;

// The SH DC coefficient the app's palette→DC conversion divides by.
// `to_dc` is that conversion; the kernel math in `project` uses the same
// constant.
pub use project::to_dc;

/// Route every thread's kernel launches through one ordered stream. cubecl's
/// default policy is per-thread, and per-thread streams don't order against
/// each other — a worker thread's writes (the segmentation tint, the Beta
/// state) would land unordered relative to the UI thread's renders, which
/// then read stale buffers. Idempotent; call once at startup.
pub fn use_single_stream() {
    cubecl_environment::stream::set_policy(cubecl_environment::stream::StreamPolicy::Single);
}

/// The `.sog` archive extension, without the dot: shared by the picker list
/// below and the loader's dispatch so the two can't drift apart.
pub(crate) const SOG_EXTENSION: &str = "sog";

/// Scene file extensions the GUI accepts — `load_scene` dispatches on them.
/// One list so a new format can't land in the parser while the GUI still
/// silently rejects it.
pub const SCENE_EXTENSIONS: [&str; 2] = ["ply", SOG_EXTENSION];

/// Single scene-file ceiling, below `fetch::MAX_DOWNLOAD_BYTES` (that one
/// covers the pinned release zip; real scenes run 14–130 MB). Hazard it
/// prevents: the browser read double-copies the file into linear memory
/// (`arrayBuffer()` + one JS→wasm copy) BEFORE the parser's 1 GiB output
/// plane budget ever runs — a multi-GB drop or download would OOM-abort the
/// wasm32 heap on an avoidable copy instead of failing as an error.
pub const MAX_SCENE_BYTES: u64 = 512 * 1024 * 1024;

/// Preflight a scene source BEFORE any read or download begins: the name's
/// extension must be one of [`SCENE_EXTENSIONS`] and the size, when known,
/// within [`MAX_SCENE_BYTES`]. One guard for every entry point (drop, file
/// picker, URL) so none can drift into accepting what the parsers reject.
/// Pure so the host tests pin the gate the wasm paths run.
pub fn preflight_scene(name: &str, size: Option<u64>) -> Result<(), String> {
    let ext = std::path::Path::new(name)
        .extension()
        .map(|e| e.to_string_lossy().into_owned())
        .unwrap_or_default();
    if !SCENE_EXTENSIONS.iter().any(|x| ext.eq_ignore_ascii_case(x)) {
        return Err(format!("{name} is not a .ply or .sog file"));
    }
    if let Some(size) = size
        && size > MAX_SCENE_BYTES
    {
        return Err(format!(
            "{name} is {} MB — the limit is {} MB",
            size / (1024 * 1024),
            MAX_SCENE_BYTES / (1024 * 1024),
        ));
    }
    Ok(())
}

/// Validate a user-supplied scene URL BEFORE fetching: http(s) only, and
/// the path (query/fragment stripped) must end in a scene-extension name —
/// the same guarantee [`preflight_scene`] gives a local file, applied
/// before any download starts. Returns the trimmed URL and the file name
/// the loader needs (the save button derives `<name>.edited.ply` from it);
/// computing the name here means the GUI's `submit_url` and the validator
/// never parse the same URL twice. Pure string parsing: rejecting the
/// guarded cases needs no URL parser, and staying free of browser
/// types keeps this host-testable.
pub fn validate_scene_url(input: &str) -> Result<(String, String), String> {
    /// The scene file name an http(s) URL tail (scheme stripped) ends in,
    /// if any: query/fragment stripped FIRST — a `?` or `#` ends the path,
    /// so a `/` inside a query must not read as one — then the authority
    /// dropped at the first `/`, base name taken. Without that `/` there
    /// is no file name ("https://model.ply" is a bare host, not a file).
    fn tail_name(rest: &str) -> Option<&str> {
        let path = rest.split(['?', '#']).next().unwrap();
        Some(path.split_once('/')?.1.rsplit('/').next().unwrap())
    }
    let err = || "enter an http(s) URL that ends in .ply or .sog".to_owned();
    let url = input.trim();
    // The scheme reads a lowercased copy — browsers accept "HTTPS://…" —
    // but the name is cut on the ORIGINAL tail: ASCII lowering preserves
    // byte length, so `rest`'s offset into `lower` is its offset into
    // `url`, and `preflight_scene`'s extension match is case-insensitive
    // anyway (an upper-case scheme must not ship an empty name — pinned).
    let lower = url.to_ascii_lowercase();
    let rest = lower
        .strip_prefix("https://")
        .or_else(|| lower.strip_prefix("http://"))
        .ok_or_else(err)?;
    let name = tail_name(&url[url.len() - rest.len()..]).ok_or_else(err)?;
    if preflight_scene(name, None).is_err() {
        return Err(err());
    }
    Ok((url.to_owned(), name.to_owned()))
}

/// Parse a scene file by extension — `.sog` archive, anything else PLY.
/// The loader rule both callers share (the GUI's byte loads and the host
/// oracle); the reader (not a path) is the input.
pub fn load_scene(
    ext: &std::ffi::OsStr,
    file: impl std::io::Read + std::io::Seek,
) -> anyhow::Result<render::CpuSplats> {
    if ext.eq_ignore_ascii_case(SOG_EXTENSION) {
        sog::parse_sog(file)
    } else {
        ply::parse_ply(std::io::BufReader::new(file))
    }
}

/// The scene-source preflight gate's pins: the extension/size/URL cases
/// every entry point (drop, picker, URL submit) funnels through.
#[cfg(test)]
mod scene_source_tests {
    use super::{MAX_SCENE_BYTES, preflight_scene, validate_scene_url};

    #[test]
    fn preflight_accepts_scene_extensions_case_insensitively() {
        assert!(preflight_scene("bear.sog", None).is_ok());
        assert!(preflight_scene("MODEL.PLY", None).is_ok());
        assert!(preflight_scene("/x/y/bear.3d71a266_sh1.sog", None).is_ok());
    }

    #[test]
    fn preflight_rejects_unsupported_types_loudly() {
        let err = preflight_scene("photo.png", None).unwrap_err();
        assert!(err.contains("photo.png") && err.contains(".ply"), "{err}");
        // A source without an extension has nothing to trust.
        assert!(preflight_scene("", None).is_err());
        assert!(preflight_scene("noext", None).is_err());
    }

    #[test]
    fn preflight_rejects_oversize_before_any_read() {
        assert!(preflight_scene("big.ply", Some(MAX_SCENE_BYTES + 1)).is_err());
        assert!(preflight_scene("big.ply", Some(MAX_SCENE_BYTES)).is_ok());
    }

    #[test]
    fn url_validation_preflights_before_fetching() {
        assert!(validate_scene_url("https://h/x/bear.sog").is_ok());
        assert!(validate_scene_url(" http://h/a.ply?token=1 ").is_ok());
        assert!(validate_scene_url("https://h/a.sog#frag").is_ok());
        assert!(validate_scene_url("ftp://h/a.ply").is_err());
        // The scheme matches case-insensitively, like a browser's.
        assert!(validate_scene_url("HTTPS://h/a.ply").is_ok());
        // No path past the host → no file name to preflight, even when the
        // host itself ends in an extension.
        assert!(validate_scene_url("https://h").is_err());
        assert!(validate_scene_url("https://model.ply").is_err());
        assert!(validate_scene_url("https://h/not-a-model").is_err());
        assert!(validate_scene_url("").is_err());
    }

    /// A `/` inside a query or fragment is not a path separator: the path
    /// ends at the first `?` or `#`, so `https://h?a=/x.ply` has NO path —
    /// the bare-host refusal must fire, not a fabricated `x.ply` name
    /// whose download would hit the server root.
    #[test]
    fn url_query_slash_is_not_a_path_separator() {
        assert!(validate_scene_url("https://h?a=/x.ply").is_err());
        assert!(validate_scene_url("https://example.com/redirect?to=/bear.sog").is_err());
        assert!(validate_scene_url("https://h#/x.sog").is_err());
        // The real path's name still wins over a query mention.
        assert!(validate_scene_url("https://h/x.ply?a=/y.sog").is_ok());
    }

    #[test]
    fn url_validation_returns_the_original_cased_name() {
        // Trimming, scheme casing, query, and path depth all stay out of
        // the name; the name itself keeps the user's casing — it narrates
        // the pill and names the `<name>.edited.ply` save.
        let (url, name) =
            validate_scene_url(" https://Host.Example/Dir/Bear.PLY?token=1 ").unwrap();
        assert_eq!(url, "https://Host.Example/Dir/Bear.PLY?token=1");
        assert_eq!(name, "Bear.PLY");
        // The scheme check runs on a lowercased copy; the name cut must
        // still land on the ORIGINAL tail when the scheme itself is
        // upper-case (an empty name here would ship a silent
        // "downloading …" with no file name).
        let (url, name) = validate_scene_url("HTTPS://Host/Dir/Bear.PLY").unwrap();
        assert_eq!(url, "HTTPS://Host/Dir/Bear.PLY");
        assert_eq!(name, "Bear.PLY");
    }

    /// The dialog's `accept` filter is a markup copy of
    /// [`super::SCENE_EXTENSIONS`] the compiler can't check across. Pin it
    /// entry-for-entry (whole entries, so `.plyx` never matches `.ply`) —
    /// a format added to one list without the other is greyed out in the
    /// OS dialog while the drop and URL paths accept it.
    #[test]
    fn picker_accept_list_matches_scene_extensions() {
        let html = include_str!("../index.html");
        let input = html
            .lines()
            .find(|line| line.contains("id=\"scene_picker\""))
            .expect("#scene_picker is gone from index.html");
        let accept = input
            .split_once("accept=\"")
            .and_then(|(_, rest)| rest.split_once('"'))
            .expect("#scene_picker has no accept attribute")
            .0;
        for ext in super::SCENE_EXTENSIONS {
            assert!(
                accept.split(',').any(|entry| entry == format!(".{ext}")),
                "{ext} is missing from #scene_picker"
            );
        }
    }
}

/// GPU test support: the sort crate's serialized test client, entered only
/// after setting the single-stream policy — the root render/seg tests'
/// cross-thread ordering assertions depend on it, which sort's own
/// single-threaded test callers don't.
#[cfg(test)]
pub(crate) mod gpu_testing {
    pub(crate) fn test_client() -> (std::sync::MutexGuard<'static, ()>, cubecl::client::Client) {
        crate::use_single_stream();
        splat_sort::tensor::test_client()
    }
}
