//! First-run model fetch for the segmentation pipeline: stream the pinned
//! release zip, verify its sha256, unpack exactly the release files against
//! the per-file pins, and hand back a [`gsam::ModelStore`].
//!
//! The network half (release zip, demo scene, user scene URLs) is the
//! browser `fetch` API — no test reaches it — while the install logic is
//! platform-neutral and host-tested.

use anyhow::{Context, Result, bail, ensure};
use gsam::{ModelStore, REQUIRED_FILES};
use std::io::Read as _;
#[cfg(test)]
use std::io::Write as _;

/// The release zip URL on the web — relative on purpose: in the pipeline
/// worker it resolves against the worker script's base URL (same directory,
/// same origin as the page), so it lands next to index.html either way.
/// `scripts/dev_server.py` serves it from the durable master copy
/// `data/splatfield-models.zip` (git-ignored): a trunk rebuild wipes
/// `dist/`, so the zip is never copied there.
#[cfg(target_arch = "wasm32")]
pub const ZIP_URL: &str = "models.zip";

/// Streamed-download byte ceiling — the stream aborts past it, so a
/// sentinel or corrupt URL pointing at some huge file can't fill the
/// tab's memory.
#[cfg(target_arch = "wasm32")]
const MAX_DOWNLOAD_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// The stream buffer's reserve ceiling: content-length is untrusted, and
/// reserving a lying 2 GiB header aborts the wasm32 heap before the first
/// byte arrives. The pinned releases are known-sized (models.zip, demo
/// scene — both well under this), so they keep their up-front reserve;
/// anything larger grows honestly while the caller's ceiling
/// (`MAX_DOWNLOAD_BYTES` for the zip, `MAX_SCENE_BYTES` for scenes) still
/// bounds the real bytes. Wasm-only: the host downloader streams
/// through ureq's own `.limit` instead of `stream_body`.
#[cfg(target_arch = "wasm32")]
const MAX_RESERVE_BYTES: u64 = 256 * 1024 * 1024;

/// SHA-256 of the release zip. The wasm install checks the whole archive
/// against it before unzipping — a corrupt or substituted download never
/// reaches the per-file pins with a confusing error.
pub const ZIP_SHA256: &str = "259ed2a989fc368b839cb32c1b43c758fa1207e4849123e603f83baedb35f740";

/// The release zip's byte length — the bytes are sha-pinned
/// ([`ZIP_SHA256`]), so the length is a release constant too. Seeds the
/// status pill's MB figures and the parts assembly's one honest reserve
/// (a wrong value costs a realloc, never correctness — the sha256 pin
/// gates the install); a new release moves it together with the pin.
#[cfg(target_arch = "wasm32")]
pub const ZIP_BYTES: u64 = 158_618_158;

/// A release past the download ceiling would stream to the ceiling then
/// die mid-install; pin the pair at compile time like the scene ceiling.
#[cfg(target_arch = "wasm32")]
const _: () = assert!(ZIP_BYTES <= MAX_DOWNLOAD_BYTES);

/// SHA-256 of every release file, keyed by its release-relative path, in
/// [`gsam::REQUIRED_FILES`] order — the two tables pair positionally (pinned
/// by tests). Verified during install; not re-hashed on launch.
const FILE_SHA256: [(&str, &str); 6] = [
    (
        "grounding_dino_tiny/onnx/model_q4f16.onnx",
        "48435b57e5a5ca01792596b9c64277260b734c10aed320109505c8e71238d6ac",
    ),
    (
        "grounding_dino_tiny/tokenizer.json",
        "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    ),
    (
        "sam2_tiny/onnx/vision_encoder_q4f16.onnx",
        "c4092f4c02a369a94b65718ba9fd274c0dc8923e4fa10e55bc908633d4a4a794",
    ),
    (
        "sam2_tiny/onnx/vision_encoder_q4f16.onnx_data",
        "48c3dc2795f70c855fc35061b9f49431d50762c00af8c644380781f2a20eb187",
    ),
    (
        "sam2_tiny/onnx/prompt_encoder_mask_decoder_q4f16.onnx",
        "0ff3ee0406764a4c5e8f05c1b5d59de4ec292c7c2a13d64f08a7d37a2d9f697b",
    ),
    (
        "sam2_tiny/onnx/prompt_encoder_mask_decoder_q4f16.onnx_data",
        "8b204a80f601cf580f81ede775a096ecbb0c25f3d49ea592b6364ac7d67cc3d0",
    ),
];

/// Download + install the pinned release into a fresh store — or build it
/// straight from the OPFS cache when one survives from an earlier load.
/// The bool answers "came from the cache": a cache hit needs no persist (the
/// bytes were just re-validated from the very store being returned), and the
/// worker skips its caching stage on it. Persisting is the caller's job (the
/// worker stages it and owns the UX replies); the zip's ~158 MB drop when
/// this returns on the download path is deliberate — the persist copies each
/// file out to JS, and holding both at once is the memory-spike class of bug
/// fixed for the detector (worker_main take).
#[cfg(target_arch = "wasm32")]
pub async fn ensure_models(on_progress: &mut dyn FnMut(f64)) -> Result<(ModelStore, bool)> {
    if let Some(store) = crate::opfs::load_cached().await {
        return Ok((store, true));
    }
    let mut store = ModelStore::default();
    let zip = download_zip(on_progress).await?;
    install_zip(&zip, &mut store).context("installing release zip")?;
    Ok((store, false))
}

/// The zip-entry name's pin slot: strip one optional top-level `models/`
/// component (whole-component only — `models_x/…` does not strip), then
/// match a pin exactly, so strays and zip-slip paths never qualify. Both
/// install walks (host fs, wasm store) share this one matcher so their
/// matching rules cannot drift.
fn pin_slot(entry_name: &str, pins: &[(&str, &str)]) -> Option<usize> {
    let name = entry_name.strip_prefix("models/").unwrap_or(entry_name);
    pins.iter().position(|(p, _)| *p == name)
}

/// GET `url` from whichever scope declares fetch — a Window on the page, a
/// WorkerGlobalScope inside the pipeline worker — and return the raw
/// `Response`, whatever its status. The status-raw variant exists for the
/// parts-manifest probe, which must distinguish a 404 (the dev server's
/// single-file shape, no manifest shipped) from a real failure; everything
/// else goes through [`fetch_ok`].
#[cfg(target_arch = "wasm32")]
pub(crate) async fn fetch_response(
    url: &str,
    cache: Option<web_sys::RequestCache>,
) -> Result<web_sys::Response> {
    use crate::opfs::js_err;
    use wasm_bindgen::JsCast;

    let init = web_sys::RequestInit::new();
    if let Some(cache) = cache {
        init.set_cache(cache);
    }
    let request = web_sys::Request::new_with_str_and_init(url, &init)
        .map_err(|e| js_err(e).context(format!("building the request for {url}")))?;
    let fetch_promise = match js_sys::global().dyn_into::<web_sys::Window>() {
        Ok(window) => window.fetch_with_request(&request),
        Err(global) => global
            .unchecked_into::<web_sys::WorkerGlobalScope>()
            .fetch_with_request(&request),
    };
    let response = wasm_bindgen_futures::JsFuture::from(fetch_promise)
        .await
        .map_err(|e| js_err(e).context(format!("fetching {url}")))?;
    response
        .dyn_into()
        .map_err(|_| anyhow::anyhow!("fetch of {url} did not yield a Response"))
}

/// GET `url` from whichever scope declares fetch — a Window on the page, a
/// WorkerGlobalScope inside the pipeline worker — and require an OK
/// `Response`. `cache`, when given, rides the request's `RequestInit`;
/// `None` leaves every field at its default (wire-identical to no init).
#[cfg(target_arch = "wasm32")]
pub(crate) async fn fetch_ok(
    url: &str,
    cache: Option<web_sys::RequestCache>,
) -> Result<web_sys::Response> {
    let response = fetch_response(url, cache).await?;
    ensure!(response.ok(), "fetch {url}: HTTP {}", response.status());
    Ok(response)
}

/// Read a response header — the only JsValue error from `Headers::get`
/// surfaces; absent returns `Ok(None)`. One site so every header read
/// shares the `reading {name}: …` wording.
#[cfg(target_arch = "wasm32")]
fn header_raw(response: &web_sys::Response, name: &str) -> Result<Option<String>> {
    response
        .headers()
        .get(name)
        .map_err(|e| anyhow::anyhow!("reading {name}: {e:?}"))
}

/// Read `response`'s body into one buffer, streaming: the progress callback
/// fires per chunk with the fraction of content-length (never called when
/// the header is absent). `max_bytes` holds per chunk, so a mislabeled or
/// endless stream can't grow past it just because the header lied. The
/// buffer is reserved up front — a >100 MB Vec growing by amortized
/// doubling memcpy's the whole body several times inside the worker.
#[cfg(target_arch = "wasm32")]
pub(crate) async fn stream_body(
    response: web_sys::Response,
    max_bytes: u64,
    on_progress: &mut dyn FnMut(f64),
) -> Result<Vec<u8>> {
    use crate::opfs::js_err;
    use wasm_bindgen::JsCast;

    let total = header_raw(&response, "content-length")?.and_then(|v| v.parse::<u64>().ok());
    let reader = response
        .body()
        .context("response carries no body")?
        .get_reader()
        .dyn_into::<web_sys::ReadableStreamDefaultReader>()
        .map_err(|_| anyhow::anyhow!("response stream is not a default reader"))?;
    let mut buf =
        Vec::with_capacity(total.map_or(0, |t| t.min(max_bytes).min(MAX_RESERVE_BYTES) as usize));
    loop {
        // The read result is a spec dictionary, not a JS class — dyn_into
        // would `instanceof` against a nonexistent global and fail the cast
        // on every chunk — so the cast from the resolved promise value is
        // unchecked; the getters below are plain property reads either way.
        let chunk: web_sys::ReadableStreamReadResult =
            wasm_bindgen_futures::JsFuture::from(reader.read())
                .await
                .map_err(|e| js_err(e).context("stream chunk unreadable"))?
                .unchecked_into();
        if chunk.get_done() == Some(true) {
            return Ok(buf);
        }
        let value: js_sys::Uint8Array = chunk
            .get_value()
            .dyn_into()
            .map_err(|_| anyhow::anyhow!("stream chunk is not a Uint8Array"))?;
        // copy_to writes straight into the buffer's tail: a to_vec+extend
        // would copy every chunk twice and allocate a throwaway Vec per
        // chunk. The ceiling check precedes the resize — one attacker-
        // sized chunk must not allocate past it (u64 sum: usize wraps).
        let start = buf.len();
        let chunk_len = value.length() as u64;
        ensure!(
            buf.len() as u64 + chunk_len <= max_bytes,
            "stream exceeds the {max_bytes}-byte ceiling"
        );
        buf.resize(start + chunk_len as usize, 0);
        value.copy_to(&mut buf[start..]);
        if let Some(total) = total {
            on_progress((buf.len() as f64 / total as f64).min(1.0));
        }
    }
}

/// The `models.zip` sibling naming convention shared with the split in
/// `scripts/ship_models.sh` — the one recipe both deploy legs run: parts
/// ship as `<base>.p0…p<N-1>` behind a `<base>.manifest.json` of
/// `{"parts": N}`. A rename here must move the script with it — pinned by
/// the url-contract test, the one anchor the compiler can't check across
/// the shell script.
#[cfg(any(target_arch = "wasm32", test))]
fn zip_part_url(base: &str, i: u32) -> String {
    format!("{base}.p{i}")
}

#[cfg(any(target_arch = "wasm32", test))]
fn zip_manifest_url(base: &str) -> String {
    format!("{base}.manifest.json")
}

/// Parse a parts manifest body: `{"parts": N}` with N in 1..=64. `None`
/// means "not a manifest" — but a manifest that EXISTS yet parses to None
/// is a broken parts deployment (under the 25 MB/file cap the single
/// archive cannot coexist), so the caller fails loud instead of falling
/// back to the single-file URL.
#[cfg(any(target_arch = "wasm32", test))]
fn parse_parts_manifest(body: &[u8]) -> Option<u32> {
    let v: serde_json::Value = serde_json::from_slice(body).ok()?;
    u32::try_from(v.get("parts")?.as_u64()?)
        .ok()
        .filter(|n| (1..=64).contains(n))
}

/// Streamed-manifest ceiling: the manifest is ~15 bytes, so anything past
/// 64 KiB is a lie — cut before the JSON parse sees it.
#[cfg(target_arch = "wasm32")]
const MAX_MANIFEST_BYTES: u64 = 64 * 1024;

/// Stream the release zip off the network with fractional progress (0..1).
/// Native builds never download: this is the browser branch.
///
/// Two deployment shapes, chosen by probing `models.zip.manifest.json`:
/// absent (HTTP 404) — the single same-origin `models.zip` that
/// `scripts/dev_server.py` serves; present — the zip ships as
/// `models.zip.p0…` parts, because EdgeOne Makers caps deploy files at
/// 25 MB and the 158 MB archive cannot exist there whole (Pages ships
/// the same parts shape, so dev is the only single-file deployment).
/// The parts join in memory and face the SAME whole-zip [`ZIP_SHA256`]
/// pin as the single download — the parts change the transport, never
/// the verification.
#[cfg(target_arch = "wasm32")]
pub async fn download_zip(on_progress: &mut dyn FnMut(f64)) -> Result<Vec<u8>> {
    // Revalidate before use: a same-origin cache entry from an earlier dev
    // session (possibly behind a different server state) otherwise shadows
    // the pinned release and the sha256 pin rejects bytes the user cannot
    // fix except by clearing browser state. With `no-cache` a stale entry
    // costs one conditional request; a fresh file answers 200 unchanged.
    let manifest = fetch_response(
        &zip_manifest_url(ZIP_URL),
        Some(web_sys::RequestCache::NoCache),
    )
    .await?;
    if manifest.status() == 404 {
        let response = fetch_ok(ZIP_URL, Some(web_sys::RequestCache::NoCache)).await?;
        return stream_body(response, MAX_DOWNLOAD_BYTES, on_progress).await;
    }
    ensure!(
        manifest.ok(),
        "fetch {}: HTTP {}",
        zip_manifest_url(ZIP_URL),
        manifest.status()
    );
    let body = stream_body(manifest, MAX_MANIFEST_BYTES, &mut |_| {}).await?;
    let parts = parse_parts_manifest(&body).with_context(|| {
        format!(
            "{} is not a parts manifest — a parts deployment ships {{\"parts\": N}} \
         next to the {} parts",
            zip_manifest_url(ZIP_URL),
            zip_part_url(ZIP_URL, 0)
        )
    })?;
    // The pinned length seeds the reserve: honest parts then never realloc
    // (a 158 MB Vec doubling by 22 MB steps memcpy's ~300 MB inside the
    // worker), and a wrong pin costs one realloc, never correctness — the
    // sha256 pin still gates the install.
    let mut zip = Vec::with_capacity(ZIP_BYTES as usize);
    for i in 0..parts {
        let response = fetch_ok(
            &zip_part_url(ZIP_URL, i),
            Some(web_sys::RequestCache::NoCache),
        )
        .await?;
        let start = zip.len() as u64;
        let chunk = stream_body(response, MAX_DOWNLOAD_BYTES, &mut |f| {
            on_progress((i as f64 + f) / parts as f64)
        })
        .await?;
        // Per-part streams are individually capped; this holds the SAME
        // ceiling over the assembly, so 64 honest-looking parts can't
        // sum past it.
        ensure!(
            start + chunk.len() as u64 <= MAX_DOWNLOAD_BYTES,
            "assembled zip exceeds the {}-byte ceiling",
            MAX_DOWNLOAD_BYTES
        );
        zip.extend_from_slice(&chunk);
    }
    Ok(zip)
}

/// The demo scene the landing card's Bear button loads — the SH1 bear on the
/// fixtures release, same assets CI tests against. Relative on purpose,
/// like `ZIP_URL`: the page's `<base>` (trunk's public URL) resolves it
/// next to index.html, where `scripts/dev_server.py` serves the durable
/// `data/` master copy and the deploy workflow ships the release asset
/// into the Pages artifact. The release URL itself is unreachable from a
/// browser: the github.com download redirect chain sends no CORS headers
/// (curl never shows this — it does not enforce CORS).
#[cfg(target_arch = "wasm32")]
pub const DEMO_SCENE_URL: &str = "bear.3d71a266_sh1.sog";

/// The demo scene's byte length — content-pinned by the fixtures-v1 tag
/// (the file name embeds the source hash), so this is a display constant
/// (the status pill's MB figures), never load-bearing. Same release
/// contract as `ZIP_BYTES`.
#[cfg(target_arch = "wasm32")]
pub const DEMO_SCENE_BYTES: u64 = 14_222_631;

/// Fetch the demo scene bytes, streaming fractional progress (0..1) to
/// `on_progress` — the status pill narrates the wait the way the models
/// download does. One buffer, no decoding. Same origin, so plain CORS
/// rules apply and no cache override is needed (unlike the models zip):
/// the content-hashed name is immutable — new fixture bytes get a new
/// name, so default HTTP caching can never serve a stale scene.
#[cfg(target_arch = "wasm32")]
pub async fn fetch_demo_scene(mut on_progress: impl FnMut(f64)) -> Result<Vec<u8>> {
    let response = fetch_ok(DEMO_SCENE_URL, None).await?;
    // A dev server that answers unknown paths with index.html (trunk
    // serve's SPA fallback) reports 200 + text/html — name that failure
    // instead of letting HTML surface as a cryptic zip error in parse.
    let content_type = header_raw(&response, "content-type")?.unwrap_or_default();
    ensure!(
        !content_type.starts_with("text/"),
        "the demo scene endpoint answered {content_type:?} — the asset is not \
         deployed next to the app; local development must serve it via \
         scripts/dev_server.py"
    );
    stream_body(response, crate::MAX_SCENE_BYTES, &mut on_progress).await
}

// The demo streams under the scene ceiling like every other scene source
// (lib.rs): a release asset larger than the ceiling must move the ceiling
// consciously, not silently stream 2 GiB into a wasm32 tab.
#[cfg(target_arch = "wasm32")]
const _: () = assert!(DEMO_SCENE_BYTES <= crate::MAX_SCENE_BYTES);

/// Fetch a user-supplied scene URL — the demo-scene pattern minus the fixed
/// endpoint: cross-origin only works when the server sends CORS headers (a
/// refusal surfaces as the JS TypeError inside `fetch_ok`'s "fetching {url}"
/// context). The URL was preflighted by `validate_scene_url` before this
/// runs; the size holds HERE too: a content-length over the scene ceiling
/// fails on the header, before the first chunk, and `stream_body`'s
/// per-chunk ceiling catches a lying header mid-stream.
#[cfg(target_arch = "wasm32")]
pub async fn fetch_scene_url(
    url: String,
    name: String,
    mut on_progress: impl FnMut(f64),
) -> Result<Vec<u8>> {
    let response = fetch_ok(&url, None).await?;
    // The lie the demo endpoint's guard (fetch_demo_scene) catches, narrowed
    // for user URLs: only text/html — a same-path SPA fallback — is
    // certainly not a model. PLY is legitimately text/plain (a plain file
    // host serves it so), so the broader text/ rejection would fail real
    // models with a wrong explanation; a mislabeled non-HTML body fails
    // loudly in parse instead.
    let content_type = header_raw(&response, "content-type")?.unwrap_or_default();
    ensure!(
        !content_type.starts_with("text/html"),
        "{url} answered {content_type:?} — that is a web page, not a model file"
    );
    let total = header_raw(&response, "content-length")?.and_then(|v| v.parse::<u64>().ok());
    if let Some(total) = total {
        // Same message a preflighted file gets for the same size violation.
        crate::preflight_scene(&name, Some(total)).map_err(anyhow::Error::msg)?;
    }
    stream_body(response, crate::MAX_SCENE_BYTES, &mut on_progress).await
}

/// Verify the zip's sha256, unzip in memory, keep exactly the required
/// release files (the zip may carry a top-level `models/` wrapper or junk),
/// and verify each against its pin. A wrong hash anywhere is a hard error —
/// nothing partial lands in `store`.
pub fn install_zip(zip_bytes: &[u8], store: &mut ModelStore) -> Result<()> {
    let got = zip_sha256(zip_bytes);
    ensure!(
        got == ZIP_SHA256,
        // {n} bytes: a truncated transfer hashes differently every time —
        // the length distinguishes "short read" from "garbled bytes".
        "sha256 of the release zip is {got} over {} bytes, expected {ZIP_SHA256} over a pinned length — corrupt download or stale pin",
        zip_bytes.len(),
    );
    install_zip_with(&FILE_SHA256, zip_bytes, store)
}

/// SHA-256 of raw bytes as lowercase hex.
fn zip_sha256(bytes: &[u8]) -> String {
    use sha2::Digest;
    format!("{:x}", sha2::Sha256::digest(bytes))
}

/// Unzip `zip_bytes` in memory and install exactly the pinned files into
/// `store`. An entry qualifies only by exact name match against a pin,
/// after one optional leading `models/` component is stripped — so zip-slip
/// paths, `__MACOSX` cruft, and strays are ignored, and no archive string
/// is ever used as a path. `pins` pairs with [`gsam::REQUIRED_FILES`]
/// positionally (pinned by tests); verified files land in the store only
/// after every pin passed — a wrong hash anywhere leaves `store` empty.
pub fn install_zip_with(
    pins: &[(&str, &str)],
    zip_bytes: &[u8],
    store: &mut ModelStore,
) -> Result<()> {
    let mut archive =
        zip::ZipArchive::new(std::io::Cursor::new(zip_bytes)).context("not a zip file")?;
    // The memory peak here is intrinsic, not a lifetime bug: the archive
    // reads lazily from `zip_bytes`, so the whole zip stays alive through
    // the loop while `slots` accumulates every extracted file (~zip +
    // extracted, a ~490 MB transient on the 158 MB release), and the
    // store-fills-only-after-every-pin-passes invariant needs the
    // bytes to coexist until the last hash passes. `insert` then MOVES them
    // out of `slots` — there is no copy to remove. Shrinking this peak
    // means a redesign (streaming zip reader, OPFS spill), not a reorder.
    let mut slots: Vec<Option<(&'static str, Vec<u8>)>> = vec![None; pins.len()];
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        if entry.is_dir() {
            continue;
        }
        let Some(slot) = pin_slot(&entry.name()?, pins) else {
            continue;
        };
        // `position` bounds `slot` below `pins.len()`.
        let (rel, want) = pins[slot];
        let Some(name) = REQUIRED_FILES.get(slot).copied() else {
            bail!(
                "pin {slot} ({rel}) exceeds the {} required files — pin table drift",
                REQUIRED_FILES.len()
            );
        };
        // Same reserve-up-front rule as download_zip (.onnx_data is ~150 MB).
        let mut bytes = Vec::with_capacity(entry.size() as usize);
        entry
            .read_to_end(&mut bytes)
            .with_context(|| format!("reading {rel}"))?;
        let got = zip_sha256(&bytes);
        ensure!(
            got == *want,
            "sha256 of {rel} is {got}, expected {want} — corrupt zip or stale pin"
        );
        slots[slot] = Some((name, bytes));
    }
    let missing: Vec<&str> = pins
        .iter()
        .zip(&slots)
        .filter(|(_, slot)| slot.is_none())
        .map(|((rel, _), _)| *rel)
        .collect();
    ensure!(
        missing.is_empty(),
        "release zip lacks {} — it must package the release layout, \
         optionally under one top-level models/ directory",
        missing.join(", ")
    );
    for (name, bytes) in slots.into_iter().flatten() {
        store.insert(name, bytes)?;
    }
    Ok(())
}

/// One in-memory zip with `SimpleFileOptions` entries, shared by the host
/// tests that fabricate archives (the install tests below and sog.rs's SOG
/// fixtures). The deflate-bomb fixture stays hand-rolled: it streams a
/// 256 MiB entry in chunks a `&[u8]`-per-entry signature can't express.
#[cfg(test)]
pub(crate) fn build_zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut w = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    for (name, data) in entries {
        if name.ends_with('/') {
            w.add_directory(name.to_string(), zip::write::SimpleFileOptions::default())
                .unwrap();
        } else {
            w.start_file(name, zip::write::SimpleFileOptions::default())
                .unwrap();
            w.write_all(data).unwrap();
        }
    }
    w.finish().unwrap().into_inner()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pin table and gsam's required-file list must describe the same
    /// release: same length, same order — the host extraction, the wasm
    /// install, and the OPFS cache all walk them side by side.
    #[test]
    fn test_pins_pair_with_required_files() {
        assert_eq!(FILE_SHA256.len(), REQUIRED_FILES.len());
        for ((rel, _), req) in FILE_SHA256.iter().zip(REQUIRED_FILES) {
            assert_eq!(*rel, req);
        }
    }

    /// Pins for one synthetic release: each required file's real sha256.
    fn pins_for(contents: &[&[u8]; 6]) -> Vec<(&'static str, String)> {
        REQUIRED_FILES
            .iter()
            .zip(contents)
            .map(|(file, bytes)| (*file, zip_sha256(bytes)))
            .collect()
    }

    fn pin_refs<'a>(pins: &'a [(&'static str, String)]) -> Vec<(&'a str, &'a str)> {
        pins.iter().map(|(p, s)| (*p, s.as_str())).collect()
    }

    /// The six release files' fixture payloads, in REQUIRED_FILES order —
    /// every install test's store contents (the hash-mismatch test's
    /// `tampered` array is the deliberate counterexample).
    const CONTENTS: [&[u8]; 6] = [b"dino", b"{}", b"enc", b"enc-data", b"dec", b"dec-data"];

    /// Those names behind one `models/` wrapper directory level.
    fn wrapped_names() -> Vec<String> {
        REQUIRED_FILES
            .iter()
            .map(|f| format!("models/{f}"))
            .collect()
    }

    /// The parts-manifest shape scripts/ship_models.sh writes: an integer
    /// `parts` in 1..=64. Both extremes parse — 1 is a legal (if pointless)
    /// parts deployment, 64 the documented sanity ceiling.
    #[test]
    fn parse_parts_manifest_accepts_the_deployed_shape() {
        assert_eq!(parse_parts_manifest(br#"{"parts":7}"#), Some(7));
        // The script's printf writes exactly this — space after the
        // colon, trailing newline — so pin it byte-for-byte.
        assert_eq!(parse_parts_manifest(b"{\"parts\": 7}\n"), Some(7));
        assert_eq!(parse_parts_manifest(br#"{"parts":1}"#), Some(1));
        assert_eq!(parse_parts_manifest(br#"{"parts": 64 }"#), Some(64));
    }

    /// Everything that is not an in-range integer part count is rejected —
    /// the caller fails loud on these rather than falling back to the
    /// single-file URL, because a manifest that exists means the
    /// deployment INTENDED parts (the single archive cannot coexist under
    /// the 25 MB/file cap).
    #[test]
    fn parse_parts_manifest_rejects_everything_else() {
        assert_eq!(parse_parts_manifest(b""), None);
        assert_eq!(parse_parts_manifest(b"not json"), None);
        assert_eq!(parse_parts_manifest(b"[]"), None);
        assert_eq!(parse_parts_manifest(br#"{"parts":0}"#), None);
        assert_eq!(parse_parts_manifest(br#"{"parts":65}"#), None);
        assert_eq!(parse_parts_manifest(br#"{"parts":-1}"#), None);
        assert_eq!(parse_parts_manifest(br#"{"parts":7.5}"#), None);
        assert_eq!(parse_parts_manifest(br#"{"parts":"7"}"#), None);
        assert_eq!(parse_parts_manifest(br#"{"count":7}"#), None);
        assert_eq!(parse_parts_manifest(br#"{"parts":7} trailing"#), None);
        assert_eq!(
            parse_parts_manifest(br#"{"parts":18446744073709551615}"#),
            None
        );
    }

    /// The URL convention the deploy workflow's split step must write and
    /// the worker reads: `<base>.p{i}` parts behind `<base>.manifest.json`.
    /// A rename on either side without the other 404s every part — this
    /// pin is the one cross-artifact anchor the compiler can't check.
    #[test]
    fn zip_part_and_manifest_urls_follow_the_deploy_convention() {
        assert_eq!(zip_part_url("models.zip", 0), "models.zip.p0");
        assert_eq!(zip_part_url("models.zip", 6), "models.zip.p6");
        assert_eq!(zip_manifest_url("models.zip"), "models.zip.manifest.json");
    }

    /// Six files through the `models/` wrapper, hashes checked byte-for-byte.
    #[test]
    fn test_install_installs_all_required_files_through_the_models_wrapper() {
        let pins = pins_for(&CONTENTS);
        let names = wrapped_names();
        let entries: Vec<(&str, &[u8])> = names
            .iter()
            .zip(&CONTENTS)
            .map(|(name, bytes)| (name.as_str(), *bytes))
            .collect();
        let zip = build_zip(&entries);

        let mut store = ModelStore::default();
        install_zip_with(&pin_refs(&pins), &zip, &mut store).unwrap();
        assert!(store.is_complete());
        for (file, bytes) in REQUIRED_FILES.iter().zip(&CONTENTS) {
            assert_eq!(*store.get(file).unwrap(), *bytes);
        }
    }

    /// The `models/` wrapper is optional: a flat zip installs identically.
    #[test]
    fn test_install_accepts_the_zip_without_the_models_wrapper() {
        let pins = pins_for(&CONTENTS);
        let entries: Vec<(&str, &[u8])> = REQUIRED_FILES
            .iter()
            .zip(&CONTENTS)
            .map(|(file, bytes)| (*file, *bytes))
            .collect();
        let zip = build_zip(&entries);

        let mut store = ModelStore::default();
        install_zip_with(&pin_refs(&pins), &zip, &mut store).unwrap();
        assert!(store.is_complete());
    }

    /// `__MACOSX` cruft, root strays, junk inside allowed roots, and
    /// directory stubs are ignored — only pinned paths reach the store.
    #[test]
    fn test_install_ignores_junk_entries() {
        let pins = pins_for(&CONTENTS);
        let names = wrapped_names();
        let mut entries: Vec<(String, &[u8])> = names
            .iter()
            .zip(&CONTENTS)
            .map(|(name, bytes)| (name.clone(), *bytes))
            .collect();
        entries.push(("__MACOSX/models/x".into(), b"cruft".as_slice()));
        entries.push(("random.txt".into(), b"no".as_slice()));
        entries.push((
            "models/grounding_dino_tiny/junk.bin".into(),
            b"no".as_slice(),
        ));
        entries.push(("models/".into(), b"".as_slice()));
        let owned: Vec<(&str, &[u8])> = entries.iter().map(|(n, b)| (n.as_str(), *b)).collect();
        let zip = build_zip(&owned);

        let mut store = ModelStore::default();
        install_zip_with(&pin_refs(&pins), &zip, &mut store).unwrap();
        assert!(store.is_complete());
    }

    /// A tampered payload is a hard error naming the file, and NOTHING
    /// lands in the store — a half-installed release would feed corrupt
    /// weights to the sessions forever.
    #[test]
    fn test_install_hash_mismatch_names_the_file_and_leaves_the_store_empty() {
        let pins = pins_for(&CONTENTS);
        let tampered: [&[u8]; 6] = [
            b"dino",
            b"{}",
            b"TAMPERED",
            b"enc-data",
            b"dec",
            b"dec-data",
        ];
        let names = wrapped_names();
        let entries: Vec<(&str, &[u8])> = names
            .iter()
            .zip(&tampered)
            .map(|(name, bytes)| (name.as_str(), *bytes))
            .collect();
        let zip = build_zip(&entries);

        let mut store = ModelStore::default();
        let err = install_zip_with(&pin_refs(&pins), &zip, &mut store).unwrap_err();
        assert!(err.to_string().contains(REQUIRED_FILES[2]), "{err}");
        assert_eq!(store.missing(), REQUIRED_FILES.to_vec());
    }

    /// A zip lacking a required file errors naming it, store untouched.
    #[test]
    fn test_install_missing_required_file_names_it_and_leaves_the_store_empty() {
        let pins = pins_for(&CONTENTS);
        let mut kept: Vec<(&'static str, &[u8])> =
            REQUIRED_FILES.iter().copied().zip(CONTENTS).collect();
        let (dropped_file, _) = kept.pop().unwrap();
        assert_eq!(dropped_file, REQUIRED_FILES[5]);
        let zip = build_zip(&kept);

        let mut store = ModelStore::default();
        let err = install_zip_with(&pin_refs(&pins), &zip, &mut store).unwrap_err();
        assert!(err.to_string().contains(REQUIRED_FILES[5]), "{err}");
        assert_eq!(store.missing(), REQUIRED_FILES.to_vec());
    }

    /// The helper is plain lowercase-hex sha256 — pinned against the NIST
    /// test vector for "abc".
    #[test]
    fn test_zip_sha256_is_lowercase_hex_sha256() {
        assert_eq!(
            zip_sha256(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    /// The one matcher behind both install walks (host fs, wasm store):
    /// one optional `models/` wrapper, then an exact pin match. `models_x`
    /// keeps its prefix, a bare `models` matches nothing, and a second
    /// wrapper never unwraps — the two spellings this replaced (host
    /// component-wise `Path::strip_prefix("models")`, wasm string
    /// `strip_prefix("models/")`) agreed on every one of these.
    #[test]
    fn pin_slot_strips_one_wrapper_and_matches_exactly() {
        let pins = [("a/b.onnx", "h1"), ("top.onnx", "h2")];
        assert_eq!(pin_slot("models/a/b.onnx", &pins), Some(0));
        assert_eq!(pin_slot("a/b.onnx", &pins), Some(0));
        assert_eq!(pin_slot("top.onnx", &pins), Some(1));
        assert_eq!(pin_slot("models_x/a.onnx", &pins), None);
        assert_eq!(pin_slot("models", &pins), None);
        assert_eq!(pin_slot("models/models/a/b.onnx", &pins), None);
        assert_eq!(pin_slot("__MACOSX/a", &pins), None);
    }

    /// The production wrapper first checks the whole zip against
    /// [`ZIP_SHA256`] — a corrupt or substituted download never reaches the
    /// unzip step.
    #[test]
    fn test_install_zip_rejects_a_zip_missing_the_release_pin() {
        let mut store = ModelStore::default();
        let err = install_zip(b"not the release zip", &mut store).unwrap_err();
        assert!(
            err.to_string().contains("sha256 of the release zip"),
            "{err}"
        );
        assert_eq!(store.missing(), REQUIRED_FILES.to_vec());
    }
}
