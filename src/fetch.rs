//! First-run model fetch for the segmentation pipeline: stream the pinned
//! release zip, verify its sha256, unpack exactly the release files against
//! the per-file pins, and hand back a [`gsam::ModelStore`].
//!
//! Split by target. The host half downloads the zip into the gsam on-disk
//! cache for the native oracle path (`seg::prompted`). The wasm half streams
//! it off a same-origin file server with the browser `fetch` API — only
//! `download_zip` and `fetch_demo_scene` touch the network there, and no
//! test reaches either; the
//! install logic is shared verbatim between the halves and host-tested, so
//! both targets verify the identical pin tables.

use anyhow::{Context, Result, bail, ensure};
use gsam::{ModelStore, REQUIRED_FILES};
#[cfg(not(target_arch = "wasm32"))]
use sha2::Digest;
#[cfg(target_arch = "wasm32")]
use std::io::Read as _;
#[cfg(not(target_arch = "wasm32"))]
use std::io::{Read, Seek, Write};
#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};

/// The release zip URL on the web — relative on purpose: in the pipeline
/// worker it resolves against the worker script's base URL (same directory,
/// same origin as the page), so it lands next to index.html either way.
/// `scripts/dev_server.py` serves it from the durable master copy
/// `data/splatfield-models.zip` (git-ignored): a trunk rebuild wipes
/// `dist/`, so the zip is never copied there.
#[cfg(target_arch = "wasm32")]
pub const ZIP_URL: &str = "models.zip";

/// The release asset URL — the tag is gsam's release identity, so a new
/// release moves the URL and the cache leaf together. The asset name is
/// pinned in the format string because the release doesn't name it after
/// the tag.
#[cfg(not(target_arch = "wasm32"))]
fn models_zip_url() -> String {
    format!(
        "https://github.com/suous/splatfield/releases/download/{0}/splatfield-models.zip",
        gsam::RELEASE_TAG
    )
}

/// Streamed-download byte ceiling — the stream aborts past it, so a
/// sentinel or corrupt URL pointing at some huge file can't fill the disk
/// (or, on the web, the tab's memory).
const MAX_DOWNLOAD_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// The stream buffer's reserve ceiling: content-length is untrusted, and
/// reserving a lying 2 GiB header aborts the wasm32 heap before the first
/// byte arrives. The pinned releases are known-sized (models.zip, demo
/// scene — both well under this), so they keep their up-front reserve;
/// anything larger grows honestly while the `MAX_DOWNLOAD_BYTES` check
/// still bounds the real bytes. Wasm-only: the host downloader streams
/// through ureq's own `.limit` instead of `stream_body`.
#[cfg(target_arch = "wasm32")]
const MAX_RESERVE_BYTES: u64 = 256 * 1024 * 1024;

/// SHA-256 of the release zip. The wasm install checks the whole archive
/// against it before unzipping — a corrupt or substituted download never
/// reaches the per-file pins with a confusing error.
pub const ZIP_SHA256: &str = "259ed2a989fc368b839cb32c1b43c758fa1207e4849123e603f83baedb35f740";

/// The release zip's byte length — the bytes are sha-pinned
/// ([`ZIP_SHA256`]), so the length is a release constant too. Display only
/// (the status pill's MB figures); a new release moves it together with the
/// pin. Never load-bearing for the install path.
#[cfg(target_arch = "wasm32")]
pub const ZIP_BYTES: u64 = 158_618_158;

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

/// Cache files the text-prompted oracle needs — grounding-dino as one
/// self-contained onnx, its tokenizer, and SAM2 encoder/decoder plus their
/// external `*.onnx_data` weights.
#[cfg(not(target_arch = "wasm32"))]
fn required_files() -> Result<Vec<PathBuf>> {
    let mut files = vec![gsam::grounding_file()?, gsam::grounding_tokenizer()?];
    for stem in ["vision_encoder", "prompt_encoder_mask_decoder"] {
        let onnx = gsam::sam_file(stem)?;
        files.push(onnx.clone());
        files.push(onnx.with_extension("onnx_data"));
    }
    Ok(files)
}

/// Make sure the oracle models exist in the gsam cache, downloading the
/// release zip when they don't. `on_progress` is never called when the cache
/// is already complete.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn ensure_models(on_progress: &mut dyn FnMut(&str)) -> Result<()> {
    let files = required_files()?;
    ensure_release(
        &gsam::models_dir()?,
        &gsam::release_dir()?,
        &files,
        &pair_manifest(&files)?,
        on_progress,
    )
}

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
    let mut store = ModelStore::new();
    let zip = download_zip(on_progress).await?;
    install_zip(&zip, &mut store).context("installing release zip")?;
    Ok((store, false))
}

/// Pair gsam's cache paths with the pins — same length, and each
/// absolute path ending in its relative twin, so layout drift fails
/// loud instead of silently skipping verification.
#[cfg(not(target_arch = "wasm32"))]
fn pair_manifest(files: &[PathBuf]) -> Result<Vec<(&'static str, &'static str)>> {
    ensure!(
        files.len() == FILE_SHA256.len(),
        "{} required files but {} pins — cache layout drift",
        files.len(),
        FILE_SHA256.len()
    );
    files
        .iter()
        .zip(FILE_SHA256)
        .map(|(f, (rel, sum))| {
            ensure!(
                f.ends_with(rel),
                "{} is not …/{rel} — cache layout drift",
                f.display()
            );
            Ok((rel, sum))
        })
        .collect()
}

/// Stream `path` through sha256 and require an exact pin match — the same
/// check guards extraction and legacy adoption, so no corrupt file
/// survives either install path.
#[cfg(not(target_arch = "wasm32"))]
fn verify_sha256(path: &Path, want: &str, leaf: &Path) -> Result<()> {
    let mut hasher = sha2::Sha256::new();
    std::io::copy(&mut std::fs::File::open(path)?, &mut hasher)?;
    let got = format!("{:x}", hasher.finalize());
    ensure!(
        got == *want,
        "sha256 of {} is {got}, expected {want} — delete {} and rerun to re-download",
        path.display(),
        leaf.display()
    );
    Ok(())
}

/// Ensure `dir` (the release leaf) holds the pinned models. A legacy flat
/// install — family dirs directly under the models `root`, the shape
/// caches had before release versioning — is adopted by whole-dir renames
/// instead of a 2 GB re-download, each moved file hash-verified so a
/// corrupt legacy install fails loud. Then the complete fast-path, else
/// download the release zip and extract exactly the manifest into `dir`.
#[cfg(not(target_arch = "wasm32"))]
fn ensure_release(
    root: &Path,
    dir: &Path,
    files: &[PathBuf],
    manifest: &[(&str, &str)],
    on_progress: &mut dyn FnMut(&str),
) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    if !files.iter().all(|f| f.is_file()) {
        // Adopt a legacy flat install by whole-dir renames instead of a
        // 2 GB re-download. Each family is verified before the next one
        // moves: the rename is one-way, so the error's delete-and-rerun
        // advice is the recovery, not an undo.
        for family in ["grounding_dino_tiny", "sam2_tiny"] {
            let (src, dst) = (root.join(family), dir.join(family));
            if !src.is_dir() || dst.exists() {
                continue;
            }
            std::fs::rename(&src, &dst)
                .with_context(|| format!("adopting legacy {family} into {}", dir.display()))?;
            // Files pair with pins positionally — the same manifest that
            // drives extraction — so every moved file has a pin here.
            for (f, (_, want)) in files.iter().zip(manifest) {
                if f.starts_with(&dst) {
                    verify_sha256(f, want, dir)?;
                }
            }
        }
    }
    if files.iter().all(|f| f.is_file()) {
        return Ok(());
    }

    // Stream to `.part` and rename only on success, so an interrupted
    // download never masquerades as a complete zip.
    let part = dir.join("models.zip.part");
    let zip = dir.join("models.zip");
    download(&part, on_progress).with_context(|| format!("fetching {}", models_zip_url()))?;

    let _ = std::fs::remove_file(&zip); // fs::rename won't replace on Windows
    std::fs::rename(&part, &zip)?;
    let mut archive = zip::ZipArchive::new(std::fs::File::open(&zip)?)
        .context("downloaded models.zip is not a zip")?;
    extract_into(&mut archive, dir, manifest, on_progress)
        .with_context(|| format!("unpacking {}", zip.display()))?;

    let missing: Vec<String> = files
        .iter()
        .filter(|f| !f.is_file())
        .map(|f| f.display().to_string())
        .collect();
    ensure!(
        missing.is_empty(),
        "models zip lacks {} — it must package the cache layout \
         (grounding_dino_tiny/, sam2_tiny/)",
        missing.join(", ")
    );
    let _ = std::fs::remove_file(&zip);
    on_progress("models ready");
    Ok(())
}

/// Stream the zip to `part`.
#[cfg(not(target_arch = "wasm32"))]
fn download(part: &Path, on_progress: &mut dyn FnMut(&str)) -> Result<()> {
    let res = ureq::get(&models_zip_url()).call()?;
    let total = res.body().content_length();
    let mut reader = res
        .into_body()
        .into_with_config()
        .limit(MAX_DOWNLOAD_BYTES)
        .reader();

    let mut out = std::io::BufWriter::new(std::fs::File::create(part)?);
    let mut buf = [0u8; 64 * 1024];
    let mut received = 0u64;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        received += n as u64;
        on_progress(&match total {
            Some(total) => format!(
                "fetching models {:.0}% ({:.1}/{:.1} MB)",
                100.0 * received as f64 / total.max(1) as f64,
                received as f64 / 1e6,
                total as f64 / 1e6
            ),
            None => format!("fetching models… {:.1} MB", received as f64 / 1e6),
        });
    }
    out.flush()?;
    Ok(())
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

/// Unpack exactly the manifest entries into `dest`: a zip path qualifies
/// only after the single top-level wrapper folder is stripped and it equals
/// a manifest path, so `..` escapes, `__MACOSX` cruft, strays, and
/// unlisted files inside allowed roots are all skipped (a `..` escape
/// never matches a pin, and the written path is the pin's own relative
/// path, never an archive string). Each entry lands on a `.part` sibling
/// and is renamed into place only after its hash passes; a mismatch wipes
/// every file this call wrote, and a wipe that itself fails is loud — a
/// leftover file would pass the launch-time presence check and feed
/// corrupt weights to the oracle forever.
#[cfg(not(target_arch = "wasm32"))]
fn extract_into(
    archive: &mut zip::ZipArchive<impl Read + Seek>,
    dest: &Path,
    manifest: &[(&str, &str)],
    on_progress: &mut dyn FnMut(&str),
) -> Result<()> {
    let mut installed: Vec<PathBuf> = Vec::new();
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let Some(slot) = pin_slot(entry.name(), manifest) else {
            continue;
        };
        let (rel, want) = manifest[slot];
        let rel = Path::new(rel);
        if entry.is_dir() {
            continue;
        }
        let out = dest.join(rel);
        std::fs::create_dir_all(out.parent().expect("zip file entries have parents"))?;
        // The final path appears only after its hash passes: a kill mid-copy
        // leaves at most a `.part` sibling, which the launch-time presence check
        // ignores. The part path aliases across entries sharing a stem
        // (`x.onnx` and `x.onnx_data` both → `x.part`) — benign because every
        // iteration consumes the part (renames or removes it) before the next
        // begins.
        let part = out.with_extension("part");
        std::io::copy(&mut entry, &mut std::fs::File::create(&part)?)
            .with_context(|| format!("writing {}", rel.display()))?;
        match verify_sha256(&part, want, dest) {
            Ok(()) => {
                std::fs::rename(&part, &out)?;
                installed.push(out);
            }
            Err(e) => {
                let _ = std::fs::remove_file(&part);
                for p in &installed {
                    std::fs::remove_file(p).with_context(|| {
                        format!(
                            "rolling back {} after a hash mismatch — a leftover file \
                             passes the presence check forever",
                            p.display()
                        )
                    })?;
                }
                // The mismatch was found on the `.part`, but the entry is
                // known by its final name.
                return Err(e.context(format!("verifying {}", rel.display())));
            }
        }
        on_progress(&format!("unpacking models… {}", rel.display()));
    }
    Ok(())
}

/// GET `url` from whichever scope declares fetch — a Window on the page, a
/// WorkerGlobalScope inside the pipeline worker — and require an OK
/// `Response`. `cache`, when given, rides the request's `RequestInit`;
/// `None` leaves every field at its default (wire-identical to no init).
#[cfg(target_arch = "wasm32")]
async fn fetch_ok(url: &str, cache: Option<web_sys::RequestCache>) -> Result<web_sys::Response> {
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
    let response: web_sys::Response = response
        .dyn_into()
        .map_err(|_| anyhow::anyhow!("fetch of {url} did not yield a Response"))?;
    ensure!(response.ok(), "fetch {url}: HTTP {}", response.status());
    Ok(response)
}

/// Read `response`'s body into one buffer, streaming: the progress callback
/// fires per chunk with the fraction of content-length (never called when
/// the header is absent). `max_bytes` holds per chunk, so a mislabeled or
/// endless stream can't grow past it just because the header lied. The
/// buffer is reserved up front — a >100 MB Vec growing by amortized
/// doubling memcpy's the whole body several times inside the worker.
#[cfg(target_arch = "wasm32")]
async fn stream_body(
    response: web_sys::Response,
    max_bytes: u64,
    on_progress: &mut dyn FnMut(f64),
) -> Result<Vec<u8>> {
    use crate::opfs::js_err;
    use wasm_bindgen::JsCast;

    let total = response
        .headers()
        .get("content-length")
        .map_err(|e| anyhow::anyhow!("reading content-length: {e:?}"))?
        .and_then(|v| v.parse::<u64>().ok());
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
        // chunk.
        let start = buf.len();
        buf.resize(start + value.length() as usize, 0);
        value.copy_to(&mut buf[start..]);
        ensure!(
            buf.len() as u64 <= max_bytes,
            "stream exceeds the {max_bytes}-byte ceiling"
        );
        if let Some(total) = total {
            on_progress((buf.len() as f64 / total as f64).min(1.0));
        }
    }
}

/// Stream the release zip off the network with fractional progress (0..1).
/// Native builds never download: this is the browser branch.
#[cfg(target_arch = "wasm32")]
pub async fn download_zip(on_progress: &mut dyn FnMut(f64)) -> Result<Vec<u8>> {
    // Revalidate before use: a same-origin cache entry from an earlier dev
    // session (possibly behind a different server state) otherwise shadows
    // the pinned release and the sha256 pin rejects bytes the user cannot
    // fix except by clearing browser state. With `no-cache` a stale entry
    // costs one conditional request; a fresh file answers 200 unchanged.
    let response = fetch_ok(ZIP_URL, Some(web_sys::RequestCache::NoCache)).await?;
    stream_body(response, MAX_DOWNLOAD_BYTES, on_progress).await
}

/// The demo scene the help panel's button loads — the SH1 bear on the
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
    let content_type = response
        .headers()
        .get("content-type")
        .map_err(|e| anyhow::anyhow!("reading content-type: {e:?}"))?
        .unwrap_or_default();
    ensure!(
        !content_type.starts_with("text/"),
        "the demo scene endpoint answered {content_type:?} — the asset is not \
         deployed next to the app; local development must serve it via \
         scripts/dev_server.py"
    );
    stream_body(response, MAX_DOWNLOAD_BYTES, &mut on_progress).await
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
        let Some(slot) = pin_slot(entry.name(), pins) else {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::io::Write;

    /// The oracle file set: dino onnx + tokenizer, and both SAM2 graphs
    /// with their external-data siblings.
    #[test]
    fn test_required_files_shape() {
        let files = required_files().unwrap();
        let names: Vec<_> = files
            .iter()
            .map(|f| f.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            names,
            [
                "model_q4f16.onnx",
                "tokenizer.json",
                "vision_encoder_q4f16.onnx",
                "vision_encoder_q4f16.onnx_data",
                "prompt_encoder_mask_decoder_q4f16.onnx",
                "prompt_encoder_mask_decoder_q4f16.onnx_data",
            ]
        );
    }

    /// The pins pair with gsam's file list positionally; a length or
    /// suffix mismatch is layout drift and must fail loud rather than
    /// silently skip verification.
    #[test]
    fn test_pair_manifest_rejects_drift() {
        let files: Vec<PathBuf> = FILE_SHA256
            .iter()
            .map(|(rel, _)| PathBuf::from("cache").join(rel))
            .collect();
        assert!(pair_manifest(&files).is_ok());

        let err = pair_manifest(&files[..files.len() - 1])
            .unwrap_err()
            .to_string();
        assert!(err.contains("layout drift"), "{err}");

        let mut wrong = files.clone();
        wrong[0] = PathBuf::from("cache").join("elsewhere/model_q4f16.onnx");
        let err = pair_manifest(&wrong).unwrap_err().to_string();
        assert!(err.contains("layout drift"), "{err}");
    }

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

    fn build_zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut w = zip::ZipWriter::new(Cursor::new(Vec::new()));
        for (name, data) in entries {
            if name.ends_with('/') {
                w.add_directory(name.to_string(), zip::write::SimpleFileOptions::default())
                    .unwrap();
            } else {
                w.start_file(name.to_string(), zip::write::SimpleFileOptions::default())
                    .unwrap();
                w.write_all(data).unwrap();
            }
        }
        w.finish().unwrap().into_inner()
    }

    fn scratch(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("splatfield-fetch-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
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

    /// A legacy flat install at `root` — the pre-release-versioning cache
    /// shape — with `weights` as the third file's on-disk bytes; the
    /// manifest pins stay honest either way. Shared setup for the two
    /// migration tests.
    fn legacy_install(
        tag: &str,
        weights: &[u8],
    ) -> (PathBuf, PathBuf, Vec<PathBuf>, Vec<(&'static str, String)>) {
        let manifest: Vec<(&'static str, String)> = [
            ("grounding_dino_tiny/onnx/model.onnx", zip_sha256(b"onnx")),
            ("grounding_dino_tiny/tokenizer.json", zip_sha256(b"{}")),
            (
                "sam2_tiny/onnx/vision_encoder.onnx_data",
                zip_sha256(b"weights"),
            ),
        ]
        .into();
        let root = scratch(tag);
        let leaf = root.join(gsam::RELEASE_TAG);
        std::fs::create_dir_all(root.join("grounding_dino_tiny/onnx")).unwrap();
        std::fs::create_dir_all(root.join("sam2_tiny/onnx")).unwrap();
        std::fs::write(root.join("grounding_dino_tiny/onnx/model.onnx"), b"onnx").unwrap();
        std::fs::write(root.join("grounding_dino_tiny/tokenizer.json"), b"{}").unwrap();
        std::fs::write(
            root.join("sam2_tiny/onnx/vision_encoder.onnx_data"),
            weights,
        )
        .unwrap();
        let files: Vec<_> = manifest.iter().map(|(rel, _)| leaf.join(rel)).collect();
        (root, leaf, files, manifest)
    }

    /// Extraction installs exactly the manifest entries — through a
    /// `models/` wrapper — and drops everything else: `..` escapes,
    /// `__MACOSX` cruft, root strays, directory stubs.
    #[test]
    fn test_extract_keeps_layout_and_skips_the_rest() {
        let onnx_sum = zip_sha256(b"onnx");
        let weights_sum = zip_sha256(b"weights");
        let manifest = [
            ("grounding_dino_tiny/onnx/model.onnx", onnx_sum.as_str()),
            (
                "sam2_tiny/onnx/vision_encoder.onnx_data",
                weights_sum.as_str(),
            ),
        ];
        let zip = build_zip(&[
            ("models/", &[]),
            ("models/grounding_dino_tiny/onnx/model.onnx", b"onnx"),
            ("models/sam2_tiny/onnx/vision_encoder.onnx_data", b"weights"),
            ("../escape.txt", b"no"),
            ("__MACOSX/junk", b"no"),
            ("stray.txt", b"no"),
        ]);
        let dir = scratch("extract");
        let mut archive = zip::ZipArchive::new(Cursor::new(zip)).unwrap();
        let mut extracted = 0usize;
        extract_into(&mut archive, &dir, &manifest, &mut |_| extracted += 1).unwrap();

        assert_eq!(
            std::fs::read(dir.join("grounding_dino_tiny/onnx/model.onnx")).unwrap(),
            b"onnx"
        );
        assert_eq!(
            std::fs::read(dir.join("sam2_tiny/onnx/vision_encoder.onnx_data")).unwrap(),
            b"weights"
        );
        assert!(!dir.join("stray.txt").exists());
        assert!(!dir.parent().unwrap().join("escape.txt").exists());
        assert_eq!(extracted, 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A file inside an allowed root but absent from the manifest is junk —
    /// exact-match filtering skips it rather than installing it.
    #[test]
    fn test_extract_skips_junk_inside_allowed_roots() {
        let tok_sum = zip_sha256(b"{}");
        let manifest = [("grounding_dino_tiny/tokenizer.json", tok_sum.as_str())];
        let zip = build_zip(&[
            ("grounding_dino_tiny/tokenizer.json", b"{}"),
            ("grounding_dino_tiny/junk.bin", b"junk"),
        ]);
        let dir = scratch("junk-root");
        let mut archive = zip::ZipArchive::new(Cursor::new(zip)).unwrap();
        let mut extracted = 0usize;
        extract_into(&mut archive, &dir, &manifest, &mut |_| extracted += 1).unwrap();

        assert_eq!(
            std::fs::read(dir.join("grounding_dino_tiny/tokenizer.json")).unwrap(),
            b"{}"
        );
        assert!(!dir.join("grounding_dino_tiny/junk.bin").exists());
        assert_eq!(extracted, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A payload that doesn't match its pinned hash aborts the install and
    /// wipes every file this extraction wrote — a half-installed cache
    /// would pass the launch-time presence check and feed corrupt weights
    /// to the oracle.
    #[test]
    fn test_extract_tampered_payload_wipes_install() {
        let good_sum = zip_sha256(b"good");
        let weights_sum = zip_sha256(b"good weights");
        let manifest = [
            ("grounding_dino_tiny/tokenizer.json", good_sum.as_str()),
            (
                "sam2_tiny/onnx/vision_encoder.onnx_data",
                weights_sum.as_str(),
            ),
        ];
        let zip = build_zip(&[
            ("grounding_dino_tiny/tokenizer.json", b"good"),
            ("sam2_tiny/onnx/vision_encoder.onnx_data", b"tampered"),
        ]);
        let dir = scratch("tampered");
        let mut archive = zip::ZipArchive::new(Cursor::new(zip)).unwrap();
        let err = extract_into(&mut archive, &dir, &manifest, &mut |_| {}).unwrap_err();
        assert!(err.to_string().contains("vision_encoder.onnx_data"));
        assert!(!dir.join("grounding_dino_tiny/tokenizer.json").exists());
        assert!(!dir.join("sam2_tiny/onnx/vision_encoder.onnx_data").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An entry whose checksum fails while it is read — a kill mid-copy —
    /// leaves no final file behind: at most a `.part` sibling the
    /// launch-time presence check ignores. Earlier verified entries stay —
    /// copy errors do not roll back.
    #[test]
    fn test_extract_corrupt_entry_leaves_no_final_file() {
        let good_sum = zip_sha256(b"good");
        let weights_sum = zip_sha256(b"good weights");
        let manifest = [
            ("grounding_dino_tiny/tokenizer.json", good_sum.as_str()),
            (
                "sam2_tiny/onnx/vision_encoder.onnx_data",
                weights_sum.as_str(),
            ),
        ];
        let mut zip = build_zip(&[
            ("grounding_dino_tiny/tokenizer.json", b"good"),
            ("sam2_tiny/onnx/vision_encoder.onnx_data", b"good weights"),
        ]);
        // Byte surgery: the central directory ends the zip, one
        // `PK\x01\x02` record per entry in file order. Flip a byte of the
        // second record's CRC-32 (sig 4 + version 2 + version 2 + flags 2 +
        // method 2 + time 2 + date 2 → CRC at +16): the entry's bytes are
        // untouched, but zip-rs now reads it against the wrong checksum and
        // the read fails mid-entry.
        let needle: &[u8] = b"PK\x01\x02";
        let record = zip
            .windows(4)
            .enumerate()
            .filter(|(_, w)| *w == needle)
            .map(|(i, _)| i)
            .nth(1)
            .expect("second central directory record");
        zip[record + 16] ^= 0xFF;

        let dir = scratch("corrupt-entry");
        let mut archive = zip::ZipArchive::new(Cursor::new(zip)).unwrap();
        let err = extract_into(&mut archive, &dir, &manifest, &mut |_| {}).unwrap_err();
        assert!(
            err.to_string().contains("vision_encoder.onnx_data"),
            "{err}"
        );
        assert!(!dir.join("sam2_tiny/onnx/vision_encoder.onnx_data").exists());
        assert!(dir.join("grounding_dino_tiny/tokenizer.json").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Pins today's release URL byte-for-byte: the asset name isn't
    /// `<tag>.zip`, so a tag bump must consciously carry the asset name
    /// along instead of silently 404ing.
    #[test]
    fn test_models_zip_url_pins_current_release() {
        assert_eq!(
            models_zip_url(),
            "https://github.com/suous/splatfield/releases/download/models-v1/splatfield-models.zip"
        );
    }

    /// A legacy flat cache (family dirs directly under the models root —
    /// the pre-versioning shape) is adopted into the release leaf by
    /// whole-dir renames instead of a 2 GB re-download.
    #[test]
    fn test_migration_moves_flat_cache_into_release_leaf() {
        let (root, leaf, files, manifest) = legacy_install("migrate", b"weights");

        ensure_release(&root, &leaf, &files, &pin_refs(&manifest), &mut |_| {}).unwrap();

        assert_eq!(
            std::fs::read(leaf.join("grounding_dino_tiny/onnx/model.onnx")).unwrap(),
            b"onnx"
        );
        assert_eq!(
            std::fs::read(leaf.join("grounding_dino_tiny/tokenizer.json")).unwrap(),
            b"{}"
        );
        assert_eq!(
            std::fs::read(leaf.join("sam2_tiny/onnx/vision_encoder.onnx_data")).unwrap(),
            b"weights"
        );
        assert!(!root.join("grounding_dino_tiny").exists());
        assert!(!root.join("sam2_tiny").exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A corrupt legacy install fails loud on the moved files — it must
    /// never pass as adopted and feed the oracle bad weights; the message
    /// says how to recover.
    #[test]
    fn test_migration_corrupt_legacy_fails_loud() {
        let (root, leaf, files, manifest) = legacy_install("migrate-corrupt", b"tampered");

        let err =
            ensure_release(&root, &leaf, &files, &pin_refs(&manifest), &mut |_| {}).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("vision_encoder.onnx_data"), "{msg}");
        assert!(msg.contains("delete"), "{msg}");
        assert!(
            leaf.join("sam2_tiny/onnx/vision_encoder.onnx_data")
                .exists()
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A complete leaf returns before any download or migration — a
    /// root-level legacy tree stays put and the leaf bytes are untouched.
    #[test]
    fn test_complete_leaf_skips_migration_and_download() {
        let (onnx_sum, tok_sum) = (zip_sha256(b"onnx"), zip_sha256(b"{}"));
        let manifest = [
            ("grounding_dino_tiny/onnx/model.onnx", onnx_sum.as_str()),
            ("grounding_dino_tiny/tokenizer.json", tok_sum.as_str()),
        ];
        let root = scratch("complete");
        let leaf = root.join(gsam::RELEASE_TAG);
        std::fs::create_dir_all(leaf.join("grounding_dino_tiny/onnx")).unwrap();
        std::fs::write(leaf.join("grounding_dino_tiny/onnx/model.onnx"), b"onnx").unwrap();
        std::fs::write(leaf.join("grounding_dino_tiny/tokenizer.json"), b"{}").unwrap();
        std::fs::create_dir_all(root.join("sam2_tiny/onnx")).unwrap();
        std::fs::write(
            root.join("sam2_tiny/onnx/vision_encoder.onnx_data"),
            b"root copy",
        )
        .unwrap();
        let files: Vec<_> = manifest.iter().map(|(rel, _)| leaf.join(rel)).collect();

        ensure_release(&root, &leaf, &files, &manifest, &mut |_| {
            panic!("on_progress called — a complete leaf must return before any download")
        })
        .unwrap();

        assert_eq!(
            std::fs::read(leaf.join("grounding_dino_tiny/onnx/model.onnx")).unwrap(),
            b"onnx"
        );
        assert!(root.join("sam2_tiny").exists());
        assert!(!leaf.join("sam2_tiny").exists());
        let _ = std::fs::remove_dir_all(&root);
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

        let mut store = ModelStore::new();
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

        let mut store = ModelStore::new();
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

        let mut store = ModelStore::new();
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

        let mut store = ModelStore::new();
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

        let mut store = ModelStore::new();
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
        let mut store = ModelStore::new();
        let err = install_zip(b"not the release zip", &mut store).unwrap_err();
        assert!(
            err.to_string().contains("sha256 of the release zip"),
            "{err}"
        );
        assert_eq!(store.missing(), REQUIRED_FILES.to_vec());
    }
}
