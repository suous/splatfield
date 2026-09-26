//! OPFS cache of the verified release: after one pinned download +
//! verification, the six model files and a manifest persist in the
//! origin-private filesystem, and later page loads build the store straight
//! from there — no network, no re-download.
//!
//! Split like fetch.rs: the manifest and its checks are pure and host-tested
//! below; the I/O half is wasm-only and best-effort on both ends — a persist
//! failure just skips caching, and a load that fails any check wipes the
//! cache and reads as a miss (the caller redownloads), so a corrupt cache
//! can never block a load or reach the sessions.

use anyhow::{Context, Result};
use gsam::{ModelStore, REQUIRED_FILES};
use serde::{Deserialize, Serialize};

/// OPFS directory holding the cache, directly under the storage root.
#[cfg(target_arch = "wasm32")]
const CACHE_DIR: &str = "models-cache";
/// The cache's manifest file — written last, so its presence is the commit
/// marker (see `persist`).
#[cfg(target_arch = "wasm32")]
const MANIFEST_FILE: &str = "manifest.bin";

/// What a valid cache holds: the zip pin these bytes came from (a new
/// release changes it and invalidates every prior cache at a glance) and
/// each file's byte length under its release-relative path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheManifest {
    zip_sha256: String,
    files: Vec<(String, u64)>,
}

impl CacheManifest {
    /// Bincode, not JSON: the pipeline wire codec is already bincode and no
    /// serde_json dependency is wanted for one file.
    pub fn encode(&self) -> Result<Vec<u8>> {
        bincode::serialize(self).context("encoding the cache manifest")
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        bincode::deserialize(bytes).context("decoding the cache manifest")
    }
}

/// Pin a manifest to the release currently being shipped: the store must be
/// complete (a partial cache must never be committed as if whole) and every
/// size comes off the bytes actually held.
pub fn build_manifest(store: &ModelStore) -> Result<CacheManifest> {
    let mut files = Vec::with_capacity(REQUIRED_FILES.len());
    for file in REQUIRED_FILES {
        let bytes = store.get(file).with_context(|| {
            format!("the store lacks {file} — refusing to cache an incomplete release")
        })?;
        files.push((file.to_string(), bytes.len() as u64));
    }
    Ok(CacheManifest {
        zip_sha256: crate::fetch::ZIP_SHA256.to_string(),
        files,
    })
}

/// A manifest qualifies only if it describes exactly the pinned release:
/// same zip pin, and a file table that is [`REQUIRED_FILES`] with nothing
/// missing, unknown, duplicated, or empty.
pub fn validate(manifest: &CacheManifest) -> Result<()> {
    anyhow::ensure!(
        manifest.zip_sha256 == crate::fetch::ZIP_SHA256,
        "cache manifest pins zip sha256 {}, expected {} — a cache from another release",
        manifest.zip_sha256,
        crate::fetch::ZIP_SHA256,
    );
    anyhow::ensure!(
        manifest.files.len() == REQUIRED_FILES.len(),
        "cache manifest lists {} files, expected {}",
        manifest.files.len(),
        REQUIRED_FILES.len(),
    );
    for (name, size) in &manifest.files {
        anyhow::ensure!(
            REQUIRED_FILES.contains(&name.as_str()),
            "cache manifest lists {name}, which is not a release file",
        );
        anyhow::ensure!(*size > 0, "cache manifest lists {name} at 0 bytes");
    }
    for file in REQUIRED_FILES {
        let count = manifest.files.iter().filter(|(n, _)| n == file).count();
        anyhow::ensure!(
            count == 1,
            "cache manifest lists {file} {count} times, expected exactly once",
        );
    }
    Ok(())
}

/// `actual` holds the byte lengths of the cache files read in
/// [`REQUIRED_FILES`] order; each must match the manifest positionally — a
/// drifted length means the cache does not hold what it says.
pub fn check_sizes(manifest: &CacheManifest, actual: &[u64]) -> Result<()> {
    anyhow::ensure!(
        actual.len() == manifest.files.len(),
        "read {} cached file lengths for a manifest of {} files",
        actual.len(),
        manifest.files.len(),
    );
    for ((name, want), got) in manifest.files.iter().zip(actual) {
        anyhow::ensure!(
            want == got,
            "cached {name} is {got} bytes, the manifest pins {want} — corrupt cache entry",
        );
    }
    Ok(())
}

/// Map a release-relative path to a flat OPFS-safe file name: the path's
/// directory separators become `__`, which no [`REQUIRED_FILES`] path
/// contains — the host test pins the mapping injective over the current
/// release, so distinct files cannot collide inside the cache directory.
pub fn cache_file_name(required_file: &str) -> String {
    required_file.replace('/', "__")
}

// --- wasm-only I/O ----------------------------------------------------------

/// Map a rejected JS promise/value onto the anyhow error; the `{:?}` keeps
/// the whole JsValue payload for diagnosis. Crate-shared: fetch.rs's
/// download path maps through this same spelling.
#[cfg(target_arch = "wasm32")]
pub(crate) fn js_err(value: wasm_bindgen::JsValue) -> anyhow::Error {
    anyhow::anyhow!("JS error: {value:?}")
}

/// Open the cache directory under the storage root — created for persist,
/// failed-on-absent for load (an absent cache is a plain miss, not an error).
#[cfg(target_arch = "wasm32")]
async fn open_cache_dir(create: bool) -> Result<web_sys::FileSystemDirectoryHandle> {
    use wasm_bindgen::JsCast;

    let root: web_sys::FileSystemDirectoryHandle =
        wasm_bindgen_futures::JsFuture::from(storage_manager().map_err(js_err)?.get_directory())
            .await
            .map_err(js_err)?
            .dyn_into()
            .map_err(|_| anyhow::anyhow!("get_directory did not yield a directory handle"))?;
    let options = web_sys::FileSystemGetDirectoryOptions::new();
    options.set_create(create);
    let dir: web_sys::FileSystemDirectoryHandle = wasm_bindgen_futures::JsFuture::from(
        root.get_directory_handle_with_options(CACHE_DIR, &options),
    )
    .await
    .map_err(js_err)?
    .dyn_into()
    .map_err(|_| anyhow::anyhow!("{CACHE_DIR} is not a directory handle"))?;
    Ok(dir)
}

/// Fast path: build the store straight from OPFS, or `None` on any problem —
/// missing manifest, wrong pin, size drift, JS error — after wiping the
/// cache directory. Never panics, never returns a partial store: the caller
/// redownloads and repins a cache that fails any check.
///
/// The fast path does NOT re-hash file bytes — a full 158 MB sha256 would
/// eat the very seconds the cache exists to save. Integrity rests on three
/// legs instead: the bytes entered OPFS fully pinned (zip sha256 +
/// per-file `FILE_SHA256` at install time), the manifest lands last as the
/// commit marker, and every length is re-checked on load. Same-length
/// silent corruption inside OPFS is the accepted residual risk.
#[cfg(target_arch = "wasm32")]
pub async fn load_cached() -> Option<ModelStore> {
    match load_cached_inner().await {
        Ok(store) => Some(store),
        Err(err) => {
            web_sys::console::warn_1(
                &format!("opfs cache unusable, redownloading: {err:#}").into(),
            );
            remove_cache_dir().await;
            None
        }
    }
}

#[cfg(target_arch = "wasm32")]
async fn load_cached_inner() -> Result<ModelStore> {
    // No create: an absent cache directory is a plain miss, not an error.
    let dir = open_cache_dir(false).await?;

    let manifest = CacheManifest::decode(&read_file(&dir, MANIFEST_FILE).await?)?;
    validate(&manifest).context("cached release rejected")?;

    let mut files = Vec::with_capacity(REQUIRED_FILES.len());
    for file in REQUIRED_FILES {
        files.push((file, read_file(&dir, &cache_file_name(file)).await?));
    }
    let lengths = files
        .iter()
        .map(|(_, b)| b.len() as u64)
        .collect::<Vec<_>>();
    check_sizes(&manifest, &lengths).context("cached release rejected")?;

    let mut store = ModelStore::new();
    for (file, bytes) in files {
        store
            .insert(file, bytes)
            .context("installing a cached release file")?;
    }
    Ok(store)
}

/// Persist a complete, verified store. THE MANIFEST IS WRITTEN LAST — it is
/// the commit marker: a crash (or killed tab) mid-write leaves model files
/// without a manifest, and a later load — which validates the manifest and
/// every size before trusting a byte — treats that half-written cache as
/// corrupt, wipes it, and redownloads. A load must never see a manifest
/// without all files complete.
#[cfg(target_arch = "wasm32")]
pub async fn persist(store: &ModelStore) -> Result<()> {
    let dir = open_cache_dir(true).await?;

    // Files first, manifest last — the commit marker (hazard above).
    for file in REQUIRED_FILES {
        let bytes = store
            .get(file)
            .with_context(|| format!("refusing to cache an incomplete release: {file} absent"))?;
        write_file(&dir, &cache_file_name(file), bytes).await?;
    }
    let manifest = build_manifest(store).and_then(|m| m.encode())?;
    write_file(&dir, MANIFEST_FILE, &manifest).await?;
    Ok(())
}

/// `navigator.storage()` off the global scope — a Window on the page, a
/// WorkerGlobalScope inside the worker (same resolution as fetch in
/// fetch.rs).
#[cfg(target_arch = "wasm32")]
fn storage_manager() -> Result<web_sys::StorageManager, wasm_bindgen::JsValue> {
    use wasm_bindgen::JsCast;

    Ok(match js_sys::global().dyn_into::<web_sys::Window>() {
        Ok(window) => window.navigator().storage(),
        Err(global) => global
            .unchecked_into::<web_sys::WorkerGlobalScope>()
            .navigator()
            .storage(),
    })
}

/// Read one file from `dir` fully into memory. A missing or unreadable file
/// is an error — the caller turns it into a cache miss.
#[cfg(target_arch = "wasm32")]
async fn read_file(dir: &web_sys::FileSystemDirectoryHandle, name: &str) -> Result<Vec<u8>> {
    use wasm_bindgen::JsCast;

    let handle: web_sys::FileSystemFileHandle =
        wasm_bindgen_futures::JsFuture::from(dir.get_file_handle(name))
            .await
            .map_err(js_err)
            .with_context(|| format!("opening {name} in the cache"))?
            .dyn_into()
            .map_err(|_| anyhow::anyhow!("{name} is not a file handle"))?;
    let file: web_sys::File = wasm_bindgen_futures::JsFuture::from(handle.get_file())
        .await
        .map_err(js_err)?
        .dyn_into()
        .map_err(|_| anyhow::anyhow!("{name} did not yield a File"))?;
    let buffer = wasm_bindgen_futures::JsFuture::from(file.array_buffer())
        .await
        .map_err(js_err)
        .with_context(|| format!("reading {name}"))?;
    Ok(js_sys::Uint8Array::new(&buffer).to_vec())
}

/// Create-or-open `name` in `dir` and replace its whole contents.
#[cfg(target_arch = "wasm32")]
async fn write_file(
    dir: &web_sys::FileSystemDirectoryHandle,
    name: &str,
    bytes: &[u8],
) -> Result<()> {
    use wasm_bindgen::JsCast;

    let create = web_sys::FileSystemGetFileOptions::new();
    create.set_create(true);
    let handle: web_sys::FileSystemFileHandle =
        wasm_bindgen_futures::JsFuture::from(dir.get_file_handle_with_options(name, &create))
            .await
            .map_err(js_err)
            .with_context(|| format!("opening {name} in the cache"))?
            .dyn_into()
            .map_err(|_| anyhow::anyhow!("{name} is not a file handle"))?;
    let stream: web_sys::FileSystemWritableFileStream =
        wasm_bindgen_futures::JsFuture::from(handle.create_writable())
            .await
            .map_err(js_err)
            .with_context(|| format!("opening {name} for write"))?
            .dyn_into()
            .map_err(|_| anyhow::anyhow!("{name} did not yield a writable stream"))?;
    wasm_bindgen_futures::JsFuture::from(stream.write_with_u8_array(bytes).map_err(js_err)?)
        .await
        .map_err(js_err)
        .with_context(|| format!("writing {name}"))?;
    // web-sys generates no close() on the stream, but without it the write
    // never commits — call the JS method through Reflect instead.
    let close: js_sys::Function = js_sys::Reflect::get(stream.as_ref(), &"close".into())
        .map_err(js_err)?
        .dyn_into()
        .map_err(|_| anyhow::anyhow!("{name}: writable stream has no close()"))?;
    let done: js_sys::Promise = close
        .call0(stream.as_ref())
        .map_err(js_err)?
        .unchecked_into();
    wasm_bindgen_futures::JsFuture::from(done)
        .await
        .map_err(js_err)
        .with_context(|| format!("committing {name}"))?;
    Ok(())
}

/// Best-effort wipe of the cache directory — every error is swallowed: a
/// failed cleanup only costs a redownload of a cache that already failed
/// its checks.
#[cfg(target_arch = "wasm32")]
async fn remove_cache_dir() {
    use wasm_bindgen::JsCast;

    let Ok(storage) = storage_manager() else {
        return;
    };
    let Ok(ready) = wasm_bindgen_futures::JsFuture::from(storage.get_directory()).await else {
        return;
    };
    let Ok(root) = ready.dyn_into::<web_sys::FileSystemDirectoryHandle>() else {
        return;
    };
    // removeEntry names a CHILD of the directory it is called on: the call
    // goes to the storage root, deleting the cache dir itself. (Calling it
    // on the cache dir would look for models-cache/models-cache — a silent
    // no-op.)
    let recursive = web_sys::FileSystemRemoveOptions::new();
    recursive.set_recursive(true);
    let _ =
        wasm_bindgen_futures::JsFuture::from(root.remove_entry_with_options(CACHE_DIR, &recursive))
            .await;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A complete store with one distinguishable byte length per file
    /// position: file `i` holds `i + 1` bytes.
    fn populated_store() -> ModelStore {
        let mut store = ModelStore::new();
        for (i, file) in REQUIRED_FILES.iter().enumerate() {
            store.insert(file, vec![i as u8; i + 1]).unwrap();
        }
        store
    }

    /// A manifest over REQUIRED_FILES (in order) with the given sizes.
    fn manifest_with(sizes: &[u64; 6]) -> CacheManifest {
        CacheManifest {
            zip_sha256: crate::fetch::ZIP_SHA256.to_string(),
            files: REQUIRED_FILES
                .iter()
                .zip(sizes)
                .map(|(f, s)| (f.to_string(), *s))
                .collect(),
        }
    }

    #[test]
    fn manifest_round_trips_through_bincode() {
        let manifest = manifest_with(&[1, 2, 3, 4, 5, 6]);
        let bytes = manifest.encode().unwrap();
        assert_eq!(CacheManifest::decode(&bytes).unwrap(), manifest);
    }

    #[test]
    fn validate_accepts_a_correct_manifest() {
        validate(&manifest_with(&[1, 2, 3, 4, 5, 6])).unwrap();
    }

    #[test]
    fn validate_rejects_a_wrong_zip_pin() {
        let mut manifest = manifest_with(&[1, 2, 3, 4, 5, 6]);
        manifest.zip_sha256 = "deadbeef".into();
        assert!(validate(&manifest).is_err());
    }

    #[test]
    fn validate_rejects_a_missing_file() {
        let mut manifest = manifest_with(&[1, 2, 3, 4, 5, 6]);
        manifest.files.remove(2);
        assert!(validate(&manifest).is_err());
    }

    /// An unknown entry displacing a required one (same table length) is
    /// rejected — and so is a duplicated entry that leaves another file
    /// unlisted.
    #[test]
    fn validate_rejects_unknown_and_duplicated_files() {
        let mut unknown = manifest_with(&[1, 2, 3, 4, 5, 6]);
        unknown.files[4] = ("junk.bin".to_string(), 5);
        assert!(validate(&unknown).is_err());

        let mut dup = manifest_with(&[1, 2, 3, 4, 5, 6]);
        dup.files[1] = dup.files[0].clone();
        assert!(validate(&dup).is_err());
    }

    #[test]
    fn validate_rejects_a_zero_size() {
        let mut manifest = manifest_with(&[1, 2, 3, 4, 5, 6]);
        manifest.files[3].1 = 0;
        assert!(validate(&manifest).is_err());
    }

    #[test]
    fn check_sizes_passes_on_a_match() {
        check_sizes(&manifest_with(&[1, 2, 3, 4, 5, 6]), &[1, 2, 3, 4, 5, 6]).unwrap();
    }

    /// Every position is checked: flipping any one length fails, naming the
    /// file at that position.
    #[test]
    fn check_sizes_fails_on_any_mismatch_and_names_the_file() {
        let manifest = manifest_with(&[1, 2, 3, 4, 5, 6]);
        for pos in 0..6 {
            let mut actual = [1, 2, 3, 4, 5, 6];
            actual[pos] += 1;
            let err = check_sizes(&manifest, &actual).unwrap_err();
            assert!(
                err.to_string().contains(REQUIRED_FILES[pos]),
                "position {pos}: {err}"
            );
        }
        // The first position names REQUIRED_FILES[0].
        let err = check_sizes(&manifest, &[9, 2, 3, 4, 5, 6]).unwrap_err();
        assert!(err.to_string().contains(REQUIRED_FILES[0]), "{err}");
    }

    #[test]
    fn check_sizes_fails_on_a_length_mismatch() {
        let manifest = manifest_with(&[1, 2, 3, 4, 5, 6]);
        assert!(check_sizes(&manifest, &[1, 2, 3, 4, 5]).is_err());
        assert!(check_sizes(&manifest, &[1, 2, 3, 4, 5, 6, 7]).is_err());
    }

    #[test]
    fn build_manifest_matches_the_store_and_validates() {
        let store = populated_store();
        let manifest = build_manifest(&store).unwrap();
        validate(&manifest).unwrap();
        for (i, (name, size)) in manifest.files.iter().enumerate() {
            assert_eq!(name, REQUIRED_FILES[i]);
            assert_eq!(*size, store.get(name).unwrap().len() as u64);
        }
    }

    /// An incomplete store has no manifest: a partial cache must never be
    /// committed as if whole.
    #[test]
    fn build_manifest_rejects_an_incomplete_store() {
        let mut store = ModelStore::new();
        for file in &REQUIRED_FILES[..5] {
            store.insert(file, vec![0; 1]).unwrap();
        }
        assert!(!store.is_complete());
        assert!(build_manifest(&store).is_err());
    }

    /// Six distinct paths map to six distinct flat names, none carrying a
    /// separator into the flat cache directory.
    #[test]
    fn cache_file_names_are_injective_and_slash_free() {
        let names: Vec<String> = REQUIRED_FILES.iter().map(|f| cache_file_name(f)).collect();
        for (file, name) in REQUIRED_FILES.iter().zip(&names) {
            assert!(!name.contains('/'), "{file} -> {name}");
            assert!(!name.is_empty(), "{file} -> {name}");
        }
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), REQUIRED_FILES.len(), "collision: {names:?}");
    }
}
