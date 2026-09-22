//! First-run model fetch for the segmentation oracle: when the gsam cache is
//! missing the ONNX exports, download the models zip into
//! [`gsam::models_dir`]. The zip packages the cache layout itself —
//! `grounding_dino_tiny/…` and `sam2_tiny/…`, optionally under one
//! top-level folder (`models/`).
//!
//! Only `download` touches the network — no test reaches it; the module's
//! tests cover the surrounding layout, pairing, and extraction logic.

use anyhow::{Context, Result, ensure};
use sha2::Digest;
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};

/// The release asset URL — the tag is gsam's release identity, so a new
/// release moves the URL and the cache leaf together. The asset name is
/// pinned in the format string because the release doesn't name it after
/// the tag.
fn models_zip_url() -> String {
    format!(
        "https://github.com/suous/splatfield/releases/download/{0}/splatfield-models.zip",
        gsam::RELEASE_TAG
    )
}

/// Zip byte ceiling — the streamed download aborts past it, so a sentinel
/// or corrupt URL pointing at some huge file can't fill the disk.
const MAX_ZIP_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Cache files the text-prompted oracle needs — grounding-dino as one
/// self-contained onnx, its tokenizer, and SAM2 encoder/decoder plus their
/// external `*.onnx_data` weights.
fn required_files() -> Result<Vec<PathBuf>> {
    let mut files = vec![gsam::grounding_file()?, gsam::grounding_tokenizer()?];
    for stem in ["vision_encoder", "prompt_encoder_mask_decoder"] {
        let onnx = gsam::sam_file(stem)?;
        files.push(onnx.clone());
        files.push(onnx.with_extension("onnx_data"));
    }
    Ok(files)
}

/// SHA-256 of every cache file, keyed by its cache-relative path, in
/// `required_files()` order. Verified during install; not re-hashed on
/// launch.
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

/// Make sure the oracle models exist in the gsam cache, downloading the
/// release zip when they don't. `on_progress` is never called when the cache
/// is already complete.
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

/// Pair gsam's cache paths with the pins — same length, and each
/// absolute path ending in its relative twin, so layout drift fails
/// loud instead of silently skipping verification.
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
fn download(part: &Path, on_progress: &mut dyn FnMut(&str)) -> Result<()> {
    let res = ureq::get(&models_zip_url()).call()?;
    let total = res.body().content_length();
    let mut reader = res
        .into_body()
        .into_with_config()
        .limit(MAX_ZIP_BYTES)
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

/// Unpack exactly the manifest entries into `dest`: a zip path qualifies
/// only after the single top-level wrapper folder is stripped and it equals
/// a manifest path, so `..` escapes (rejected by `enclosed_name`),
/// `__MACOSX` cruft, strays, and unlisted files inside allowed roots are
/// all skipped. Each entry lands on a `.part` sibling and is renamed into
/// place only after its hash passes; a mismatch wipes every file this call
/// wrote, and a wipe that itself fails is loud — a leftover file would pass
/// the launch-time presence check and feed corrupt weights to the oracle
/// forever.
fn extract_into(
    archive: &mut zip::ZipArchive<impl Read + Seek>,
    dest: &Path,
    manifest: &[(&str, &str)],
    on_progress: &mut dyn FnMut(&str),
) -> Result<()> {
    let mut installed: Vec<PathBuf> = Vec::new();
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let Some(rel) = entry.enclosed_name() else {
            continue;
        };
        // `strip_prefix` removes whole components only, so `models_x/…`
        // stays; a bare `models` reduces to an empty path, matching no
        // manifest key.
        let rel = rel.strip_prefix("models").unwrap_or(&rel);
        let Some((_, want)) = manifest.iter().find(|(p, _)| Path::new(p) == rel) else {
            continue;
        };
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

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

    fn sha(data: &[u8]) -> String {
        use sha2::Digest;
        format!("{:x}", sha2::Sha256::digest(data))
    }

    /// Extraction installs exactly the manifest entries — through a
    /// `models/` wrapper — and drops everything else: `..` escapes,
    /// `__MACOSX` cruft, root strays, directory stubs.
    #[test]
    fn test_extract_keeps_layout_and_skips_the_rest() {
        let onnx_sum = sha(b"onnx");
        let weights_sum = sha(b"weights");
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
        let tok_sum = sha(b"{}");
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
        let good_sum = sha(b"good");
        let weights_sum = sha(b"good weights");
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
        let good_sum = sha(b"good");
        let weights_sum = sha(b"good weights");
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
        let (onnx_sum, tok_sum, weights_sum) = (sha(b"onnx"), sha(b"{}"), sha(b"weights"));
        let manifest = [
            ("grounding_dino_tiny/onnx/model.onnx", onnx_sum.as_str()),
            ("grounding_dino_tiny/tokenizer.json", tok_sum.as_str()),
            (
                "sam2_tiny/onnx/vision_encoder.onnx_data",
                weights_sum.as_str(),
            ),
        ];
        let root = scratch("migrate");
        let leaf = root.join(gsam::RELEASE_TAG);
        std::fs::create_dir_all(root.join("grounding_dino_tiny/onnx")).unwrap();
        std::fs::create_dir_all(root.join("sam2_tiny/onnx")).unwrap();
        std::fs::write(root.join("grounding_dino_tiny/onnx/model.onnx"), b"onnx").unwrap();
        std::fs::write(root.join("grounding_dino_tiny/tokenizer.json"), b"{}").unwrap();
        std::fs::write(
            root.join("sam2_tiny/onnx/vision_encoder.onnx_data"),
            b"weights",
        )
        .unwrap();
        let files: Vec<_> = manifest.iter().map(|(rel, _)| leaf.join(rel)).collect();

        ensure_release(&root, &leaf, &files, &manifest, &mut |_| {}).unwrap();

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
        let (onnx_sum, tok_sum, weights_sum) = (sha(b"onnx"), sha(b"{}"), sha(b"weights"));
        let manifest = [
            ("grounding_dino_tiny/onnx/model.onnx", onnx_sum.as_str()),
            ("grounding_dino_tiny/tokenizer.json", tok_sum.as_str()),
            (
                "sam2_tiny/onnx/vision_encoder.onnx_data",
                weights_sum.as_str(),
            ),
        ];
        let root = scratch("migrate-corrupt");
        let leaf = root.join(gsam::RELEASE_TAG);
        std::fs::create_dir_all(root.join("grounding_dino_tiny/onnx")).unwrap();
        std::fs::create_dir_all(root.join("sam2_tiny/onnx")).unwrap();
        std::fs::write(root.join("grounding_dino_tiny/onnx/model.onnx"), b"onnx").unwrap();
        std::fs::write(root.join("grounding_dino_tiny/tokenizer.json"), b"{}").unwrap();
        std::fs::write(
            root.join("sam2_tiny/onnx/vision_encoder.onnx_data"),
            b"tampered",
        )
        .unwrap();
        let files: Vec<_> = manifest.iter().map(|(rel, _)| leaf.join(rel)).collect();

        let err = ensure_release(&root, &leaf, &files, &manifest, &mut |_| {}).unwrap_err();
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
        let (onnx_sum, tok_sum) = (sha(b"onnx"), sha(b"{}"));
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
}
