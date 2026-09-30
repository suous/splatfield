use crate::layout::{ATTR_PLANES, PLANE_OPACITY, PLANE_QW, PLANE_SX, PLANE_X};
use crate::ply::MAX_PLANE_BYTES;
use crate::render::CpuSplats;
use anyhow::{Context, Result};
use serde::Deserialize;
use std::io::{Cursor, Read, Seek};
use zip::ZipArchive;

/// Cap on one WebP entry's decompressed bytes: a crafted deflate stream is
/// a bomb vector — the archive itself is size-preflighted, but its
/// entries' decompressed length is not (the central-directory size is a
/// lie the reader never consults). The worst legit entry is the least
/// compressible: a budget-max no-shN scene's means sheet is ~77 MB decoded,
/// and lossless noise's bitstream runs ≈ decoded size, so this matches the
/// sheet cap and keeps ~3× headroom.
const MAX_ENTRY_BYTES: u64 = 256 * 1024 * 1024;

/// Cap on the decompressed `meta.json` stream: the codebooks are
/// `Vec<f32>` parsed straight off it, so an uncapped read grows a Vec to
/// whatever the deflate stream emits — the same bomb vector as the sheet
/// entries. Legit meta.json is ~20 KB (codebooks are 256 floats), so this
/// is ~50× headroom; a value past the cap hits EOF mid-parse and errors.
const MAX_META_BYTES: u64 = 1024 * 1024;

/// Cap on one sheet's decoded RGBA buffer: the bitstream declares its own
/// width×height, and a 16383² lossless sheet decodes to 1 GB from a few KB
/// of compressed bytes. A means sheet is ≥ one pixel per splat, so the
/// worst legit case — a no-shN scene at the 1 GiB plane budget, ~19.2M
/// splats — needs ~77 MB: ~3× headroom.
const MAX_SHEET_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Deserialize)]
struct Quantized {
    codebook: Vec<f32>,
    files: Vec<String>,
}

#[derive(Deserialize)]
struct Meta {
    count: usize,
    means: Means,
    scales: Quantized,
    quats: Files,
    sh0: Quantized,
    #[serde(rename = "shN")]
    sh_n: Option<ShN>,
}

#[derive(Deserialize)]
struct Means {
    mins: [f32; 3],
    maxs: [f32; 3],
    files: [String; 2],
}

#[derive(Deserialize)]
struct Files {
    files: [String; 1],
}

#[derive(Deserialize)]
struct ShN {
    bands: usize,
    codebook: Vec<f32>,
    files: [String; 2],
}

fn decode_rgba<R: Read + Seek>(
    zip: &mut ZipArchive<R>,
    name: &str,
    min_pixels: usize,
) -> Result<image::RgbaImage> {
    use image::ImageDecoder;

    let file = zip
        .by_name(name)
        .with_context(|| format!("missing {name}"))?;
    // Both vectors behind this read grow from attacker-chosen lengths, so
    // each is capped before it exists: `take` bounds the decompressed read
    // (read_to_end would grow to whatever the stream emits), and the header
    // dims are budgeted before the RGBA buffer is sized.
    let mut buf = Vec::new();
    file.take(MAX_ENTRY_BYTES + 1).read_to_end(&mut buf)?;
    anyhow::ensure!(
        buf.len() as u64 <= MAX_ENTRY_BYTES,
        "{name} exceeds the {MAX_ENTRY_BYTES}-byte entry cap"
    );
    // The bitstream declares its own width×height — a 16383² lossless
    // sheet decodes to 1 GB from a few KB — so budget the header dims
    // before the decode allocates. The probe parses the container first:
    // an extended VP8X file scans its chunk list here (bounded by the
    // capped entry above, and image-webp's chunk map dedupes by kind, so
    // the scan costs time, not allocation). ×4 bounds either pixel format
    // (lossy no-alpha sheets decode RGB, 3 B/px).
    let (w, h) = image::codecs::webp::WebPDecoder::new(Cursor::new(&buf))
        .with_context(|| format!("decode {name}"))?
        .dimensions();
    let pixels = w as u64 * h as u64;
    anyhow::ensure!(
        pixels * 4 <= MAX_SHEET_BYTES,
        "{name}: {w}x{h} exceeds the {MAX_SHEET_BYTES}-byte sheet budget"
    );
    if pixels < min_pixels as u64 {
        anyhow::bail!("{name}: {w}x{h} < {min_pixels} pixels");
    }
    let img = image::load_from_memory_with_format(&buf, image::ImageFormat::WebP)
        .with_context(|| format!("decode {name}"))?;

    Ok(img.into_rgba8())
}

fn inv_log(v: f32) -> f32 {
    v.signum() * v.abs().exp_m1()
}

fn logit(y: f32) -> f32 {
    let e = y.clamp(1e-6, 1.0 - 1e-6);
    (e / (1.0 - e)).ln()
}

fn codebook<'a>(cb: &'a [f32], what: &str) -> Result<&'a [f32; 256]> {
    cb.try_into()
        .with_context(|| format!("{what} codebook: 256 floats expected, got {}", cb.len()))
}

fn unpack_quat(px: u8, py: u8, pz: u8, tag: u8) -> [f32; 4] {
    let u = |v: u8| (v as f32 / u8::MAX as f32 * 2.0 - 1.0) / core::f32::consts::SQRT_2;
    let (a, b, c) = (u(px), u(py), u(pz));
    let d = (1.0 - a * a - b * b - c * c).max(0.0).sqrt();
    match tag {
        252 => [d, a, b, c],
        253 => [a, d, b, c],
        254 => [a, b, d, c],
        _ => [a, b, c, d], // 255: z is the omitted component
    }
}

/// The RGBA8 pixels of a plane.
fn rgba_pixels(px: &[u8]) -> &[[u8; 4]] {
    px.as_chunks::<4>().0
}

pub fn parse_sog(reader: impl Read + Seek) -> Result<CpuSplats> {
    let mut zip = ZipArchive::new(reader).context("not a SOG archive")?;
    let meta: Meta = {
        let mut meta_file = zip
            .by_name("meta.json")
            .context("missing meta.json")?
            .take(MAX_META_BYTES + 1);
        serde_json::from_reader(&mut meta_file).context("invalid meta.json")?
    };

    let n = meta.count;
    if n == 0 {
        anyhow::bail!("SOG contains no splats");
    }
    let sh_per_ch = match meta.sh_n.as_ref() {
        None => 1,
        Some(s) => match s.bands {
            1 => 4,
            2 => 9,
            3 => 16,
            b => anyhow::bail!("shN.bands {b} outside 1..=3"),
        },
    };
    // Guard absurd metadata before allocating the output planes — the same
    // budget as parse_ply's. `count` comes straight from meta.json, and on
    // wasm32 an unchecked vec![0f32; …] of that size is an alloc-failure
    // abort before the first sheet is even read (an overflowing usize
    // product wraps into a tiny buffer and an index abort instead).
    anyhow::ensure!(
        (n as u64)
            .checked_mul((ATTR_PLANES + sh_per_ch * 3) as u64 * 4)
            .is_some_and(|bytes| bytes <= MAX_PLANE_BYTES),
        "SOG splat count {n} exceeds the {MAX_PLANE_BYTES}-byte output-plane budget"
    );
    let mut attributes = vec![0f32; n * ATTR_PLANES];

    let lo = decode_rgba(&mut zip, &meta.means.files[0], n)?;
    let hi = decode_rgba(&mut zip, &meta.means.files[1], n)?;
    let mins = glam::Vec3::from_array(meta.means.mins);
    let spans = glam::Vec3::from_array(meta.means.maxs) - mins;

    for (i, (lc, hc)) in rgba_pixels(lo.as_raw())
        .iter()
        .zip(rgba_pixels(hi.as_raw()))
        .take(n)
        .enumerate()
    {
        for k in 0..3 {
            let t = u16::from_le_bytes([lc[k], hc[k]]) as f32 / u16::MAX as f32;
            attributes[(PLANE_X + k) * n + i] = inv_log(mins[k] + spans[k] * t);
        }
    }

    let sl = decode_rgba(&mut zip, &meta.scales.files[0], n)?;
    let scale_cb = codebook(&meta.scales.codebook, "scales")?;
    for (i, c) in rgba_pixels(sl.as_raw()).iter().enumerate().take(n) {
        for k in 0..3 {
            attributes[(PLANE_SX + k) * n + i] = scale_cb[c[k] as usize];
        }
    }

    let qr = decode_rgba(&mut zip, &meta.quats.files[0], n)?;
    for (i, c) in rgba_pixels(qr.as_raw()).iter().enumerate().take(n) {
        let tag = c[3];
        let q = match tag {
            252..=255 => unpack_quat(c[0], c[1], c[2], tag),
            _ => anyhow::bail!("quats: invalid rotation tag {tag}"),
        };
        for k in 0..4 {
            attributes[(PLANE_QW + k) * n + i] = q[k];
        }
    }

    let c0 = decode_rgba(&mut zip, &meta.sh0.files[0], n)?;
    let mut sh_coeffs = vec![0f32; n * sh_per_ch * 3];
    let sh0_cb = codebook(&meta.sh0.codebook, "sh0")?;

    for (i, c) in rgba_pixels(c0.as_raw()).iter().enumerate().take(n) {
        for k in 0..3 {
            sh_coeffs[k * n + i] = sh0_cb[c[k] as usize];
        }
        attributes[PLANE_OPACITY * n + i] = logit(c[3] as f32 / u8::MAX as f32);
    }

    if let Some(ref sh_n) = meta.sh_n {
        // (bands+1)^2 channels incl. DC; the palette sheet holds the rest.
        let sh_coeffs_per_ch = sh_per_ch - 1;
        // The centroid sheet is a palette, not per-splat data: its pixel
        // count is unbounded, but its geometry is checked once decoded.
        let centroids = decode_rgba(&mut zip, &sh_n.files[0], 1)?;
        let labels = decode_rgba(&mut zip, &sh_n.files[1], n)?;
        let shn_cb = codebook(&sh_n.codebook, "shN")?;
        let cw = centroids.width() as usize;
        // Rows hold whole palettes, so the sheet width must tile the coeffs
        // exactly: a remainder either floors palette_offset's per-row count
        // to zero (`label % 0` panic) or walks row tails past the sheet
        // (index panic) — a crafted file must bail, not abort the wasm module.
        anyhow::ensure!(
            cw.is_multiple_of(sh_coeffs_per_ch),
            "shN centroid sheet: width {cw} not a multiple of {sh_coeffs_per_ch}"
        );
        let centroids = centroids.as_raw();
        // Rows hold whole palettes: total palettes is just pixels / coeffs.
        let palette_count = centroids.len() / 4 / sh_coeffs_per_ch;

        for (i, c) in rgba_pixels(labels.as_raw()).iter().enumerate().take(n) {
            let label = c[0] as usize | (c[1] as usize) << 8;
            if label >= palette_count {
                anyhow::bail!("shN label {label} >= palette size {palette_count}");
            }
            let (base_x, base_y) = palette_offset(label, cw, sh_coeffs_per_ch);

            for j in 0..sh_coeffs_per_ch {
                let p = (base_y * cw + base_x + j) * 4;
                for k in 0..3 {
                    sh_coeffs[((j + 1) * 3 + k) * n + i] = shn_cb[centroids[p + k] as usize];
                }
            }
        }
    }

    Ok(CpuSplats {
        attributes,
        sh_coeffs,
    })
}

/// Column (in pixels) and row of a label's palette entry in the centroid sheet.
fn palette_offset(label: usize, cw: usize, sh_coeffs_per_ch: usize) -> (usize, usize) {
    let per_row = cw / sh_coeffs_per_ch;
    (label % per_row * sh_coeffs_per_ch, label / per_row)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fetch::build_zip;
    use std::io::{Cursor, Write};

    /// Lossless-encode an RGBA8 buffer as WebP — the parser's strict input
    /// format — so palette-walk fixtures need no data/ assets.
    fn webp_bytes(img: &image::RgbaImage) -> Vec<u8> {
        let mut out = Vec::new();
        image::codecs::webp::WebPEncoder::new_lossless(&mut out)
            .encode(
                img.as_raw(),
                img.width(),
                img.height(),
                image::ExtendedColorType::Rgba8,
            )
            .unwrap();
        out
    }

    /// A minimal SOG archive over `n` all-zero splats with identity
    /// codebooks (coefficient == byte value), so shN palette reads are
    /// assertable byte-for-byte. `sh_bands` adds the shN half with a
    /// `sheet_w`×`sheet_h` centroid sheet — malformed geometry is the
    /// point of some callers. Labels address palette 0 by default;
    /// `labels` overrides the per-splat label values.
    fn fixture_sog(
        n: u32,
        sh_bands: Option<usize>,
        sheet_w: u32,
        sheet_h: u32,
        labels: &dyn Fn(u32) -> [u8; 4],
    ) -> Vec<u8> {
        let solid = |rgba: [u8; 4], w: u32, h: u32| {
            image::RgbaImage::from_fn(w, h, |_, _| image::Rgba(rgba))
        };
        let codebook: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let mut meta = serde_json::json!({
            "count": n,
            "means": {
                "mins": [0.0, 0.0, 0.0],
                "maxs": [0.0, 0.0, 0.0],
                "files": ["means_lo.webp", "means_hi.webp"],
            },
            "scales": {"codebook": codebook.clone(), "files": ["scales.webp"]},
            "quats": {"files": ["quats.webp"]},
            "sh0": {"codebook": codebook.clone(), "files": ["sh0.webp"]},
        });
        if let Some(bands) = sh_bands {
            meta["shN"] = serde_json::json!({
                "bands": bands,
                "codebook": codebook,
                "files": ["shN_cent.webp", "shN_lab.webp"],
            });
        }

        // Quats carry tag 255 (z omitted) → the quaternion [0,0,0,1].
        let quats_img = solid([0, 0, 0, 255], n, 1);
        let label_img = image::RgbaImage::from_fn(n, 1, |x, _| image::Rgba(labels(x)));
        let sheet_img = sh_bands.map(|_| {
            // Palette row p holds pixel bytes (p*10 + j*3 + k) for entry
            // j's channel k — distinct per (palette, entry, channel).
            image::RgbaImage::from_fn(sheet_w, sheet_h, |j, p| {
                let b = |k: u8| p as u8 * 10 + j as u8 * 3 + k;
                image::Rgba([b(0), b(1), b(2), 255])
            })
        });
        let meta = meta.to_string();
        let blank = webp_bytes(&solid([0; 4], n, 1));
        let quats = webp_bytes(&quats_img);
        let label = webp_bytes(&label_img);
        let sheet = sheet_img.as_ref().map(webp_bytes);
        let mut files: Vec<(&str, &[u8])> = vec![
            ("meta.json", meta.as_bytes()),
            ("means_lo.webp", &blank),
            ("means_hi.webp", &blank),
            ("scales.webp", &blank),
            ("quats.webp", &quats),
            ("sh0.webp", &blank),
        ];
        if let Some(sheet) = &sheet {
            files.push(("shN_cent.webp", sheet));
            files.push(("shN_lab.webp", &label));
        }
        build_zip(&files)
    }

    /// A `meta.json`-only archive (plus optional raw entries) — enough to
    /// drive every check that runs before the first sheet read.
    fn meta_only_sog(count: u64, entries: &[(&str, Vec<u8>)]) -> Vec<u8> {
        let meta = serde_json::json!({
            "count": count,
            "means": {
                "mins": [0.0, 0.0, 0.0],
                "maxs": [0.0, 0.0, 0.0],
                "files": ["means_lo.webp", "means_hi.webp"],
            },
            "scales": {"codebook": [], "files": ["scales.webp"]},
            "quats": {"files": ["quats.webp"]},
            "sh0": {"codebook": [], "files": ["sh0.webp"]},
        });
        let meta = meta.to_string();
        let mut files: Vec<(&str, &[u8])> = vec![("meta.json", meta.as_bytes())];
        files.extend(
            entries
                .iter()
                .map(|(name, bytes)| (*name, bytes.as_slice())),
        );
        build_zip(&files)
    }

    #[test]
    fn test_sog_over_budget_count_bails_before_any_read() {
        // 2e9 splats × 56 B (no shN) ≈ 112 GB of planes — the budget must
        // fire before the first sheet is read, on an archive with no
        // sheets at all.
        let bytes = meta_only_sog(2_000_000_000, &[]);
        let err = parse_sog(Cursor::new(bytes)).unwrap_err().to_string();
        assert!(err.contains("budget"), "{err}");

        // A count whose product overflows u64 (host usize) must take the
        // checked_mul `None` arm and bail, not wrap into a small
        // allocation.
        let bytes = meta_only_sog(u64::MAX / 2, &[]);
        let err = parse_sog(Cursor::new(bytes)).unwrap_err().to_string();
        assert!(err.contains("budget"), "{err}");
    }

    #[test]
    fn test_sog_entry_cap_bombs_bail() {
        // A deflate bomb: ~260 KB compressed in, MAX_ENTRY_BYTES + 64 KiB
        // of zeros out — the take cap bounds the read instead of growing
        // to the stream's real length.
        let bomb = (MAX_ENTRY_BYTES as usize / 65536 + 2) * 65536;
        let chunk = vec![0u8; 65536];
        let mut zw = zip::ZipWriter::new(Cursor::new(Vec::new()));
        zw.start_file("meta.json", zip::write::SimpleFileOptions::default())
            .unwrap();
        zw.write_all(
            serde_json::json!({
                "count": 1,
                "means": {
                    "mins": [0.0, 0.0, 0.0],
                    "maxs": [0.0, 0.0, 0.0],
                    "files": ["means_lo.webp", "means_hi.webp"],
                },
                "scales": {"codebook": [], "files": ["scales.webp"]},
                "quats": {"files": ["quats.webp"]},
                "sh0": {"codebook": [], "files": ["sh0.webp"]},
            })
            .to_string()
            .as_bytes(),
        )
        .unwrap();
        zw.start_file("means_lo.webp", zip::write::SimpleFileOptions::default())
            .unwrap();
        for _ in 0..(bomb / 65536) {
            zw.write_all(&chunk).unwrap();
        }
        let bytes = zw.finish().unwrap().into_inner();
        let err = parse_sog(Cursor::new(bytes)).unwrap_err().to_string();
        assert!(err.contains("entry cap"), "{err}");
    }

    #[test]
    fn test_sog_sheet_budget_bails_not_allocates() {
        // A lossless WebP declares its own dims: a 16383² sheet is a ~1 GB
        // decode from a 20-byte file — the header is budgeted before the
        // pixel buffer exists. (16383, not 16384: image-webp clamps VP8L
        // dims to 14 bits post-increment, so the spec max wraps to 0.)
        let payload: [u8; 9] = [
            0x2f, // lossless signature
            0xfe, 0xbf, 0xff, 0x0f, // dims: (w-1) | (h-1) << 14, w = h = 16383
            0, 0, 0, 0,
        ];
        let mut webp = b"RIFF".to_vec();
        webp.extend_from_slice(&((4 + 8 + payload.len()) as u32).to_le_bytes());
        webp.extend_from_slice(b"WEBPVP8L");
        webp.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        webp.extend_from_slice(&payload);
        let bytes = meta_only_sog(1, &[("means_lo.webp", webp)]);
        let err = parse_sog(Cursor::new(bytes)).unwrap_err().to_string();
        assert!(err.contains("sheet budget"), "{err}");
    }

    #[test]
    fn test_sog_meta_cap_bombs_bail() {
        // A numeric codebook padded past the meta cap (a string element
        // would type-error at the first byte and never reach the cap): the
        // parse must hit the capped EOF as an error, not grow the Vec to
        // the stream's real length. Without the take the parse SUCCEEDS
        // and the failure moves to codebook()'s 256-float check, so the
        // assert pins the cap, not the codebook length.
        let codebook = vec![0.0f64; 600_000];
        let meta = serde_json::json!({
            "count": 1,
            "means": {
                "mins": [0.0, 0.0, 0.0],
                "maxs": [0.0, 0.0, 0.0],
                "files": ["means_lo.webp", "means_hi.webp"],
            },
            "scales": {"codebook": codebook, "files": ["scales.webp"]},
            "quats": {"files": ["quats.webp"]},
            "sh0": {"codebook": [], "files": ["sh0.webp"]},
        });
        let meta = meta.to_string();
        let bytes = build_zip(&[("meta.json", meta.as_bytes())]);
        let err = parse_sog(Cursor::new(bytes)).unwrap_err().to_string();
        assert!(err.contains("invalid meta.json"), "{err}");
    }

    #[test]
    fn test_palette_offset_derives_row_width() {
        // 128px-wide centroid sheet, 8 coeffs per channel → 16 palettes per row.
        let (col, row) = palette_offset(20, 128, 8);
        assert_eq!((col, row), (4 * 8, 1)); // col in pixels, not palette index
    }

    #[test]
    fn test_shn_narrow_sheet_bails_not_panics() {
        // bands=3 → 15 coeffs; a 4px-wide sheet floors per_row to 0, so the
        // walk would `label % 0` — the file must be rejected, not abort.
        let bytes = fixture_sog(1, Some(3), 4, 10, &|_| [0, 0, 0, 255]);
        let err = parse_sog(Cursor::new(bytes)).unwrap_err().to_string();
        assert!(err.contains("centroid sheet"), "{err}");
    }

    #[test]
    fn test_shn_untiled_sheet_bails_not_panics() {
        // bands=1 → 3 coeffs; a 7px sheet has one row-tail pixel, so label 6
        // (still < palette_count) would index one row past the sheet.
        let bytes = fixture_sog(1, Some(1), 7, 3, &|_| [6, 0, 0, 255]);
        let err = parse_sog(Cursor::new(bytes)).unwrap_err().to_string();
        assert!(err.contains("centroid sheet"), "{err}");
    }

    #[test]
    fn test_shn_palette_walk_reads_named_palettes() {
        // Valid geometry: 2 palettes (3px rows), two splats pointing at
        // different palettes — the coeffs must equal the sheet's byte values.
        let bytes = fixture_sog(2, Some(1), 3, 2, &|i| [i as u8, 0, 0, 255]);
        let parsed = parse_sog(Cursor::new(bytes)).unwrap();
        assert_eq!(parsed.sh_coeffs.len(), 2 * 4 * 3);
        let b = |p: u8, j: u8, k: u8| (p * 10 + j * 3 + k) as f32;
        for j in 0..3u8 {
            for k in 0..3u8 {
                let base = ((j as usize + 1) * 3 + k as usize) * 2;
                assert_eq!(parsed.sh_coeffs[base], b(0, j, k));
                assert_eq!(parsed.sh_coeffs[base + 1], b(1, j, k));
            }
        }
    }

    #[test]
    fn test_unpack_quat_unit_all_tags() {
        for tag in 252u8..=255 {
            let q = unpack_quat(200, 10, 77, tag);
            let norm: f32 = q.iter().map(|x| x * x).sum();
            assert!((norm - 1.0).abs() < 1e-5, "tag {tag}: {norm}");
        }
    }

    #[test]
    fn test_inv_log_logit_roundtrip() {
        for &v in &[-3.0f32, -0.5, 0.0, 0.5, 3.0] {
            let y = 1.0 / (1.0 + (-v).exp());
            assert!((logit(y) - v).abs() < 1e-4);
        }
    }

    #[test]
    fn test_parse_sog_fixtures() {
        let files = [
            "data/bear.3d71a266.sog",
            "data/bear.3d71a266_sh1.sog",
            "data/bear.3d71a266_sh2.sog",
        ];
        if !files.iter().all(|f| std::path::Path::new(f).exists()) {
            if std::env::var_os("SPLATFIELD_ALLOW_MISSING_ASSETS").is_none() {
                panic!(
                    "missing sog fixtures {files:?} — this test is the only parse_sog coverage; set SPLATFIELD_ALLOW_MISSING_ASSETS=1 to skip"
                );
            }
            eprintln!("skipping: no sog fixtures");
            return;
        }
        let parsed: Vec<CpuSplats> = files
            .iter()
            .map(|f| parse_sog(std::fs::File::open(f).unwrap()).unwrap())
            .collect();
        // Measured fixtures: the base bear.sog is a bands=3 model (970_948 splats);
        // _sh1/_sh2 are bands=1/2 models sharing 944_830 splats — not bands 0/1/2
        // of one model, so geometry equality only holds within the _shN pair.
        let ns: Vec<usize> = parsed
            .iter()
            .map(|p| p.attributes.len() / ATTR_PLANES)
            .collect();
        assert!(ns.iter().all(|&n| n > 100_000));
        assert_eq!(ns[1], ns[2], "sh1/sh2 share geometry");
        let chs: Vec<usize> = parsed
            .iter()
            .zip(&ns)
            .map(|(p, &n)| p.sh_coeffs.len() / n / 3)
            .collect();
        assert_eq!(chs, vec![16, 4, 9], "bands 3/1/2 → channels 16/4/9");
        assert!(parsed.iter().all(|p| {
            p.sh_coeffs
                .chunks(3)
                .all(|c| c.iter().all(|x| x.is_finite()))
        }));
    }
}
