use crate::render::CpuSplats;
use anyhow::{Context, Result};
use serde::Deserialize;
use std::io::{Read, Seek};
use zip::ZipArchive;

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
    files: Vec<String>,
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
) -> Result<(Vec<u8>, usize)> {
    let mut file = zip
        .by_name(name)
        .with_context(|| format!("missing {name}"))?;
    // Zip entries aren't Seek but the WebP decoder needs it — buffer the entry.
    let mut buf = Vec::with_capacity(file.size() as usize);
    file.read_to_end(&mut buf)?;
    let img = image::load_from_memory_with_format(&buf, image::ImageFormat::WebP)
        .with_context(|| format!("decode {name}"))?;

    let rgba = img.into_rgba8();
    let (w, h) = (rgba.width() as usize, rgba.height() as usize);
    if w * h < min_pixels {
        anyhow::bail!("{name}: {w}x{h} < {min_pixels} pixels");
    }
    Ok((rgba.into_raw(), w))
}

fn inv_log(v: f32) -> f32 {
    v.signum() * v.abs().exp_m1()
}

fn logit(y: f32) -> f32 {
    let e = y.clamp(1e-6, 1.0 - 1e-6);
    (e / (1.0 - e)).ln()
}

fn unpack_quat(px: u8, py: u8, pz: u8, tag: u8) -> [f32; 4] {
    let sqrt2 = std::f32::consts::SQRT_2;
    let a = (px as f32 / u8::MAX as f32 * 2.0 - 1.0) / sqrt2;
    let b = (py as f32 / u8::MAX as f32 * 2.0 - 1.0) / sqrt2;
    let c = (pz as f32 / u8::MAX as f32 * 2.0 - 1.0) / sqrt2;
    let d = (1.0 - a * a - b * b - c * c).max(0.0).sqrt();
    match tag.wrapping_sub(252) {
        0 => [d, a, b, c],
        1 => [a, d, b, c],
        2 => [a, b, d, c],
        _ => [a, b, c, d],
    }
}

/// Iterate an RGBA8 plane as `(pixel index, pixel)` for the first `n` pixels.
fn rgba_pixels(px: &[u8]) -> impl Iterator<Item = (usize, [u8; 4])> + '_ {
    px.chunks_exact(4)
        .enumerate()
        .map(|(i, c)| (i, c.try_into().unwrap()))
}

pub fn parse_sog(reader: impl Read + Seek) -> Result<CpuSplats> {
    let mut zip = ZipArchive::new(reader)?;
    let meta: Meta = serde_json::from_reader(zip.by_name("meta.json")?)?;

    let n = meta.count;
    let mut attributes = vec![0f32; n * 11];

    let (lo, _) = decode_rgba(&mut zip, &meta.means.files[0], n)?;
    let (hi, _) = decode_rgba(&mut zip, &meta.means.files[1], n)?;
    let mins = glam::Vec3::from_array(meta.means.mins);
    let spans = glam::Vec3::from_array(meta.means.maxs) - mins;

    for ((i, lc), (_, hc)) in rgba_pixels(&lo).zip(rgba_pixels(&hi)).take(n) {
        // Field-major output (see CpuSplats docs): plane k of splat i at [k*n + i].
        attributes[i] =
            inv_log(mins.x + spans.x * u16::from_le_bytes([lc[0], hc[0]]) as f32 / u16::MAX as f32);
        attributes[n + i] =
            inv_log(mins.y + spans.y * u16::from_le_bytes([lc[1], hc[1]]) as f32 / u16::MAX as f32);
        attributes[2 * n + i] =
            inv_log(mins.z + spans.z * u16::from_le_bytes([lc[2], hc[2]]) as f32 / u16::MAX as f32);
    }

    let (sl, _) = decode_rgba(&mut zip, &meta.scales.files[0], n)?;
    let scale_cb = &meta.scales.codebook;
    for (i, c) in rgba_pixels(&sl).take(n) {
        for k in 0..3 {
            attributes[(7 + k) * n + i] = scale_cb[c[k] as usize];
        }
    }

    let (qr, _) = decode_rgba(&mut zip, &meta.quats.files[0], n)?;
    for (i, c) in rgba_pixels(&qr).take(n) {
        let tag = c[3];
        let q = match tag {
            252..=255 => unpack_quat(c[0], c[1], c[2], tag),
            _ => [1.0, 0.0, 0.0, 0.0],
        };
        for k in 0..4 {
            attributes[(3 + k) * n + i] = q[k];
        }
    }

    let (c0, _) = decode_rgba(&mut zip, &meta.sh0.files[0], n)?;
    let sh_per_ch = meta.sh_n.as_ref().map_or(1, |s| (s.bands + 1).pow(2));
    let mut sh_coeffs = vec![0f32; n * sh_per_ch * 3];
    let sh0_cb = &meta.sh0.codebook;

    for (i, c) in rgba_pixels(&c0).take(n) {
        for k in 0..3 {
            sh_coeffs[k * n + i] = sh0_cb[c[k] as usize];
        }
        attributes[10 * n + i] = logit(c[3] as f32 / u8::MAX as f32);
    }

    if let Some(ref sh_n) = meta.sh_n {
        let bands = sh_n.bands;
        let sh_coeffs_per_ch = (bands + 1).pow(2) - 1;
        // centroids is the SH palette, not per-splat data — don't gate it on n
        let (centroids, cw) = decode_rgba(&mut zip, &sh_n.files[0], 0)?;
        let (labels, _) = decode_rgba(&mut zip, &sh_n.files[1], n)?;
        let codebook = &sh_n.codebook;
        // Rows hold whole palettes: total palettes is just pixels / coeffs.
        let palette_count = centroids.len() / 4 / sh_coeffs_per_ch;

        for (i, c) in rgba_pixels(&labels).take(n) {
            let label = c[0] as usize | (c[1] as usize) << 8;
            if label >= palette_count {
                continue;
            }
            let (base_x, base_y) = palette_offset(label, cw, sh_coeffs_per_ch);

            for j in 0..sh_coeffs_per_ch {
                let p = (base_y * cw + base_x + j) * 4;
                for k in 0..3 {
                    sh_coeffs[((j + 1) * 3 + k) * n + i] = codebook[centroids[p + k] as usize];
                }
            }
        }
    }

    // Release the archive (holds the full dropped-file bytes on wasm) before upload.
    drop(zip);
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

    #[test]
    fn test_palette_offset_derives_row_width() {
        // 128px-wide centroid sheet, 8 coeffs per channel → 16 palettes per row.
        let (col, row) = palette_offset(20, 128, 8);
        assert_eq!((col, row), (4 * 8, 1)); // col in pixels, not palette index
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
        if !std::path::Path::new(files[0]).exists() {
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
        let ns: Vec<usize> = parsed.iter().map(|p| p.attributes.len() / 11).collect();
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
