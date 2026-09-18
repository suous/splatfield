use crate::layout::{ATTR_PLANES, PLANE_OPACITY, PLANE_QW, PLANE_SX, PLANE_X};
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
    Ok(rgba)
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
    let sqrt2 = core::f32::consts::SQRT_2;
    let a = (px as f32 / u8::MAX as f32 * 2.0 - 1.0) / sqrt2;
    let b = (py as f32 / u8::MAX as f32 * 2.0 - 1.0) / sqrt2;
    let c = (pz as f32 / u8::MAX as f32 * 2.0 - 1.0) / sqrt2;
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
    let meta: Meta =
        serde_json::from_reader(zip.by_name("meta.json").context("missing meta.json")?)
            .context("invalid meta.json")?;

    let n = meta.count;
    if n == 0 {
        anyhow::bail!("SOG contains no splats");
    }
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

    let sh_per_ch = match meta.sh_n.as_ref() {
        None => 1,
        Some(s) => match s.bands {
            1 => 4,
            2 => 9,
            3 => 16,
            b => anyhow::bail!("shN.bands {b} outside 1..=3"),
        },
    };
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
        // The centroid sheet is a palette, not per-splat data: any size is valid.
        let centroids = decode_rgba(&mut zip, &sh_n.files[0], 1)?;
        let labels = decode_rgba(&mut zip, &sh_n.files[1], n)?;
        let shn_cb = codebook(&sh_n.codebook, "shN")?;
        let cw = centroids.width() as usize;
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
