use crate::render::Splats;
use anyhow::{Context, Result};
use cubecl::{client::ComputeClient, wgpu::WgpuRuntime};
use serde::Deserialize;
use std::io::{Cursor, Read};
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

fn decode_rgba(
    zip: &mut ZipArchive<Cursor<&[u8]>>,
    name: &str,
    min_pixels: usize,
) -> Result<(Vec<u8>, usize)> {
    let mut file = zip
        .by_name(name)
        .with_context(|| format!("missing {name}"))?;
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
    let a = (px as f32 / 255.0 * 2.0 - 1.0) / sqrt2;
    let b = (py as f32 / 255.0 * 2.0 - 1.0) / sqrt2;
    let c = (pz as f32 / 255.0 * 2.0 - 1.0) / sqrt2;
    let d = (1.0 - a * a - b * b - c * c).max(0.0).sqrt();
    match tag.wrapping_sub(252) {
        0 => [d, a, b, c],
        1 => [a, d, b, c],
        2 => [a, b, d, c],
        _ => [a, b, c, d],
    }
}

pub fn load_sog(data: &[u8], client: &ComputeClient<WgpuRuntime>) -> Result<Splats> {
    let mut zip = ZipArchive::new(Cursor::new(data))?;
    let meta: Meta = serde_json::from_reader(zip.by_name("meta.json")?)?;

    let n = meta.count;
    let mut attributes = vec![0f32; n * 11];
    let mut min = glam::Vec3::splat(f32::MAX);
    let mut max = glam::Vec3::splat(f32::MIN);

    let (lo, _) = decode_rgba(&mut zip, &meta.means.files[0], n)?;
    let (hi, _) = decode_rgba(&mut zip, &meta.means.files[1], n)?;
    let mins = glam::Vec3::from_array(meta.means.mins);
    let spans = glam::Vec3::from_array(meta.means.maxs) - mins;

    for ((attr, lc), hc) in attributes
        .chunks_exact_mut(11)
        .zip(lo.chunks_exact(4))
        .zip(hi.chunks_exact(4))
    {
        let p = glam::Vec3::new(
            inv_log(mins.x + spans.x * u16::from_le_bytes([lc[0], hc[0]]) as f32 / 65535.0),
            inv_log(mins.y + spans.y * u16::from_le_bytes([lc[1], hc[1]]) as f32 / 65535.0),
            inv_log(mins.z + spans.z * u16::from_le_bytes([lc[2], hc[2]]) as f32 / 65535.0),
        );

        attr[0..3].copy_from_slice(&p.to_array());
        min = min.min(p);
        max = max.max(p);
    }

    let (sl, _) = decode_rgba(&mut zip, &meta.scales.files[0], n)?;
    let scale_cb = &meta.scales.codebook;
    for (attr, chunk) in attributes.chunks_exact_mut(11).zip(sl.chunks_exact(4)) {
        for i in 0..3 {
            attr[7 + i] = scale_cb[chunk[i] as usize];
        }
    }

    let (qr, _) = decode_rgba(&mut zip, &meta.quats.files[0], n)?;
    for (attr, chunk) in attributes.chunks_exact_mut(11).zip(qr.chunks_exact(4)) {
        let tag = chunk[3];
        let q = match tag {
            252..=255 => unpack_quat(chunk[0], chunk[1], chunk[2], tag),
            _ => [1.0, 0.0, 0.0, 0.0],
        };
        attr[3..7].copy_from_slice(&q);
    }

    let (c0, _) = decode_rgba(&mut zip, &meta.sh0.files[0], n)?;
    let sh_per_ch = meta.sh_n.as_ref().map_or(1, |s| (s.bands + 1).pow(2));
    let mut sh_coeffs = vec![0f32; n * sh_per_ch * 3];
    let sh0_cb = &meta.sh0.codebook;

    for ((attr, chunk), sh_chunk) in attributes
        .chunks_exact_mut(11)
        .zip(c0.chunks_exact(4))
        .zip(sh_coeffs.chunks_exact_mut(sh_per_ch * 3))
    {
        for i in 0..3 {
            sh_chunk[i] = sh0_cb[chunk[i] as usize];
        }
        attr[10] = logit(chunk[3] as f32 / 255.0);
    }

    if let Some(ref sh_n) = meta.sh_n {
        let bands = sh_n.bands;
        let sh_coeffs_per_ch = (bands + 1).pow(2) - 1;
        let (centroids, cw) = decode_rgba(&mut zip, &sh_n.files[0], n)?;
        let (labels, _) = decode_rgba(&mut zip, &sh_n.files[1], n)?;
        let codebook = &sh_n.codebook;
        let palette_count = (cw / sh_coeffs_per_ch) * (centroids.len() / 4 / cw);

        for (label_chunk, sh_chunk) in labels
            .chunks_exact(4)
            .zip(sh_coeffs.chunks_exact_mut(sh_per_ch * 3))
        {
            let label = label_chunk[0] as usize | (label_chunk[1] as usize) << 8;
            if label >= palette_count {
                continue;
            }
            let base_x = (label % 64) * sh_coeffs_per_ch;
            let base_y = label / 64;

            for j in 0..sh_coeffs_per_ch {
                let p = (base_y * cw + base_x + j) * 4;
                let c_idx = (j + 1) * 3;
                for k in 0..3 {
                    sh_chunk[c_idx + k] = codebook[centroids[p + k] as usize];
                }
            }
        }
    }

    Ok(Splats::new(&attributes, &sh_coeffs, client, (min, max)))
}
