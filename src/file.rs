use crate::render::Splats;
use cubecl::{client::ComputeClient, wgpu::WgpuRuntime};
use glam::Vec3;
use serde::{
    Deserialize,
    de::{DeserializeSeed, Error},
};
use serde_ply::{DeserializeError, PlyChunkedReader, RowVisitor};
use std::collections::HashMap;
use std::io::Read;

#[derive(Deserialize, Default)]
struct PlyGaussian {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub scale_0: f32,
    pub scale_1: f32,
    pub scale_2: f32,
    pub opacity: f32,
    pub rot_0: f32,
    pub rot_1: f32,
    pub rot_2: f32,
    pub rot_3: f32,
    #[serde(default)]
    pub f_dc_0: f32,
    #[serde(default)]
    pub f_dc_1: f32,
    #[serde(default)]
    pub f_dc_2: f32,
    #[serde(flatten)]
    pub sh: HashMap<String, f32>,
}

fn interleave_coeffs(sh_dc: &[f32], sh_rest: &[f32], result: &mut Vec<f32>) {
    let n = sh_rest.len() / 3;
    result.extend_from_slice(sh_dc);
    result.extend((0..n).flat_map(|i| (0..3).map(move |j| sh_rest[j * n + i])));
}

pub fn load_ply(
    mut reader: impl Read,
    client: &ComputeClient<WgpuRuntime>,
) -> Result<Splats, DeserializeError> {
    let mut file = PlyChunkedReader::new();
    let mut buf = [0u8; 64 * 1024];

    let (vertex_count, rest_keys) = loop {
        if let Some(header) = file.header() {
            let vertex = header
                .get_element("vertex")
                .ok_or_else(|| DeserializeError::custom("Missing vertex element"))?;
            let rest_keys: Vec<_> = vertex
                .properties
                .iter()
                .filter(|&p| p.name.starts_with("f_rest_"))
                .map(|p| p.name.clone())
                .collect();
            break (vertex.count, rest_keys);
        }
        let n = reader.read(&mut buf)?;
        if n == 0 {
            return Err(DeserializeError::custom("Unexpected EOF before PLY header"));
        }
        file.buffer_mut().extend_from_slice(&buf[..n]);
    };

    let mut attributes = Vec::with_capacity(vertex_count * 11);
    let mut shs = Vec::with_capacity(vertex_count * (rest_keys.len() + 3));
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    let mut rb = Vec::with_capacity(rest_keys.len());

    let mut row_visitor = RowVisitor::new(|gs: PlyGaussian| {
        let q = glam::Vec4::new(gs.rot_0, gs.rot_1, gs.rot_2, gs.rot_3).normalize_or(glam::Vec4::X);
        let pos = Vec3::new(gs.x, gs.y, gs.z);
        min = min.min(pos);
        max = max.max(pos);

        attributes.extend_from_slice(&[
            gs.x, gs.y, gs.z, q.x, q.y, q.z, q.w, gs.scale_0, gs.scale_1, gs.scale_2, gs.opacity,
        ]);

        rb.clear();
        rb.extend(
            rest_keys
                .iter()
                .map(|k| gs.sh.get(k).copied().unwrap_or(0.0)),
        );
        interleave_coeffs(&[gs.f_dc_0, gs.f_dc_1, gs.f_dc_2], &rb, &mut shs);
    });

    loop {
        (&mut row_visitor).deserialize(&mut file)?;
        if file.current_element().is_none() {
            break;
        }
        let n = reader.read(&mut buf)?;
        if n == 0 {
            return Err(DeserializeError::custom("Unexpected EOF while reading PLY"));
        }
        file.buffer_mut().extend_from_slice(&buf[..n]);
    }

    Ok(Splats::new(&attributes, &shs, client, (min, max)))
}
