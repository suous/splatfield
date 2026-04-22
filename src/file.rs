use crate::render::Splats;
use cubecl::wgpu::WgpuDevice;
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

fn interleave_coeffs(sh_dc: Vec3, sh_rest: &[f32], result: &mut Vec<f32>) {
    let n = sh_rest.len() / 3;
    result.extend([sh_dc.x, sh_dc.y, sh_dc.z]);
    result.extend((0..n).flat_map(|i| (0..3).map(move |j| sh_rest[j * n + i])));
}

pub fn load_ply(mut reader: impl Read, device: &WgpuDevice) -> Result<Splats, DeserializeError> {
    let mut file = PlyChunkedReader::new();
    reader.read_to_end(file.buffer_mut())?;
    let header = file
        .header()
        .ok_or_else(|| DeserializeError::custom("Missing PLY header"))?;

    let vertex = header
        .get_element("vertex")
        .ok_or_else(|| DeserializeError::custom("Missing vertex element"))?;

    let total = vertex.count;
    let sh_count = vertex
        .properties
        .iter()
        .filter(|x| x.name.starts_with("f_rest_") || x.name.starts_with("f_dc_"))
        .count();

    let mut attributes = Vec::with_capacity(total * 11);
    let mut shs = Vec::with_capacity(total * sh_count.max(3));
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);

    let rc = sh_count.saturating_sub(3);
    let mut rb = Vec::with_capacity(rc);
    let mut count = 0;
    while count < total {
        RowVisitor::new(|gs: PlyGaussian| {
            count += 1;
            let q = glam::Vec4::new(gs.rot_0, gs.rot_1, gs.rot_2, gs.rot_3);
            let q = q.normalize_or(glam::Vec4::X);
            let pos = Vec3::new(gs.x, gs.y, gs.z);
            min = min.min(pos);
            max = max.max(pos);
            attributes.extend([
                gs.x, gs.y, gs.z, q.x, q.y, q.z, q.w, gs.scale_0, gs.scale_1, gs.scale_2,
                gs.opacity,
            ]);

            rb.clear();
            rb.extend((0..rc).map(|i| gs.sh.get(&format!("f_rest_{i}")).copied().unwrap_or(0.0)));
            interleave_coeffs(Vec3::new(gs.f_dc_0, gs.f_dc_1, gs.f_dc_2), &rb, &mut shs);
        })
        .deserialize(&mut file)?;
    }

    Ok(Splats::new(&attributes, &shs, device, (min, max)))
}
