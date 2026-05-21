use crate::render::Splats;
use anyhow::{Context, Result, anyhow};
use cubecl::{client::ComputeClient, wgpu::WgpuRuntime};
use std::io::{BufRead, BufReader, Read};

pub fn load_ply(reader: impl Read, client: &ComputeClient<WgpuRuntime>) -> Result<Splats> {
    let mut reader = BufReader::new(reader);
    let mut vertex_count = 0;
    let mut properties = Vec::new();
    let mut line = String::new();

    while reader.read_line(&mut line)? > 0 {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        match tokens.as_slice() {
            ["end_header", ..] => break,
            ["element", "vertex", count] => {
                vertex_count = count.parse().map_err(|_| anyhow!("Invalid vertex count"))?
            }
            ["property", "float", name] => properties.push(name.to_string()),
            _ => {}
        }
        line.clear();
    }

    let get_idx = |name: &str| {
        properties
            .iter()
            .position(|p| p == name)
            .ok_or_else(|| anyhow!("Missing property: {name}"))
    };

    let idx_x = get_idx("x")?;
    let idx_y = get_idx("y")?;
    let idx_z = get_idx("z")?;
    let idx_s0 = get_idx("scale_0")?;
    let idx_s1 = get_idx("scale_1")?;
    let idx_s2 = get_idx("scale_2")?;
    let idx_op = get_idx("opacity")?;
    let idx_r0 = get_idx("rot_0")?;
    let idx_r1 = get_idx("rot_1")?;
    let idx_r2 = get_idx("rot_2")?;
    let idx_r3 = get_idx("rot_3")?;
    let idx_dc0 = get_idx("f_dc_0")?;
    let idx_dc1 = get_idx("f_dc_1")?;
    let idx_dc2 = get_idx("f_dc_2")?;

    let mut rest_keys: Vec<usize> = properties
        .iter()
        .enumerate()
        .filter(|(_, name)| name.starts_with("f_rest_"))
        .map(|(idx, _)| idx)
        .collect();

    rest_keys.sort_by_key(|&idx| {
        properties[idx]
            .strip_prefix("f_rest_")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0)
    });

    let stride = properties.len();
    let mut buf = vec![0u8; vertex_count * stride * 4];
    reader.read_exact(&mut buf).context("Failed read vertex")?;

    let float_data: &[f32] = bytemuck::cast_slice(&buf);
    let mut attributes = Vec::with_capacity(vertex_count * 11);
    let mut shs = Vec::with_capacity(vertex_count * (rest_keys.len() + 3));
    let mut min = glam::Vec3::splat(f32::MAX);
    let mut max = glam::Vec3::splat(f32::MIN);
    let n = rest_keys.len() / 3;

    for d in float_data.chunks(stride).take(vertex_count) {
        let p = glam::Vec3::new(d[idx_x], d[idx_y], d[idx_z]);
        let q = glam::Quat::from_xyzw(d[idx_r1], d[idx_r2], d[idx_r3], d[idx_r0]).normalize();
        attributes.extend_from_slice(&[
            p.x, p.y, p.z, q.w, q.x, q.y, q.z, d[idx_s0], d[idx_s1], d[idx_s2], d[idx_op],
        ]);

        shs.extend_from_slice(&[d[idx_dc0], d[idx_dc1], d[idx_dc2]]);
        for i in 0..n {
            shs.push(d[rest_keys[i]]);
            shs.push(d[rest_keys[n + i]]);
            shs.push(d[rest_keys[2 * n + i]]);
        }
        (min, max) = (min.min(p), max.max(p));
    }

    Ok(Splats::new(&attributes, &shs, client, (min, max)))
}
