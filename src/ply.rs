use crate::render::{CpuSplats, Splats};
use anyhow::{Context, Result, anyhow};
use cubecl::{client::ComputeClient, wgpu::WgpuRuntime};
use std::io::{BufRead, BufReader, Read};

pub fn load_ply(reader: impl Read, client: &ComputeClient<WgpuRuntime>) -> Result<Splats> {
    let cpu = parse_ply(reader)?;
    Ok(Splats::new(cpu.attributes, cpu.sh_coeffs, client))
}

pub fn parse_ply(reader: impl Read) -> Result<CpuSplats> {
    let mut reader = BufReader::new(reader);
    let mut vertex_count = 0;
    let mut properties = Vec::new();

    for line in reader.by_ref().lines() {
        let line = line?;
        let tokens: Vec<&str> = line.split_whitespace().collect();
        match tokens.as_slice() {
            ["end_header", ..] => break,
            ["element", "vertex", count] => {
                vertex_count = count.parse().map_err(|_| anyhow!("Invalid vertex count"))?
            }
            ["property", "float", name] => properties.push(name.to_string()),
            _ => {}
        }
    }

    if vertex_count == 0 {
        return Err(anyhow!("PLY contains no vertices"));
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

    let mut rest_keys: Vec<(usize, usize)> = properties
        .iter()
        .enumerate()
        .filter_map(|(idx, name)| {
            name.strip_prefix("f_rest_")
                .and_then(|s| s.parse::<usize>().ok())
                .map(|n| (idx, n))
        })
        .collect();
    rest_keys.sort_by_key(|&(_, n)| n);
    let rest_keys: Vec<usize> = rest_keys.into_iter().map(|(idx, _)| idx).collect();

    let stride = properties.len();

    let mut buf = vec![0u8; vertex_count * stride * 4];
    reader.read_exact(&mut buf).context("Failed read vertex")?;
    // Free the file bytes before the big allocations: wasm32 linear memory is
    // capped at 4 GiB, so every large buffer is released as early as possible.
    drop(reader);

    let float_data: &[f32] = bytemuck::cast_slice(&buf);
    let mut attributes = Vec::with_capacity(vertex_count * 11);
    let mut shs = Vec::with_capacity(vertex_count * (rest_keys.len() + 3));
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
    }

    drop(buf);
    Ok(CpuSplats {
        attributes,
        sh_coeffs: shs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE_PROPS: [&str; 14] = [
        "x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3", "scale_0", "scale_1", "scale_2",
        "opacity", "f_dc_0", "f_dc_1", "f_dc_2",
    ];

    fn synthetic_ply(n: usize, extra_props: &[&str]) -> Vec<u8> {
        let props = BASE_PROPS
            .iter()
            .chain(extra_props)
            .map(|p| format!("property float {p}\n"))
            .collect::<String>();
        let mut bytes = format!(
            "ply\nformat binary_little_endian 1.0\nelement vertex {n}\n{props}end_header\n"
        )
        .into_bytes();
        let stride = BASE_PROPS.len() + extra_props.len();
        for i in 0..n {
            for j in 0..stride {
                let v = (i * 31 + j) as f32 * 0.25 - 8.0;
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        bytes
    }

    #[test]
    fn test_parse_ply_layout() {
        let bytes = synthetic_ply(3, &["f_rest_0", "f_rest_2", "f_rest_1"]);
        let cpu = parse_ply(&bytes[..]).unwrap();
        assert_eq!(cpu.attributes.len(), 3 * 11);
        // 3 DC + 1 rest per channel (the fixture declares one rest prop per channel).
        assert_eq!(cpu.sh_coeffs.len(), 3 * 6);

        // `end_header` is followed by a newline before the binary data.
        let data_off = bytes.windows(10).position(|w| w == b"end_header").unwrap() + 11;
        let stride = 17;
        let file = |i: usize, j: usize| {
            let start = data_off + (i * stride + j) * 4;
            f32::from_le_bytes(bytes[start..start + 4].try_into().unwrap())
        };

        // Attribute layout: x, y, z, q(wxyz), scale, opacity.
        assert_eq!(cpu.attributes[0], file(0, 0));
        assert_eq!(cpu.attributes[1], file(0, 1));
        assert_eq!(cpu.attributes[2], file(0, 2));
        assert_eq!(cpu.attributes[7], file(0, 7));
        assert_eq!(cpu.attributes[8], file(0, 8));
        assert_eq!(cpu.attributes[9], file(0, 9));
        assert_eq!(cpu.attributes[10], file(0, 10));

        // SH is channel-interleaved: [dc_r, dc_g, dc_b, rest_r, rest_g, rest_b].
        // File property order rest_0, rest_2, rest_1 is sorted to rest_0, rest_1, rest_2,
        // i.e. file indices 14, 16, 15 feed rest slots R, G, B in that order.
        let sorted_rest = [14usize, 16, 15];
        for i in 0..3usize {
            for (c, &rest) in sorted_rest.iter().enumerate() {
                assert_eq!(cpu.sh_coeffs[i * 6 + c], file(i, 11 + c));
                assert_eq!(cpu.sh_coeffs[i * 6 + 3 + c], file(i, rest));
            }
        }
    }

    #[test]
    fn test_parse_ply_missing_property() {
        let bytes = "ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nend_header\n"
            .to_string()
            .into_bytes();
        let err = parse_ply(&bytes[..]).unwrap_err().to_string();
        assert!(err.contains("Missing property"), "{err}");
    }

    #[test]
    fn test_parse_ply_empty_rejected() {
        let bytes = "ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty float x\nend_header\n"
            .to_string()
            .into_bytes();
        assert!(parse_ply(&bytes[..]).is_err());
    }
}
