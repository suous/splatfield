use crate::layout::{
    ATTR_PLANES, PLANE_OPACITY, PLANE_QW, PLANE_QX, PLANE_QY, PLANE_QZ, PLANE_SX, PLANE_SY,
    PLANE_SZ, PLANE_X, PLANE_Y, PLANE_Z,
};
use crate::render::CpuSplats;
use anyhow::{Context, Result, anyhow, bail};
use std::io::BufRead;

pub fn parse_ply(mut reader: impl BufRead) -> Result<CpuSplats> {
    let mut vertex_count = 0;
    let mut properties = Vec::new();

    for line in reader.by_ref().lines() {
        let line = line?;
        let tokens: Vec<&str> = line.split_whitespace().collect();
        match tokens.as_slice() {
            ["end_header", ..] => break,
            ["format", fmt, ..] if *fmt != "binary_little_endian" => {
                bail!("Unsupported PLY format: {fmt}")
            }
            ["element", "vertex", count] => {
                vertex_count = count.parse().context("Invalid vertex count")?
            }
            ["property", ty, name] => {
                if *ty != "float" {
                    bail!("Unsupported property type {ty}: {name}")
                }
                properties.push(name.to_string());
            }
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

    let (idx_x, idx_y, idx_z) = (get_idx("x")?, get_idx("y")?, get_idx("z")?);
    let (idx_s0, idx_s1, idx_s2) = (
        get_idx("scale_0")?,
        get_idx("scale_1")?,
        get_idx("scale_2")?,
    );
    let idx_op = get_idx("opacity")?;
    let (idx_r0, idx_r1, idx_r2, idx_r3) = (
        get_idx("rot_0")?,
        get_idx("rot_1")?,
        get_idx("rot_2")?,
        get_idx("rot_3")?,
    );
    let (idx_dc0, idx_dc1, idx_dc2) = (get_idx("f_dc_0")?, get_idx("f_dc_1")?, get_idx("f_dc_2")?);

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

    let mut data = vec![0f32; vertex_count * stride];
    reader
        .read_exact(bytemuck::cast_slice_mut(&mut data))
        .with_context(|| format!("failed to read {vertex_count}x{stride} float vertices"))?;
    // Free the file bytes before the big allocations: wasm32 linear memory is
    // capped at 4 GiB, so every large buffer is released as early as possible.
    drop(reader);

    let n = rest_keys.len() / 3;
    let vc = vertex_count;
    let mut attributes = vec![0f32; vc * ATTR_PLANES];
    let mut shs = vec![0f32; vc * (rest_keys.len() + 3)];

    for (i, d) in data.chunks(stride).enumerate() {
        let q = glam::Quat::from_xyzw(d[idx_r1], d[idx_r2], d[idx_r3], d[idx_r0]).normalize();
        attributes[PLANE_X * vc + i] = d[idx_x];
        attributes[PLANE_Y * vc + i] = d[idx_y];
        attributes[PLANE_Z * vc + i] = d[idx_z];
        attributes[PLANE_QW * vc + i] = q.w;
        attributes[PLANE_QX * vc + i] = q.x;
        attributes[PLANE_QY * vc + i] = q.y;
        attributes[PLANE_QZ * vc + i] = q.z;
        attributes[PLANE_SX * vc + i] = d[idx_s0];
        attributes[PLANE_SY * vc + i] = d[idx_s1];
        attributes[PLANE_SZ * vc + i] = d[idx_s2];
        attributes[PLANE_OPACITY * vc + i] = d[idx_op];

        shs[i] = d[idx_dc0];
        shs[vc + i] = d[idx_dc1];
        shs[2 * vc + i] = d[idx_dc2];
        for j in 0..n {
            shs[((j + 1) * 3) * vc + i] = d[rest_keys[j]];
            shs[((j + 1) * 3 + 1) * vc + i] = d[rest_keys[n + j]];
            shs[((j + 1) * 3 + 2) * vc + i] = d[rest_keys[2 * n + j]];
        }
    }

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
        assert_eq!(cpu.attributes.len(), 3 * ATTR_PLANES);
        // 3 DC + 1 rest per channel (the fixture declares one rest prop per channel).
        assert_eq!(cpu.sh_coeffs.len(), 3 * 6);

        // `end_header` is followed by a newline before the binary data.
        let data_off = bytes.windows(10).position(|w| w == b"end_header").unwrap() + 11;
        let stride = 17;
        let file = |i: usize, j: usize| {
            let start = data_off + (i * stride + j) * 4;
            f32::from_le_bytes(bytes[start..start + 4].try_into().unwrap())
        };

        // Field-major attribute planes: [x | y | z | qw.. | sx.. | opacity],
        // plane k of splat i at attributes[k * n + i] (n = 3 here).
        assert_eq!(cpu.attributes[0], file(0, 0)); // x, splat 0
        assert_eq!(cpu.attributes[3], file(0, 1)); // y, splat 0
        assert_eq!(cpu.attributes[6], file(0, 2)); // z, splat 0
        assert_eq!(cpu.attributes[7 * 3], file(0, 7)); // scale_0, splat 0
        assert_eq!(cpu.attributes[8 * 3], file(0, 8)); // scale_1, splat 0
        assert_eq!(cpu.attributes[9 * 3], file(0, 9)); // scale_2, splat 0
        assert_eq!(cpu.attributes[10 * 3], file(0, 10)); // opacity, splat 0

        // SH planes: dc channel c at c * n + i, rest coefficient 1 channel c
        // at (3 + c) * n + i. File property order rest_0, rest_2, rest_1 is
        // sorted to rest_0, rest_1, rest_2, i.e. file indices 14, 16, 15 feed
        // rest planes R, G, B in that order.
        let sorted_rest = [14usize, 16, 15];
        for i in 0..3usize {
            for (c, &rest) in sorted_rest.iter().enumerate() {
                assert_eq!(cpu.sh_coeffs[c * 3 + i], file(i, 11 + c));
                assert_eq!(cpu.sh_coeffs[(3 + c) * 3 + i], file(i, rest));
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

    #[test]
    fn test_parse_ply_rejects_other_formats() {
        // A big-endian header would silently byte-swap every float below.
        let bytes =
            "ply\nformat binary_big_endian 1.0\nelement vertex 1\nproperty float x\nend_header\n"
                .to_string()
                .into_bytes();
        let err = parse_ply(&bytes[..]).unwrap_err().to_string();
        assert!(err.contains("Unsupported PLY format"), "{err}");
    }

    #[test]
    fn test_parse_ply_rejects_non_float_properties() {
        // A non-float column would undercount the row stride and shift every
        // following vertex's bytes.
        let bytes = "ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nproperty uchar red\nend_header\n"
            .to_string()
            .into_bytes();
        let err = parse_ply(&bytes[..]).unwrap_err().to_string();
        assert!(err.contains("Unsupported property type"), "{err}");
    }
}
