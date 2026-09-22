use crate::layout::{
    ATTR_PLANES, PLANE_OPACITY, PLANE_QW, PLANE_QX, PLANE_QY, PLANE_QZ, PLANE_SX, PLANE_SY,
    PLANE_SZ, PLANE_X, PLANE_Y, PLANE_Z,
};
use crate::render::CpuSplats;
use anyhow::{Context, Result, anyhow, bail, ensure};
use std::io::{BufRead, BufWriter, Write};

pub(crate) fn parse_ply(mut reader: impl BufRead) -> Result<CpuSplats> {
    let mut vertex_count: usize = 0;
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
        bail!("PLY contains no vertices");
    }

    let get_idx = |name: &str| {
        properties
            .iter()
            .position(|p| p == name)
            .ok_or_else(|| anyhow!("Missing property: {name}"))
    };

    // Checked left-to-right: the first missing name is the one reported.
    #[rustfmt::skip]
    let [idx_x, idx_y, idx_z, idx_s0, idx_s1, idx_s2, idx_op, idx_r0, idx_r1, idx_r2, idx_r3, idx_dc0, idx_dc1, idx_dc2] =
        ["x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity", "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2"]
            .into_iter()
            .map(get_idx)
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .unwrap();

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

    let stride = properties.len();

    // Guard absurd headers before allocating the output planes.
    vertex_count
        .checked_mul(stride)
        .context("PLY vertex count overflows address space")?;

    let n = rest_keys.len() / 3;
    let mut attributes = vec![0f32; vertex_count * ATTR_PLANES];
    let mut shs = vec![0f32; vertex_count * (rest_keys.len() + 3)];

    // Row-by-row streaming: load peaks at the outputs alone (270 → 140 MiB on a 130 MiB file).
    let mut row = vec![0f32; stride];
    for i in 0..vertex_count {
        reader
            .read_exact(bytemuck::cast_slice_mut(&mut row))
            .with_context(|| format!("failed to read vertex {i} of {vertex_count}"))?;

        let q =
            glam::Quat::from_xyzw(row[idx_r1], row[idx_r2], row[idx_r3], row[idx_r0]).normalize();
        attributes[PLANE_X * vertex_count + i] = row[idx_x];
        attributes[PLANE_Y * vertex_count + i] = row[idx_y];
        attributes[PLANE_Z * vertex_count + i] = row[idx_z];
        attributes[PLANE_QW * vertex_count + i] = q.w;
        attributes[PLANE_QX * vertex_count + i] = q.x;
        attributes[PLANE_QY * vertex_count + i] = q.y;
        attributes[PLANE_QZ * vertex_count + i] = q.z;
        attributes[PLANE_SX * vertex_count + i] = row[idx_s0];
        attributes[PLANE_SY * vertex_count + i] = row[idx_s1];
        attributes[PLANE_SZ * vertex_count + i] = row[idx_s2];
        attributes[PLANE_OPACITY * vertex_count + i] = row[idx_op];

        shs[i] = row[idx_dc0];
        shs[vertex_count + i] = row[idx_dc1];
        shs[2 * vertex_count + i] = row[idx_dc2];
        for j in 0..n {
            shs[((j + 1) * 3) * vertex_count + i] = row[rest_keys[j].0];
            shs[((j + 1) * 3 + 1) * vertex_count + i] = row[rest_keys[n + j].0];
            shs[((j + 1) * 3 + 2) * vertex_count + i] = row[rest_keys[2 * n + j].0];
        }
    }

    Ok(CpuSplats {
        attributes,
        sh_coeffs: shs,
    })
}

impl CpuSplats {
    /// Binary little-endian 3DGS PLY of this scene — the exact inverse of
    /// `parse_ply`'s input contract: canonical INRIA property order, zero
    /// normals, rot with w first, `f_rest` channel-major. The fields are
    /// exactly what `parse_ply` produces: `attributes` field-major
    /// [`ATTR_PLANES`] planes, `sh_coeffs` field-major 3·`k_per_ch` planes.
    pub fn write_ply(&self, out: impl Write) -> Result<()> {
        let n = self.count();
        ensure!(n > 0, "no splats to write");
        let k_per_ch = self.sh_coeffs.len() / (3 * n);
        let attr = &self.attributes;
        let sh = &self.sh_coeffs;
        ensure!(attr.len() == n * ATTR_PLANES);
        ensure!(sh.len() == n * k_per_ch * 3);
        ensure!(k_per_ch > 0);
        let rest = (k_per_ch - 1) * 3;
        let floats = 17 + rest; // xyz nnn dc3 rest op1 scale3 rot4

        let mut props = Vec::from(
            [
                "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
            ]
            .map(String::from),
        );
        props.extend((0..rest).map(|j| format!("f_rest_{j}")));
        props.extend(
            [
                "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
            ]
            .map(String::from),
        );
        let mut header = String::with_capacity(1024 + rest * 24);
        header.push_str("ply\nformat binary_little_endian 1.0\nelement vertex ");
        header.push_str(&n.to_string());
        header.push('\n');
        header.extend(props.iter().map(|p| format!("property float {p}\n")));
        header.push_str("end_header\n");

        let mut out = BufWriter::new(out);
        out.write_all(header.as_bytes())?;
        let mut row = vec![0f32; floats];
        for i in 0..n {
            let a = |plane: usize| attr[plane * n + i];
            row[0] = a(PLANE_X);
            row[1] = a(PLANE_Y);
            row[2] = a(PLANE_Z);
            row[3..6].fill(0.0);
            row[6] = sh[i];
            row[7] = sh[n + i];
            row[8] = sh[2 * n + i];
            for j in 0..rest {
                let (c, coef) = (j / (rest / 3), j % (rest / 3));
                row[9 + j] = sh[((coef + 1) * 3 + c) * n + i];
            }
            row[9 + rest] = a(PLANE_OPACITY);
            row[10 + rest] = a(PLANE_SX);
            row[11 + rest] = a(PLANE_SY);
            row[12 + rest] = a(PLANE_SZ);
            row[13 + rest] = a(PLANE_QW);
            row[14 + rest] = a(PLANE_QX);
            row[15 + rest] = a(PLANE_QY);
            row[16 + rest] = a(PLANE_QZ);
            out.write_all(bytemuck::cast_slice(&row))?;
        }
        out.flush()?;
        Ok(())
    }
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

    /// Writer→parser roundtrip: a written file parses back to the exact
    /// planes, for every SH degree in use. Rotation must be stored
    /// normalized — parse re-normalizes on read.
    #[test]
    fn test_write_ply_roundtrips_parse() {
        for k_per_ch in [1, 3] {
            let n = 3usize;
            let mut attr = vec![0f32; n * ATTR_PLANES];
            for (i, a) in attr.iter_mut().enumerate() {
                *a = ((i * 7 % 23) as f32 - 11.0) / 8.0;
            }
            for i in 0..n {
                attr[PLANE_QW * n + i] = 1.0;
                attr[PLANE_QX * n + i] = 0.0;
                attr[PLANE_QY * n + i] = 0.0;
                attr[PLANE_QZ * n + i] = 0.0;
            }
            let sh: Vec<f32> = (0..n * k_per_ch * 3)
                .map(|i| i as f32 * 0.25 - 9.0)
                .collect();

            let scene = CpuSplats {
                attributes: attr,
                sh_coeffs: sh,
            };
            let mut bytes = Vec::new();
            scene.write_ply(&mut bytes).unwrap();
            let cpu = parse_ply(&bytes[..]).unwrap();
            assert_eq!(cpu.attributes, scene.attributes);
            assert_eq!(cpu.sh_coeffs, scene.sh_coeffs);
        }
    }
}
