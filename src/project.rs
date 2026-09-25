//! Project splats to 2D.
//!
//! References:
//! - <https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu>
//! - <https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/rasterizer_impl.cu>
use crate::layout::{
    PLANE_OPACITY, PLANE_QW, PLANE_QX, PLANE_QY, PLANE_QZ, PLANE_SX, PLANE_SY, PLANE_SZ, PLANE_X,
    PLANE_Y, PLANE_Z, PROJ_FLOATS, TILE_WIDTH, Vec2F, Vec3F,
};
use cubecl::prelude::*;

const ALPHA_CUTOFF: f32 = 10.0 / u8::MAX as f32;

#[derive(CubeType, Clone, Copy)]
#[expand(derive(Clone, Copy))]
struct Vec4F {
    pub w: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(CubeType, Clone, Copy)]
struct TileBBox {
    pub min_x: u32,
    pub min_y: u32,
    pub max_x: u32,
    pub max_y: u32,
}

#[cube]
fn sigmoid(x: f32) -> f32 {
    1.0f32 / (1.0f32 + (-x).exp())
}

#[cube]
fn tile_bbox(mean: Vec2F, ext: Vec2F, bounds: Vec2F) -> TileBBox {
    let inv_tile = 1.0f32 / TILE_WIDTH as f32;
    TileBBox {
        min_x: ((mean.x - ext.x) * inv_tile).clamp(0.0, bounds.x) as u32,
        min_y: ((mean.y - ext.y) * inv_tile).clamp(0.0, bounds.y) as u32,
        max_x: ((mean.x + ext.x) * inv_tile + 1.0).clamp(0.0, bounds.x) as u32,
        max_y: ((mean.y + ext.y) * inv_tile + 1.0).clamp(0.0, bounds.y) as u32,
    }
}

// counters layout: [total intersection emissions, visible splats]

/// Full float-precision depth key: camera-space z is always positive here, so
/// the raw bit pattern sorts depth monotonically, and keys tie only on exact
/// float equality. Truncated keys once tied whole surface patches, and the
/// racy atomic compaction order re-rolled the blend order of ties on every
/// launch — visible as heavy flicker on camera-neutral repaints.
#[cube]
fn depth_key(z: f32) -> u32 {
    z.to_bits()
}

/// Per-frame camera state, passed as kernel scalars. `rot`/`trans` are the
/// world-to-camera transform; `world_pos` is the camera position in world
/// space (the SH evaluation direction).
#[derive(CubeType, CubeLaunch, Clone, Copy)]
#[expand(derive(Clone, Copy))]
pub(crate) struct CameraView {
    pub rot0: Vec3F,
    pub rot1: Vec3F,
    pub rot2: Vec3F,
    pub trans: Vec3F,
    pub focal: Vec2F,
    pub world_pos: Vec3F,
    pub img: Vec2F,
    pub tile_bounds: Vec2F,
}

impl CameraViewLaunch {
    /// Kernel view of `camera` for an `img_size` frame of `tile_bounds` tiles.
    pub(crate) fn for_camera(
        camera: &crate::camera::Camera,
        img_size: glam::UVec2,
        tile_bounds: glam::UVec2,
    ) -> Self {
        let w2c = camera.w2c();
        let rot = w2c.matrix3.transpose();
        Self::new(
            rot.x_axis.into(),
            rot.y_axis.into(),
            rot.z_axis.into(),
            w2c.translation.into(),
            camera.focal(img_size).into(),
            camera.position.into(),
            img_size.as_vec2().into(),
            tile_bounds.as_vec2().into(),
        )
    }
}

#[cube]
fn dot3(a: Vec3F, b: Vec3F) -> f32 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

#[cube]
fn normalize(v: Vec3F) -> Vec3F {
    let inv = dot3(v, v).inverse_sqrt();
    Vec3F {
        x: v.x * inv,
        y: v.y * inv,
        z: v.z * inv,
    }
}

/// Camera-space position: the w2c rotation rows dotted with `pos`, plus the
/// w2c translation.
#[cube]
fn to_camera_space(view: &CameraView, pos: Vec3F) -> Vec3F {
    Vec3F {
        x: dot3(view.rot0, pos) + view.trans.x,
        y: dot3(view.rot1, pos) + view.trans.y,
        z: dot3(view.rot2, pos) + view.trans.z,
    }
}

#[cube]
fn quat_to_rotation(q: Vec4F) -> (Vec3F, Vec3F, Vec3F) {
    let x2 = q.x * q.x;
    let y2 = q.y * q.y;
    let z2 = q.z * q.z;
    let xy = q.x * q.y;
    let xz = q.x * q.z;
    let yz = q.y * q.z;
    let wx = q.w * q.x;
    let wy = q.w * q.y;
    let wz = q.w * q.z;

    (
        Vec3F {
            x: 1.0f32 - 2.0f32 * (y2 + z2),
            y: 2.0f32 * (xy - wz),
            z: 2.0f32 * (xz + wy),
        },
        Vec3F {
            x: 2.0f32 * (xy + wz),
            y: 1.0f32 - 2.0f32 * (x2 + z2),
            z: 2.0f32 * (yz - wx),
        },
        Vec3F {
            x: 2.0f32 * (xz - wy),
            y: 2.0f32 * (yz + wx),
            z: 1.0f32 - 2.0f32 * (x2 + y2),
        },
    )
}

#[cube]
fn scale_components(v: Vec3F, s: Vec3F) -> Vec3F {
    Vec3F {
        x: v.x * s.x,
        y: v.y * s.y,
        z: v.z * s.z,
    }
}

/// Vector-matrix product: `v` dotted with each column of the row-major
/// `[m0; m1; m2]`.
#[cube]
fn mat_vec(v: Vec3F, m0: Vec3F, m1: Vec3F, m2: Vec3F) -> Vec3F {
    Vec3F {
        x: v.x * m0.x + v.y * m1.x + v.z * m2.x,
        y: v.x * m0.y + v.y * m1.y + v.z * m2.y,
        z: v.x * m0.z + v.y * m1.z + v.z * m2.z,
    }
}

#[cube]
fn compute_cov2d(
    scale: Vec3F,
    quat: Vec4F,
    view: &CameraView,
    cam: Vec3F,
    inv_cam_z: f32,
) -> Vec3F {
    let rot0 = view.rot0;
    let rot1 = view.rot1;
    let rot2 = view.rot2;

    let (r0, r1, r2) = quat_to_rotation(quat);
    let m0 = scale_components(r0, scale);
    let m1 = scale_components(r1, scale);
    let m2 = scale_components(r2, scale);

    let lim_x = 1.3f32 * view.img.x / (2.0f32 * view.focal.x);
    let lim_y = 1.3f32 * view.img.y / (2.0f32 * view.focal.y);
    let u = (cam.x * inv_cam_z).clamp(-lim_x, lim_x);
    let v = (cam.y * inv_cam_z).clamp(-lim_y, lim_y);

    let fx_inv_z = view.focal.x * inv_cam_z;
    let fu_inv_z = fx_inv_z * u;
    let t0 = Vec3F {
        x: fx_inv_z * rot0.x - fu_inv_z * rot2.x,
        y: fx_inv_z * rot0.y - fu_inv_z * rot2.y,
        z: fx_inv_z * rot0.z - fu_inv_z * rot2.z,
    };

    let fy_inv_z = view.focal.y * inv_cam_z;
    let fv_inv_z = fy_inv_z * v;
    let t1 = Vec3F {
        x: fy_inv_z * rot1.x - fv_inv_z * rot2.x,
        y: fy_inv_z * rot1.y - fv_inv_z * rot2.y,
        z: fy_inv_z * rot1.z - fv_inv_z * rot2.z,
    };

    let j0 = mat_vec(t0, m0, m1, m2);
    let j1 = mat_vec(t1, m0, m1, m2);

    Vec3F {
        x: dot3(j0, j0) + 0.3,
        y: dot3(j0, j1),
        z: dot3(j1, j1) + 0.3,
    }
}

#[cube]
fn compute_conic(a: f32, b: f32, c: f32) -> Vec3F {
    let det = (a * c - b * b).max(1e-6);
    let inv_det = det.recip();
    Vec3F {
        x: c * inv_det,
        y: -b * inv_det,
        z: a * inv_det,
    }
}

#[cube(launch)]
pub(crate) fn project_splats(
    view: CameraView,
    attributes: &[f32],
    sh_coeffs: &[f32],
    #[comptime] sh_per_ch: u32,
    depth_order: &mut [u32],
    depth_keys: &mut [u32],
    projected_splats: &mut [f32],
    counters: &[Atomic<u32>],
    tile_counts: &mut [u32],
    packed_bbox: &mut [u32],
) {
    if ABSOLUTE_POS_X >= depth_order.len() as u32 {
        terminate!();
    }
    let n = depth_order.len();
    let i = ABSOLUTE_POS_X as usize;
    let mean = Vec3F {
        x: attributes[PLANE_X * n + i],
        y: attributes[PLANE_Y * n + i],
        z: attributes[PLANE_Z * n + i],
    };
    let quat = Vec4F {
        w: attributes[PLANE_QW * n + i],
        x: attributes[PLANE_QX * n + i],
        y: attributes[PLANE_QY * n + i],
        z: attributes[PLANE_QZ * n + i],
    };
    let scale = Vec3F {
        x: attributes[PLANE_SX * n + i].exp(),
        y: attributes[PLANE_SY * n + i].exp(),
        z: attributes[PLANE_SZ * n + i].exp(),
    };
    let opacity = sigmoid(attributes[PLANE_OPACITY * n + i]);
    if opacity < ALPHA_CUTOFF {
        terminate!();
    }

    let cam = to_camera_space(&view, mean);
    if cam.z <= 0.1f32 {
        terminate!();
    }
    let inv_cam_z = cam.z.recip();
    let cov2d = compute_cov2d(scale, quat, &view, cam, inv_cam_z);
    let conic = compute_conic(cov2d.x, cov2d.y, cov2d.z);

    let vis_slot = counters[1].fetch_add(1u32);
    depth_order[vis_slot as usize] = ABSOLUTE_POS_X;
    depth_keys[vis_slot as usize] = depth_key(cam.z);

    let dir = Vec3F {
        x: mean.x - view.world_pos.x,
        y: mean.y - view.world_pos.y,
        z: mean.z - view.world_pos.z,
    };
    let (r, g, b) = sh_to_rgb(
        sh_per_ch,
        normalize(dir),
        ABSOLUTE_POS_X,
        n as u32,
        sh_coeffs,
    );

    let mean2d = Vec2F {
        x: view.focal.x * cam.x * inv_cam_z + view.img.x * 0.5,
        y: view.focal.y * cam.y * inv_cam_z + view.img.y * 0.5,
    };
    let out_base = i * PROJ_FLOATS;
    projected_splats[out_base] = mean2d.x;
    projected_splats[out_base + 1] = mean2d.y;
    projected_splats[out_base + 2] = conic.x;
    projected_splats[out_base + 3] = conic.y;
    projected_splats[out_base + 4] = conic.z;
    projected_splats[out_base + 5] = r + 0.5f32;
    projected_splats[out_base + 6] = g + 0.5f32;
    projected_splats[out_base + 7] = b + 0.5f32;
    projected_splats[out_base + 8] = opacity;

    let cutoff = (opacity / ALPHA_CUTOFF).ln();
    let ext = Vec2F {
        x: (2.0 * cutoff * cov2d.x).sqrt(),
        y: (2.0 * cutoff * cov2d.z).sqrt(),
    };
    let bb = tile_bbox(mean2d, ext, view.tile_bounds);
    // Count, don't emit: the map kernel re-emits from this exact packed bbox
    // at prefix-sum offsets, so count and emission can never diverge.
    let num_tiles = (bb.max_x - bb.min_x) * (bb.max_y - bb.min_y);
    tile_counts[i] = num_tiles;
    counters[0].fetch_add(num_tiles);
    let pack = i * 2;
    packed_bbox[pack] = bb.min_x | (bb.min_y << 16u32);
    packed_bbox[pack + 1] = bb.max_x | (bb.max_y << 16u32);
}

pub(crate) const SH_C0: f32 = 0.282_094_8_f32;

/// Display color → SH DC coefficient: rgb = C0·f_dc + 0.5 — the palette
/// conversion shared by the GUI's tints and the seg tests.
pub fn to_dc(color: [f32; 3]) -> [f32; 3] {
    color.map(|c| (c - 0.5) / SH_C0)
}
const SH_C1: f32 = 0.488_602_52_f32;
const SH_C2_0: f32 = 0.946_174_7_f32;
const SH_C2_1: f32 = 0.315_391_57_f32;
const SH_C2_2: f32 = -1.092_548_5_f32;
const SH_C2_3: f32 = 0.546_274_24_f32;

#[rustfmt::skip]
#[cube]
fn sh_to_rgb(chs: u32, dir: Vec3F, splat: u32, n: u32, shs: &[f32]) -> (f32, f32, f32) {
    let s = splat as usize;
    let stride = n as usize;
    let mut r = SH_C0 * shs[s];
    let mut g = SH_C0 * shs[stride + s];
    let mut b = SH_C0 * shs[2 * stride + s];

    if chs >= 4 {
        r += SH_C1 * (-dir.y * shs[3 * stride + s] + dir.z * shs[6 * stride + s] - dir.x * shs[9 * stride + s]);
        g += SH_C1 * (-dir.y * shs[4 * stride + s] + dir.z * shs[7 * stride + s] - dir.x * shs[10 * stride + s]);
        b += SH_C1 * (-dir.y * shs[5 * stride + s] + dir.z * shs[8 * stride + s] - dir.x * shs[11 * stride + s]);
    }

    if chs >= 9 {
        let b4 = SH_C2_2 * dir.z * dir.x;
        let b5 = SH_C2_2 * dir.z * dir.y;
        let b6 = SH_C2_0 * dir.z * dir.z - SH_C2_1;
        let b7 = SH_C2_3 * 2.0 * dir.x * dir.y;
        let b8 = SH_C2_3 * (dir.x * dir.x - dir.y * dir.y);

        r += b6 * shs[18 * stride + s] + b7 * shs[12 * stride + s] + b5 * shs[15 * stride + s] + b4 * shs[21 * stride + s] + b8 * shs[24 * stride + s];
        g += b6 * shs[19 * stride + s] + b7 * shs[13 * stride + s] + b5 * shs[16 * stride + s] + b4 * shs[22 * stride + s] + b8 * shs[25 * stride + s];
        b += b6 * shs[20 * stride + s] + b7 * shs[14 * stride + s] + b5 * shs[17 * stride + s] + b4 * shs[23 * stride + s] + b8 * shs[26 * stride + s];
    }

    if chs >= 16 {
        let sh_c1x = dir.x * dir.x - dir.y * dir.y;
        let sh_c1y = 2.0f32 * dir.x * dir.y;
        let tmp0c = -2.285_229f32 * dir.z * dir.z + 0.457_045_8;
        let tmp1b = 1.445_305_7f32 * dir.z;
        let b9 = -0.590_043_6f32 * dir.x * sh_c1y + dir.y * sh_c1x;
        let b10 = -0.590_043_6f32 * dir.x * sh_c1x - dir.y * sh_c1y;
        let b11 = tmp0c * dir.y;
        let b12 = tmp0c * dir.x;
        let b13 = dir.z * (1.865_881_7f32 * dir.z * dir.z - 1.119_529);
        let b14 = tmp1b * sh_c1y;
        let b15 = tmp1b * sh_c1x;

        r += b9 * shs[27 * stride + s] + b10 * shs[30 * stride + s] + b11 * shs[33 * stride + s] + b12 * shs[36 * stride + s] + b13 * shs[39 * stride + s] + b14 * shs[42 * stride + s] + b15 * shs[45 * stride + s];
        g += b9 * shs[28 * stride + s] + b10 * shs[31 * stride + s] + b11 * shs[34 * stride + s] + b12 * shs[37 * stride + s] + b13 * shs[40 * stride + s] + b14 * shs[43 * stride + s] + b15 * shs[46 * stride + s];
        b += b9 * shs[29 * stride + s] + b10 * shs[32 * stride + s] + b11 * shs[35 * stride + s] + b12 * shs[38 * stride + s] + b13 * shs[41 * stride + s] + b14 * shs[44 * stride + s] + b15 * shs[47 * stride + s];
    }

    (r, g, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::calculate_cube_count_elemwise;
    use splat_sort::tensor::GpuTensor;

    const SENTINEL: u32 = 0xDEAD_BEEF;

    #[test]
    fn test_project_counts_tiles_and_packs_bbox() {
        let (_gpu, client) = crate::gpu_testing::test_client();
        // One splat at the image center, scale ~1, near-full opacity: with a
        // 64x64 image and 16x16 tiles its bbox covers all 16 tiles.
        let attributes: Vec<f32> = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0];
        let sh: Vec<f32> = vec![0.0; 3];
        let attrs_t = GpuTensor::from(&client, [1, 11], &attributes[..]);
        let sh_t = GpuTensor::from(&client, [1, 1, 3], &sh[..]);
        let depth_order = GpuTensor::empty(&client, [1]);
        let depth_keys = GpuTensor::empty(&client, [1]);
        let projected = GpuTensor::empty(&client, [1, 9]);
        let counters = GpuTensor::from(&client, [2], &[0u32, 0][..]);
        let tile_counts = GpuTensor::from(&client, [1], &[SENTINEL][..]);
        let tile_bbox = GpuTensor::from(&client, [1, 2], &[SENTINEL; 2][..]);

        project_splats::launch(
            &client,
            calculate_cube_count_elemwise(&client, 1, CubeDim::new_1d(256)),
            CubeDim::new_1d(256),
            CameraViewLaunch::for_camera(
                &crate::camera::Camera::default(),
                glam::uvec2(64, 64),
                glam::uvec2(4, 4),
            ),
            attrs_t.as_buffer_arg(),
            sh_t.as_buffer_arg(),
            1,
            depth_order.as_buffer_arg(),
            depth_keys.as_buffer_arg(),
            projected.as_buffer_arg(),
            counters.as_buffer_arg(),
            tile_counts.as_buffer_arg(),
            tile_bbox.as_buffer_arg(),
        );

        let c: Vec<u32> = counters.read_vec();
        assert_eq!(c[1], 1, "one visible splat");
        assert_eq!(c[0], 16, "all 16 tiles counted");
        let tc: Vec<u32> = tile_counts.read_vec();
        assert_eq!(tc, vec![16]);
        let bb: Vec<u32> = tile_bbox.read_vec();
        assert_eq!(
            bb,
            vec![0u32, 4 | 4 << 16],
            "bbox must cover the full 4x4 tile grid"
        );
    }
}
