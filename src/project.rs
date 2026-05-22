//! Project splats to 2D.
//!
//! References:
//! - <https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu>
//! - <https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/rasterizer_impl.cu>
use crate::helpers::{self, Covariance3D, Mat3, Vec2F, Vec3F, Vec4F};
use cubecl::prelude::*;

const ALPHA_CUTOFF: f32 = 10.0 / 255.0;

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

#[cube]
fn sym_mul_row(row: Vec3F, s: Covariance3D) -> Vec3F {
    Vec3F {
        x: row.x * s.xx + row.y * s.xy + row.z * s.xz,
        y: row.x * s.xy + row.y * s.yy + row.z * s.yz,
        z: row.x * s.xz + row.y * s.yz + row.z * s.zz,
    }
}

#[cube]
fn to_camera_space(viewmat: &Array<f32>, pos: Vec3F) -> (Vec3F, Mat3) {
    let rot = Mat3 {
        row0: Vec3F {
            x: viewmat[0],
            y: viewmat[1],
            z: viewmat[2],
        },
        row1: Vec3F {
            x: viewmat[4],
            y: viewmat[5],
            z: viewmat[6],
        },
        row2: Vec3F {
            x: viewmat[8],
            y: viewmat[9],
            z: viewmat[10],
        },
    };

    let cam = Vec3F {
        x: dot3(rot.row0, pos) + viewmat[3],
        y: dot3(rot.row1, pos) + viewmat[7],
        z: dot3(rot.row2, pos) + viewmat[11],
    };

    (cam, rot)
}

#[cube]
fn quat_to_rotation(q: Vec4F) -> Mat3 {
    let x2 = q.x * q.x;
    let y2 = q.y * q.y;
    let z2 = q.z * q.z;
    let xy = q.x * q.y;
    let xz = q.x * q.z;
    let yz = q.y * q.z;
    let wx = q.w * q.x;
    let wy = q.w * q.y;
    let wz = q.w * q.z;

    Mat3 {
        row0: Vec3F {
            x: 1.0 - 2.0 * (y2 + z2),
            y: 2.0 * (xy - wz),
            z: 2.0 * (xz + wy),
        },
        row1: Vec3F {
            x: 2.0 * (xy + wz),
            y: 1.0 - 2.0 * (x2 + z2),
            z: 2.0 * (yz - wx),
        },
        row2: Vec3F {
            x: 2.0 * (xz - wy),
            y: 2.0 * (yz + wx),
            z: 1.0 - 2.0 * (x2 + y2),
        },
    }
}

#[cube]
fn scale_components(v: Vec3F, s: Vec3F) -> Vec3F {
    Vec3F {
        x: v.x * s.x,
        y: v.y * s.y,
        z: v.z * s.z,
    }
}

#[cube]
fn compute_cov3d(scale: Vec3F, quat: Vec4F) -> Covariance3D {
    let r = quat_to_rotation(quat);
    let m0 = scale_components(r.row0, scale);
    let m1 = scale_components(r.row1, scale);
    let m2 = scale_components(r.row2, scale);
    Covariance3D {
        xx: dot3(m0, m0),
        xy: dot3(m0, m1),
        xz: dot3(m0, m2),
        yy: dot3(m1, m1),
        yz: dot3(m1, m2),
        zz: dot3(m2, m2),
    }
}

#[cube]
fn compute_cov2d(cov3d: Covariance3D, rot: &Mat3, focal: Vec2F, cam: Vec3F, img: Vec2F) -> Vec3F {
    let inv_cam_z = cam.z.recip();
    let lim_x = 1.3 * img.x / (2.0 * focal.x);
    let lim_y = 1.3 * img.y / (2.0 * focal.y);
    let u = (cam.x * inv_cam_z).clamp(-lim_x, lim_x);
    let v = (cam.y * inv_cam_z).clamp(-lim_y, lim_y);

    let fx_inv_z = focal.x * inv_cam_z;
    let fu_inv_z = fx_inv_z * u;
    let t_r0 = Vec3F {
        x: fx_inv_z * rot.row0.x - fu_inv_z * rot.row2.x,
        y: fx_inv_z * rot.row0.y - fu_inv_z * rot.row2.y,
        z: fx_inv_z * rot.row0.z - fu_inv_z * rot.row2.z,
    };

    let fy_inv_z = focal.y * inv_cam_z;
    let fv_inv_z = fy_inv_z * v;
    let t_r1 = Vec3F {
        x: fy_inv_z * rot.row1.x - fv_inv_z * rot.row2.x,
        y: fy_inv_z * rot.row1.y - fv_inv_z * rot.row2.y,
        z: fy_inv_z * rot.row1.z - fv_inv_z * rot.row2.z,
    };

    let jc_r0 = sym_mul_row(t_r0, cov3d);
    let jc_r1 = sym_mul_row(t_r1, cov3d);

    Vec3F {
        x: dot3(jc_r0, t_r0) + 0.3,
        y: dot3(jc_r0, t_r1),
        z: dot3(jc_r1, t_r1) + 0.3,
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
    viewmat: &Array<f32>,
    focal: Vec2F,
    camera_pos: Vec3F,
    attributes: &Array<f32>,
    sh_coeffs: &Array<f32>,
    sh_per_ch: u32,
    tile_bounds: Vec2F,
    img_size: Vec2F,
    depth_order: &mut Array<u32>,
    depth_keys: &mut Array<u32>,
    projected_splats: &mut Array<f32>,
    counters: &Array<Atomic<u32>>,
    tile_ids: &mut Array<u32>,
    gaussian_ids: &mut Array<u32>,
) {
    if ABSOLUTE_POS_X < depth_order.len() as u32 {
        // Splat attributes layout: [x, y, z, qw, qx, qy, qz, sx, sy, sz, opacity]
        let base = (ABSOLUTE_POS_X * 11u32) as usize;
        let mean = Vec3F {
            x: attributes[base],
            y: attributes[base + 1],
            z: attributes[base + 2],
        };
        let quat = Vec4F {
            w: attributes[base + 3],
            x: attributes[base + 4],
            y: attributes[base + 5],
            z: attributes[base + 6],
        };
        let scale = Vec3F {
            x: attributes[base + 7].exp(),
            y: attributes[base + 8].exp(),
            z: attributes[base + 9].exp(),
        };
        let opacity = helpers::sigmoid(attributes[base + 10]);
        if opacity < ALPHA_CUTOFF {
            terminate!();
        }

        let (cam, rot) = to_camera_space(viewmat, mean);
        if cam.z <= 0.1f32 {
            terminate!();
        }
        let cov3d = compute_cov3d(scale, quat);
        let cov2d = compute_cov2d(cov3d, &rot, focal, cam, img_size);
        let conic = compute_conic(cov2d.x, cov2d.y, cov2d.z);

        let vis_slot = counters[1].fetch_add(1u32);
        depth_order[vis_slot as usize] = vis_slot;
        depth_keys[vis_slot as usize] = cam.z.to_bits();

        let dir = Vec3F {
            x: mean.x - camera_pos.x,
            y: mean.y - camera_pos.y,
            z: mean.z - camera_pos.z,
        };
        let (r, g, b) = helpers::sh_to_rgb(sh_per_ch, normalize(dir), sh_coeffs);

        let inv_cam_z = cam.z.recip();
        let mean2d = Vec2F {
            x: focal.x * cam.x * inv_cam_z + img_size.x * 0.5,
            y: focal.y * cam.y * inv_cam_z + img_size.y * 0.5,
        };
        let out_base = vis_slot as usize * 9;
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
        let tile_bbox = helpers::tile_bbox(mean2d, ext, tile_bounds);
        for ty in tile_bbox.min_y..tile_bbox.max_y {
            for tx in tile_bbox.min_x..tile_bbox.max_x {
                let slot = counters[0].fetch_add(1u32);
                tile_ids[slot as usize] = tx + ty * tile_bounds.x as u32;
                gaussian_ids[slot as usize] = vis_slot;
            }
        }
    }
}
