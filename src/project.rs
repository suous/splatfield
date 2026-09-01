//! Project splats to 2D.
//!
//! References:
//! - <https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu>
//! - <https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/rasterizer_impl.cu>
use crate::helpers::{self, Mat3, Vec2F, Vec3F, Vec4F};
use cubecl::prelude::*;

const ALPHA_CUTOFF: f32 = 10.0 / u8::MAX as f32;

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
fn to_camera_space(viewmat: &[f32], pos: Vec3F) -> (Vec3F, Mat3) {
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
            x: 1.0f32 - 2.0f32 * (y2 + z2),
            y: 2.0f32 * (xy - wz),
            z: 2.0f32 * (xz + wy),
        },
        row1: Vec3F {
            x: 2.0f32 * (xy + wz),
            y: 1.0f32 - 2.0f32 * (x2 + z2),
            z: 2.0f32 * (yz - wx),
        },
        row2: Vec3F {
            x: 2.0f32 * (xz - wy),
            y: 2.0f32 * (yz + wx),
            z: 1.0f32 - 2.0f32 * (x2 + y2),
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
fn compute_cov2d(
    scale: Vec3F,
    quat: Vec4F,
    rot: &Mat3,
    focal: Vec2F,
    cam: Vec3F,
    img: Vec2F,
) -> Vec3F {
    let r = quat_to_rotation(quat);
    let m0 = scale_components(r.row0, scale);
    let m1 = scale_components(r.row1, scale);
    let m2 = scale_components(r.row2, scale);

    let inv_cam_z = cam.z.recip();
    let lim_x = 1.3f32 * img.x / (2.0f32 * focal.x);
    let lim_y = 1.3f32 * img.y / (2.0f32 * focal.y);
    let u = (cam.x * inv_cam_z).clamp(-lim_x, lim_x);
    let v = (cam.y * inv_cam_z).clamp(-lim_y, lim_y);

    let fx_inv_z = focal.x * inv_cam_z;
    let fu_inv_z = fx_inv_z * u;
    let t0 = Vec3F {
        x: fx_inv_z * rot.row0.x - fu_inv_z * rot.row2.x,
        y: fx_inv_z * rot.row0.y - fu_inv_z * rot.row2.y,
        z: fx_inv_z * rot.row0.z - fu_inv_z * rot.row2.z,
    };

    let fy_inv_z = focal.y * inv_cam_z;
    let fv_inv_z = fy_inv_z * v;
    let t1 = Vec3F {
        x: fy_inv_z * rot.row1.x - fv_inv_z * rot.row2.x,
        y: fy_inv_z * rot.row1.y - fv_inv_z * rot.row2.y,
        z: fy_inv_z * rot.row1.z - fv_inv_z * rot.row2.z,
    };

    let j0 = Vec3F {
        x: t0.x * m0.x + t0.y * m1.x + t0.z * m2.x,
        y: t0.x * m0.y + t0.y * m1.y + t0.z * m2.y,
        z: t0.x * m0.z + t0.y * m1.z + t0.z * m2.z,
    };
    let j1 = Vec3F {
        x: t1.x * m0.x + t1.y * m1.x + t1.z * m2.x,
        y: t1.x * m0.y + t1.y * m1.y + t1.z * m2.y,
        z: t1.x * m0.z + t1.y * m1.z + t1.z * m2.z,
    };

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
    viewmat: &[f32],
    focal: Vec2F,
    camera_pos: Vec3F,
    attributes: &[f32],
    sh_coeffs: &[f32],
    sh_per_ch: u32,
    tile_bounds: Vec2F,
    img_size: Vec2F,
    depth_order: &mut [u32],
    depth_keys: &mut [u32],
    projected_splats: &mut [f32],
    counters: &[Atomic<u32>],
    max_isects: u32,
    tile_ids: &mut [u32],
    gaussian_ids: &mut [u32],
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
        let cov2d = compute_cov2d(scale, quat, &rot, focal, cam, img_size);
        let conic = compute_conic(cov2d.x, cov2d.y, cov2d.z);

        let vis_slot = counters[1].fetch_add(1u32);
        depth_order[vis_slot as usize] = vis_slot;
        depth_keys[vis_slot as usize] = cam.z.to_bits() >> 4;

        let dir = Vec3F {
            x: mean.x - camera_pos.x,
            y: mean.y - camera_pos.y,
            z: mean.z - camera_pos.z,
        };
        let (r, g, b) = helpers::sh_to_rgb(sh_per_ch, normalize(dir), ABSOLUTE_POS_X, sh_coeffs);

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
                if slot < max_isects {
                    tile_ids[slot as usize] = tx + ty * tile_bounds.x as u32;
                    gaussian_ids[slot as usize] = vis_slot;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::{Vec2FLaunch, Vec3FLaunch};
    use crate::tensor::GpuTensor;
    use cubecl::calculate_cube_count_elemwise;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    const SENTINEL: u32 = 0xDEAD_BEEF;

    #[test]
    fn test_project_respects_isect_cap() {
        let client = WgpuRuntime::client(&WgpuDevice::default());
        // One splat at the image center, scale ~1, near-full opacity: with a
        // 64x64 image and 16x16 tiles it intersects all 16 tiles.
        let attributes: Vec<f32> = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0];
        let sh: Vec<f32> = vec![0.0; 3];
        let attrs_t = GpuTensor::from(&client, [1, 11], &attributes[..]);
        let sh_t = GpuTensor::from(&client, [1, 1, 3], &sh[..]);
        let depth_order = GpuTensor::empty(&client, [1]);
        let depth_keys = GpuTensor::empty(&client, [1]);
        let projected = GpuTensor::empty(&client, [1, 9]);
        let counters = GpuTensor::from(&client, [2], &[0u32, 0][..]);
        let tile_ids = GpuTensor::from(&client, [32], &vec![SENTINEL; 32][..]);
        let gaussian_ids = GpuTensor::from(&client, [32], &vec![SENTINEL; 32][..]);
        let viewmat: Vec<f32> = glam::Mat4::IDENTITY.to_cols_array().to_vec();
        let viewmat_t = GpuTensor::from(&client, [16], &viewmat[..]);

        project_splats::launch::<WgpuRuntime>(
            &client,
            calculate_cube_count_elemwise(&client, 1, CubeDim::new_1d(256)),
            CubeDim::new_1d(256),
            viewmat_t.as_buffer_arg(),
            Vec2FLaunch::new(32.0, 32.0),
            Vec3FLaunch::new(0.0, 0.0, 0.0),
            attrs_t.as_buffer_arg(),
            sh_t.as_buffer_arg(),
            1,
            Vec2FLaunch::new(4.0, 4.0),
            Vec2FLaunch::new(64.0, 64.0),
            depth_order.as_buffer_arg(),
            depth_keys.as_buffer_arg(),
            projected.as_buffer_arg(),
            counters.as_buffer_arg(),
            2, // max_isects
            tile_ids.as_buffer_arg(),
            gaussian_ids.as_buffer_arg(),
        );

        let c: Vec<u32> = counters.read_vec();
        assert_eq!(c[1], 1, "one visible splat");
        assert_eq!(c[0], 16, "all 16 tile emissions counted");
        let tids: Vec<u32> = tile_ids.read_vec();
        let gids: Vec<u32> = gaussian_ids.read_vec();
        assert!(
            tids[..2].iter().all(|&t| t != SENTINEL),
            "capped slots written"
        );
        assert!(
            tids[2..].iter().all(|&t| t == SENTINEL),
            "slots beyond max_isects must be untouched, got {:?}",
            &tids[2..6]
        );
        assert!(gids[2..].iter().all(|&g| g == SENTINEL));
    }
}
