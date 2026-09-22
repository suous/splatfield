//! Object localization and camera-free candidate view generation.
//!
//! B3-Seg never touches the reconstruction cameras: from the current
//! posterior it estimates where the object is —
//!
//!   c_obj = Σ_{fg} m_i·μ_i / Σ_{fg} m_i,   r_obj = Σ_{fg} m_i·‖μ_i − c_obj‖ / Σ_{fg} m_i
//!
//! — places N_cand candidate cameras on the sphere of radius
//! 1.5·r_obj / tan(fov/2) centered on c_obj (each looking at c_obj), and lets
//! the EIG ranking pick among them.

use super::beta::BetaState;
use crate::render::{Splats, splat_position};
use glam::Vec3;

/// Where the object is, per the current posterior.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ObjectLocalization {
    pub center: Vec3,
    pub radius: f32,
}

/// Weighted center and radius of the current foreground set (Gaussians with
/// a > b, weighted by m = a/(a+b)) — the paper's c_obj and r_obj. `None`
/// while no Gaussian is foreground — the caller decides the fallback (e.g.
/// scene bounds).
///
/// Once-per-round host math (the EIG ranking renders ~20 views per round):
/// one a/b readback (8 B/splat; positions come from the RAM copy on
/// `Splats`) and two f32 passes. The result only places candidate cameras;
/// it never feeds EIG or labels.
pub(crate) fn localize(splats: &Splats, state: &BetaState) -> Option<ObjectLocalization> {
    let n = splats.attributes.shape[0];
    let attr = &splats.positions;
    let a: Vec<f32> = state.a.read_vec();
    let b: Vec<f32> = state.b.read_vec();

    let mut sums = [0f32; 4];
    for i in 0..n {
        if a[i] > b[i] {
            let m = a[i] / (a[i] + b[i]);
            let p = splat_position(attr, n, i);
            sums[0] += m;
            sums[1] += m * p.x;
            sums[2] += m * p.y;
            sums[3] += m * p.z;
        }
    }
    let [sw, sx, sy, sz] = sums;
    if sw <= 0.0 {
        return None;
    }
    let center = Vec3::new(sx / sw, sy / sw, sz / sw);

    let mut total = 0f32;
    for i in 0..n {
        if a[i] > b[i] {
            let m = a[i] / (a[i] + b[i]);
            let d = (splat_position(attr, n, i) - center).length();
            total += m * d;
        }
    }
    Some(ObjectLocalization {
        center,
        radius: total / sw,
    })
}

/// Vertical fov of every camera the loop generates (radians): the candidate
/// sphere shares the viewer's reference fov by construction.
const CANDIDATE_FOV: f32 = crate::camera::Camera::BASE_FOV.x;

/// `count` cameras on a Fibonacci sphere around `center` at the paper's
/// distance 1.5·radius/tan(fov/2), each looking at the center. The camera
/// frame looks along its local +Z (see `project::to_camera_space`), so the
/// minimal arc from +Z to the view direction orients each camera.
pub(crate) fn candidates(
    localization: &ObjectLocalization,
    count: usize,
) -> Vec<crate::camera::Camera> {
    let golden = (5.0f32.sqrt() - 1.0) / 2.0;
    // A degenerate localization (one foreground splat: r_obj = 0) would put
    // every camera inside the object and NaN the look-at quaternion.
    let radius = localization.radius.max(1e-3);
    let dist = 1.5 * radius / (CANDIDATE_FOV * 0.5).tan();
    (0..count)
        .map(|k| {
            let z = 1.0 - 2.0 * (k as f32 + 0.5) / count as f32;
            let r = (1.0 - z * z).sqrt();
            let theta = 2.0 * core::f32::consts::PI * golden * k as f32;
            let dir = Vec3::new(r * theta.cos(), r * theta.sin(), z);
            let position = localization.center + dist * dir;
            crate::camera::Camera::look_at(position, localization.center)
        })
        .collect()
}

/// Deterministic pseudo-random offset for splat `i`, in [-0.5, 0.5]³ — the
/// shared scatter of the seg test fixtures.
#[cfg(test)]
pub(crate) fn jitter(i: usize) -> Vec3 {
    glam::vec3(
        ((i * 73) % 17) as f32 / 17.0 - 0.5,
        ((i * 151) % 13) as f32 / 13.0 - 0.5,
        ((i * 201) % 11) as f32 / 11.0 - 0.5,
    )
}

/// Splats per cluster in [`cluster_scene`].
#[cfg(test)]
pub(crate) const PER_CLUSTER: usize = 40;

/// A scene of three well-separated spherical clusters; cluster 0 is the
/// segmentation target. The active-loop and cli suites share this exact
/// geometry, so pass thresholds tuned to it stay meaningful everywhere.
#[cfg(test)]
pub(crate) fn cluster_scene() -> (Vec<f32>, Vec<(glam::Vec3, f32)>) {
    let centers = [
        (glam::vec3(1.2, 0.0, 0.0), 0.30),
        (glam::vec3(-1.6, 0.6, 0.4), 0.45),
        (glam::vec3(0.1, -0.4, 1.9), 0.40),
    ];
    let n = centers.len() * PER_CLUSTER;
    let a = crate::render::sample_opaque_attributes(n, |i| {
        let (c, r) = centers[i / PER_CLUSTER];
        c + 1.4 * r * jitter(i)
    });
    (a, centers.to_vec())
}

/// A perfect oracle for tests, as the loop's sensor closure: the "object"
/// is a set of spheres; a pixel is foreground iff its camera ray hits any
/// sphere.
#[cfg(test)]
pub(crate) fn sphere_oracle(
    spheres: Vec<(Vec3, f32)>,
) -> impl FnMut(&[u8], glam::UVec2, &crate::camera::Camera) -> anyhow::Result<Vec<u8>> {
    move |_rgb: &[u8], size: glam::UVec2, camera: &crate::camera::Camera| {
        let focal = camera.focal(size);
        let center = size.as_vec2() * 0.5;
        let mut mask = vec![0u8; (size.x * size.y) as usize];
        for y in 0..size.y {
            for x in 0..size.x {
                // The renderer's own pixel convention:
                // screen = focal·cam.xy/cam.z + size/2.
                let cam = glam::vec3(
                    (x as f32 + 0.5 - center.x) / focal.x,
                    (y as f32 + 0.5 - center.y) / focal.y,
                    1.0,
                );
                let dir = camera.rotation * cam.normalize();
                let hit = spheres.iter().any(|&(c, r)| {
                    let oc = c - camera.position;
                    let t = oc.dot(dir);
                    // Center behind the camera: only a hit if the camera
                    // sits inside the sphere.
                    let d2 = if t < 0.0 {
                        oc.length_squared()
                    } else {
                        oc.length_squared() - t * t
                    };
                    d2 <= r * r
                });
                mask[(y * size.x + x) as usize] = hit as u8;
            }
        }
        Ok(mask)
    }
}

/// A perfect oracle for one sphere — the tests' segmentation target.
#[cfg(test)]
pub(crate) fn target(
    center: Vec3,
    radius: f32,
) -> impl FnMut(&[u8], glam::UVec2, &crate::camera::Camera) -> anyhow::Result<Vec<u8>> {
    sphere_oracle(vec![(center, radius)])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Attributes for `n_fg` foreground splats clustered at `fg_center` and
    /// `n_bg` background splats at `bg_center`.
    fn two_cluster_attributes(
        n_fg: usize,
        fg_center: Vec3,
        n_bg: usize,
        bg_center: Vec3,
        spread: f32,
    ) -> Vec<f32> {
        crate::render::sample_opaque_attributes(n_fg + n_bg, |i| {
            let c = if i < n_fg { fg_center } else { bg_center };
            c + spread * jitter(i)
        })
    }

    #[test]
    fn test_localize_single_fg_splat_clamps_radius() {
        let (_gpu, client) = crate::gpu_testing::test_client();
        let n = 8usize;
        let attributes =
            two_cluster_attributes(1, Vec3::new(2.0, -1.0, 0.5), n - 1, Vec3::NEG_ONE, 0.0);
        let splats = crate::render::opaque_splats(&client, attributes);
        // Posteriors: the single splat at (2,-1,0.5) decided foreground.
        let (a, b): (Vec<f32>, Vec<f32>) = (0..n)
            .map(|i| if i == 0 { (3.0, 1.0) } else { (1.0, 3.0) })
            .unzip();
        let state = crate::seg::beta::state(&client, &a, &b);
        let loc = localize(&splats, &state).expect("one fg splat");
        // One point mass: the mean-distance radius collapses to ~0 — the
        // guard clamp in `candidates` must lift it to the floor, NaN-free.
        assert!(loc.radius.is_finite() && loc.radius < 1e-3);
        let cams = candidates(&loc, 5);
        let want_dist = 1.5 * loc.radius.max(1e-3) / (CANDIDATE_FOV * 0.5).tan();
        for (k, cam) in cams.iter().enumerate() {
            let off = cam.position - loc.center;
            assert!(
                (off.length() - want_dist).abs() < 1e-6,
                "cam {k} distance {} vs {want_dist}",
                off.length()
            );
            assert!(cam.position.is_finite() && (cam.rotation * Vec3::Z).is_finite());
        }
    }

    #[test]
    fn test_localize_none_without_fg() {
        let (_gpu, client) = crate::gpu_testing::test_client();
        let attributes = two_cluster_attributes(10, Vec3::ONE, 10, Vec3::NEG_ONE, 0.1);
        let splats = crate::render::opaque_splats(&client, attributes);
        let state = BetaState::new_uniform(&client, 20);
        // a=b=1: nothing is foreground yet.
        assert!(localize(&splats, &state).is_none());
    }

    /// The CPU moments must reproduce the paper's definitions exactly:
    /// c_obj is the m-weighted centroid, r_obj the m-weighted mean distance
    /// to it, and background splats contribute nothing. Equal weights make
    /// both quantities plain averages, so the expectations are hand-computed
    /// and the tolerance is f32-exact-tight.
    #[test]
    fn test_localize_matches_hand_computed_moments() {
        let (_gpu, client) = crate::gpu_testing::test_client();
        let pos = [
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(50.0, -50.0, 50.0),
            Vec3::new(-40.0, 10.0, 5.0),
        ];
        let n = pos.len();
        let attributes = crate::render::sample_opaque_attributes(n, |i| pos[i]);
        let splats = crate::render::opaque_splats(&client, attributes);
        // First three decided foreground (a=3,b=1 → m=0.75), rest background.
        let (a, b): (Vec<f32>, Vec<f32>) = (0..n)
            .map(|i| if i < 3 { (3.0, 1.0) } else { (1.0, 3.0) })
            .unzip();
        let state = crate::seg::beta::state(&client, &a, &b);
        let loc = localize(&splats, &state).expect("fg exists");

        let m = 0.75f32;
        let w = 3.0 * m;
        let want_c = (pos[0] + pos[1] + pos[2]) * (m / w);
        assert!(
            (loc.center - want_c).length() < 1e-6,
            "center {:?} vs {want_c:?}",
            loc.center
        );
        let want_r = pos[..3].iter().map(|p| (p - want_c).length()).sum::<f32>() * (m / w);
        assert!(
            (loc.radius - want_r).abs() < 1e-6,
            "radius {} vs {want_r}",
            loc.radius
        );
    }

    #[test]
    fn test_candidates_on_sphere_looking_at_center() {
        let loc = ObjectLocalization {
            center: Vec3::new(1.0, 2.0, 3.0),
            radius: 0.5,
        };
        let want_dist = 1.5 * loc.radius / (CANDIDATE_FOV * 0.5).tan();
        let cams = candidates(&loc, 20);
        assert_eq!(cams.len(), 20);
        for (k, cam) in cams.iter().enumerate() {
            let off = cam.position - loc.center;
            assert!(
                (off.length() - want_dist).abs() < 1e-4,
                "cam {k} distance {} vs {want_dist}",
                off.length()
            );
            let forward = (cam.rotation * Vec3::Z).normalize();
            let to_center = (-off).normalize();
            assert!(
                forward.dot(to_center) > 1.0 - 1e-4,
                "cam {k} must look at the center"
            );
        }
        // Fibonacci spread: z coordinates monotonically decreasing.
        let zs: Vec<f32> = cams
            .iter()
            .map(|c| (c.position - loc.center).normalize().z)
            .collect();
        for w in zs.windows(2) {
            assert!(w[0] > w[1], "fibonacci z must decrease");
        }
    }
}
