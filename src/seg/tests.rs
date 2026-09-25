//! The seg GPU test suites over `mod.rs`'s evidence and paint paths, plus
//! the cross-thread tint test. The satellite modules' tests live with their
//! modules (`active`, `beta`, `views`).

use super::*;
use crate::render::{opaque_splats, sample_opaque_attributes};
use crate::seg::beta::BetaState;

/// Attributes for `n` opaque splats in a row along camera-space x at
/// depth z: splat k lands at screen x ≈ focal·offset(k)/z + size/2.
fn row_attributes(n: usize, z: f32, spacing: f32) -> Vec<f32> {
    sample_opaque_attributes(n, |i| {
        glam::vec3((i as f32 - (n - 1) as f32 * 0.5) * spacing, 0.0, z)
    })
}

/// Render `attributes` at 64×64 from the default camera; return
/// (Σ_p alpha/255 from the RGB bitmap, Σ_i ε_i, per-Gaussian ε).
fn alpha_sum_and_eps(client: &Client, attributes: Vec<f32>) -> (f32, f32, Vec<f32>) {
    let n = attributes.len() / crate::layout::ATTR_PLANES;
    let splats = opaque_splats(client, attributes);
    let mut scratch = RenderScratch::new(client, n, glam::uvec2(64, 64));
    let bitmap = splats.render_with(&mut scratch, &Camera::default(), glam::uvec2(64, 64));
    let px: Vec<u32> = bitmap.read_vec();
    let alpha_sum: f32 = px.iter().map(|p| (p >> 24) as f32 / 255.0).sum();

    let mut acc = Accumulators::new(client, n);
    splats.render_responsibility(
        &mut scratch,
        &mut acc,
        &Camera::default(),
        glam::uvec2(64, 64),
    );
    let eps = acc.read_f32();
    (alpha_sum, eps.iter().sum(), eps)
}

/// Center-pixel RGB of a 64×64 default-camera render — the tint tests'
/// on-screen probe.
fn center_rgb(splats: &Splats, scratch: &mut RenderScratch) -> [u8; 3] {
    let img = glam::uvec2(64, 64);
    let bitmap = splats.render_with(scratch, &Camera::default(), img);
    let rgb = crate::seg::active::bitmap_to_rgb(&bitmap, img.x);
    let px = ((img.y / 2) * img.x + img.x / 2) as usize * 3;
    [rgb[px], rgb[px + 1], rgb[px + 2]]
}

/// Left-half mask over a `size` frame: pixels with x < size.x/2.
fn left_half_mask(client: &Client, size: glam::UVec2) -> Mask {
    let bytes: Vec<u8> = (0..size.x * size.y)
        .map(|p| ((p % size.x) < size.x / 2) as u8)
        .collect();
    Mask::from_bytes(client, size, &bytes)
}

/// The dense overdraw scene: 40 staggered splats crowding the central
/// tile at 70×70 — pixels saturate (whole-tile early exit), 70 not being
/// a multiple of the 16-wide tile keeps out-of-bounds units in the edge
/// tiles, and the staggered depths make the sort total.
fn overdraw_scene(client: &Client) -> (Splats, RenderScratch, Accumulators) {
    let n = 40usize;
    let attributes = sample_opaque_attributes(n, |i| {
        let x = (i % 4) as f32 - 1.5;
        let y = (i % 3) as f32 - 1.0;
        glam::vec3(x * 0.04, y * 0.04, 1.0 + i as f32 * 0.5)
    });
    let splats = opaque_splats(client, attributes);
    let scratch = RenderScratch::new(client, n, glam::uvec2(70, 70));
    let acc = Accumulators::new(client, n);
    (splats, scratch, acc)
}

/// Two opaque splats in a row at 64×64 under the default camera with a
/// left-half mask (covers splat 0 only): the shared scene of the
/// evidence-split and Beta-update tests, both renders already done —
/// total ε in `bits`, the mask split in `fg`/`bg` of one accumulator.
fn split_scene(client: &Client) -> (Splats, RenderScratch, Accumulators) {
    let size = glam::uvec2(64, 64);
    let n = 2usize;
    let attributes = row_attributes(n, 2.0, 1.0);
    let splats = opaque_splats(client, attributes);
    let mut scratch = RenderScratch::new(client, n, size);
    let camera = Camera::default();
    let mut acc = Accumulators::new(client, n);

    splats.render_responsibility(&mut scratch, &mut acc, &camera, size);

    let mask = left_half_mask(client, size);
    let isects = splats.prepare_isects(&mut scratch, &camera, size);
    splats.accumulate_prepared(&scratch, &isects, size, &mut acc, Some(&mask));
    (splats, scratch, acc)
}

/// Conservation: every pixel's accumulated per-splat weights equal the
/// pixel's final opacity, 1 − T. The RGB path truncates per pixel at
/// T < 1/255 and quantizes alpha to 8 bits, so allow ~3 LSB per pixel.
#[test]
fn test_responsibility_conserves_pixel_weights() {
    let (_gpu, client) = crate::gpu_testing::test_client();
    let n = 5usize;
    let attributes = sample_opaque_attributes(n, |i| glam::vec3(0.0, 0.0, 1.0 + i as f32 * 0.5));
    let (alpha_sum, eps_sum, _eps) = alpha_sum_and_eps(&client, attributes);
    let tol = 64.0 * 64.0 * 3.0 / 255.0;
    assert!(
        (alpha_sum - eps_sum).abs() < tol,
        "Σ ε ({eps_sum}) must match Σ_p (1−T) ({alpha_sum}), tol {tol}"
    );
}

/// A front splat grabs strictly more weight than the one it occludes,
/// and a splat behind the camera accumulates nothing.
#[test]
fn test_responsibility_orders_depth_and_culls() {
    let (_gpu, client) = crate::gpu_testing::test_client();
    let n = 6usize;
    // Five opaque splats stacked in depth, plus one behind the camera at
    // z = −3 that projection must cull.
    let attributes = sample_opaque_attributes(n, |i| {
        glam::vec3(0.0, 0.0, if i == 5 { -3.0 } else { 1.0 + i as f32 * 0.5 })
    });
    let (_alpha_sum, _eps_sum, eps) = alpha_sum_and_eps(&client, attributes);
    for k in 0..n - 2 {
        assert!(
            eps[k] > eps[k + 1],
            "front splat {k} ({}) must outweigh occluded {} ({})",
            eps[k],
            eps[k + 1],
            k + 1
        );
    }
    assert!(eps[5] == 0.0, "behind-camera splat must have ε = 0");
}

/// The early-exit path must be deterministic: rendering the same dense
/// overdraw scene into a reused buffer yields bit-identical u32 words
/// every time. Depths are staggered so the depth sort is total:
/// equal-depth ties blend in schedule-dependent order (see project.rs),
/// which legitimately redistributes ε between tied splats.
#[test]
fn test_responsibility_is_bit_deterministic_across_renders() {
    let (_gpu, client) = crate::gpu_testing::test_client();
    let (splats, mut scratch, mut acc) = overdraw_scene(&client);

    let mut reference = None;
    for _ in 0..8 {
        splats.render_responsibility(
            &mut scratch,
            &mut acc,
            &Camera::default(),
            glam::uvec2(70, 70),
        );
        let bits: Vec<u32> = acc.bits.read_vec();
        match &reference {
            None => {
                assert!(bits.iter().any(|&q| q > 0), "overdraw scene must collect ε");
                reference = Some(bits);
            }
            Some(want) => assert_eq!(&bits, want, "ε must be bit-identical across renders"),
        }
    }
}

/// The ε finalize's u32 words are pinned exactly on the overdraw scene,
/// so a traversal reorganization that shifts any fixed-point
/// contribution fails here rather than inside the conservation test's
/// tolerance.
#[test]
fn test_eps_bits_match_golden() {
    let (_gpu, client) = crate::gpu_testing::test_client();
    let (splats, mut scratch, mut acc) = overdraw_scene(&client);
    splats.render_responsibility(
        &mut scratch,
        &mut acc,
        &Camera::default(),
        glam::uvec2(70, 70),
    );
    let bits: Vec<u32> = acc.bits.read_vec();
    // Golden from the current kernel on this device; splats past the
    // last visible one collect nothing.
    const GOLDEN: [u32; 40] = [
        411225576, 64983932, 17882249, 5200864, 773093, 333408, 108187, 61572, 12526, 2462, 1466,
        1461, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    assert_eq!(bits, GOLDEN, "ε fixed-point words drifted");
}

/// The mask splits each Gaussian's total responsibility exactly: e₁ + e₀
/// == ε bit-for-bit (same weights, same fixed-point quantization), with
/// splats fully inside/outside the mask landing entirely in fg/bg.
#[test]
fn test_evidence_splits_responsibility_exactly() {
    let (_gpu, client) = crate::gpu_testing::test_client();
    let (_splats, _scratch, acc) = split_scene(&client);
    let eps = acc.read_f32();
    assert!(eps[0] > 0.0 && eps[1] > 0.0, "both splats visible");

    let (fg, bg) = acc.read_evidence_f32();
    let fg_bits: Vec<u32> = acc.fg.read_vec();
    let bg_bits: Vec<u32> = acc.bg.read_vec();
    let total_bits: Vec<u32> = acc.bits.read_vec();
    for i in 0..eps.len() {
        assert_eq!(fg_bits[i] + bg_bits[i], total_bits[i], "splat {i}: split");
    }
    // Splat 0 (left) is entirely inside; splat 1 (right) entirely out.
    assert_eq!(bg_bits[0], 0, "left splat must collect no background");
    assert_eq!(fg_bits[1], 0, "right splat must collect no foreground");
    assert!(fg[0] > 0.5 * eps[0]);
    assert!(bg[1] > 0.5 * eps[1]);
}

/// The reuse path must be bit-identical to re-preparing: accumulating
/// evidence over intersections kept from the view's RGB render (what the
/// active loop does per observation round) produces exactly the u32
/// fixed-point buffers a fresh pipeline run for the same camera produces.
#[test]
fn test_evidence_reuse_matches_prepared() {
    let (_gpu, client) = crate::gpu_testing::test_client();
    let size = glam::uvec2(64, 64);
    let n = 4usize;
    let splats = opaque_splats(&client, row_attributes(n, 2.0, 0.7));
    let mut scratch = RenderScratch::new(&client, n, size);
    let camera = Camera::default();
    let mask = left_half_mask(&client, size);

    let mut fresh = Accumulators::new(&client, n);
    let isects = splats.prepare_isects(&mut scratch, &camera, size);
    splats.accumulate_prepared(&scratch, &isects, size, &mut fresh, Some(&mask));
    assert!(
        fresh.fg.read_vec::<u32>().iter().any(|&q| q > 0),
        "scene must collect foreground evidence"
    );

    let isects = splats.prepare_isects(&mut scratch, &camera, size);
    splats.finalize(&scratch, &isects, size, crate::render::Finalize::Rgb);
    let mut reused = Accumulators::new(&client, n);
    splats.accumulate_prepared(&scratch, &isects, size, &mut reused, Some(&mask));

    assert_eq!(
        fresh.fg.read_vec::<u32>(),
        reused.fg.read_vec::<u32>(),
        "fg fixed-point bits must match"
    );
    assert_eq!(
        fresh.bg.read_vec::<u32>(),
        reused.bg.read_vec::<u32>(),
        "bg fixed-point bits must match"
    );
}

/// The fixed-point scale tracks the render resolution: an accumulator
/// reused across a resolution change must produce exactly the u32 words
/// a fresh accumulator produces at the new size.
#[test]
fn test_scale_tracks_resolution_change() {
    let (_gpu, client) = crate::gpu_testing::test_client();
    let n = 2usize;
    let splats = opaque_splats(&client, row_attributes(n, 2.0, 1.0));
    let mut scratch = RenderScratch::new(&client, n, glam::uvec2(64, 64));
    let camera = Camera::default();
    let mut reused = Accumulators::new(&client, n);

    splats.render_responsibility(&mut scratch, &mut reused, &camera, glam::uvec2(64, 64));
    splats.render_responsibility(&mut scratch, &mut reused, &camera, glam::uvec2(32, 32));

    let mut fresh = Accumulators::new(&client, n);
    splats.render_responsibility(&mut scratch, &mut fresh, &camera, glam::uvec2(32, 32));

    assert_eq!(
        reused.bits.read_vec::<u32>(),
        fresh.bits.read_vec::<u32>(),
        "scale must track the resolution before the next accumulate"
    );
}

/// The conjugate update moves the posterior in the mask's direction: a
/// fully-inside splat ends up foreground (a > b), a fully-outside one
/// background.
#[test]
fn test_beta_update_uses_evidence() {
    let (_gpu, client) = crate::gpu_testing::test_client();
    let (_splats, _scratch, ev) = split_scene(&client);

    let state = BetaState::new_uniform(&client, 2);
    state.update(&ev);
    let a: Vec<f32> = state.a.read_vec();
    let b: Vec<f32> = state.b.read_vec();
    assert!(a[0] > b[0], "inside splat must become foreground");
    assert!(b[1] > a[1], "outside splat must become background");
    let (fg, bg) = ev.read_evidence_f32();
    assert!((a[0] - (1.0 + fg[0])).abs() < 1e-5);
    assert!((b[1] - (1.0 + bg[1])).abs() < 1e-5);
}

/// A posterior left over from a different model (model-swap race) must
/// not paint out of bounds into the projected buffer: render_posterior
/// falls back to the normal color render instead.
#[test]
fn test_render_posterior_ignores_size_mismatch() {
    let (_gpu, client) = crate::gpu_testing::test_client();
    let n = 2usize;
    let splats = opaque_splats(&client, row_attributes(n, 2.0, 1.0));
    let mut scratch = RenderScratch::new(&client, n, glam::uvec2(64, 64));
    // Sized for 3 splats — a model that no longer matches the scene.
    let a = GpuTensor::from(&client, [3], [2.0f32, 1.0, 3.0]);
    let b = GpuTensor::from(&client, [3], [1.0f32, 3.0, 1.0]);

    let stale = splats.render_posterior(
        &mut scratch,
        &a,
        &b,
        &Camera::default(),
        glam::uvec2(64, 64),
    );
    let normal = splats.render_with(&mut scratch, &Camera::default(), glam::uvec2(64, 64));
    assert_eq!(
        stale.read_vec::<u32>(),
        normal.read_vec::<u32>(),
        "mismatched posterior must degrade to the plain render"
    );
}

/// The heatmap paints the posterior mean as grayscale (the paper's
/// "render mean image of Beta dist."): every pixel has R == G == B, and
/// the m = 0.75 splat out-brightens the m = 0.25 one.
#[test]
fn test_posterior_heatmap_is_grayscale() {
    let (_gpu, client) = crate::gpu_testing::test_client();
    let img = glam::uvec2(64, 64);
    let n = 2usize;
    let splats = opaque_splats(&client, row_attributes(n, 2.0, 1.0));
    let mut scratch = RenderScratch::new(&client, n, img);
    let a = GpuTensor::from(&client, [n], [3.0f32, 1.0]);
    let b = GpuTensor::from(&client, [n], [1.0f32, 3.0]);

    let bitmap = splats.render_posterior(&mut scratch, &a, &b, &Camera::default(), img);
    let rgb = crate::seg::active::bitmap_to_rgb(&bitmap, img.x);
    let mut lum = [0u8; 2];
    for (p, px) in rgb.as_chunks::<3>().0.iter().enumerate() {
        assert_eq!(px[0], px[1], "R == G at pixel {p}");
        assert_eq!(px[1], px[2], "G == B at pixel {p}");
        let half = (p % img.x as usize) >= img.x as usize / 2;
        lum[half as usize] = lum[half as usize].max(px[0]);
    }
    assert!(
        lum[0] > lum[1],
        "m=0.75 splat ({}) must out-brighten m=0.25 ({})",
        lum[0],
        lum[1]
    );
}

/// The app's recolor sequence end-to-end: render → tint → render. The
/// next frame must show the new DC colors on screen.
#[test]
fn test_tint_is_visible_in_next_render() {
    let (_gpu, client) = crate::gpu_testing::test_client();
    let n = 1usize;
    let attributes = sample_opaque_attributes(n, |_| glam::vec3(0.0, 0.0, 5.0));
    let splats = opaque_splats(&client, attributes);
    let mut scratch = crate::render::RenderScratch::new(&client, n, glam::uvec2(64, 64));

    let before = center_rgb(&splats, &mut scratch);
    let a = GpuTensor::from(&client, [1], [2.0f32]);
    let b = GpuTensor::from(&client, [1], [1.0f32]);
    // Palette red: dc = (display − 0.5)/C0, the app's exact conversion.
    let red = crate::to_dc([0.937, 0.267, 0.267]);
    splats.tint(&a, &b, red);
    let after = center_rgb(&splats, &mut scratch);

    assert!(after[0] > before[0] + 40, "red up: {before:?} → {after:?}");
    assert!(
        after[1] < before[1] - 10,
        "green down: {before:?} → {after:?}"
    );
    assert!(
        after[2] < before[2] - 10,
        "blue down: {before:?} → {after:?}"
    );
}

/// save_colors snapshots only the DC planes of a higher-degree SH model,
/// and the tint → save → tint-overwrite → restore cycle leaves every
/// rest-plane coefficient untouched. A snapshot aliasing sh_coeffs would
/// be clobbered by the overwrite tint; a restore overrunning 3·total
/// floats would corrupt the rest planes.
#[test]
fn test_save_colors_roundtrips_dc_of_high_degree_sh() {
    let (_gpu, client) = crate::gpu_testing::test_client();
    let n = 3usize;
    let k_per_ch = 2usize;
    let attributes = sample_opaque_attributes(n, |i| glam::vec3(0.0, 0.0, 2.0 + i as f32));
    // Field-major SH: channel c of splat i at c·n + i — three DC planes
    // first, then (k_per_ch − 1) rest planes per channel, all distinct.
    let planes = 3 * k_per_ch;
    let mut sh = vec![0f32; planes * n];
    for c in 0..planes {
        for i in 0..n {
            sh[c * n + i] = 1.0 + (c * n + i) as f32;
        }
    }
    let splats = crate::render::CpuSplats {
        attributes,
        sh_coeffs: sh.clone(),
    }
    .upload(&client);

    let early = splats.save_colors();
    assert_eq!(early.shape.as_slice(), &[3, n], "DC snapshot shape");
    let snapshot: Vec<f32> = early.read_vec();
    for c in 0..3 {
        for i in 0..n {
            assert_eq!(
                snapshot[c * n + i],
                sh[c * n + i],
                "DC plane {c}, splat {i}"
            );
        }
    }

    let a = GpuTensor::from(&client, [3], [2.0f32, 1.0, 3.0]);
    let b = GpuTensor::from(&client, [3], [1.0f32, 3.0, 1.0]);
    splats.tint(&a, &b, [1.0, 0.0, 0.0]);
    let saved = splats.save_colors();

    let all_fg = GpuTensor::from(&client, [3], [2.0f32, 2.0, 2.0]);
    let ones = GpuTensor::from(&client, [3], [1.0f32, 1.0, 1.0]);
    splats.tint(&all_fg, &ones, [9.0, 9.0, 9.0]);
    splats.restore_colors(&saved);

    let all: Vec<f32> = splats.sh_coeffs.read_vec();
    // Foreground splats restore to the tinted DC, the background one to
    // its original DC.
    for i in [0usize, 2] {
        assert_eq!(all[i], 1.0, "splat {i} red restored");
        assert_eq!(all[n + i], 0.0, "splat {i} green restored");
        assert_eq!(all[2 * n + i], 0.0, "splat {i} blue restored");
    }
    for c in 0..3 {
        assert_eq!(
            all[c * n + 1],
            sh[c * n + 1],
            "background splat DC plane {c}"
        );
    }
    // Rest planes: never written by tint, never overrun by restore.
    for c in 3..planes {
        for i in 0..n {
            assert_eq!(all[c * n + i], sh[c * n + i], "rest plane {c}, splat {i}");
        }
    }
    // The early snapshot must still hold the pre-tint DC: no aliasing.
    let early_dc: Vec<f32> = early.read_vec();
    for c in 0..3 {
        for i in 0..n {
            assert_eq!(
                early_dc[c * n + i],
                sh[c * n + i],
                "early plane {c}, splat {i}"
            );
        }
    }
}

/// The app tints on the worker thread and renders on the UI thread.
/// The next frame on THIS thread must observe the worker's writes.
#[test]
fn test_cross_thread_tint_visible_in_render() {
    let (_gpu, client) = crate::gpu_testing::test_client();
    let n = 1usize;
    let splats = std::sync::Arc::new(opaque_splats(
        &client,
        sample_opaque_attributes(n, |_| glam::vec3(0.0, 0.0, 5.0)),
    ));
    let mut scratch = RenderScratch::new(&client, n, glam::uvec2(64, 64));

    // The worker: tint off-thread, then signal through a channel —
    // exactly the app's Progress/Done pattern.
    let (tx, rx) = std::sync::mpsc::channel::<Vec<f32>>();
    let s = splats.clone();
    std::thread::spawn(move || {
        let a = splat_sort::tensor::GpuTensor::from(&s.attributes.client, [1], [2.0f32]);
        let b = splat_sort::tensor::GpuTensor::from(&s.attributes.client, [1], [1.0f32]);
        let red = crate::to_dc([0.937, 0.267, 0.267]);
        s.tint(&a, &b, red);
        let dc: Vec<f32> = s.sh_coeffs.read_vec();
        tx.send(dc).ok();
    });
    let dc = rx.recv().unwrap();
    let expect = ((0.937f64 - 0.5) / f64::from(crate::project::SH_C0)) as f32;
    assert!(
        (dc[0] - expect).abs() < 1e-5,
        "worker wrote dc: {} vs {expect}",
        dc[0]
    );

    let after = center_rgb(&splats, &mut scratch);
    assert!(
        after[0] > 150 && after[1] < 120 && after[2] < 120,
        "render must show red: {after:?}"
    );
}
