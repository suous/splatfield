use crate::camera::Camera;
use crate::layout;
use crate::project::{CameraViewLaunch, project_splats};
use crate::raster::{map_isects, rasterize_kernel};
use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;
use splat_sort::scan::{ScanScratch, exclusive_scan_gather};
use splat_sort::sort::{RadixScratch, bits_for};
use splat_sort::tensor::GpuTensor;

// Memory ceiling for the intersection buffers, not a derived bound: ~67M
// entries ≈ 0.5 GiB for the id pair, plus the tile sort's ping-pong scratch.
// prepare clamps capacity here; map_isects truncates emissions past it.
const INTERSECTS_UPPER_BOUND: usize = 2 * 512 * 65535;
// First-frame capacity cap, raised toward the ceiling only when a frame
// actually emits more (see prepare).
const INITIAL_ISECTS_CAP: usize = 1 << 22;

/// Host-side splat payload produced by the PLY/SOG parsers, uploaded once by
/// `Splats::new`. Field-major layout: see the `PLANE_*` planes in `layout`.
/// The app retains it as the pristine master copy that box-select deletion
/// gathers from — the GPU upload consumes the parser's buffers.
#[derive(Clone, Debug)]
pub struct CpuSplats {
    pub attributes: Vec<f32>,
    pub sh_coeffs: Vec<f32>,
}

impl CpuSplats {
    pub fn upload(self, client: &ComputeClient<WgpuRuntime>) -> Splats {
        Splats::new(self.attributes, self.sh_coeffs, client)
    }

    /// Splat count (`ATTR_PLANES` floats per splat, field-major).
    pub(crate) fn count(&self) -> usize {
        self.attributes.len() / layout::ATTR_PLANES
    }

    /// The splats kept by `keep` (in `keep`'s order), gathered plane-wise —
    /// both buffers are field-major (plane p of splat i at p·n + i), so
    /// deletion is the same gather over each plane.
    pub fn gather(&self, keep: &[usize]) -> CpuSplats {
        let n = self.count();
        CpuSplats {
            attributes: gather_planes(&self.attributes, layout::ATTR_PLANES, n, keep),
            sh_coeffs: gather_planes(&self.sh_coeffs, self.sh_coeffs.len() / n, n, keep),
        }
    }

    /// The display keep-list: non-removed master indices, ascending. The
    /// GPU scene shows exactly these masters, and a splat's display index
    /// is its position here (materialized by [`Self::gather`]).
    pub fn kept(&self, removed: &[usize]) -> Vec<usize> {
        let n = self.count();
        (0..n)
            .filter(|i| removed.binary_search(i).is_err())
            .collect()
    }

    /// Display indices of the kept splats whose projected center falls
    /// inside the viewport-pixel rect (`min`/`max`, inclusive), from
    /// `camera` at `pixel` resolution — the same world→pixel map the
    /// rendered view shows, so what's inside the drawn box is what's
    /// selected. Indices are display numbering — positions in
    /// [`Self::kept`] — the numbering the GPU buffers use.
    pub fn select_in_rect(
        &self,
        camera: &Camera,
        pixel: glam::UVec2,
        min: glam::Vec2,
        max: glam::Vec2,
        removed: &[usize],
    ) -> Vec<usize> {
        let n = self.count();
        let mut selected = Vec::new();
        let mut d = 0usize;
        for i in 0..n {
            if removed.binary_search(&i).is_err() {
                let inside = crate::camera::guide_point(
                    camera,
                    splat_position(&self.attributes, n, i),
                    pixel,
                )
                .is_some_and(|uv| {
                    // guide_point yields pixel-index space; the drag rect is
                    // continuous, where pixel i spans [i, i+1).
                    let q = uv * pixel.as_vec2() + 0.5;
                    q.x >= min.x && q.x <= max.x && q.y >= min.y && q.y <= max.y
                });
                if inside {
                    selected.push(d);
                }
                d += 1;
            }
        }
        selected
    }
}

/// Gather `keep` out of a field-major buffer with `planes` strided planes
/// of `n` splats each.
fn gather_planes(src: &[f32], planes: usize, n: usize, keep: &[usize]) -> Vec<f32> {
    let m = keep.len();
    let mut out = vec![0f32; planes * m];
    for (j, &i) in keep.iter().enumerate() {
        for p in 0..planes {
            out[p * m + j] = src[p * n + i];
        }
    }
    out
}

/// Per-frame GPU buffers, reused across frames so steady-state rendering
/// allocates nothing new. Rebuilt when the frame shape changes.
pub struct RenderScratch {
    depth_order: GpuTensor,
    depth_keys: GpuTensor,
    pub(crate) projected: GpuTensor,
    counters: GpuTensor,
    tile_counts: GpuTensor,
    tile_bbox: GpuTensor,
    tile_ids: GpuTensor,
    gaussian_ids: GpuTensor,
    pub(crate) bitmap: GpuTensor,
    // Ping-pong scratch for the two sorts. The tile sort must not share the
    // depth sort's: an odd pass count returns buffers aliasing the scratch.
    sort_depth: RadixScratch,
    sort_tile: RadixScratch,
    scan: ScanScratch,
    img_size: glam::UVec2,
    // The previous prepare's raw intersection emission, stashed for the
    // next capacity check — growth only ever happens between frames.
    last_isects_raw: u32,
}

/// Tile grid for `img_size`: `(bounds, count)`.
fn tile_grid(img_size: glam::UVec2) -> (glam::UVec2, usize) {
    let bounds = img_size.map(|c| c.div_ceil(layout::TILE_WIDTH));
    (bounds, (bounds.x * bounds.y) as usize)
}

impl RenderScratch {
    pub fn new(client: &ComputeClient<WgpuRuntime>, total: usize, img_size: glam::UVec2) -> Self {
        let (_, num_tiles) = tile_grid(img_size);
        let isect_capacity =
            (num_tiles.saturating_mul(total).min(INITIAL_ISECTS_CAP)).next_power_of_two();
        // 256-byte rows: wgpu buffer→texture copies align rows to
        // COPY_BYTES_PER_ROW_ALIGNMENT; texture.rs copies with this stride.
        let row_stride = (img_size.x * 4).next_multiple_of(256) / 4;
        Self {
            depth_order: GpuTensor::empty(client, [total]),
            depth_keys: GpuTensor::empty(client, [total]),
            projected: GpuTensor::empty(client, [total, layout::PROJ_FLOATS]),
            counters: GpuTensor::empty(client, [2]),
            tile_counts: GpuTensor::empty(client, [total]),
            tile_bbox: GpuTensor::empty(client, [total, 2]),
            tile_ids: GpuTensor::empty(client, [isect_capacity]),
            gaussian_ids: GpuTensor::empty(client, [isect_capacity]),
            bitmap: GpuTensor::empty(client, [img_size.y as usize, row_stride as usize]),
            sort_depth: RadixScratch::new(client, total),
            sort_tile: RadixScratch::new(client, isect_capacity),
            scan: ScanScratch::new(client, total),
            img_size,
            last_isects_raw: 0,
        }
    }

    /// Rebuild if the frame shape changed, then grow the isect buffers from
    /// last frame's emission. Returns the tile grid `(bounds, count)`.
    fn prepare(
        &mut self,
        client: &ComputeClient<WgpuRuntime>,
        total: usize,
        img_size: glam::UVec2,
    ) -> (glam::UVec2, usize) {
        // Read the growth hint before a rebuild resets it: the first frame
        // at a new shape must still grow to the previous shape's emission,
        // else map_isects silently truncates it.
        let raw = self.last_isects_raw;
        if self.depth_order.shape[0] != total || self.img_size != img_size {
            *self = Self::new(client, total, img_size);
        }
        let new_cap = (raw as usize)
            .next_power_of_two()
            .min(INTERSECTS_UPPER_BOUND);
        if new_cap > self.tile_ids.shape[0] {
            eprintln!(
                "intersection capacity {} reached ({raw} emitted); growing to {new_cap}",
                self.tile_ids.shape[0]
            );
            self.tile_ids = GpuTensor::empty(client, [new_cap]);
            self.gaussian_ids = GpuTensor::empty(client, [new_cap]);
            self.sort_tile = RadixScratch::new(client, new_cap);
        }
        tile_grid(img_size)
    }
}

pub struct Splats {
    pub attributes: GpuTensor,
    pub sh_coeffs: GpuTensor,
    pub bounds: (glam::Vec3, glam::Vec3),
    /// Host copy of the xyz planes. Positions are immutable after upload
    /// (no kernel takes `&mut` to `attributes`; edits re-upload), so
    /// per-round localization reads RAM instead of pulling the planes
    /// back over PCIe.
    pub(crate) positions: Vec<f32>,
}

/// Splat `i`'s world position from the field-major attribute planes.
pub(crate) fn splat_position(attr: &[f32], n: usize, i: usize) -> glam::Vec3 {
    glam::vec3(
        attr[layout::PLANE_X * n + i],
        attr[layout::PLANE_Y * n + i],
        attr[layout::PLANE_Z * n + i],
    )
}

/// Which buffers one finalize dispatch fills — the rasterize_kernel comptime
/// `(rgb, evidence)` mode pair. ε variants carry the fixed-point scale and
/// the live accumulator tensors; every mode-dead buffer slot aliases the
/// `tile_ids` input (buffer arg 0): comptime-dead args are not always dropped
/// from the launch signature, and cubecl binds an alias without a second
/// wgpu binding (see `raster::rasterize_kernel`).
pub(crate) enum Finalize<'a> {
    /// Blend the projected SH colors into `scratch.bitmap`.
    Rgb,
    /// Accumulate total rendering responsibility into `resp`.
    Total { scale: f32, resp: &'a GpuTensor },
    /// Split the weights by the packed `mask` into `fg`/`bg` pseudo-counts.
    Evidence {
        scale: f32,
        mask: &'a GpuTensor,
        words_per_row: u32,
        fg: &'a GpuTensor,
        bg: &'a GpuTensor,
    },
}

impl Splats {
    pub(crate) fn new(
        attributes: Vec<f32>,
        sh_coeffs: Vec<f32>,
        client: &ComputeClient<WgpuRuntime>,
    ) -> Self {
        let n = attributes.len() / layout::ATTR_PLANES;
        assert!(n > 0, "Splats::new: zero splats");
        let n_coeffs = sh_coeffs.len() / n;
        // xyz are the first three planes.
        let positions = attributes[..3 * n].to_vec();

        let mut min = glam::Vec3::splat(f32::MAX);
        let mut max = glam::Vec3::splat(f32::MIN);
        for i in 0..n {
            let p = splat_position(&positions, n, i);
            min = min.min(p);
            max = max.max(p);
        }

        Self {
            // The shape names splat-major rows, but the bytes are plane-major
            // (layout.rs); shapes here are consumed only as counts.
            attributes: GpuTensor::from(client, [n, layout::ATTR_PLANES], attributes),
            sh_coeffs: GpuTensor::from(client, [n, n_coeffs / 3, 3], sh_coeffs),
            bounds: (min, max),
            positions,
        }
    }

    /// Shared front half of the render paths: project, depth-sort, emit and
    /// tile-sort the intersections. Returns everything a per-pixel finalize
    /// kernel (RGB blending, responsibility accumulation, ...) needs.
    pub(crate) fn prepare_isects(
        &self,
        scratch: &mut RenderScratch,
        camera: &Camera,
        img_size: glam::UVec2,
    ) -> TileIsects {
        let client = &self.attributes.client;
        let total = self.attributes.shape[0];
        let (tile_bounds, num_tiles) = scratch.prepare(client, total, img_size);
        let max_isects = scratch.tile_ids.shape[0] as u32;
        let cube_dim = CubeDim::new_1d(layout::TILE_SIZE);

        let sh_per_ch = self.sh_coeffs.shape[1] as u32;
        let view = CameraViewLaunch::for_camera(camera, img_size, tile_bounds);

        // Stream-ordered before the project launch; saves a kernel dispatch.
        scratch.counters.write([0u32, 0]);

        project_splats::launch::<WgpuRuntime>(
            client,
            calculate_cube_count_elemwise(client, total, cube_dim),
            cube_dim,
            view,
            self.attributes.as_buffer_arg(),
            self.sh_coeffs.as_buffer_arg(),
            sh_per_ch,
            scratch.depth_order.as_buffer_arg(),
            scratch.depth_keys.as_buffer_arg(),
            scratch.projected.as_buffer_arg(),
            scratch.counters.as_buffer_arg(),
            scratch.tile_counts.as_buffer_arg(),
            scratch.tile_bbox.as_buffer_arg(),
        );

        let [num_isects_raw, num_visible] = scratch.counters.read_vec::<u32>().try_into().unwrap();
        let num_isects = num_isects_raw.min(max_isects);
        let (_sorted_keys, depth_order) = scratch.sort_depth.argsort(
            &scratch.depth_keys,
            &scratch.depth_order,
            num_visible,
            32,
            false,
        );

        // Emission offsets in depth order; dead depth_keys doubles as the
        // offsets scratch.
        exclusive_scan_gather(
            &depth_order,
            &scratch.tile_counts,
            &scratch.depth_keys,
            num_visible,
            &scratch.scan,
        );

        map_isects::launch::<WgpuRuntime>(
            client,
            calculate_cube_count_elemwise(client, num_visible as usize, cube_dim),
            cube_dim,
            depth_order.as_buffer_arg(),
            scratch.tile_bbox.as_buffer_arg(),
            scratch.depth_keys.as_buffer_arg(),
            tile_bounds.x,
            max_isects,
            scratch.tile_ids.as_buffer_arg(),
            scratch.gaussian_ids.as_buffer_arg(),
            num_visible,
        );

        // Isects are emitted in depth order, so this single stable sort by
        // tile id yields tile-major, depth-minor order — no separate gid sort
        // or composite key needed.
        let tile_bits = bits_for(num_tiles as u32);
        let (tile_ids, gaussian_ids) = scratch.sort_tile.argsort(
            &scratch.tile_ids,
            &scratch.gaussian_ids,
            num_isects,
            tile_bits,
            true,
        );

        // Stash the emission count for the next capacity check — growth only
        // ever happens between frames.
        scratch.last_isects_raw = num_isects_raw;
        TileIsects {
            tile_ids,
            gaussian_ids,
            num_isects,
            tile_bounds,
            row_stride: scratch.bitmap.shape[1] as u32,
        }
    }

    /// Render one frame into `scratch`'s buffers; the returned bitmap aliases
    /// `scratch.bitmap` and is valid until the next `render_with` on it.
    ///
    /// A `scratch` that doesn't match the splat count or image size is rebuilt
    /// in place.
    pub fn render_with(
        &self,
        scratch: &mut RenderScratch,
        camera: &Camera,
        img_size: glam::UVec2,
    ) -> GpuTensor {
        let isects = self.prepare_isects(scratch, camera, img_size);
        self.finalize(scratch, &isects, img_size, Finalize::Rgb);
        scratch.bitmap.clone()
    }

    /// The one `rasterize_kernel` finalize dispatch shared by the RGB and ε
    /// paths: `mode` resolves the kernel's ten mode-dependent slots, the
    /// remaining six arguments are the shared tile-pipeline plumbing. The
    /// RGB mode blends into `scratch.bitmap`, which callers clone to hand
    /// the bitmap out (it aliases the scratch).
    pub(crate) fn finalize(
        &self,
        scratch: &RenderScratch,
        isects: &TileIsects,
        img_size: glam::UVec2,
        mode: Finalize<'_>,
    ) {
        let dead = || BufferArg::alias(0, isects.num_isects as usize);
        let (rgb, evidence, scale, row_stride, mask, words, bitmap, resp, fg, bg) = match mode {
            Finalize::Rgb => (
                true,
                false,
                0.0,
                isects.row_stride,
                dead(),
                0,
                scratch.bitmap.as_buffer_arg(),
                dead(),
                dead(),
                dead(),
            ),
            Finalize::Total { scale, resp } => (
                false,
                false,
                scale,
                0,
                dead(),
                0,
                dead(),
                resp.as_buffer_arg(),
                dead(),
                dead(),
            ),
            Finalize::Evidence {
                scale,
                mask,
                words_per_row,
                fg,
                bg,
            } => (
                false,
                true,
                scale,
                0,
                mask.as_buffer_arg(),
                words_per_row,
                dead(),
                dead(),
                fg.as_buffer_arg(),
                bg.as_buffer_arg(),
            ),
        };
        rasterize_kernel::launch::<WgpuRuntime>(
            &self.attributes.client,
            CubeCount::new_2d(isects.tile_bounds.x, isects.tile_bounds.y),
            CubeDim::new_2d(layout::TILE_WIDTH, layout::TILE_WIDTH),
            img_size.x,
            img_size.y,
            row_stride,
            isects.num_isects,
            scale,
            isects.tile_ids.as_buffer_arg(),
            isects.gaussian_ids.as_buffer_arg(),
            scratch.projected.as_buffer_arg(),
            rgb,
            evidence,
            mask,
            words,
            bitmap,
            resp,
            fg,
            bg,
        );
    }
}

/// Sorted per-tile intersection lists plus the frame geometry the finalize
/// kernels traverse them with.
pub(crate) struct TileIsects {
    pub(crate) tile_ids: GpuTensor,
    pub(crate) gaussian_ids: GpuTensor,
    pub(crate) num_isects: u32,
    pub(crate) tile_bounds: glam::UVec2,
    pub(crate) row_stride: u32,
}

/// Field-major attributes for `n` opaque splats — `position(i)` places splat
/// i; every other plane is inert-but-valid (identity rotation, tiny scale,
/// high opacity) so tests vary geometry only. Test-only: shared by the
/// render and seg tests.
#[cfg(test)]
pub fn sample_opaque_attributes(
    n: usize,
    mut position: impl FnMut(usize) -> glam::Vec3,
) -> Vec<f32> {
    let mut a = vec![0f32; n * layout::ATTR_PLANES];
    for (i, p) in (0..n).map(|i| (i, position(i))) {
        a[layout::PLANE_X * n + i] = p.x;
        a[layout::PLANE_Y * n + i] = p.y;
        a[layout::PLANE_Z * n + i] = p.z;
        a[layout::PLANE_QW * n + i] = 1.0;
        a[layout::PLANE_SX * n + i] = -2.0;
        a[layout::PLANE_SY * n + i] = -2.0;
        a[layout::PLANE_SZ * n + i] = -2.0;
        a[layout::PLANE_OPACITY * n + i] = 8.0;
    }
    a
}

/// Opaque-DC test fixture: [`sample_opaque_attributes`] + zero SH — tests
/// vary geometry only.
#[cfg(test)]
pub(crate) fn opaque_splats(client: &ComputeClient<WgpuRuntime>, attributes: Vec<f32>) -> Splats {
    let n = attributes.len() / layout::ATTR_PLANES;
    Splats::new(attributes, vec![0.0; n * 3], client)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Golden from current rasterizer; regenerate only with documented behavior change.
    const GOLDEN_CENTER: [i32; 3] = [127, 127, 127];

    /// A 3-splat CPU scene: splat i at (i as f32, 0, 5) — left/center/right
    /// in the default camera's 32×32 frame.
    fn three_splat_cpu() -> CpuSplats {
        CpuSplats {
            attributes: sample_opaque_attributes(3, |i| glam::vec3(i as f32 - 1.0, 0.0, 5.0)),
            sh_coeffs: vec![0.0; 3 * 3],
        }
    }

    /// gather is a pure plane-wise selection: identity for all indices, and
    /// for a subset every plane keeps only the kept columns in order.
    #[test]
    fn test_gather_selects_planes() {
        let mut cpu = three_splat_cpu();
        // Stamp distinct values so plane order mismatches would show.
        for (k, v) in cpu.attributes.iter_mut().enumerate() {
            *v = k as f32;
        }
        assert_eq!(cpu.count(), 3);

        let all = cpu.gather(&(0..3).collect::<Vec<_>>());
        assert_eq!(all.attributes, cpu.attributes);
        assert_eq!(all.sh_coeffs, cpu.sh_coeffs);

        let some = cpu.gather(vec![2, 0].as_slice());
        let n = 3;
        for k in 0..layout::ATTR_PLANES {
            for (j, &i) in [2usize, 0].iter().enumerate() {
                assert_eq!(
                    some.attributes[k * 2 + j],
                    cpu.attributes[k * n + i],
                    "plane {k}"
                );
            }
        }
        // 3 DC planes mirrored in sh_coeffs.
        assert_eq!(some.sh_coeffs.len(), 3 * 2);
    }

    /// select_in_rect mirrors the rendered view: the rect keeps splats whose
    /// projected centers land inside it, drops off-frame and behind-camera
    /// ones, and returns ascending indices.
    #[test]
    fn test_select_in_rect_projects_like_the_view() {
        let cpu = three_splat_cpu();
        let cam = Camera::default();
        let px = glam::uvec2(32, 32);
        // Center column of the frame: only the middle splat (x = 0).
        let sel = cpu.select_in_rect(&cam, px, glam::vec2(12.0, 0.0), glam::vec2(20.0, 32.0), &[]);
        assert_eq!(sel, vec![1]);

        // Full frame: every splat in front of the camera.
        let sel = cpu.select_in_rect(&cam, px, glam::Vec2::ZERO, glam::vec2(32.0, 32.0), &[]);
        assert_eq!(sel, vec![0, 1, 2]);

        // Nothing behind the camera is ever selected, wherever the rect is.
        let mut behind = three_splat_cpu();
        let n = 3;
        behind.attributes[layout::PLANE_Z * n] = -5.0;
        let sel = behind.select_in_rect(&cam, px, glam::Vec2::ZERO, glam::vec2(32.0, 32.0), &[]);
        assert_eq!(sel, vec![1, 2]);
    }

    /// Selection is reported in display numbering (master minus the removed
    /// set) even though the projection runs over master positions — the
    /// numbering the GPU buffers and the delete mapping use. This is the
    /// select-after-delete contract: master 0 is gone, so the middle splat
    /// (master 1) is display 0.
    #[test]
    fn test_select_in_rect_reports_display_numbering() {
        let cpu = three_splat_cpu();
        let cam = Camera::default();
        let px = glam::uvec2(32, 32);
        let sel = cpu.select_in_rect(
            &cam,
            px,
            glam::vec2(12.0, 0.0),
            glam::vec2(20.0, 32.0),
            &[0],
        );
        assert_eq!(sel, vec![0]);
        let sel = cpu.select_in_rect(&cam, px, glam::Vec2::ZERO, glam::vec2(32.0, 32.0), &[0]);
        assert_eq!(sel, vec![0, 1]);
    }

    /// The delete fold the app performs — map the display selection through
    /// the keep-list into master numbering, merge into the removed-set —
    /// traced on a 5-splat master with master 2 already removed: selecting
    /// display [0, 2, 3] (masters 0, 3, 4) leaves only master 1. The tail
    /// display index maps to the last master.
    #[test]
    fn test_delete_fold_through_kept() {
        let cpu = CpuSplats {
            attributes: sample_opaque_attributes(5, |_| glam::Vec3::ZERO),
            sh_coeffs: vec![0.0; 5 * 3],
        };
        let removed = vec![2usize];
        let sel = [0usize, 2, 3]; // as select_in_rect would report

        let kept = cpu.kept(&removed);
        assert_eq!(kept, vec![0, 1, 3, 4]);
        assert_eq!(kept[3], 4, "the last display splat is the last master");

        let mut removed = removed;
        removed.extend(sel.iter().map(|&d| kept[d]));
        removed.sort_unstable();
        assert_eq!(removed, vec![0, 2, 3, 4]);
        assert_eq!(cpu.kept(&removed), vec![1]);
    }

    /// The delete→undo cycle through the GPU path, exactly as the app does
    /// it: deletion re-gathers the master minus the removed-set and changes
    /// the frame; undo restores the removed-set and the frame returns
    /// bit-for-bit (the master is the source of truth, uploads are
    /// deterministic).
    #[test]
    fn test_delete_then_undo_restores_render_bitwise() {
        let (_gpu, client) = crate::gpu_testing::test_client();
        let cpu = three_splat_cpu();
        let cam = Camera::default();
        let px = glam::uvec2(32, 32);
        let render = |keep: &[usize]| {
            let splats = cpu.gather(keep).upload(&client);
            let mut scratch = RenderScratch::new(&client, splats.attributes.shape[0], px);
            splats.render_with(&mut scratch, &cam, px).read_vec::<u32>()
        };
        let all: Vec<usize> = (0..cpu.count()).collect();
        let before = render(&all);

        // Delete the left splat: the left of the frame empties.
        let after_delete = render(&[1, 2]);
        assert_ne!(after_delete, before, "deletion must change the frame");

        // Undo restores the removed-set: the keep-list is whole again and
        // the frame matches bit-for-bit.
        assert_eq!(render(&all), before);
    }

    /// Unpack a packed RGBA8 pixel (r | g<<8 | b<<16 | a<<24).
    fn rgba(px: u32) -> [i32; 4] {
        let [r, g, b, a] = px.to_le_bytes();
        [r as i32, g as i32, b as i32, a as i32]
    }

    /// Render `n` splats with `attributes` under `camera` into a 32×32 frame;
    /// returns (center pixel, corner pixel).
    fn render_corner_pixels(
        client: &ComputeClient<WgpuRuntime>,
        n: usize,
        attributes: Vec<f32>,
        camera: &Camera,
    ) -> (u32, u32) {
        let splats = opaque_splats(client, attributes);
        let mut scratch = RenderScratch::new(client, n, glam::uvec2(32, 32));
        let bitmap = splats.render_with(&mut scratch, camera, glam::uvec2(32, 32));
        let px: Vec<u32> = bitmap.read_vec();
        let stride = bitmap.shape[1] as usize;
        (px[16 * stride + 16], px[0])
    }

    #[test]
    fn test_render_golden_center() {
        let (_gpu, client) = crate::gpu_testing::test_client();
        let n = 5usize;
        // Five opaque splats stacked in depth at the same XY, plus the same
        // stack seen by a camera yawed 90° and translated to (-2, 0, 0): a
        // transposed rotation row or a flipped/dropped translation sends the
        // stack behind the camera, which the identity case cannot distinguish.
        let yawed = Camera {
            position: glam::Vec3::new(-2.0, 0.0, 0.0),
            rotation: glam::Quat::from_rotation_y(core::f32::consts::FRAC_PI_2),
            ..Camera::default()
        };
        let cases = [
            (
                "stacked",
                sample_opaque_attributes(n, |i| glam::vec3(0.0, 0.0, 1.0 + i as f32 * 0.5)),
                Camera::default(),
            ),
            (
                "yawed",
                sample_opaque_attributes(n, |_| glam::Vec3::ZERO),
                yawed,
            ),
        ];
        for (name, attributes, camera) in cases {
            let (center, corner) = render_corner_pixels(&client, n, attributes, &camera);
            let [r, g, b, a] = rgba(center);
            // quantize_u8 truncates; f32 residual transmittance leaves 255-1 LSB.
            assert!(a >= 254, "{name}: opaque stack must saturate alpha ({a})");
            assert_eq!(
                corner & 0xFF00_0000,
                0,
                "{name}: uncovered pixel must be transparent"
            );
            for (got, want) in [r, g, b].iter().zip(GOLDEN_CENTER) {
                assert!((got - want).abs() <= 1, "{name}: center color {r},{g},{b}");
            }
        }
    }

    #[test]
    fn test_render_reuses_scratch() {
        let (_gpu, client) = crate::gpu_testing::test_client();
        let attributes: Vec<f32> = (0..50 * 11)
            .map(|i| (i as f32 * 0.13).sin() * 0.5)
            .collect();
        let sh = vec![0.1f32; 50 * 3];
        let splats = Splats::new(attributes, sh, &client);
        let camera = crate::camera::Camera {
            position: glam::Vec3::new(0.0, -1.0, 0.0),
            ..Camera::default()
        };
        let mut scratch = RenderScratch::new(&client, 50, glam::uvec2(64, 64));
        let b = splats
            .render_with(&mut scratch, &camera, glam::uvec2(64, 64))
            .read_vec::<u32>();
        let c = splats
            .render_with(&mut scratch, &camera, glam::uvec2(64, 64))
            .read_vec::<u32>();
        assert_eq!(b, c);
    }

    #[test]
    fn test_scratch_stops_allocating_after_first_frame() {
        let (_gpu, client) = crate::gpu_testing::test_client();
        let attributes: Vec<f32> = (0..100 * 11)
            .map(|i| (i as f32 * 0.07).sin() * 0.4)
            .collect();
        let splats = opaque_splats(&client, attributes);
        let camera = crate::camera::Camera {
            position: glam::Vec3::new(0.0, -1.2, 0.0),
            ..Camera::default()
        };
        let mut scratch = RenderScratch::new(&client, 100, glam::uvec2(64, 64));
        splats.render_with(&mut scratch, &camera, glam::uvec2(64, 64));
        // The runtime caches one client per device, so other tests' deferred
        // frees and pool reclaim land inside the measurement window and can
        // spike a single reading. Fail only on sustained growth — systematic
        // per-frame allocation trips every attempt.
        let mut prev = client.memory_usage().unwrap().bytes_in_use;
        let mut steady = false;
        for _ in 0..5 {
            splats.render_with(&mut scratch, &camera, glam::uvec2(64, 64));
            let cur = client.memory_usage().unwrap().bytes_in_use;
            if cur <= prev + 65_536 {
                steady = true;
                break;
            }
            prev = cur;
        }
        assert!(steady, "steady-state frames keep allocating memory");
    }

    /// A shape rebuild must not discard the growth hint: the first prepare
    /// at the new shape still grows the isect buffers toward the previous
    /// emission, else that frame's map_isects silently truncates.
    #[test]
    fn test_shape_change_keeps_isects_capacity_hint() {
        let (_gpu, client) = crate::gpu_testing::test_client();
        let mut scratch = RenderScratch::new(&client, 1, glam::uvec2(64, 64));
        // A scene that outgrew the initial capacity at the old shape.
        scratch.last_isects_raw = INITIAL_ISECTS_CAP as u32 + 1;

        scratch.prepare(&client, 1, glam::uvec2(32, 32));

        assert!(
            scratch.tile_ids.shape[0] > INITIAL_ISECTS_CAP,
            "rebuild must keep the growth hint: capacity {}",
            scratch.tile_ids.shape[0]
        );
    }

    /// 200 splats at one exact depth are the tie stress case —
    /// every depth key is equal, so the racy atomic compaction order and the
    /// radix sort's tie handling are fully exercised. The splats are
    /// attribute-identical, so blend order among ties cannot change the
    /// result: repeated renders — through the same scratch and through a
    /// fresh one — must be bit-identical.
    #[test]
    fn test_render_ties_deterministic() {
        let (_gpu, client) = crate::gpu_testing::test_client();
        let n = 200usize;
        let splats = opaque_splats(
            &client,
            sample_opaque_attributes(n, |_| glam::vec3(0.0, 0.0, 1.0)),
        );
        let camera = crate::camera::Camera::default();
        let mut scratch = RenderScratch::new(&client, n, glam::uvec2(64, 64));
        let a = splats
            .render_with(&mut scratch, &camera, glam::uvec2(64, 64))
            .read_vec::<u32>();
        let b = splats
            .render_with(&mut scratch, &camera, glam::uvec2(64, 64))
            .read_vec::<u32>();
        let mut fresh = RenderScratch::new(&client, n, glam::uvec2(64, 64));
        let c = splats
            .render_with(&mut fresh, &camera, glam::uvec2(64, 64))
            .read_vec::<u32>();
        assert_eq!(a, b, "same-scratch re-render must be bit-identical");
        assert_eq!(a, c, "fresh-scratch render must be bit-identical");
    }

    /// Test-only probe of the rasterizer's per-tile range lookup, sharing
    /// the production `lower_bound`.
    #[cube(launch)]
    fn probe_tile_range(ids: &[u32], num_isects: u32, tile: u32, range: &mut [u32]) {
        if UNIT_POS == 0 {
            range[0] = crate::raster::lower_bound(ids, num_isects, tile);
            range[1] = crate::raster::lower_bound(ids, num_isects, tile + 1u32);
        }
    }

    #[test]
    fn test_tile_ranges_binary_search_matches_cpu() {
        let (_gpu, client) = crate::gpu_testing::test_client();
        // Sorted tile ids: tiles 0, 1, 2, 5 have intersections; 3, 4, 6, 7
        // are empty. Every tile's rasterizer range must match a CPU
        // lower-bound pair over the same ids.
        let ids: Vec<u32> = vec![0, 0, 1, 2, 2, 2, 5, 5];
        let ids_t = GpuTensor::from(&client, [ids.len()], &ids[..]);

        let lower_bound_host = |target: u32| ids.partition_point(|&k| k < target) as u32;

        for tile in 0..8u32 {
            let range = GpuTensor::empty(&client, [2]);
            probe_tile_range::launch::<WgpuRuntime>(
                &client,
                CubeCount::new_single(),
                CubeDim::new_1d(layout::TILE_SIZE),
                ids_t.as_buffer_arg(),
                ids.len() as u32,
                tile,
                range.as_buffer_arg(),
            );
            let r: Vec<u32> = range.read_vec();
            let expect = [lower_bound_host(tile), lower_bound_host(tile + 1)];
            assert_eq!(r, expect, "tile {tile}: rasterizer range");
        }
    }
}
