//! The GUI's scene state: the loaded model, the box-select / delete / undo
//! / cut edits over it, and the viewport render path. Declared from
//! `main.rs` — binary code, so library items go through `splatfield::`.

use std::sync::Arc;

use cubecl::client::ComputeClient;
use cubecl::wgpu::WgpuRuntime;
use eframe::egui;
use splat_sort::tensor::GpuTensor;
use splatfield::seg::beta::map_labels;
use splatfield::{camera, render, texture, to_dc};

use super::{App, SELECT_GREEN};

/// A loaded scene: the GPU model, the pristine CPU master it rebuilds
/// from, the untouched DC color snapshot (for the reset button), and the
/// source path the save button writes alongside. Fields are always set
/// together, so a half-loaded state is unrepresentable.
pub(crate) struct Scene {
    pub(crate) splats: Arc<render::Splats>,
    /// Pristine CPU master the box-select deletion gathers from — the GPU
    /// upload consumes the parser's buffers, so this is the only host copy.
    pub(crate) cpu: render::CpuSplats,
    pub(crate) colors: GpuTensor,
    /// Selection-paint scratch sized to the live splat count: the mask is
    /// rewritten per selection via stream-ordered `write` (no GPU alloc),
    /// the zeros are uploaded once and only ever read by the tint.
    pub(crate) sel_mask: GpuTensor,
    pub(crate) sel_zeros: GpuTensor,
    pub(crate) path: std::path::PathBuf,
}

/// The loaded scene and its "reframe the camera" flag under one lock.
/// `load_gen` counts load requests: whichever was requested last wins, and
/// a stale late finish can't clobber a newer model.
#[derive(Default)]
pub(crate) struct Loaded {
    pub(crate) scene: Option<Scene>,
    pub(crate) reframe: bool,
    pub(crate) load_gen: u64,
    /// Load failures (drag-drop), surfaced in the status pill — a GUI
    /// launch has no stderr to read.
    pub(crate) load_error: Option<String>,
}

/// GPU frame state: the presentation texture and the per-frame scratch
/// (built on first use, reused every frame after).
pub(crate) struct FrameGpu {
    pub(crate) backbuffer: texture::GpuTexture,
    pub(crate) scratch: Option<render::RenderScratch>,
}

impl App {
    pub(crate) fn load_file(&self, path: &std::path::Path, ctx: egui::Context) {
        let client = self.client.clone();
        let splats = Arc::clone(&self.splats);

        // Claim the next generation up front: of two racing loads, the one
        // requested LAST wins and a stale late finish is discarded.
        let load_id = {
            let mut slot = splats.lock().unwrap();
            slot.load_gen += 1;
            slot.load_gen
        };

        let path = path.to_owned();
        let save_path = path.clone();
        let on_loaded = move |result: anyhow::Result<render::CpuSplats>| {
            let payload = result.map(|cpu| {
                let data = Arc::new(cpu.clone().upload(&client));
                let colors = data.save_colors();
                (cpu, data, colors)
            });
            match payload {
                Ok((cpu, data, colors)) => {
                    let mut slot = splats.lock().unwrap();
                    if slot.load_gen != load_id {
                        return; // superseded by a newer load request
                    }
                    let n = data.attributes.shape[0];
                    slot.scene = Some(Scene {
                        splats: data,
                        cpu,
                        colors,
                        sel_mask: GpuTensor::empty(&client, [n]),
                        sel_zeros: GpuTensor::from(&client, [n], vec![0f32; n]),
                        path: save_path,
                    });
                    slot.reframe = true;
                    drop(slot);
                }
                Err(e) => {
                    eprintln!("Failed to load splat: {e:#}");
                    let mut slot = splats.lock().unwrap();
                    if slot.load_gen != load_id {
                        return;
                    }
                    slot.load_error = Some(format!("{e:#}"));
                    drop(slot);
                }
            }
            ctx.request_repaint();
        };

        std::thread::spawn(move || {
            on_loaded(splatfield::load_scene_file(&path));
        });
    }

    /// Box-select: keep the splats whose projected centers fall in the
    /// viewport-pixel rect and repaint the highlight.
    pub(crate) fn select_in_rect(&mut self, pixel: glam::UVec2, min: glam::Vec2, max: glam::Vec2) {
        if self.seg.busy() {
            return;
        }
        {
            let slot = self.splats.lock().unwrap();
            let Some(scene) = &slot.scene else {
                return;
            };
            self.sel =
                scene
                    .cpu
                    .select_in_rect(&self.controller.camera, pixel, min, max, &self.removed);
        }
        self.paint_selection();
    }

    /// Repaint the selection highlight: restore the pristine colors, then
    /// tint the selected splats green. An empty selection is a plain
    /// restore.
    pub(crate) fn paint_selection(&mut self) {
        let slot = self.splats.lock().unwrap();
        let Some(scene) = &slot.scene else {
            return;
        };
        let splats = &scene.splats;
        splats.restore_colors(&scene.colors);
        if !self.sel.is_empty() {
            let n = splats.attributes.shape[0];
            let mut mask = vec![0f32; n];
            for &i in &self.sel {
                mask[i] = 1.0;
            }
            scene.sel_mask.write(mask);
            splats.tint(
                &scene.sel_mask,
                &scene.sel_zeros,
                to_dc(SELECT_GREEN.map(|c| f32::from(c) / 255.0)),
            );
        }
        drop(slot);
        self.paint_dirty = true;
    }

    /// Delete the selected splats: fold them into the removed-set (their
    /// previous state is the single undo level) and rebuild the scene from
    /// the pristine CPU master.
    pub(crate) fn delete_selection(&mut self) {
        if self.seg.busy() || self.sel.is_empty() {
            return;
        }
        let slot = self.splats.lock().unwrap();
        let Some(scene) = &slot.scene else {
            return;
        };
        // sel indexes the display: positions in the keep-list over the
        // master. Fold them into master numbering and merge into the
        // removed-set. A selection that doesn't fit the live model — stale
        // by at most one frame — or that would delete everything (the
        // renderer needs n > 0) is dropped, never applied.
        let kept = scene.cpu.kept(&self.removed);
        if kept.len() != scene.splats.attributes.shape[0]
            || self.sel.len() >= kept.len()
            || self.sel.last().is_some_and(|&d| d >= kept.len())
        {
            return;
        }
        let masters: Vec<usize> = self.sel.iter().map(|&d| kept[d]).collect();
        drop(slot);
        self.apply_removal(masters, None);
    }

    /// Fold `masters` into the removed-set as one undo level (a cut also
    /// snapshots its pre-cut DC `colors`), clear the selection, rebuild.
    /// The caller has validated non-destruction and dropped the scene lock.
    pub(crate) fn apply_removal(&mut self, masters: Vec<usize>, colors: Option<GpuTensor>) {
        let mut removed = std::mem::take(&mut self.removed);
        self.undo.push((removed.clone(), colors));
        removed.extend(masters);
        removed.sort_unstable();
        self.removed = removed;
        self.sel.clear();
        self.rebuild();
    }

    /// Undo the last delete or cut by restoring its removed-set; a cut also
    /// repaints its pre-cut colors. One level per keystroke, last action
    /// first.
    pub(crate) fn undo_delete(&mut self) {
        if self.seg.busy() {
            return;
        }
        if let Some((prev, colors)) = self.undo.pop() {
            self.removed = prev;
            self.sel.clear();
            self.rebuild();
            if let Some(colors) = colors
                && let Some(scene) = self.splats.lock().unwrap().scene.as_ref()
            {
                scene.splats.restore_colors(&colors);
            }
        }
    }

    /// Cut the segmented object out of the scene: drop every splat the
    /// posterior puts in the background, rebuilding from the pristine CPU
    /// master so the extracted object shows its original colors. The
    /// pre-cut removed-set and DC colors become the undo level.
    pub(crate) fn cut_object(&mut self) {
        if self.seg.busy() {
            return;
        }
        let Some((a, b)) = &self.seg.result else {
            return;
        };
        let slot = self.splats.lock().unwrap();
        let Some(scene) = &slot.scene else {
            return;
        };
        // A posterior from before a box-delete no longer matches the live
        // scene's numbering — cut only when the counts agree.
        let kept = scene.cpu.kept(&self.removed);
        if kept.len() != scene.splats.attributes.shape[0] || kept.len() != a.shape[0] {
            return;
        }
        let (a, b): (Vec<f32>, Vec<f32>) = (a.read_vec(), b.read_vec());
        let labels = map_labels(&a, &b);
        let cut: Vec<usize> = (0..kept.len())
            .filter_map(|i| (!labels[i]).then_some(kept[i]))
            .collect();
        // A no-op cut — everything object or everything background — must
        // not spend the undo level; removing everything starves the renderer.
        if cut.is_empty() || cut.len() == kept.len() {
            return;
        }
        let colors = scene.splats.save_colors();
        drop(slot);
        self.apply_removal(cut, Some(colors));
    }

    /// Rebuild the GPU scene from the CPU master minus the removed-set. A
    /// fresh Arc repaints automatically; the color snapshot is retaken at
    /// the new count.
    pub(crate) fn rebuild(&mut self) {
        let mut slot = self.splats.lock().unwrap();
        let Some(scene) = &mut slot.scene else {
            return;
        };
        let kept = scene.cpu.kept(&self.removed);
        let splats = Arc::new(scene.cpu.gather(&kept).upload(&self.client));
        scene.colors = splats.save_colors();
        let n = splats.attributes.shape[0];
        scene.sel_mask = GpuTensor::empty(&self.client, [n]);
        scene.sel_zeros = GpuTensor::from(&self.client, [n], vec![0f32; n]);
        scene.splats = splats;
    }

    /// Save the live scene — current colors and removals exactly as shown —
    /// as a PLY next to the source file. Every plane except SH DC is the
    /// pristine master's gather (tint/restore mutate only DC on the GPU),
    /// so only the 3n DC floats are read back; a full-scene readback synced
    /// the UI thread for tens of MB for nothing.
    pub(crate) fn save_ply(&mut self) {
        if self.seg.busy() {
            return;
        }
        let (path, cpu) = {
            let slot = self.splats.lock().unwrap();
            let Some(scene) = &slot.scene else {
                return;
            };
            let kept = scene.cpu.kept(&self.removed);
            let mut cpu = scene.cpu.gather(&kept);
            let dc = scene.splats.save_colors().read_vec();
            cpu.sh_coeffs[..dc.len()].copy_from_slice(&dc);
            (scene.path.clone(), cpu)
        };
        let out = path.with_extension("edited.ply");
        let saved = || -> anyhow::Result<()> { cpu.write_ply(std::fs::File::create(&out)?) }();
        let (msg, error) = match saved {
            Ok(()) => (format!("saved {}", out.display()), false),
            Err(e) => (format!("save failed: {e}"), true),
        };
        self.set_status(msg, error);
    }
}

/// Render `splats` into the frame's scratch and present the bitmap to the
/// backbuffer; the pipeline syncs on the counters readback, on the UI thread.
/// With `posterior` set, splats are painted by their Beta posterior mean
/// instead of their SH color.
pub(crate) fn render_frame(
    client: &ComputeClient<WgpuRuntime>,
    frame: &mut FrameGpu,
    splats: &render::Splats,
    camera: &camera::Camera,
    pixel: glam::UVec2,
    posterior: Option<&(GpuTensor, GpuTensor)>,
) {
    let img = {
        let scratch = frame.scratch.get_or_insert_with(|| {
            render::RenderScratch::new(client, splats.attributes.shape[0], pixel)
        });
        match posterior {
            Some((a, b)) => splats.render_posterior(scratch, a, b, camera, pixel),
            None => splats.render_with(scratch, camera, pixel),
        }
    };
    frame.backbuffer.update_texture(&img, pixel);
}
