//! The GUI's scene state: the loaded model, the box-select / delete / undo
//! / cut edits over it, and the viewport render path. Declared from
//! `main.rs` — binary code, so library items go through `splatfield::`.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

#[cfg(target_arch = "wasm32")]
use eframe::egui;
#[cfg(target_arch = "wasm32")]
use eframe::wasm_bindgen::JsCast;
use splat_sort::tensor::GpuTensor;
#[cfg(target_arch = "wasm32")]
use splatfield::fetch;
#[cfg(target_arch = "wasm32")]
use splatfield::seg::beta::map_labels;
use splatfield::{camera, render, texture, to_dc};

use super::{App, SELECT_GREEN};

/// A loaded scene: the GPU model, the pristine CPU master it rebuilds
/// from, and the untouched DC color snapshot (for the reset button).
/// Fields are always set together, so a half-loaded state is unrepresentable.
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
    /// The source FILE NAME (web has no directories): the save button
    /// derives `<name>.edited.ply` from it.
    pub(crate) name: String,
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
    /// Finished saves report (message, error) here — the async save task
    /// can't borrow `self`, so the next frame drains it into the status
    /// pill, mirroring `load_error`.
    pub(crate) save_result: Option<(String, bool)>,
    /// The wasm cut's two-phase state (see `cut_object`): `cut_in_flight`
    /// from click until the staged outcome drains, then `cut` holds the
    /// outcome for the next frame's UI drain. Both live under this lock so
    /// the release rule is one testable place: the lock re-opens only when
    /// a staged cut actually lands — an empty drain means the readback is
    /// still running and every numbering-sensitive edit stays out.
    pub(crate) cut_in_flight: bool,
    /// The wasm cut task's staged outcome, drained by the UI like
    /// `save_result`. (Host never stages — the host cut stub is a no-op —
    /// so the field sits empty there.)
    pub(crate) cut: Option<CutStaged>,
}

/// GPU frame state: the presentation texture and the per-frame scratch
/// (built on first use, reused every frame after).
pub(crate) struct FrameGpu {
    pub(crate) backbuffer: texture::GpuTexture,
    pub(crate) scratch: Option<render::RenderScratch>,
}

/// Staged outcome of a cut's async readback (wasm): `None` apply is a no-op
/// cut (everything object or everything background — it must not spend the
/// undo level). The readback task stages it; the next frame's UI drain
/// applies it on the UI thread, where the undo/removed bookkeeping lives.
pub(crate) struct CutStaged {
    pub(crate) apply: Option<(Vec<usize>, GpuTensor)>,
}

impl Loaded {
    /// Open the cut lock: the readback task may now stage against the
    /// scene, and every numbering-sensitive edit is shut out until
    /// [`Loaded::drain_cut`] lands a staged outcome.
    pub(crate) fn begin_cut(&mut self) {
        self.cut_in_flight = true;
    }

    /// The readback task publishes its outcome (a no-op cut stages
    /// `apply: None` — it still releases the lock). Wasm-only caller.
    pub(crate) fn stage_cut(&mut self, staged: CutStaged) {
        self.cut = Some(staged);
    }

    /// Take the staged cut, releasing the lock ONLY when one actually
    /// landed. Clearing on an empty drain would re-open the edits while the
    /// readback is still in flight: a delete/undo in that window folds the
    /// landing cut over a renumbered scene, and a box-select's tint lands
    /// in the cut's undo-level colors.
    pub(crate) fn drain_cut(&mut self) -> Option<CutStaged> {
        let staged = self.cut.take();
        if staged.is_some() {
            self.cut_in_flight = false;
        }
        staged
    }
}

/// The render task moves the frame state out for the duration of a frame, so
/// no borrow is held across its await; `None` means a render is in flight.
pub(crate) type FrameSlot = Rc<RefCell<Option<FrameGpu>>>;

impl App {
    /// Parse and upload scene bytes off the UI thread; of two racing loads
    /// the one requested LAST wins. The `bytes` future is the only
    /// difference between callers: the drop path reads a browser file
    /// handle, the demo button fetches the same-origin demo scene.
    #[cfg(target_arch = "wasm32")]
    pub(crate) fn load_bytes(
        &self,
        name: String,
        ctx: egui::Context,
        bytes: impl std::future::Future<Output = anyhow::Result<Vec<u8>>> + 'static,
    ) {
        let client = self.client.clone();
        let splats = Arc::clone(&self.splats);

        // Claim the next generation up front: of two racing loads, the one
        // requested LAST wins and a stale late finish is discarded.
        let load_id = {
            let mut slot = splats.lock().unwrap();
            slot.load_gen += 1;
            slot.load_gen
        };

        // The source FILE NAME (web has no directories) — the save button
        // derives `<name>.edited.ply` from it — and the extension
        // `load_scene` dispatches on.
        let ext = std::path::Path::new(&name)
            .extension()
            .unwrap_or_default()
            .to_owned();

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
                        name,
                    });
                    slot.reframe = true;
                    drop(slot);
                }
                Err(e) => {
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

        wasm_bindgen_futures::spawn_local(async move {
            match bytes.await {
                Ok(bytes) => on_loaded(splatfield::load_scene(&ext, std::io::Cursor::new(bytes))),
                Err(e) => on_loaded(Err(e)),
            }
        });
    }

    /// A drag-and-dropped file: read its bytes through the browser handle.
    /// `file.path()` on wasm is the file NAME (eframe sets it from
    /// File::name) — extension detection still works.
    #[cfg(target_arch = "wasm32")]
    pub(crate) fn load_file(&self, file: egui::DroppedFileHandle, ctx: egui::Context) {
        let name = std::path::Path::new(file.path())
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "scene".into());
        self.load_bytes(name, ctx, async move {
            file.bytes_async()
                .await
                .map_err(|e| anyhow::anyhow!("Failed to read dropped file: {e}"))
        });
    }

    /// The help panel's demo row — fetch the fixtures-release bear and load
    /// it like a drop — shared by the hover tooltip and the click-pinned
    /// panel: egui keeps tooltips containing interactive widgets
    /// interactable, so the button is clickable in both. Disabled like a
    /// drop is guarded: loading during a run would leave the worker tinting
    /// a stale model.
    #[cfg(target_arch = "wasm32")]
    pub(crate) fn demo_button(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        if ui
            .add_enabled(!self.locked(), egui::Button::new("load demo scene (bear)"))
            .clicked()
        {
            self.set_status("fetching demo scene…", false);
            // Dismiss the pinned panel; the status pill narrates from here.
            egui::Popup::close_all(ui.ctx());
            // The URL's file name: the save button derives
            // `<name>.edited.ply` from it, exactly like a dropped file.
            let name = fetch::DEMO_SCENE_URL.rsplit('/').next().unwrap().to_owned();
            self.load_bytes(name, ui.ctx().clone(), fetch::fetch_demo_scene());
        }
    }

    /// Box-select: keep the splats whose projected centers fall in the
    /// viewport-pixel rect and repaint the highlight.
    pub(crate) fn select_in_rect(&mut self, pixel: glam::UVec2, min: glam::Vec2, max: glam::Vec2) {
        if self.locked() {
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
        if self.locked() || self.sel.is_empty() {
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
        if self.locked() {
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
    ///
    /// The a/b readback is blocking (panics on wasm) and cannot borrow the
    /// app across an await, so on wasm the labels are computed in a task
    /// and the outcome is staged for the next frame's UI drain — all App
    /// mutation stays on the UI thread (the `save_result` pattern).
    /// [`App::locked`] keeps a second cut — and every
    /// numbering-sensitive edit — out until it lands, so `kept` (captured
    /// here) still describes the scene when the drain applies it.
    #[cfg(target_arch = "wasm32")]
    pub(crate) fn cut_object(&mut self) {
        if self.locked() {
            return;
        }
        let Some((a, b)) = &self.seg.result else {
            return;
        };
        let kept = {
            let slot = self.splats.lock().unwrap();
            let Some(scene) = &slot.scene else {
                return;
            };
            // A posterior from before a box-delete no longer matches the
            // live scene's numbering — cut only when the counts agree.
            let kept = scene.cpu.kept(&self.removed);
            if kept.len() != scene.splats.attributes.shape[0] || kept.len() != a.shape[0] {
                return;
            }
            kept
        };
        self.splats.lock().unwrap().begin_cut();
        let (a, b) = (a.clone(), b.clone());
        let splats = Arc::clone(&self.splats);
        wasm_bindgen_futures::spawn_local(async move {
            let (pa, pb) = (a.read_vec_async().await, b.read_vec_async().await);
            let labels = map_labels(&pa, &pb);
            let cut: Vec<usize> = (0..kept.len())
                .filter_map(|i| (!labels[i]).then_some(kept[i]))
                .collect();
            let apply = if cut.is_empty() || cut.len() == kept.len() {
                None
            } else {
                // Snapshot the pre-cut DC colors on the live scene; a scene
                // that vanished mid-readback leaves nothing to cut.
                let colors = splats
                    .lock()
                    .unwrap()
                    .scene
                    .as_ref()
                    .map(|s| s.splats.save_colors());
                colors.map(|colors| (cut, colors))
            };
            splats.lock().unwrap().stage_cut(CutStaged { apply });
        });
    }

    /// Native main is a tombstone — the real cut runs on wasm.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn cut_object(&mut self) {}

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
    /// as a PLY the browser downloads. Every plane except SH DC is the
    /// pristine master's gather (tint/restore mutate only DC on the GPU),
    /// so only the 3n DC floats are read back; a full-scene readback synced
    /// the UI thread for tens of MB for nothing.
    #[cfg(target_arch = "wasm32")]
    pub(crate) fn save_ply(&mut self) {
        if self.locked() {
            return;
        }
        let (name, colors, cpu) = {
            let slot = self.splats.lock().unwrap();
            let Some(scene) = &slot.scene else {
                return;
            };
            let kept = scene.cpu.kept(&self.removed);
            let cpu = scene.cpu.gather(&kept);
            let colors = scene.splats.save_colors();
            (scene.name.clone(), colors, cpu)
        };
        // std's with_extension: the source's extension is replaced, so
        // `bear.3d71a266.sog` saves as `bear.3d71a266.edited.ply` — the
        // native flow's exact naming.
        let out = std::path::PathBuf::from(&name)
            .with_extension("edited.ply")
            .to_string_lossy()
            .into_owned();
        self.set_status("saving models…", false);
        let splats = Arc::clone(&self.splats);
        wasm_bindgen_futures::spawn_local(async move {
            let saved = async {
                // The pill must paint before the main thread blocks on the
                // readback/write/copy pipeline, so yield one macrotask first.
                let delay = js_sys::eval("new Promise((resolve) => setTimeout(resolve, 50))")
                    .map_err(|e| anyhow::anyhow!("scheduling the save: {e:?}"))?;
                wasm_bindgen_futures::JsFuture::from(delay.unchecked_into::<js_sys::Promise>())
                    .await
                    .map_err(|e| anyhow::anyhow!("save interrupted: {e:?}"))?;
                let dc = colors.read_vec_async::<f32>().await;
                let mut cpu = cpu;
                cpu.sh_coeffs[..dc.len()].copy_from_slice(&dc);
                // One reservation instead of Vec doubling: the output is
                // tens to hundreds of MB and every doubling re-memcpys the
                // whole file inside linear memory.
                let n = cpu.attributes.len() / splatfield::layout::ATTR_PLANES;
                let floats = 17 + (cpu.sh_coeffs.len() / (3 * n) - 1) * 3;
                let mut bytes = Vec::with_capacity(n * floats * 4 + 1024 + (floats - 17) * 24);
                cpu.write_ply(&mut bytes)?;
                download_bytes(&bytes, &out)
            }
            .await;
            let result = match saved {
                Ok(()) => (format!("saved {out} — check your downloads"), false),
                Err(e) => (format!("save failed: {e:#}"), true),
            };
            splats.lock().unwrap().save_result = Some(result);
        });
    }

    /// Native main is a tombstone — the UI never runs here.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn save_ply(&mut self) {}
}

/// Hand `bytes` to the browser as a file download named `name`: a blob URL
/// on a detached anchor, clicked and revoked — detached anchors download
/// fine, no DOM insertion needed.
///
/// The Uint8Array goes INSIDE a JS array: `new Blob(parts)` takes a
/// sequence of BlobPart, and a bare Uint8Array is itself iterable —
/// passing it directly makes the browser walk it byte-by-byte and fall
/// through to the USVString branch, stringifying every value: slowly
/// assembling a text file of decimal digits while the page sits frozen.
#[cfg(target_arch = "wasm32")]
fn download_bytes(bytes: &[u8], name: &str) -> anyhow::Result<()> {
    let document = web_sys::window()
        .ok_or_else(|| anyhow::anyhow!("no window"))?
        .document()
        .ok_or_else(|| anyhow::anyhow!("no document"))?;
    let parts = js_sys::Array::of1(&js_sys::Uint8Array::from(bytes).into());
    let blob = web_sys::Blob::new_with_u8_array_sequence(parts.as_ref())
        .map_err(|e| anyhow::anyhow!("create blob: {e:?}"))?;
    let url = web_sys::Url::create_object_url_with_blob(&blob)
        .map_err(|e| anyhow::anyhow!("create object url: {e:?}"))?;
    let anchor = document
        .create_element("a")
        .map_err(|e| anyhow::anyhow!("create anchor: {e:?}"))?
        .dyn_into::<web_sys::HtmlAnchorElement>()
        .map_err(|e| anyhow::anyhow!("anchor is not an <a>: {e:?}"))?;
    anchor.set_href(&url);
    anchor.set_download(name);
    anchor.click();
    web_sys::Url::revoke_object_url(&url).map_err(|e| anyhow::anyhow!("revoke url: {e:?}"))?;
    Ok(())
}

/// Holds the frame state for the duration of a render and returns it to the
/// slot on drop. The drop path matters: a panic mid-pipeline would otherwise
/// leave the slot empty forever, and on wasm every later frame would see a
/// busy slot and render nothing — a frozen app with no error.
struct SlotGuard {
    slot: FrameSlot,
    frame: Option<FrameGpu>,
}

impl Drop for SlotGuard {
    fn drop(&mut self) {
        *self.slot.borrow_mut() = self.frame.take();
    }
}

/// Render `splats` into the frame's scratch and present the bitmap to the
/// backbuffer. With `posterior` set, splats are painted by their Beta
/// posterior mean instead of their SH color. The frame state moves out of
/// the slot for the duration of the pipeline's counters readback, so no
/// borrow is held across the await.
pub(crate) async fn render_frame(
    slot: &FrameSlot,
    splats: &render::Splats,
    camera: &camera::Camera,
    pixel: glam::UVec2,
    posterior: Option<(GpuTensor, GpuTensor)>,
) {
    let Some(frame) = slot.borrow_mut().take() else {
        return;
    };
    let mut guard = SlotGuard {
        slot: Rc::clone(slot),
        frame: Some(frame),
    };
    if let Some(frame) = guard.frame.as_mut() {
        let client = &splats.attributes.client;
        let scratch = frame.scratch.get_or_insert_with(|| {
            render::RenderScratch::new(client, splats.attributes.shape[0], pixel)
        });
        let img = match &posterior {
            Some((a, b)) => {
                splats
                    .render_posterior_async(scratch, a, b, camera, pixel)
                    .await
            }
            None => splats.render_with_async(scratch, camera, pixel).await,
        };
        frame.backbuffer.update_texture(&img, pixel);
    }
}

#[cfg(test)]
mod cut_lock_tests {
    use super::{CutStaged, Loaded};

    /// The cut lock releases only when a staged cut actually drains: a
    /// frame whose drain finds nothing (readback still in flight) must keep
    /// `cut_in_flight` set, or the numbering-sensitive edits re-open in the
    /// window before the outcome lands.
    #[test]
    fn cut_lock_releases_only_when_a_staged_cut_drains() {
        let mut loaded = Loaded::default();
        loaded.begin_cut();
        // The readback is still running: the drain finds nothing and the
        // lock must hold.
        assert!(loaded.drain_cut().is_none());
        assert!(loaded.cut_in_flight);
        // The outcome lands — here the no-op cut, `apply: None`, which
        // carries no GPU payload and still releases the lock.
        loaded.stage_cut(CutStaged { apply: None });
        assert!(loaded.drain_cut().is_some());
        assert!(!loaded.cut_in_flight);
    }
}
