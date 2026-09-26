//! The egui shell: the B3-Seg panel, input routing, and the viewport
//! frame. Declared from `main.rs` — binary code, so library items go
//! through `splatfield::`.

use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use eframe::egui::{self, Color32, Rect};

use super::App;
use crate::SELECT_GREEN;
use crate::scene::render_frame;

const UV_RECT: Rect = Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0));

/// How long an idle status message stays in the pill before dismissing.
const STATUS_LIFETIME: std::time::Duration = std::time::Duration::from_secs(4);
/// Failures outlive outcomes — long enough to read a long message — but
/// stay bounded: an error that never dismisses reads as a stuck state.
const ERROR_STATUS_LIFETIME: std::time::Duration = std::time::Duration::from_secs(16);

/// Whether a drag-and-dropped file is a loadable scene.
#[cfg(target_arch = "wasm32")]
fn is_scene(path: &std::path::Path) -> bool {
    path.extension().is_some_and(|e| {
        splatfield::SCENE_EXTENSIONS
            .iter()
            .any(|x| e.eq_ignore_ascii_case(x))
    })
}

/// The selection box: translucent green fill, wide rounded stroke around it
/// — visible over both dark renders and colored tint.
fn draw_selection_box(painter: &egui::Painter, rect: egui::Rect) {
    let [r, g, b] = SELECT_GREEN;
    let color = Color32::from_rgb(r, g, b);
    painter.rect_filled(rect, 4.0, color.gamma_multiply(0.15));
    painter.rect_stroke(
        rect,
        4.0,
        egui::Stroke::new(2.5, color),
        egui::StrokeKind::Middle,
    );
}

/// The help text shared by the `?` hover tooltip and the click-pinned popup.
fn help_body(ui: &mut egui::Ui) {
    ui.set_max_width(340.0);
    ui.strong("SplatField — text-prompted 3DGS segmentation");
    ui.add_space(4.0);
    egui::Grid::new("help")
        .num_columns(2)
        .spacing([12.0, 3.0])
        .show(ui, |ui| {
            ui.strong("Scene");
            ui.label("drag & drop a .ply / .sog file");
            ui.end_row();
            ui.strong("View");
            ui.label("drag orbits · middle/right-drag pans · scroll zooms");
            ui.end_row();
            ui.strong("Select");
            ui.label("Shift + drag a box (Esc cancels)");
            ui.end_row();
            ui.strong("Delete");
            ui.label("Del / Backspace · undo with ⌘/Ctrl + Z");
            ui.end_row();
            ui.strong("Segment");
            ui.label("type a prompt, segment, then cut extracts the object");
            ui.end_row();
            ui.strong("Reset / save");
            ui.label("initial model · <source>.edited.ply");
            ui.end_row();
        });
    ui.separator();
    ui.hyperlink("https://sony.github.io/B3-Seg-project");
}

impl App {
    /// Draw the docked B3-Seg panel; returns the user's triggers
    /// (run_requested, reset_clicked, cut_clicked, save_clicked,
    /// cancel_clicked). `busy` drives the run/stop pair; `locked` — runs
    /// plus in-flight cut readbacks — drives the edit buttons.
    pub(crate) fn seg_panel(
        &mut self,
        ui: &mut egui::Ui,
        busy: bool,
        has_model: bool,
        locked: bool,
    ) -> (bool, bool, bool, bool, bool) {
        let mut run = false;
        let mut reset_clicked = false;
        let mut cut_clicked = false;
        let mut save_clicked = false;
        let mut cancel_clicked = false;
        let mut editing = false;

        egui::Panel::bottom("b3seg").show(ui, |ui| {
            ui.horizontal(|ui| {
                let btn = |ui: &mut egui::Ui, label: &str, enabled: bool| {
                    ui.add_enabled(enabled, egui::Button::new(label)).clicked()
                };
                let edit = ui.add(
                    egui::TextEdit::singleline(&mut self.seg.prompt)
                        .hint_text("text prompt, e.g. “bear”")
                        .desired_width(220.0),
                );
                editing = edit.has_focus();
                if edit.lost_focus()
                    && ui.input(|i| i.key_pressed(egui::Key::Enter))
                    && !self.seg.prompt.trim().is_empty()
                {
                    run = true;
                }
                ui.add(
                    egui::DragValue::new(&mut self.seg.iters)
                        .range(1..=20)
                        .suffix(" iters"),
                );
                ui.checkbox(&mut self.seg.heatmap, "heatmap");
                run |= btn(
                    ui,
                    if busy { "segmenting…" } else { "segment" },
                    has_model && !busy && !self.seg.prompt.trim().is_empty(),
                );
                cancel_clicked |= btn(ui, "stop", busy);
                reset_clicked |= btn(ui, "reset", has_model && !locked);
                cut_clicked |= btn(ui, "cut", has_model && !locked && self.seg.result.is_some());
                save_clicked |= btn(ui, "save", has_model && !locked);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let help = ui.add(egui::Button::new("?").small());
                    let help = help.on_hover_ui(|ui| self.help_panel(ui));
                    // The tooltip only appears after egui's hover delay, and
                    // new users click instead — so a click pins the same
                    // panel open; clicking anywhere else dismisses it.
                    egui::Popup::from_toggle_button_response(&help)
                        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
                        .show(|ui| self.help_panel(ui));
                });
            });
        });
        // Delete removes the selected splats and Cmd/Ctrl+Z undoes the last
        // delete — but never while the user is typing in the prompt. The
        // Mac's delete key is Backspace; accept both.
        if !editing {
            let delete = ui
                .input(|i| i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace));
            if delete {
                self.delete_selection();
            }
            if ui.input(|i| i.key_pressed(egui::Key::Z) && i.modifiers.command) {
                self.undo_delete();
            }
        }
        (
            run,
            reset_clicked,
            cut_clicked,
            save_clicked,
            cancel_clicked,
        )
    }

    /// The help body plus the demo row (wasm) — one panel shared by the `?`
    /// hover tooltip and the click-pinned popup. The interactive demo button
    /// makes egui keep the tooltip interactable, so the row is clickable on
    /// hover too.
    fn help_panel(&mut self, ui: &mut egui::Ui) {
        help_body(ui);
        #[cfg(target_arch = "wasm32")]
        self.demo_button(ui);
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _: &mut eframe::Frame) {
        // Worker replies accumulated since last frame fold into the seg
        // state first, so the panel and pill draw this frame's busy/progress.
        // Taken before the loop: the cell guard must not live across a fold.
        #[cfg(target_arch = "wasm32")]
        {
            let responses = std::mem::take(&mut *self.seg.inbox.borrow_mut());
            for response in responses {
                self.fold_response(response);
            }
            // Then the loop task's publications (progress, terminal).
            let msgs = std::mem::take(&mut *self.seg.msgs.borrow_mut());
            for msg in msgs {
                self.fold_seg_msg(msg);
            }
            // A finished cut readback applies here, on the UI thread where
            // the undo bookkeeping lives; the lock re-opens only when a
            // staged cut actually landed (an empty drain means the readback
            // is still in flight — see `Loaded::drain_cut`).
            let staged = self.splats.lock().unwrap().drain_cut();
            if let Some(crate::scene::CutStaged {
                apply: Some((cut, colors)),
            }) = staged
            {
                self.apply_removal(cut, Some(colors));
            }
        }

        // Drag-and-drop loading is wasm-only: the load reads the dropped
        // file's bytes through the browser handle.
        #[cfg(target_arch = "wasm32")]
        let dropped = ui.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .find(|f| is_scene(f.path()))
                .cloned()
        });
        // A drop during a run or a cut readback would leave the worker
        // tinting a stale model or the staged cut folding over new
        // numbering — reject until both are done.
        #[cfg(target_arch = "wasm32")]
        if let Some(file) = dropped
            && !self.locked()
        {
            self.load_file(file, ui.ctx().clone());
        }

        // B3-Seg sidebar: docked bottom panel with the text-prompt controls.
        let busy = self.seg.busy;
        let has_model = self.splats.lock().unwrap().scene.is_some();
        // The edit buttons wait for runs and in-flight cut readbacks alike
        // (they all stage against the live scene's numbering).
        let locked = self.locked();
        let heatmap_before = self.seg.heatmap;
        let (run, reset_clicked, cut_clicked, save_clicked, cancel_clicked) =
            self.seg_panel(ui, busy, has_model, locked);
        if self.seg.heatmap != heatmap_before {
            // Toggling the heatmap switches what the viewport paints.
            self.paint_dirty = true;
        }
        // Esc cancels the live selection drag (unfreezing the camera).
        if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.seg.box_drag = None;
        }
        if reset_clicked {
            // Back to the initial model: pristine colors, nothing removed,
            // no undo history. Scenes without removals skip the re-upload.
            self.sel.clear();
            self.undo.clear();
            if self.removed.is_empty() {
                self.paint_selection();
            } else {
                self.removed.clear();
                self.rebuild();
            }
        }
        if cut_clicked {
            self.cut_object();
        }
        if save_clicked {
            self.save_ply();
        }
        if cancel_clicked {
            // Store the flag the loop task's on_round polls — false stops
            // the run at the round boundary (a delivered round is a valid
            // partial result). During the fetch phase no flag exists yet:
            // EnsureModels cannot be cancelled, the queued run still starts.
            if let Some(cancel) = &self.seg.cancel {
                cancel.store(true, Ordering::Relaxed);
            }
            self.set_status("cancelling…", false);
        }

        // A finished save reports through the pill; the lock drops before
        // set_status borrows the app.
        let save_result = self.splats.lock().unwrap().save_result.take();
        if let Some((msg, error)) = save_result {
            self.set_status(msg, error);
        }

        // The pill speaks whenever the engine has something to say: live
        // progress while a run is active, idle outcomes self-dismissing.
        // Failures get a longer window (and red text) instead of living
        // forever. The app repaints only when dirty, so the dismissal
        // deadline schedules its own wake-up. A load in flight holds the
        // pill past any lifetime: a slow fetch can stall longer than the
        // failure window, and a vanished progress line reads as a crash.
        let busy = self.seg.busy;
        let loading = self.splats.lock().unwrap().load_in_flight;
        let lifetime = if self.seg.status_error {
            ERROR_STATUS_LIFETIME
        } else {
            STATUS_LIFETIME
        };
        if !busy && !loading {
            match lifetime.checked_sub(self.seg.status_at.elapsed()) {
                None => self.seg.status.clear(),
                Some(left) => ui.ctx().request_repaint_after(left),
            }
        }
        if !self.seg.status.is_empty() {
            let error = self.seg.status_error;
            let status = &self.seg.status;
            let response = egui::Area::new(egui::Id::new("seg-progress"))
                .anchor(egui::Align2::CENTER_TOP, [0.0, 14.0])
                .show(ui, |ui| {
                    egui::Frame::popup(ui.style())
                        .corner_radius(10.0)
                        .inner_margin(egui::Margin::symmetric(14, 8))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                if busy {
                                    ui.add(egui::Spinner::new());
                                }
                                if error {
                                    ui.colored_label(ui.style().visuals.error_fg_color, status);
                                } else {
                                    ui.label(status);
                                }
                            })
                            .response
                        })
                        .response
                })
                .response;
            // Clicking the pill dismisses it now instead of waiting out
            // the lifetime.
            if response.clicked() {
                self.seg.status.clear();
            }
        }

        let mut slot = self.splats.lock().unwrap();
        let load_failed = slot
            .load_error
            .take()
            .map(|e| (format!("load failed: {e}"), true));
        if slot.reframe
            && let Some(s) = &slot.scene
        {
            self.controller.frame_bounds(s.splats.bounds);
            slot.reframe = false;
            // The landed scene ends its fetch narrative: drop the pill now
            // instead of letting "fetching … 100%" linger out the lifetime.
            self.seg.status.clear();
            // A fresh model resets the edit state staged against the old one.
            self.sel.clear();
            self.removed.clear();
            self.undo.clear();
            self.seg.result = None;
        }
        let Some(splats) = slot.scene.as_ref().map(|s| s.splats.clone()) else {
            // No scene yet: a failed first load must still reach the pill —
            // wasm has no stderr to fall back on, and this path returns
            // before the with-scene set_status below. The repaint schedules
            // the frame the pill is drawn in.
            drop(slot);
            if let Some((msg, error)) = load_failed {
                self.set_status(msg, error);
                ui.ctx().request_repaint();
            }
            ui.centered_and_justified(|ui| ui.heading("Drag and drop a .ply or .sog file"));
            return;
        };
        drop(slot);
        if let Some((msg, error)) = load_failed {
            self.set_status(msg, error);
        }

        let size = ui.available_size();
        let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());
        let pixel = (glam::vec2(size.x, size.y) * ui.pixels_per_point()).as_uvec2();

        // Below ~8px the aspect is 0/0: fit_fov would poison the camera with
        // NaN for every later frame, so skip input + render entirely.
        if pixel.x > 8 && pixel.y > 8 {
            let shift = ui.input(|i| i.modifiers.shift);
            if response.drag_started_by(egui::PointerButton::Primary) && shift {
                self.seg.box_drag = response.interact_pointer_pos();
            }
            // A box drag freezes the camera for the whole gesture — even if
            // shift is released mid-drag — so the selection tracks a static
            // view.
            let box_drag = self.seg.box_drag.is_some();
            let moved = !box_drag && self.controller.tick(&response, ui);
            self.controller.camera.fit_fov(pixel);

            if box_drag && response.dragged() {
                // Redraw the overlay only — the bitmap behind it is unchanged.
                ui.ctx().request_repaint();
            }
            if response.drag_stopped_by(egui::PointerButton::Primary)
                && let Some(start) = self.seg.box_drag.take()
                && let Some(end) = response.interact_pointer_pos()
            {
                let ppp = ui.pixels_per_point();
                // Viewport-relative physical pixels: the overlay draws in
                // screen coords, but the render's pixels start at the
                // rect's corner.
                let (a, b) = (
                    (start - response.rect.min) * ppp,
                    (end - response.rect.min) * ppp,
                );
                let (min, max) = (a.min(b), a.max(b));
                // Below ~2 px on both axes it's a stray click, not a box.
                if max.x - min.x > 2.0 || max.y - min.y > 2.0 {
                    self.select_in_rect(pixel, glam::vec2(min.x, min.y), glam::vec2(max.x, max.y));
                }
            }

            // Button and Enter both run the text prompt.
            if run && !busy && !self.seg.prompt.trim().is_empty() {
                let prompt = self.seg.prompt.trim().to_owned();
                self.request_segmentation(prompt);
            }

            // A camera-neutral event (e.g. a bare click) would re-run the
            // whole ~30-launch pipeline only to repaint an identical bitmap —
            // skip it. Any load brings a fresh Arc, so pointer inequality
            // covers reframes too; the pose comparison covers motion that
            // arrived while a wasm render was in flight, and the segmentation
            // tint and posterior updates mutate in place, so they set
            // paint_dirty instead.
            let stale = moved
                || self.paint_dirty
                || self
                    .rendered
                    .as_ref()
                    .is_none_or(|(last_px, last_splats, last_cam)| {
                        *last_px != pixel
                            || !Arc::ptr_eq(last_splats, &splats)
                            || *last_cam != self.controller.camera
                    });
            if stale {
                // Record the pose this flight renders — not the current
                // controller state, which may drift further while it runs.
                let camera = self.controller.camera;
                let gpu = Rc::clone(&self.gpu);
                // Only a stale frame consumes the posterior — cloning the
                // tensor-pair handles every frame would be pure waste.
                let posterior = self
                    .seg
                    .posterior
                    .as_ref()
                    .filter(|_| self.seg.heatmap)
                    .cloned();

                // Single-flight on wasm: the slot holds `None` while a render
                // is in flight, so back-to-back stale frames coalesce, and the
                // task's request_repaint brings the loop back to schedule
                // whatever camera state the flight missed. Native drives the
                // same render inline (tombstone build), where the slot is
                // never busy.
                if gpu.borrow().is_some() {
                    #[cfg(not(target_arch = "wasm32"))]
                    cubecl::future::block_on(render_frame(
                        &gpu, &splats, &camera, pixel, posterior,
                    ));

                    #[cfg(target_arch = "wasm32")]
                    {
                        let ctx = ui.ctx().clone();
                        let splats = Arc::clone(&splats);
                        wasm_bindgen_futures::spawn_local(async move {
                            render_frame(&gpu, &splats, &camera, pixel, posterior).await;
                            ctx.request_repaint();
                        });
                    }

                    self.rendered = Some((pixel, Arc::clone(&splats), camera));
                    self.paint_dirty = false;
                }
            }
        }

        ui.painter()
            .image(self.tex_id, rect, UV_RECT, Color32::WHITE);

        // Selection overlay: the live Shift+drag box.
        if let (Some(start), Some(end)) = (self.seg.box_drag, response.interact_pointer_pos()) {
            draw_selection_box(ui.painter(), Rect::from_two_pos(start, end));
        }
    }
}
