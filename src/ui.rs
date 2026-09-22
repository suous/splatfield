//! The egui shell: the B3-Seg panel, input routing, and the viewport
//! frame. Declared from `main.rs` — binary code, so library items go
//! through `splatfield::`.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use eframe::egui::{self, Color32, Rect};

use super::App;
use crate::SELECT_GREEN;
use crate::scene::render_frame;

const UV_RECT: Rect = Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0));

/// How long an idle status message stays in the pill before dismissing.
const STATUS_LIFETIME: std::time::Duration = std::time::Duration::from_secs(4);

/// Whether a drag-and-dropped file is a loadable scene.
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

impl App {
    /// Draw the docked B3-Seg panel; returns the user's triggers
    /// (run_requested, reset_clicked, cut_clicked, save_clicked,
    /// cancel_clicked).
    pub(crate) fn seg_panel(
        &mut self,
        ui: &mut egui::Ui,
        busy: bool,
        has_model: bool,
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
                reset_clicked |= btn(ui, "reset", has_model && !busy);
                cut_clicked |= btn(ui, "cut", has_model && !busy && self.seg.result.is_some());
                save_clicked |= btn(ui, "save", has_model && !busy);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let help = ui.add(egui::Button::new("?").small());
                    help.on_hover_ui(|ui| {
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
                    });
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
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _: &mut eframe::Frame) {
        let dropped = ui.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .find(|f| is_scene(f.path()))
                .cloned()
        });
        // A drop during a run would leave the worker tinting the old model
        // and reporting old counts — reject until it finishes.
        if let Some(file) = dropped
            && !self.seg.busy()
        {
            self.load_file(file.path(), ui.ctx().clone());
        }

        // B3-Seg sidebar: docked bottom panel with the text-prompt controls.
        let busy = self.seg.busy();
        let has_model = self.splats.lock().unwrap().scene.is_some();
        let heatmap_before = self.seg.heatmap;
        let (run, reset_clicked, cut_clicked, save_clicked, cancel_clicked) =
            self.seg_panel(ui, busy, has_model);
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
            if let Some(c) = &self.seg.cancel {
                c.store(true, Ordering::Relaxed);
            }
            self.set_status("cancelling…", false);
        }
        self.drain_segmentation();

        // The pill speaks whenever the engine has something to say: live
        // progress while a run is active, idle outcomes self-dismissing.
        // Failures persist until replaced — vanishing errors get missed.
        // The app repaints only when dirty, so the dismissal deadline
        // schedules its own wake-up.
        let busy = self.seg.busy();
        if !busy && !self.seg.status_error {
            match STATUS_LIFETIME.checked_sub(self.seg.status_at.elapsed()) {
                None => self.seg.status.clear(),
                Some(left) => ui.ctx().request_repaint_after(left),
            }
        }
        if !self.seg.status.is_empty() {
            egui::Area::new(egui::Id::new("seg-progress"))
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
                                ui.label(&self.seg.status);
                            });
                        });
                });
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
            // A fresh model resets the edit state staged against the old one.
            self.sel.clear();
            self.removed.clear();
            self.undo.clear();
            self.seg.result = None;
        }
        let Some(splats) = slot.scene.as_ref().map(|s| s.splats.clone()) else {
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

            let posterior = self.seg.posterior.as_ref().filter(|_| self.seg.heatmap);
            // A camera-neutral event (e.g. a bare click) would re-run the
            // whole ~30-launch pipeline only to repaint an identical bitmap —
            // skip it. Any load brings a fresh Arc, so pointer inequality
            // covers reframes too; the segmentation tint and posterior
            // updates mutate in place, so they set paint_dirty instead.
            let stale = moved
                || self.paint_dirty
                || self.rendered.as_ref().is_none_or(|(last_px, last_splats)| {
                    *last_px != pixel || !Arc::ptr_eq(last_splats, &splats)
                });
            if stale {
                render_frame(
                    &self.client,
                    &mut self.gpu,
                    &splats,
                    &self.controller.camera,
                    pixel,
                    posterior,
                );
                self.rendered = Some((pixel, Arc::clone(&splats)));
                self.paint_dirty = false;
            }
        }

        ui.painter()
            .image(self.gpu.backbuffer.id, rect, UV_RECT, Color32::WHITE);

        // Selection overlay: the live Shift+drag box.
        if let (Some(start), Some(end)) = (self.seg.box_drag, response.interact_pointer_pos()) {
            draw_selection_box(ui.painter(), Rect::from_two_pos(start, end));
        }
    }
}
