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
use crate::settings;

const UV_RECT: Rect = Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0));

/// How long an idle status message stays in the pill before dismissing.
const STATUS_LIFETIME: std::time::Duration = std::time::Duration::from_secs(4);
/// Failures outlive outcomes — long enough to read a long message — but
/// stay bounded: an error that never dismisses reads as a stuck state.
const ERROR_STATUS_LIFETIME: std::time::Duration = std::time::Duration::from_secs(16);

/// The one refusal wording for a scene source arriving mid-run or mid-cut
/// readback, shared by the three paths that can (url, pick, drop).
const BUSY_WHILE_LOCKED: &str = "busy — wait for the current run to finish";

/// Panel width below which the bar stacks: the single row's widgets measure
/// ≈530 px with egui's default fonts (≈500 idle, +31 for the busy label:
/// prompt 220 + segmenting… 89.8 + stop 33.8 + reset 37.7 + cut 26.4 + save
/// 33.5 + five 8 px gaps ≈ 481, + 8 px to the right-aligned ⚙ 19.7 / ? 12.9
/// pair + its internal 8 px gap), so the gate keeps the old ~80 px margin
/// for font/platform variance instead of clipping a trailing button — below
/// it the prompt takes the full first row and the controls wrap to a
/// second. (The settings panel moved the old 51 px iters DragValue and
/// 70 px heatmap checkbox out of the row and put the 20 px gear in.)
const NARROW_PANEL_WIDTH: f32 = 610.0;

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
        let mut editing = false;
        let (run, reset_clicked, cut_clicked, save_clicked, cancel_clicked) =
            egui::Panel::bottom("b3seg")
                .show(ui, |ui| {
                    if ui.available_width() < NARROW_PANEL_WIDTH {
                        let enter = self.prompt_row(ui, f32::INFINITY, &mut editing);
                        ui.horizontal_wrapped(|ui| {
                            self.seg_controls(ui, busy, has_model, locked, enter)
                        })
                        .inner
                    } else {
                        ui.horizontal(|ui| {
                            let enter = self.prompt_row(ui, 220.0, &mut editing);
                            self.seg_controls(ui, busy, has_model, locked, enter)
                        })
                        .inner
                    }
                })
                .inner;
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

    /// The prompt row: the text edit plus Enter-to-run. `width` is
    /// `f32::INFINITY` on stacked phones (fill the row) and the desktop's
    /// 220 px in the single-row layout. Writes the field's focus into
    /// `*editing` — the caller's delete/undo key guard — and returns
    /// whether Enter submitted the prompt.
    fn prompt_row(&mut self, ui: &mut egui::Ui, width: f32, editing: &mut bool) -> bool {
        let edit = ui.add(
            egui::TextEdit::singleline(&mut self.seg.prompt)
                .hint_text("text prompt, e.g. “bear”")
                .desired_width(width),
        );
        *editing = edit.has_focus();
        edit.lost_focus()
            && ui.input(|i| i.key_pressed(egui::Key::Enter))
            && !self.seg.prompt.trim().is_empty()
    }

    /// The controls after the prompt, in one row: run/stop, the edit
    /// buttons, then the right-aligned settings gear and help. Shared by the
    /// desktop single row and the stacked phone's second row. `enter` is the
    /// prompt row's Enter trigger, folded into the run trigger. Returns
    /// (run, reset, cut, save, cancel).
    fn seg_controls(
        &mut self,
        ui: &mut egui::Ui,
        busy: bool,
        has_model: bool,
        locked: bool,
        enter: bool,
    ) -> (bool, bool, bool, bool, bool) {
        let btn = |ui: &mut egui::Ui, label: &str, enabled: bool| {
            ui.add_enabled(enabled, egui::Button::new(label)).clicked()
        };
        // Non-short-circuiting `|`: the segment button must be drawn (and
        // its own click seen) even on frames where Enter already fired.
        let run = enter
            | btn(
                ui,
                if busy { "segmenting…" } else { "segment" },
                has_model && !busy && !self.seg.prompt.trim().is_empty(),
            );
        let cancel_clicked = btn(ui, "stop", busy);
        let reset_clicked = btn(ui, "reset", has_model && !locked);
        let cut_clicked = btn(ui, "cut", has_model && !locked && self.seg.result.is_some());
        let save_clicked = btn(ui, "save", has_model && !locked);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            // Right-to-left places the first widget rightmost: the help `?`
            // hugs the corner and the settings gear lands to its left.
            let help = ui.add(egui::Button::new("?").small());
            let help = help.on_hover_ui(help_body);
            // The tooltip only appears after egui's hover delay, and
            // new users click instead — so a click pins the same
            // panel open; clicking anywhere else dismisses it.
            egui::Popup::from_toggle_button_response(&help)
                .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
                .show(help_body);
            // The settings panel shares the help popup's click-pinned
            // pattern. CloseOnClickOutside is load-bearing twice over — it
            // keeps the panel open while the user drags the iters value,
            // and the drag's pointer grab never registers as an outside
            // click.
            let gear = ui.add(egui::Button::new("⚙").small());
            egui::Popup::from_toggle_button_response(&gear)
                .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
                .show(|ui| self.settings_body(ui));
        });
        (
            run,
            reset_clicked,
            cut_clicked,
            save_clicked,
            cancel_clicked,
        )
    }

    /// The settings panel's body: inside a floating panel width isn't
    /// scarce, so each control carries a real label. `iters` feeds each
    /// run's config at spawn (the pill's total is captured then, so a
    /// mid-run edit is already safe), and the heatmap toggle's repaint
    /// rides `App::ui`'s heatmap_before comparison. Any actual change
    /// persists immediately (settings.rs) — a refresh must not lose the
    /// pair, and the panel stays open through the edit.
    fn settings_body(&mut self, ui: &mut egui::Ui) {
        ui.set_max_width(340.0);
        ui.horizontal(|ui| {
            // The label says "iterations per run", so the DragValue carries
            // no suffix — "7 iters iterations per run" would stutter.
            ui.add(egui::DragValue::new(&mut self.seg.iters).range(settings::ITERS_RANGE));
            ui.label("iterations per run");
        });
        ui.checkbox(&mut self.seg.heatmap, "heatmap preview");
        // Save on change: compare against the last-saved pair — never a
        // per-frame localStorage read. Best-effort and wasm-only (the host
        // tombstone has no storage and never draws this).
        let now = settings::Settings {
            iters: self.seg.iters,
            heatmap: self.seg.heatmap,
        };
        if now != self.saved_settings {
            #[cfg(target_arch = "wasm32")]
            settings::save(&now);
            self.saved_settings = now;
        }
    }

    /// The no-scene landing card — the reference mock's layout grammar
    /// (data/reference.png) in the app's dark theme: a large dashed drop
    /// zone with a Browse Files button, a URL download row, and the demo
    /// scene. The status pill above is the one narrator: every accepted or
    /// refused source lands there. What the controls name is what the
    /// parsers accept.
    fn landing_zone(&mut self, ui: &mut egui::Ui) {
        let outer = ui.available_rect_before_wrap();
        // Height hugs the content block (~350 px + padding): a taller card
        // sits with a dead band under the demo button, and main-align
        // centering a top_down Ui spreads its widgets apart instead of
        // grouping them.
        let size = egui::vec2(
            (outer.width() * 0.9).min(640.0),
            (outer.height() * 0.9).min(410.0),
        );
        let card = egui::Rect::from_center_size(outer.center(), size);
        let drag = ui.input(|i| !i.raw.hovered_files.is_empty());
        let [r, g, b] = SELECT_GREEN;
        let green = Color32::from_rgb(r, g, b);
        // The card is paint-only — Browse Files is the one click target —
        // so dragging any file over the window tints it (the selection
        // overlay's tint recipe) and otherwise it sits in fixed grays
        // (the app forces Dark in main.rs).
        let (fill, stroke) = if drag {
            (green.gamma_multiply(0.08), egui::Stroke::new(2.5, green))
        } else {
            (
                Color32::from_gray(22),
                egui::Stroke::new(1.5, Color32::from_gray(64)),
            )
        };
        // The border rides the card's edge — widen the clip rect by the
        // stroke so its outer half isn't clipped away.
        let painter = ui.painter_at(card.expand(2.0));
        painter.rect_filled(card, 16.0, fill);
        // epaint ships the rounded-rect outline sampler (CIRCLE_64 arcs);
        // dashes take a closed point path, so close it and dash. Dash 6 /
        // gap 5: the reference mock's rhythm.
        let mut path = Vec::new();
        egui::epaint::tessellator::path::rounded_rectangle(&mut path, card, 16.0.into());
        path.push(path[0]);
        painter.extend(egui::Shape::dashed_line(&path, stroke, 6.0, 5.0));

        let mut inner = ui.new_child(
            egui::UiBuilder::new()
                .max_rect(card.shrink(24.0))
                .layout(egui::Layout::top_down(egui::Align::Center)),
        );

        // Upload badge: the circle + up-arrow as a non-interactive button —
        // egui's bundled NotoEmoji ships ⬆ compiled into the binary, so the
        // glyph is identical on every platform.
        inner.add(
            egui::Button::new(
                egui::RichText::new("⬆")
                    .size(28.0)
                    .color(Color32::from_gray(210)),
            )
            .fill(Color32::from_gray(34))
            .stroke(egui::Stroke::NONE)
            .min_size(egui::vec2(56.0, 56.0))
            .corner_radius(28.0)
            .sense(egui::Sense::hover()),
        );
        inner.add_space(10.0);

        inner.heading("Load a Gaussian Splat");
        inner.weak("Drag & drop a .ply or .sog file here");
        inner.add_space(14.0);
        let browse = inner.add(
            egui::Button::new(
                egui::RichText::new("+  Browse Files")
                    .strong()
                    .color(Color32::from_gray(12)),
            )
            .fill(green)
            .min_size(egui::vec2(180.0, 34.0)),
        );
        if browse.clicked() {
            self.picker.open();
        }
        inner.add_space(10.0);
        inner.separator();
        inner.add_space(8.0);
        inner.weak("Load from url (.ply or .sog)");
        inner.add_space(6.0);
        inner.horizontal(|ui| {
            const ROW_H: f32 = 32.0;
            let button_w = 116.0;
            let edit_w = ui.available_width() - button_w - ui.spacing().item_spacing.x;
            // TextEdit's builder min_size only binds the WIDTH (the height
            // derives from the text's line height plus the text margin), so
            // pad the margin from the measured line height up to the row
            // height — matching the download button exactly.
            let line_h =
                ui.text_style_height(&egui::TextStyle::Body) + ui.spacing().extra_text_line_spacing;
            let pad_y = ((ROW_H - line_h) / 2.0).round().max(2.0) as i8;
            let edit = ui.add(
                egui::TextEdit::singleline(&mut self.landing_url)
                    .hint_text("https://example.com/model.ply")
                    .desired_width(edit_w)
                    .margin(egui::Margin::symmetric(8, pad_y)),
            );
            // The button evaluates FIRST and unconditionally: `|` binds
            // tighter than `&&`, so `lost_focus && enter | button` would
            // parse as `lost_focus && (enter | button)` — short-circuiting
            // the button's own draw away on every frame the edit neither
            // lost focus nor pressed Enter, leaving an invisible widget
            // whose clicks fell through to the zone.
            let submit = ui
                .add(egui::Button::new("download").min_size(egui::vec2(button_w, ROW_H)))
                .clicked()
                | (edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)));
            if submit {
                #[cfg(target_arch = "wasm32")]
                self.submit_url(ui);
            }
        });
        inner.add_space(8.0);
        inner.separator();
        inner.add_space(8.0);
        inner.weak("Load a demo scene");
        inner.add_space(6.0);
        // The mock's example row: an outlined secondary pill below the line,
        // icon + label like the Browse button above. egui's bundled
        // NotoEmoji makes 🐻 monochrome — an icon, not a sticker. The
        // outline and the hover tint come from the per-state widget visuals
        // (a Button::fill override would freeze them — "override any
        // on-hover effects", says the builder doc).
        let demo = inner
            .scope(|ui| {
                let w = &mut ui.visuals_mut().widgets;
                w.inactive.weak_bg_fill = Color32::TRANSPARENT;
                w.inactive.bg_stroke = egui::Stroke::new(1.0, Color32::from_gray(64));
                w.hovered.weak_bg_fill = Color32::from_gray(30);
                w.hovered.bg_stroke = egui::Stroke::new(1.0, Color32::from_gray(110));
                w.active.weak_bg_fill = Color32::from_gray(24);
                ui.add_enabled(
                    !self.locked(),
                    egui::Button::new(egui::RichText::new("🐻  Bear").strong())
                        .corner_radius(8.0)
                        .min_size(egui::vec2(140.0, 32.0)),
                )
            })
            .inner;
        if demo.clicked() {
            #[cfg(target_arch = "wasm32")]
            self.load_demo(ui.ctx().clone());
        }
    }

    /// The landing URL row's submit: validate (type, before any download),
    /// then hand to the loader. wasm-only — the host tombstone never draws.
    #[cfg(target_arch = "wasm32")]
    fn submit_url(&mut self, ui: &mut egui::Ui) {
        let (url, name) = match splatfield::validate_scene_url(&self.landing_url) {
            Ok((url, name)) => (url, name),
            Err(msg) => {
                // The validator owns the wording — one literal to keep in
                // sync with its rules.
                self.set_status(msg, true);
                return;
            }
        };
        if self.locked() {
            self.set_status(BUSY_WHILE_LOCKED, true);
            return;
        }
        self.load_url(url, name, ui.ctx().clone());
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

        // Every local-file source — a drop, or a pick through the picker
        // bridge — drains here, BEFORE the pill draws: set later, a status
        // went unpainted for the frame and expired unread. No repaint is
        // owed — the pill paints it now (mid-run, the run's next line
        // replaces it). Both refuse a source during a run or cut readback,
        // which would tint a stale model or fold a cut over new numbering.
        // Not gated on an empty scene: Browse is the only way to queue a
        // pick, and the card that owns it draws only without one.
        #[cfg(target_arch = "wasm32")]
        {
            let dropped = ui.input(|i| i.raw.dropped_files.first().cloned());
            // The closure's return type is the unsize-coercion site
            // (`Arc<Picked>` → `DroppedFileHandle`); `collect` alone can't
            // infer the element type, so both annotations are load-bearing.
            let picked: Vec<egui::DroppedFileHandle> = self
                .picker
                .take_picked()
                .into_iter()
                .map(|f| -> egui::DroppedFileHandle { Arc::new(crate::scene::Picked::new(f)) })
                .collect();
            for file in dropped.into_iter().chain(picked) {
                if self.locked() {
                    self.set_status(BUSY_WHILE_LOCKED, true);
                    continue;
                }
                // Type from the name, size from the browser handle, both
                // known BEFORE any byte is read — an unsupported or
                // oversized file fails with a message instead of a late OOM.
                let name = std::path::Path::new(file.path())
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                match splatfield::preflight_scene(&name, file.web_file().map(|f| f.size() as u64)) {
                    Ok(()) => {
                        self.set_status(format!("loading {name}…"), false);
                        self.load_file(file, name, ui.ctx().clone());
                    }
                    Err(msg) => self.set_status(msg, true),
                }
            }
        }

        let busy = self.seg.busy;
        let has_model = self.splats.lock().unwrap().scene.is_some();
        // The edit buttons wait for runs and in-flight cut readbacks alike
        // (they all stage against the live scene's numbering).
        let locked = self.locked();
        let heatmap_before = self.seg.heatmap;
        let (run, reset_clicked, cut_clicked, save_clicked, cancel_clicked) =
            self.seg_panel(ui, busy, has_model, locked);
        if self.seg.heatmap != heatmap_before {
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

        // Failures get a longer window (and red text) instead of living
        // forever. The app repaints only when dirty, so the dismissal
        // deadline schedules its own wake-up. A load in flight holds the
        // pill past any lifetime: a slow fetch can stall longer than the
        // failure window, and a vanished progress line reads as a crash.
        // The `busy` read above the panel is current here: only the
        // frame-top folds and the run trigger — below the pill — write it.
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
                                // The spinner rides runs and in-flight loads
                                // alike — a load has no other activity cue.
                                if busy || loading {
                                    ui.add(egui::Spinner::new());
                                }
                                if error {
                                    ui.colored_label(ui.style().visuals.error_fg_color, status);
                                } else {
                                    ui.label(status);
                                }
                            })
                        })
                        .inner
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
            self.landing_zone(ui);
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

        // Selection overlay: the live Shift+drag box — translucent green
        // fill, wide rounded stroke, visible over dark renders and tint.
        if let (Some(start), Some(end)) = (self.seg.box_drag, response.interact_pointer_pos()) {
            let [r, g, b] = SELECT_GREEN;
            let color = Color32::from_rgb(r, g, b);
            ui.painter().rect(
                Rect::from_two_pos(start, end),
                4.0,
                color.gamma_multiply(0.15),
                egui::Stroke::new(2.5, color),
                egui::StrokeKind::Middle,
            );
        }
    }
}
