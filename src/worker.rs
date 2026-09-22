//! The segmentation worker: sidebar state, the thread protocol, and the
//! App methods that spawn and drain the run. Declared from `main.rs` —
//! binary code, so library items go through `splatfield::`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;

use eframe::egui;
use splat_sort::tensor::GpuTensor;
use splatfield::seg::active::Config as SegConfig;
use splatfield::seg::prompted::run_text;
use splatfield::{camera, render, to_dc};

use super::App;

/// Display colors for successive segmentation runs, cycled through. Stored
/// 0..1; converted to SH DC coefficients before tinting (rgb = C0·f_dc + 0.5).
const PALETTE: [[f32; 3]; 8] = [
    [0.937, 0.267, 0.267], // red
    [0.976, 0.451, 0.086], // orange
    [0.961, 0.620, 0.043], // amber
    [0.133, 0.773, 0.369], // green
    [0.024, 0.714, 0.831], // cyan
    [0.231, 0.510, 0.965], // blue
    [0.545, 0.361, 0.965], // violet
    [0.925, 0.286, 0.600], // pink
];

/// One message from the segmentation worker thread to the UI.
pub(crate) enum SegMsg {
    /// A progress line for the pill.
    Status(String),
    /// The posterior-so-far, for the live heatmap.
    Progress {
        done: usize,
        eig: f32,
        ab: Box<(GpuTensor, GpuTensor)>,
    },
    Done,
    Failed(String),
}

/// B3-Seg sidebar state: what the user asked for plus whatever the worker
/// published so far.
pub(crate) struct SegUi {
    pub(crate) prompt: String,
    pub(crate) heatmap: bool,
    pub(crate) iters: usize,
    /// Total rounds of the active run, captured at spawn: `iters` is
    /// live-editable, and a mid-run drag must not rewrite the pill's
    /// progress total.
    pub(crate) run_total: usize,
    /// Start corner of the live Shift+drag box selection, if any.
    pub(crate) box_drag: Option<egui::Pos2>,
    pub(crate) rx: Option<mpsc::Receiver<SegMsg>>,
    /// Set while a run is active: polled by the worker's `on_round`, stored
    /// by the UI's stop button.
    pub(crate) cancel: Option<std::sync::Arc<AtomicBool>>,
    /// Engine message, spoken by the top pill: live progress while a run is
    /// active, idle outcomes self-dismissing. `status_error` messages
    /// persist until replaced — vanishing errors get missed.
    pub(crate) status: String,
    pub(crate) status_error: bool,
    pub(crate) status_at: std::time::Instant,
    pub(crate) posterior: Option<(GpuTensor, GpuTensor)>,
    /// The last completed run's final posterior — the cut button's labels.
    /// Unlike `posterior` it survives `Done`, whose tint retires the
    /// heatmap preview.
    pub(crate) result: Option<(GpuTensor, GpuTensor)>,
}

impl Default for SegUi {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            // The posterior heatmap is the loop's live feedback — on unless
            // the user turns it off.
            heatmap: true,
            iters: SegConfig::default().iterations,
            run_total: 0,
            box_drag: None,
            rx: None,
            cancel: None,
            status: String::new(),
            status_error: false,
            status_at: std::time::Instant::now(),
            posterior: None,
            result: None,
        }
    }
}

impl SegUi {
    /// A segmentation run is active: editing and re-runs are locked out.
    pub(crate) fn busy(&self) -> bool {
        self.rx.is_some()
    }
}

impl App {
    /// Validate and launch a segmentation run. The loop starts from
    /// the current viewport pose, so round 0 observes what the user sees.
    pub(crate) fn request_segmentation(&mut self, prompt: String) {
        if self.seg.busy() {
            return;
        }
        // The worker paints its own tint on top of pristine colors; drop
        // the selection highlight so it can't half-survive the run.
        self.sel.clear();
        self.paint_selection();
        let Some(splats) = self
            .splats
            .lock()
            .unwrap()
            .scene
            .as_ref()
            .map(|s| s.splats.clone())
        else {
            return;
        };
        let cancel = Arc::new(AtomicBool::new(false));
        self.seg.cancel = Some(cancel.clone());
        // Square fov (the narrower axis), so a square segmentation render is
        // inscribed in what the user sees.
        let mut cam = self.controller.camera;
        cam.fov = glam::Vec2::splat(cam.fov.x.min(cam.fov.y));
        self.run_segmentation(splats, prompt, cam, cancel);
    }

    /// Spawn the active-loop worker: loads the 2D oracle and runs T
    /// observation rounds, publishing per-iteration progress (and the
    /// posterior buffers for the heatmap) over the channel, then colors the
    /// foreground splats with the next palette color on the GPU.
    fn run_segmentation(
        &mut self,
        splats: Arc<render::Splats>,
        prompt: String,
        camera: camera::Camera,
        cancel: Arc<AtomicBool>,
    ) {
        let (tx, rx) = std::sync::mpsc::channel();
        // Display color → SH DC coefficient.
        let color = to_dc(PALETTE[self.next_color % PALETTE.len()]);
        self.next_color += 1;
        let iterations = self.seg.iters;
        self.seg.run_total = iterations;
        self.seg.rx = Some(rx);
        self.seg.posterior = None;

        let seg_cfg = SegConfig {
            iterations,
            ..SegConfig::default()
        };
        std::thread::spawn(move || {
            let run = || -> anyhow::Result<()> {
                if cancel.load(Ordering::Relaxed) {
                    return Ok(());
                }
                tx.send(SegMsg::Status("loading 2D oracle…".into())).ok();
                // The loop owns the stop rules; this closure only feeds the
                // status row and the progress pill's heatmap, and polls the
                // cancel flag — false stops the run at the round boundary.
                let mut done = 0usize;
                let segmenter = run_text(
                    splats.clone(),
                    &prompt,
                    seg_cfg,
                    camera,
                    &mut |p| {
                        tx.send(SegMsg::Status(p.into())).ok();
                    },
                    &mut |round, state| {
                        // The delivered round always publishes — its evidence
                        // is already folded into the state the tint uses, so
                        // `result` must not lag the visible color.
                        done += 1;
                        tx.send(SegMsg::Progress {
                            done,
                            eig: round.eig,
                            ab: Box::new((state.a.clone(), state.b.clone())),
                        })
                        .ok();
                        !cancel.load(Ordering::Relaxed)
                    },
                )?;
                let l = segmenter.segmentation();
                let fg = l.iter().filter(|&&l| l).count();
                if fg > 0 {
                    splats.tint(&segmenter.state.a, &segmenter.state.b, color);
                }
                tx.send(SegMsg::Status(format!("colored {fg}/{} splats", l.len())))
                    .ok();
                Ok(())
            };
            let outcome = run();
            match outcome {
                Ok(()) => tx.send(SegMsg::Done).ok(),
                Err(e) => tx.send(SegMsg::Failed(format!("{e:#}"))).ok(),
            };
        });
    }

    /// A terminal worker message (Done or Failed): retire the run — the
    /// permanent tint (or failure) replaces the heatmap preview, and a stale
    /// posterior must never feed an out-of-bounds paint once a smaller model
    /// loads.
    fn seg_finished(&mut self) {
        self.seg.posterior = None;
        self.seg.rx = None;
        self.seg.cancel = None;
        self.paint_dirty = true;
    }

    /// Post a status message: the pill shows it, dismissing idle outcomes
    /// after the status pill's lifetime unless they are failures.
    pub(crate) fn set_status(&mut self, msg: impl Into<String>, error: bool) {
        self.seg.status = msg.into();
        self.seg.status_error = error;
        self.seg.status_at = std::time::Instant::now();
    }

    pub(crate) fn drain_segmentation(&mut self) {
        // Collect first: assigning `self.seg.rx` inside the loop would fight
        // the receiver borrow.
        let mut msgs = Vec::new();
        let mut disconnected = false;
        if let Some(rx) = &self.seg.rx {
            while let Ok(m) = rx.try_recv() {
                msgs.push(m);
            }
            // Single consumer: an empty channel whose sender is gone means
            // the worker is done. A message racing into this probe joins
            // this frame's batch — dropping it could lose the terminal
            // Done/Failed; crash detection just waits a frame.
            match rx.try_recv() {
                Ok(m) => msgs.push(m),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => disconnected = true,
                Err(_) => {}
            }
        }
        let total = self.seg.run_total;
        for msg in msgs {
            match msg {
                SegMsg::Status(s) => self.set_status(s, false),
                SegMsg::Progress { done, eig, ab } => {
                    self.set_status(format!("iteration {done}/{total}, EIG {eig:.3}"), false);
                    self.seg.posterior = Some(*ab);
                    // The heatmap paints the new buffers.
                    self.paint_dirty = true;
                }
                SegMsg::Done => {
                    // A cancel before round 0 delivers an empty posterior;
                    // that must not clobber the previous run's cut labels.
                    if let Some(ab) = self.seg.posterior.take() {
                        self.seg.result = Some(ab);
                    }
                    self.seg_finished();
                }
                SegMsg::Failed(e) => {
                    eprintln!("segmentation failed: {e}");
                    self.set_status(format!("failed: {e}"), true);
                    self.seg_finished();
                }
            }
        }
        // A panicked worker drops its sender without a terminal message;
        // re-enable the button instead of staying busy forever.
        if disconnected && self.seg.busy() {
            eprintln!("segmentation worker crashed");
            self.set_status("segmentation crashed", true);
            self.seg_finished();
        }
    }
}
