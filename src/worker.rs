//! The segmentation sidebar state and the App methods that drive it, plus
//! the page side of the pipeline worker link (spawn, send, response sink —
//! wasm only). Declared from `main.rs` — binary code, so library items go
//! through `splatfield::`.
//!
//! On wasm a run is two waits stitched on the UI thread: the loop task
//! (`start_segmentation_run`) drives `Segmenter::run_with_async`, whose
//! oracle awaits one `Request::Segment` round trip per round over a
//! promise ticket, and publishes progress/terminal through `SegMsg`s that
//! `App::ui` folds — the native thread flow's channel semantics, rebuilt
//! on spawn_local. Single-flight is one flag chain: `busy` covers the
//! whole fetch+run window and the buttons gate on it.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use eframe::egui;
use splat_sort::tensor::GpuTensor;
use splatfield::pipeline::Response;

// wasm-only machinery (the loop task, the ticket) — the host build never
// runs the UI, so these imports exist only under the wasm cfg.
#[cfg(target_arch = "wasm32")]
use glam::UVec2;
#[cfg(target_arch = "wasm32")]
use splatfield::camera::Camera;
#[cfg(target_arch = "wasm32")]
use splatfield::seg::active::{Config as SegConfig, Iteration};
#[cfg(target_arch = "wasm32")]
use splatfield::seg::beta::map_labels;
#[cfg(target_arch = "wasm32")]
use splatfield::{pipeline, to_dc};
#[cfg(target_arch = "wasm32")]
use std::sync::atomic::Ordering;

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

/// B3-Seg sidebar state: what the user asked for plus whatever the last
/// run published.
pub(crate) struct SegUi {
    pub(crate) prompt: String,
    pub(crate) heatmap: bool,
    pub(crate) iters: usize,
    /// Start corner of the live Shift+drag box selection, if any.
    pub(crate) box_drag: Option<egui::Pos2>,
    /// Set while a run is active: polled by the loop task's `on_round`,
    /// stored by the UI's stop button. Created per run at loop start.
    pub(crate) cancel: Option<Arc<AtomicBool>>,
    /// The prompt whose run is queued behind EnsureModels (first run): when
    /// ModelsReady folds, it starts the loop with this. `Some` keeps `busy`
    /// set — the fetch and the run are one user action.
    pub(crate) pending_run: Option<String>,
    /// The worker answered ModelsReady at least once — later runs skip the
    /// EnsureModels hop (the worker's answer is idempotent, but the page
    /// knows without asking).
    pub(crate) models_ready: bool,
    /// Engine message, spoken by the top pill: live progress while a run is
    /// active, idle outcomes self-dismissing. `status_error` messages
    /// persist until replaced — vanishing errors get missed.
    pub(crate) status: String,
    pub(crate) status_error: bool,
    pub(crate) status_at: web_time::Instant,
    /// A pipeline request is in flight (wasm: real, from the worker round
    /// trip; host: permanently false — nothing sets it).
    pub(crate) busy: bool,
    /// Worker replies accumulated since the last frame, drained by
    /// `App::ui` (wasm). `Rc<RefCell<>>` because the worker sink outlives
    /// no particular borrow: it pushes from a JS callback outside the egui
    /// borrow.
    pub(crate) inbox: Rc<RefCell<Vec<Response>>>,
    /// The loop task's publications (progress, terminal), drained by
    /// `App::ui` — the task can't borrow `self`, so it stages through here
    /// exactly like the native worker thread staged through its channel.
    pub(crate) msgs: Rc<RefCell<Vec<SegMsg>>>,
    pub(crate) posterior: Option<(GpuTensor, GpuTensor)>,
    /// The last completed run's final posterior — the cut button's labels.
    /// Unlike `posterior` it survived the run's Done, whose tint retired
    /// the heatmap preview.
    pub(crate) result: Option<(GpuTensor, GpuTensor)>,
}

/// One message from the loop task to the UI (the native worker thread's
/// channel, rebuilt over an `Rc<RefCell<Vec>>` drain).
pub(crate) enum SegMsg {
    /// A progress line for the pill.
    Status(String),
    /// One delivered round: the pill's progress line and the
    /// posterior-so-far for the live heatmap.
    Progress {
        status: String,
        ab: Box<(GpuTensor, GpuTensor)>,
    },
    Done,
    Failed(String),
}

impl Default for SegUi {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            // The posterior heatmap is the loop's live feedback — on unless
            // the user turns it off.
            heatmap: true,
            // The live-editable iteration count; every run feeds it into
            // seg::active::Config's iterations (the total in the pill is
            // captured at spawn, so a mid-run drag can't rewrite it).
            iters: 20,
            box_drag: None,
            cancel: None,
            pending_run: None,
            models_ready: false,
            status: String::new(),
            status_error: false,
            status_at: web_time::Instant::now(),
            busy: false,
            inbox: Rc::new(RefCell::new(Vec::new())),
            msgs: Rc::new(RefCell::new(Vec::new())),
            posterior: None,
            result: None,
        }
    }
}

impl App {
    /// Validate and launch a segmentation run. On the web the first run
    /// starts with the model pipeline: EnsureModels downloads/verifies the
    /// pinned release (or reads the OPFS cache) before any inference, and
    /// ModelsReady hands the queued prompt to the loop task.
    pub(crate) fn request_segmentation(&mut self, prompt: String) {
        #[cfg(target_arch = "wasm32")]
        {
            if self.seg.busy {
                return;
            }
            // The worker paints its own tint on top of pristine colors; drop
            // the selection highlight so it can't half-survive the run.
            self.sel.clear();
            self.paint_selection();
            if self.splats.lock().unwrap().scene.is_none() {
                return;
            }
            self.seg.busy = true;
            if self.seg.models_ready {
                self.start_segmentation_run(prompt);
            } else {
                // First run: fetch before inference. ModelsReady folds into
                // start_segmentation_run through `pending_run`; busy stays
                // set across the handoff — clearing it would open a
                // second-flight window (never re-send EnsureModels while a
                // Segment can be in flight; EnsureModels is answered from
                // the worker's READY cache anyway).
                self.seg.pending_run = Some(prompt);
                self.set_status("fetching models…", false);
                send(&pipeline::Request::EnsureModels);
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = prompt;
            self.set_status(
                "segmentation runs on wasm — the host build is a tombstone",
                true,
            );
        }
    }

    /// True while the numbering-sensitive edits must wait: a run is active
    /// or a cut's async readback is still staging (the flag lives on
    /// `Loaded`, next to the staged outcome it guards). The edit paths used
    /// to check only `busy`; a cut landing under a concurrent delete / undo
    /// / reset / save / box-select would fold master indices against a
    /// numbering that changed mid-readback.
    pub(crate) fn locked(&self) -> bool {
        self.seg.busy || self.splats.lock().unwrap().cut_in_flight
    }

    /// Post a status message: the pill shows it, dismissing idle outcomes
    /// after the status pill's lifetime unless they are failures.
    pub(crate) fn set_status(&mut self, msg: impl Into<String>, error: bool) {
        self.seg.status = msg.into();
        self.seg.status_error = error;
        self.seg.status_at = web_time::Instant::now();
    }
}

#[cfg(target_arch = "wasm32")]
impl App {
    /// The worker-protocol fold pins the Response → pill mapping: progress
    /// and stages speak live; Ready/Failed manage the fetch phase. The
    /// Segment frames are consumed by the in-flight oracle's ticket at the
    /// sink (`web::complete_segment_ticket`) — the loop task publishes
    /// progress and the terminal state through [`SegMsg`]s instead, so
    /// these arms stay no-ops.
    pub(crate) fn fold_response(&mut self, response: Response) {
        match response {
            Response::DownloadProgress(p) => {
                // The zip's byte length is content-pinned
                // (fetch::ZIP_SHA256), so the MB figures are a release
                // constant, not a guess.
                let mb = splatfield::fetch::ZIP_BYTES as f64 / 1e6;
                self.set_status(
                    format!(
                        "fetching models {:.0}% ({:.0}/{mb:.0} MB)",
                        p * 100.0,
                        p * mb
                    ),
                    false,
                );
            }
            Response::Stage(stage) => self.set_status(stage, false),
            Response::ModelsReady {
                fetch_ms,
                sam2_ms,
                provenance,
            } => {
                self.seg.models_ready = true;
                if let Some(prompt) = self.seg.pending_run.take() {
                    // A run was queued behind the fetch: start it. busy is
                    // NOT cleared — the run owns it now.
                    self.start_segmentation_run(prompt);
                } else {
                    self.seg.busy = false;
                    self.set_status(
                        format!(
                            "models ready — {provenance} (fetch {:.1}s, sam2 {:.1}s)",
                            fetch_ms / 1e3,
                            sam2_ms / 1e3
                        ),
                        false,
                    );
                }
            }
            Response::ModelsFailed(error) => {
                // A failed fetch strands the queued run — clear both; the
                // user retries from the segment button.
                self.seg.pending_run = None;
                self.seg.busy = false;
                self.set_status(error, true);
            }
            Response::SegmentDone { .. } | Response::SegmentFailed(_) => {}
        }
    }

    /// Fold one loop-task message — the native `drain_segmentation`'s
    /// matching arm set, kept semantics-for-semantics.
    pub(crate) fn fold_seg_msg(&mut self, msg: SegMsg) {
        match msg {
            SegMsg::Status(s) => self.set_status(s, false),
            SegMsg::Progress { status, ab } => {
                self.set_status(status, false);
                self.seg.posterior = Some(*ab);
                // The heatmap paints the new buffers.
                self.paint_dirty = true;
            }
            SegMsg::Done => {
                // Row-32 hazard: an empty-posterior Done (nothing ever
                // delivered) must not clobber the previous run's cut labels.
                if let Some(ab) = self.seg.posterior.take() {
                    self.seg.result = Some(ab);
                }
                self.seg_finished();
            }
            SegMsg::Failed(e) => {
                self.set_status(format!("failed: {e}"), true);
                self.seg_finished();
            }
        }
    }

    /// A terminal message (Done or Failed): retire the run — the permanent
    /// tint (or failure) replaces the heatmap preview, and a stale
    /// posterior must never feed an out-of-bounds paint once a smaller
    /// model loads. A Failed never touches `result` (row-32 hazard).
    fn seg_finished(&mut self) {
        self.seg.posterior = None;
        self.seg.cancel = None;
        self.seg.busy = false;
        self.paint_dirty = true;
    }

    /// Spawn the active-loop task: the current viewport pose (fov squared —
    /// a square segmentation render inscribed in what the user sees, the
    /// native flow's exact capture) runs T observation rounds, publishing
    /// per-iteration progress over [`SegMsg`], then colors the foreground
    /// splats with the next palette color on the GPU. Terminal semantics
    /// (native parity): Ok → "colored N/M splats" + Done, whose fold moves
    /// the posterior into `result`; Err → Failed, which preserves
    /// `result`.
    fn start_segmentation_run(&mut self, prompt: String) {
        let Some(splats) = self
            .splats
            .lock()
            .unwrap()
            .scene
            .as_ref()
            .map(|s| s.splats.clone())
        else {
            // The model vanished between the request and here (a load can't
            // race — busy gates it, but a failed load clears the scene);
            // release the run instead of hanging busy.
            self.seg.busy = false;
            return;
        };
        let cancel = Arc::new(AtomicBool::new(false));
        self.seg.cancel = Some(cancel.clone());
        let mut cam = self.controller.camera;
        cam.fov = glam::Vec2::splat(cam.fov.x.min(cam.fov.y));
        let iterations = self.seg.iters;
        self.seg.posterior = None;
        // Display color → SH DC coefficient.
        let color = to_dc(PALETTE[self.next_color % PALETTE.len()]);
        self.next_color += 1;
        let ctx = self.ctx.clone();
        let msgs = Rc::clone(&self.seg.msgs);

        wasm_bindgen_futures::spawn_local(async move {
            let outcome: anyhow::Result<String> = async {
                let mut segmenter = splatfield::seg::active::Segmenter::new(
                    Arc::clone(&splats),
                    SegConfig {
                        resolution: 512,
                        candidates: 20,
                        iterations,
                    },
                    cam,
                )?;
                let mut done = 0usize;
                // The loop owns the stop rules; this closure only feeds the
                // status row and the progress pill's heatmap, and polls the
                // cancel flag — false stops the run at the round boundary.
                let oracle = move |rgb: &[u8], size: UVec2, _camera: &Camera| {
                    // The sensor's future must not borrow the loop's render
                    // buffer (one Fut type across the higher-ranked calls),
                    // so the bytes move into the request before the await.
                    let request = pipeline::Request::Segment {
                        rgb: rgb.to_vec(),
                        width: size.x,
                        height: size.y,
                        prompt: prompt.clone(),
                    };
                    async move { oracle_round(request, size).await }
                };
                segmenter
                    .run_with_async(oracle, |round: &Iteration, state| {
                        done += 1;
                        msgs.borrow_mut().push(SegMsg::Progress {
                            status: format!("iteration {done}/{iterations}, EIG {:.3}", round.eig),
                            ab: Box::new((state.a.clone(), state.b.clone())),
                        });
                        ctx.request_repaint();
                        !cancel.load(Ordering::Relaxed)
                    })
                    .await?;
                // Terminal Ok: the permanent tint replaces the heatmap
                // preview — MAP-label the async posterior readback and
                // color only when something is foreground.
                let (a, b) = segmenter.posteriors_async().await;
                let labels = map_labels(&a, &b);
                let fg = labels.iter().filter(|&&l| l).count();
                if fg > 0 {
                    splats.tint(&segmenter.state.a, &segmenter.state.b, color);
                }
                Ok(format!("colored {fg}/{} splats", labels.len()))
            }
            .await;
            {
                let mut sink = msgs.borrow_mut();
                match outcome {
                    Ok(status) => {
                        sink.push(SegMsg::Status(status));
                        sink.push(SegMsg::Done);
                    }
                    // {e:#} — the native flow's error chain format.
                    Err(e) => sink.push(SegMsg::Failed(format!("{e:#}"))),
                }
            }
            ctx.request_repaint();
        });
    }
}

/// One oracle round over the worker wire: post the Segment request through
/// a promise ticket, await the reply, bincode-decode, and page-side
/// validate the mask — a length that disagrees with the frame is a wire
/// bug and fails the round loud.
#[cfg(target_arch = "wasm32")]
async fn oracle_round(request: pipeline::Request, size: UVec2) -> anyhow::Result<Vec<u8>> {
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;

    let promise = web::segment_ticket(request);
    let replied = JsFuture::from(promise)
        .await
        .map_err(|e| anyhow::anyhow!("segment request failed: {e:?}"))?;
    let bytes = replied
        .dyn_into::<js_sys::Uint8Array>()
        .map_err(|_| anyhow::anyhow!("segment reply is not a Uint8Array"))?
        .to_vec();
    let pixels = (size.x * size.y) as usize;
    match pipeline::decode_response(&bytes)? {
        Response::SegmentDone { mask, .. } => {
            anyhow::ensure!(
                mask.len() == pixels,
                "segment mask is {} bytes, want {pixels}",
                mask.len()
            );
            Ok(mask)
        }
        Response::SegmentFailed(error) => Err(anyhow::anyhow!(error)),
        other => Err(anyhow::anyhow!("unexpected worker reply: {other:?}")),
    }
}

/// The page side of the worker link: spawns the pipeline worker from
/// `splatfield-worker_loader.js` (trunk's `data-type="worker"` output) and
/// routes [`Response`]s to the registered sink. The app sends
/// [`splatfield::pipeline::Request`]s via [`web::send`]; the app's inbox is
/// fed through [`web::on_response`].
#[cfg(target_arch = "wasm32")]
mod web {
    use super::Response;
    use splatfield::pipeline::{decode_response, encode};
    use std::cell::RefCell;
    use wasm_bindgen::closure::Closure;
    use wasm_bindgen::{JsCast, JsValue};
    use web_sys::console;
    use web_sys::{MessageEvent, Worker};

    const WORKER_SCRIPT: &str = "splatfield-worker_loader.js";

    /// The app's response consumer, registered once from `App::new`.
    type ResponseSink = Box<dyn FnMut(Response)>;

    thread_local! {
        /// One worker per page; `None` means the spawn failed (logged) and
        /// requests are dropped with a console error instead of hanging.
        static WORKER: Option<Worker> = spawn_worker();
        /// wasm-bindgen closures are dropped when unreferenced; the thread_local
        /// keeps the handlers alive for the page's lifetime.
        static HANDLERS: Handlers = Handlers::new();
        /// The app's response sink. Set once from `App::new`; replies arriving
        /// before registration only hit the console.
        static SINK: RefCell<Option<ResponseSink>> = const { RefCell::new(None) };
    }

    /// Message handling is registered once: every reply feeds the app sink.
    struct Handlers {
        on_message: Closure<dyn FnMut(MessageEvent)>,
        on_error: Closure<dyn FnMut(JsValue)>,
    }

    impl Handlers {
        fn new() -> Self {
            let on_message = Closure::<dyn FnMut(MessageEvent)>::new(|e: MessageEvent| {
                let bytes = e
                    .data()
                    .dyn_into::<js_sys::Uint8Array>()
                    .ok()
                    .map(|a| a.to_vec());
                let Some(bytes) = bytes else {
                    console::error_1(&"[worker] reply is not a Uint8Array".into());
                    return;
                };
                match decode_response(&bytes) {
                    Ok(resp) => {
                        // Progress fires per stream chunk and SegmentDone carries
                        // the whole mask — Debug-formatting either floods the
                        // console (the latter builds a megabyte-scale string).
                        if !matches!(
                            resp,
                            Response::DownloadProgress(_) | Response::SegmentDone { .. }
                        ) {
                            console::log_1(&format!("[worker] {resp:?}").into());
                        }
                        // A Segment frame completes the in-flight oracle's
                        // ticket here — at the sink, not a frame later in
                        // the fold — so the loop task resumes immediately.
                        if matches!(
                            resp,
                            Response::SegmentDone { .. } | Response::SegmentFailed(_)
                        ) {
                            complete_segment_ticket(&bytes);
                        }
                        SINK.with_borrow_mut(|sink| {
                            if let Some(sink) = sink.as_mut() {
                                sink(resp);
                            }
                        });
                    }
                    Err(err) => {
                        console::error_1(&format!("[worker] undecodable reply: {err:#}").into());
                    }
                }
            });
            let on_error = Closure::<dyn FnMut(JsValue)>::new(|e: JsValue| {
                console::error_1(&format!("[worker] worker error: {e:?}").into());
            });
            Self {
                on_message,
                on_error,
            }
        }
    }

    pub(crate) fn install() {
        // Lazily spawns the worker. HANDLERS is initialized first so
        // spawn_worker's initializer finds it ready instead of nesting one
        // thread_local init inside another.
        HANDLERS.with(|_| {});
        WORKER.with(|w| w.is_some());
    }

    /// Register the app's response sink (called once from `App::new`).
    pub(crate) fn on_response(sink: ResponseSink) {
        SINK.with_borrow_mut(|slot| *slot = Some(sink));
    }

    thread_local! {
        /// The single-flight Segment ticket: the resolve side of the promise
        /// the in-flight oracle awaits. Only the loop task creates tickets
        /// and `SegUi::busy` gates it to one run at a time, so the slot
        /// holds at most one resolver — a second ticket would strand the
        /// first's await.
        static SEGMENT_RESOLVE: RefCell<Option<js_sys::Function>> =
            const { RefCell::new(None) };
    }

    /// Send a Segment request and hand back the promise its reply
    /// completes. The resolver parks here; the on_message sink takes it
    /// when the SegmentDone/SegmentFailed frame arrives. The reply crosses
    /// as the raw wire bytes — the awaiting oracle bincode-decodes them.
    pub(crate) fn segment_ticket(request: splatfield::pipeline::Request) -> js_sys::Promise {
        js_sys::Promise::new(&mut |resolve, _reject| {
            SEGMENT_RESOLVE.with_borrow_mut(|slot| *slot = Some(resolve));
            send(&request);
        })
    }

    /// Complete the parked ticket with the raw wire bytes of the Segment
    /// frame. A Segment frame with no parked ticket (a stray after a
    /// crashed run) is dropped here — the fold's matching arms are no-ops
    /// for the same reason.
    pub(crate) fn complete_segment_ticket(bytes: &[u8]) {
        SEGMENT_RESOLVE.with_borrow_mut(|slot| {
            if let Some(resolve) = slot.take() {
                let _ = resolve.call1(&js_sys::global(), &js_sys::Uint8Array::from(bytes));
            }
        });
    }

    /// Fire a request at the worker. Failure to post (worker absent or the
    /// structured clone failed) logs instead of panicking — the UI stays
    /// interactive; the missing reply surfaces as a stuck pill in the console.
    pub(crate) fn send(request: &splatfield::pipeline::Request) {
        WORKER.with(|w| {
            let Some(worker) = w.as_ref() else {
                console::error_1(&"[worker] cannot send — worker did not spawn".into());
                return;
            };
            match encode(request) {
                Ok(bytes) => {
                    if let Err(err) =
                        worker.post_message(&js_sys::Uint8Array::from(bytes.as_slice()))
                    {
                        console::error_1(&format!("[worker] post_message failed: {err:?}").into());
                    }
                }
                Err(err) => {
                    console::error_1(&format!("[worker] request encode failed: {err:#}").into())
                }
            }
        });
    }

    fn spawn_worker() -> Option<Worker> {
        match Worker::new(WORKER_SCRIPT) {
            Ok(worker) => {
                HANDLERS.with(|h| {
                    worker.set_onmessage(Some(h.on_message.as_ref().unchecked_ref()));
                    worker.set_onerror(Some(h.on_error.as_ref().unchecked_ref()));
                });
                console::log_1(&format!("[worker] spawned {WORKER_SCRIPT}").into());
                Some(worker)
            }
            Err(err) => {
                console::error_1(&format!("[worker] cannot spawn {WORKER_SCRIPT}: {err:?}").into());
                None
            }
        }
    }
}

/// Page-side worker link (wasm only): `App::new` installs the worker and
/// registers the inbox sink.
#[cfg(target_arch = "wasm32")]
pub(crate) use web::{install, on_response, send};
