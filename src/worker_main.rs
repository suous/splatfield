//! splatfield-worker — the pipeline worker.
//!
//! Spawned from the page via `new Worker("splatfield-worker_loader.js")`
//! (src/worker.rs); trunk's `data-type="worker"` link in index.html builds
//! the loader + wasm. Owns the whole model pipeline — EnsureModels
//! downloads/verifies the pinned release and loads the SAM2 sessions,
//! Segment runs detector → detect → encode → decode. Every request is
//! answered — never silently dropped — and one request runs at a time (the
//! busy guard); a second concurrent request fails loud. A wasm panic is an
//! abort, so handlers route every error into a `*Failed` response by
//! construction: nothing in the request path panics.

#[cfg(target_arch = "wasm32")]
mod wasm {
    use gsam::ortweb::Ep;
    use gsam::{Detector, ModelStore, Sam2};
    use splatfield::fetch;
    use splatfield::opfs;
    use splatfield::pipeline::{Request, Response, decode_request, encode};
    use std::cell::{Cell, RefCell};
    use wasm_bindgen::{JsCast, JsValue, closure::Closure};
    use wasm_bindgen_futures::{JsFuture, spawn_local};
    use web_sys::console;
    use web_sys::{DedicatedWorkerGlobalScope, MessageEvent};
    use web_time::Instant;

    /// The ort pin's single home (the main thread never loads ORT — see
    /// index.html). One pinned dist, three artifacts, loaded as an ES
    /// module: the ./webgpu entry paired with the asyncify wasm is ort
    /// web's non-JSEP WebGPU stack — the only one of its two shipped
    /// kernel stacks that executes this model family correctly (the JSEP
    /// stack behind the ort.min.js classic script this pin used before
    /// mis-executes dino at every precision and drifted SAM2's high-res
    /// features to whole-box masks; docs/websam_evaluation.md,
    /// "transformers.js probe" + D4). The same entry's wasm EP is the CPU
    /// fallback, verified at parity with the old path (fp16 dino 15.5s,
    /// correct). A classic worker cannot importScripts an ES module, so
    /// the module is dynamic-imported and published as `globalThis.ort` —
    /// the name the gsam ortweb bridge resolves at session-create time.
    const ORT_MODULE_URL: &str =
        "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.bundle.min.mjs";
    const ORT_WASM_ASYNCIFY_MJS: &str = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort-wasm-simd-threaded.asyncify.mjs";
    const ORT_WASM_ASYNCIFY_WASM: &str = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort-wasm-simd-threaded.asyncify.wasm";

    /// Everything the pipeline owns: the verified release, the loaded SAM2
    /// sessions, and the detector cache (one prompt — a different prompt
    /// replaces the cached detector outright).
    struct Pipeline {
        store: ModelStore,
        sam: Sam2,
        /// The execution provider the detector session loads under —
        /// picked once at EnsureModels (the EP the SAM2 sessions
        /// proved), inherited by every per-prompt detector create.
        detector_ep: Ep,
        detector: Option<(String, Detector)>,
    }

    thread_local! {
        /// Taken out for the duration of one request: the cell is never
        /// held across an await, and a request that finds it gone replies
        /// "models not loaded" instead of interleaving sessions.
        static PIPELINE: RefCell<Option<Pipeline>> = const { RefCell::new(None) };
        /// One request at a time — sessions must not interleave.
        static BUSY: Cell<bool> = const { Cell::new(false) };
        /// The successful EnsureModels result — a repeat request answers
        /// with it instead of refetching (idempotent, original timings and
        /// the EP they were loaded under).
        static READY: RefCell<Option<(f64, f64, String)>> = const { RefCell::new(None) };
    }

    fn acquire() -> bool {
        BUSY.with(|b| {
            if b.get() {
                false
            } else {
                b.set(true);
                true
            }
        })
    }

    fn release() {
        BUSY.with(|b| b.set(false));
    }

    pub fn start() {
        console_error_panic_hook::set_once();
        let scope: DedicatedWorkerGlobalScope = js_sys::global()
            .dyn_into()
            .expect("splatfield-worker must run as a dedicated worker");
        let on_message = {
            // The handler lives for the worker's whole lifetime: it keeps a
            // clone of the scope, the original goes to set_onmessage below.
            let scope = scope.clone();
            Closure::<dyn FnMut(MessageEvent)>::new(move |e: MessageEvent| {
                let bytes = e
                    .data()
                    .dyn_into::<js_sys::Uint8Array>()
                    .ok()
                    .map(|a| a.to_vec());
                let Some(bytes) = bytes else {
                    console::error_1(&"[worker] frame is not a Uint8Array".into());
                    return;
                };
                let Ok(request) = decode_request(&bytes) else {
                    console::error_1(&"[worker] undecodable frame".into());
                    return;
                };
                match request {
                    Request::EnsureModels => {
                        let scope = scope.clone();
                        spawn_local(async move {
                            // Idempotent: already loaded, original timings
                            // and the EP they were loaded under. (Read
                            // before any await — safe to borrow here.)
                            let ready = READY.with_borrow(|r| r.clone());
                            if let Some((fetch_ms, sam2_ms, provenance)) = ready {
                                reply(
                                    &scope,
                                    Response::ModelsReady {
                                        fetch_ms,
                                        sam2_ms,
                                        provenance,
                                    },
                                );
                                return;
                            }
                            if !acquire() {
                                reply(
                                    &scope,
                                    Response::ModelsFailed(
                                        "worker busy — another request is running".into(),
                                    ),
                                );
                                return;
                            }
                            let resp = load_models(&scope).await;
                            release();
                            reply(&scope, resp);
                        });
                    }
                    Request::Segment {
                        rgb,
                        width,
                        height,
                        prompt,
                    } => {
                        let scope = scope.clone();
                        spawn_local(async move {
                            if !acquire() {
                                reply(
                                    &scope,
                                    Response::SegmentFailed(
                                        "worker busy — another request is running".into(),
                                    ),
                                );
                                return;
                            }
                            let resp = segment(&scope, rgb, width, height, prompt).await;
                            release();
                            reply(&scope, resp);
                        });
                    }
                }
            })
        };
        scope.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
        on_message.forget();
        console::log_1(&"[worker] splatfield-worker ready".into());
    }

    fn reply(scope: &DedicatedWorkerGlobalScope, resp: Response) {
        match encode(&resp) {
            Ok(bytes) => {
                if let Err(err) = scope.post_message(&js_sys::Uint8Array::from(bytes.as_slice())) {
                    console::error_1(&format!("[worker] post_message failed: {err:?}").into());
                }
            }
            // A response that cannot encode is a protocol bug; say so and drop.
            Err(err) => {
                console::error_1(&format!("[worker] response encode failed: {err:#}").into())
            }
        }
    }

    fn ms(t0: Instant) -> f64 {
        t0.elapsed().as_secs_f64() * 1e3
    }

    /// Download + verify the pinned release, load the SAM2 sessions, and
    /// become ready. Progress and stage updates stream back while it runs.
    async fn load_models(scope: &DedicatedWorkerGlobalScope) -> Response {
        let t0 = Instant::now();
        // Progress fires per stream chunk; a fractional step of ≥1% (or the
        // final 1.0) is all the status pill can use — per-chunk replies
        // flood the main thread's message queue.
        let mut last = 0.0f64;
        let (store, from_cache) = match fetch::ensure_models(&mut |p| {
            if p - last >= 0.01 || p >= 1.0 {
                last = p;
                reply(scope, Response::DownloadProgress(p));
            }
        })
        .await
        {
            Ok(store) => store,
            Err(e) => return Response::ModelsFailed(format!("model fetch: {e:#}")),
        };
        let fetch_ms = ms(t0);
        // Before any session create: the bridge resolves globalThis.ort at
        // call time, so the module must be in place first. Fallible — a
        // failed ort load answers ModelsFailed, it must not abort.
        if let Err(e) = load_ort_once().await {
            return Response::ModelsFailed(format!("ort load: {e}"));
        }
        // One probe, all three slots follow it: on the non-JSEP ort stack
        // this pin loads, every model passed the real-image webgpu gate
        // (see the ORT URL comment); without an adapter everything
        // collapses to wasm.
        let ep = probe_webgpu().await;
        let ep_name = ep.as_str();
        let provenance = format!("dino:{ep_name} enc:{ep_name} dec:{ep_name}");
        console::log_1(&format!("[worker] execution providers: {provenance}").into());
        reply(scope, Response::Stage("loading SAM2…".into()));
        let t1 = Instant::now();
        let sam = match Sam2::load(&store, ep).await {
            Ok(sam) => sam,
            Err(e) => return Response::ModelsFailed(format!("SAM2 load: {e:#}")),
        };
        let sam2_ms = ms(t1);
        // Cache best-effort, AFTER the sessions proved loadable — a release
        // that cannot open its sessions must not be cached as good. Only a
        // fresh download persists: a cache hit was just re-validated from
        // these very bytes, and rewriting 158 MB would erase the win. A
        // persist failure (quota, private mode, OPFS disabled) must not fail
        // the request either: the release is verified and in hand; the only
        // cost is redownloading next page load. Staged visibly: a persist
        // that hangs or dies shows "caching models…" instead of a fake
        // ready.
        if !from_cache {
            reply(scope, Response::Stage("caching models…".into()));
            if let Err(err) = opfs::persist(&store).await {
                console::warn_1(&format!("opfs cache persist failed: {err:#}").into());
            }
        }
        READY.with_borrow_mut(|r| {
            *r = Some((fetch_ms, sam2_ms, provenance.clone()));
        });
        PIPELINE.with_borrow_mut(|slot| {
            *slot = Some(Pipeline {
                store,
                sam,
                detector_ep: ep,
                detector: None,
            });
        });
        Response::ModelsReady {
            fetch_ms,
            sam2_ms,
            provenance,
        }
    }

    /// Detector (cached per prompt) → detect → encode → decode → mask
    /// bytes. The pipeline is taken out of its cell for the duration so no
    /// await runs holding the borrow, and always put back.
    async fn segment(
        scope: &DedicatedWorkerGlobalScope,
        rgb: Vec<u8>,
        width: u32,
        height: u32,
        prompt: String,
    ) -> Response {
        let Some(mut pipeline) = PIPELINE.with_borrow_mut(|p| p.take()) else {
            return Response::SegmentFailed("models not loaded — send EnsureModels first".into());
        };
        let resp = run_segment(scope, &mut pipeline, &rgb, width, height, &prompt).await;
        PIPELINE.with_borrow_mut(|p| *p = Some(pipeline));
        resp
    }

    async fn run_segment(
        scope: &DedicatedWorkerGlobalScope,
        pipeline: &mut Pipeline,
        rgb: &[u8],
        width: u32,
        height: u32,
        prompt: &str,
    ) -> Response {
        // Take first: dropping the replaced session BEFORE the new create
        // keeps two 151 MB graphs from being alive at once — a wasm
        // linear-memory spike that showed up as a 2.5× detect slowdown.
        let cached = pipeline.detector.take();
        let mut detector = match cached {
            Some((cached_prompt, detector)) if cached_prompt == prompt => detector,
            // A different prompt replaces the cached detector outright —
            // the cache holds exactly one prompt per session.
            Some(_) | None => {
                reply(scope, Response::Stage("loading detector…".into()));
                match Detector::load(&pipeline.store, prompt, pipeline.detector_ep).await {
                    Ok(detector) => detector,
                    Err(e) => return Response::SegmentFailed(format!("detector load: {e:#}")),
                }
            }
        };

        // No per-round stage messages here (detect/encode/decode): the
        // loop's "iteration k/N, EIG x" pill line is the progress surface —
        // per-round replies would overwrite it ~3× per round, flickering
        // (the native oracle was silent per round too).
        let t0 = Instant::now();
        let detected = detector.detect(rgb, width, height).await;
        let detect_ms = ms(t0);
        // Cached even on an error/empty result — the graph and its prompt
        // constants are prompt-scoped, not result-scoped.
        pipeline.detector = Some((prompt.to_string(), detector));
        let detections = match detected {
            Ok(detections) => detections,
            Err(e) => return Response::SegmentFailed(format!("detect: {e:#}")),
        };
        // Detections come back best-confidence first. An empty result is NOT
        // a failure: the only consumer is the B3-Seg loop, whose stop rule
        // reads the mask's zero foreground count (src/seg/active.rs
        // `missed && !any_fg`), exactly like the native oracle
        // (seg/prompted.rs: "nothing found means an all-background mask and
        // no decode"). A SegmentFailed here would force the loop to
        // string-match prose. Parity means an all-background mask, zero
        // confidence, and no decode — encode+decode are skipped entirely
        // (encode_ms/decode_ms = 0.0 marks the empty path; see pipeline.rs).
        let Some(detection) = detections.first() else {
            return Response::SegmentDone {
                box_px: [0.0; 4],
                confidence: 0.0,
                mask: vec![0; (width * height) as usize],
                detect_ms,
                encode_ms: 0.0,
                decode_ms: 0.0,
            };
        };
        let (box_px, confidence) = (detection.xyxy, detection.conf);

        let t1 = Instant::now();
        if let Err(e) = pipeline.sam.encode(rgb, width, height).await {
            return Response::SegmentFailed(format!("SAM2 encode: {e:#}"));
        }
        let encode_ms = ms(t1);

        let t2 = Instant::now();
        let mask = match pipeline.sam.decode(box_px, width, height).await {
            Ok(mask) => mask,
            Err(e) => return Response::SegmentFailed(format!("SAM2 decode: {e:#}")),
        };
        Response::SegmentDone {
            box_px,
            confidence,
            mask,
            encode_ms,
            detect_ms,
            decode_ms: ms(t2),
        }
    }

    /// One navigator.gpu adapter request: [`Ep::WebGpu`] iff an adapter
    /// resolves, wasm otherwise — a null adapter is a GPU the page cannot
    /// use, the same fallback as no GPU at all. Every JS touch is guarded;
    /// the probe's whole output defaults to [`Ep::Wasm`] (a wasm panic is
    /// an abort, so nothing in here may unwind).
    async fn probe_webgpu() -> Ep {
        async {
            let nav = js_sys::Reflect::get(&js_sys::global(), &"navigator".into())
                .ok()?
                .dyn_into::<js_sys::Object>()
                .ok()?;
            let gpu = js_sys::Reflect::get(&nav, &"gpu".into()).ok()?;
            if gpu.is_undefined() || gpu.is_null() {
                return None;
            }
            let gpu = gpu.dyn_into::<js_sys::Object>().ok()?;
            let request_adapter: js_sys::Function =
                js_sys::Reflect::get(&gpu, &"requestAdapter".into())
                    .ok()?
                    .dyn_into()
                    .ok()?;
            let promise: js_sys::Promise = request_adapter.call0(&gpu).ok()?.into();
            let adapter = JsFuture::from(promise).await.ok()?;
            adapter.is_truthy().then_some(Ep::WebGpu)
        }
        .await
        .unwrap_or(Ep::Wasm)
    }

    /// Dynamic-import the pinned ort ES module once and publish it as
    /// `globalThis.ort`, then point `ort.env.wasm` at the asyncify pair
    /// before the first session create (ort resolves its .wasm against
    /// `wasmPaths`; the worker is not cross-origin isolated, so numThreads
    /// stays 1). Fallible by construction — a wasm panic is an abort, so a
    /// failed load returns Err and the caller answers ModelsFailed.
    async fn load_ort_once() -> Result<(), String> {
        thread_local! {
            static LOADED: Cell<bool> = const { Cell::new(false) };
        }
        if LOADED.with(Cell::get) {
            return Ok(());
        }
        // import() is syntax, not a callable — eval is the classic-worker
        // route to a dynamic module import (there is no bundler on this
        // side; the URL is absolute so base-URL resolution is moot).
        let promise: js_sys::Promise = js_sys::eval(&format!("import('{ORT_MODULE_URL}')"))
            .map_err(|e| format!("ort import() eval failed: {e:?}"))?
            .dyn_into()
            .map_err(|_| "import() did not yield a promise".to_string())?;
        let module = JsFuture::from(promise)
            .await
            .map_err(|e| format!("ort module load failed: {e:?}"))?;
        let has_session = js_sys::Reflect::get(&module, &"InferenceSession".into())
            .map(|v| !v.is_undefined())
            .unwrap_or(false);
        if !has_session {
            return Err("ort module has no InferenceSession export".into());
        }
        js_sys::Reflect::set(&js_sys::global(), &"ort".into(), &module)
            .map_err(|e| format!("cannot publish globalThis.ort: {e:?}"))?;
        let err = |what: &str| format!("ort env setup failed at {what}");
        let wasm_env = js_sys::Reflect::get(&module, &"env".into())
            .map_err(|_e| err("env"))?
            .dyn_into::<js_sys::Object>()
            .map_err(|_| err("env object"))?;
        let wasm = js_sys::Reflect::get(&wasm_env, &"wasm".into())
            .map_err(|_e| err("env.wasm"))?
            .dyn_into::<js_sys::Object>()
            .map_err(|_| err("env.wasm object"))?;
        let wasm_paths = js_sys::Object::new();
        js_sys::Reflect::set(&wasm_paths, &"mjs".into(), &ORT_WASM_ASYNCIFY_MJS.into())
            .map_err(|_e| err("wasmPaths.mjs"))?;
        js_sys::Reflect::set(&wasm_paths, &"wasm".into(), &ORT_WASM_ASYNCIFY_WASM.into())
            .map_err(|_e| err("wasmPaths.wasm"))?;
        js_sys::Reflect::set(&wasm, &"wasmPaths".into(), &wasm_paths.into())
            .map_err(|_e| err("wasmPaths set"))?;
        js_sys::Reflect::set(&wasm, &"numThreads".into(), &JsValue::from(1))
            .map_err(|_e| err("numThreads set"))?;
        LOADED.with(|l| l.set(true));
        Ok(())
    }
}

// cargo requires a bin entry point; on wasm it never runs — the
// #[wasm_bindgen(start)] function below does, via __wbindgen_start.
#[cfg(target_arch = "wasm32")]
fn main() {}

// Named anything but `main`: the macro exports a shim under the fn's own
// name, which would collide with rustc's entry symbol.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
fn worker_start() {
    wasm::start();
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    eprintln!(
        "splatfield-worker is worker-only — built by trunk's data-type=\"worker\" link (wasm32)"
    );
}
