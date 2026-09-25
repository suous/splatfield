// The wasm32 binary is the product; on host this crate only has to keep
// compiling for `cargo test`/`clippy`, where the wasm entry is cfg'd out
// and nothing is reachable from the tombstone main.
#![cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]

mod scene;
mod ui;
mod worker;

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

#[cfg(target_arch = "wasm32")]
use eframe::{wasm_bindgen::JsCast, web_sys};

use cubecl::wgpu::{MemoryConfiguration, RuntimeOptions, WgpuSetup, init_device};
use cubecl::{Device, client::Client};
use eframe::egui;
use eframe::wgpu;
use splat_sort::tensor::GpuTensor;
use splatfield::{camera, render, texture};

use scene::{FrameGpu, FrameSlot, Loaded};
use worker::SegUi;

/// Selection highlight color: the box overlay and the selected splats' tint.
const SELECT_GREEN: [u8; 3] = [0x30, 0xff, 0x55];

struct App {
    /// Clone of the egui context: async tasks (the segmentation loop) that
    /// publish outside a frame request repaints through it.
    ctx: egui::Context,
    gpu: FrameSlot,
    // Stable for the texture's lifetime — recreate_texture reuses the id — so
    // the paint path never borrows `gpu`.
    tex_id: eframe::egui::TextureId,
    controller: camera::Controller,
    client: Client,
    splats: Arc<Mutex<Loaded>>,
    seg: SegUi,
    /// Palette index for the next segmentation's color.
    next_color: usize,
    /// Selected splats (ascending, current display numbering) and the
    /// box-select edit state: the removed-set in master numbering plus the
    /// undo stack — one (prior removed-set, cut's DC colors) level per edit,
    /// popped last-action-first.
    sel: Vec<usize>,
    removed: Vec<usize>,
    undo: Vec<(Vec<usize>, Option<GpuTensor>)>,
    // What the backbuffer currently shows: the frame size, the model, and
    // the camera pose it was rendered from. Fresh Arcs cover new loads; the
    // pose comparison catches motion that arrived while a wasm render was
    // in flight (the slot was busy, so the moved pose was never scheduled).
    // In-place GPU mutations that a fresh Arc can't cover — the selection
    // tint, posterior updates, heatmap/reset toggles — go through
    // paint_dirty instead.
    rendered: Option<(glam::UVec2, Arc<render::Splats>, camera::Camera)>,
    paint_dirty: bool,
}

fn wgpu_config() -> eframe::egui_wgpu::WgpuConfiguration {
    eframe::egui_wgpu::WgpuConfiguration {
        wgpu_setup: eframe::egui_wgpu::WgpuSetup::CreateNew(
            eframe::egui_wgpu::WgpuSetupCreateNew {
                device_descriptor: Arc::new(|adapter: &wgpu::Adapter| wgpu::DeviceDescriptor {
                    required_features: adapter.features().difference(
                        wgpu::Features::MAPPABLE_PRIMARY_BUFFERS
                            | wgpu::Features::all_experimental_mask(),
                    ),
                    required_limits: adapter.limits(),
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                    ..Default::default()
                }),
                ..eframe::egui_wgpu::WgpuSetupCreateNew::without_display_handle()
            },
        ),
        ..Default::default()
    }
}

impl App {
    fn new(cc: &eframe::CreationContext) -> Self {
        // Stick to dark: the 3D viewport renders on black, and a light-mode
        // OS shouldn't flip the panel on top of it.
        cc.egui_ctx.set_theme(egui::Theme::Dark);
        // One ordered submission stream: the segmentation worker and the UI
        // thread share GPU buffers (tint, posterior), and cubecl's default
        // per-thread streams don't order against each other.
        splatfield::use_single_stream();
        let render_state = cc.wgpu_render_state.as_ref().expect("Must use wgpu");
        // wgpu's default uncaptured-error handler panics; device-loss isn't
        // even observable without one. Log instead — a lost/OOM device then
        // shows up as an explanatory stderr line instead of a downstream
        // egui staging-buffer panic with misleading size numbers.
        render_state.device.on_uncaptured_error(Arc::new(|err| {
            eprintln!("splatfield: wgpu device error: {err}");
        }));
        let device = init_device(
            WgpuSetup {
                instance: render_state.instance.clone(),
                adapter: render_state.adapter.clone(),
                device: render_state.device.clone(),
                queue: render_state.queue.clone(),
                backend: render_state.adapter.get_info().backend,
            },
            RuntimeOptions {
                tasks_max: 64,
                memory_config: MemoryConfiguration::ExclusivePages,
            },
        );

        let backbuffer = texture::GpuTexture::new(
            render_state.renderer.clone(),
            render_state.device.clone(),
            render_state.queue.clone(),
        );
        // Take the texture id BEFORE the backbuffer moves into the slot.
        let tex_id = backbuffer.id;
        let seg = SegUi::default();
        // The pipeline worker's replies feed the seg inbox and repaint —
        // without the repaint request the pill would never draw progress.
        // (The spawn itself is lazy; installing here starts it.)
        #[cfg(target_arch = "wasm32")]
        {
            let inbox = Rc::clone(&seg.inbox);
            let ctx = cc.egui_ctx.clone();
            worker::on_response(Box::new(move |response| {
                inbox.borrow_mut().push(response);
                ctx.request_repaint();
            }));
            worker::install();
        }
        Self {
            ctx: cc.egui_ctx.clone(),
            gpu: Rc::new(RefCell::new(Some(FrameGpu {
                backbuffer,
                scratch: None,
            }))),
            tex_id,
            controller: camera::Controller::default(),
            client: Device::Wgpu(device).client(),
            splats: Arc::new(Mutex::new(Loaded::default())),
            seg,
            next_color: 0,
            sel: Vec::new(),
            removed: Vec::new(),
            undo: Vec::new(),
            rendered: None,
            paint_dirty: false,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    eprintln!(
        "splatfield: native support removed — this build targets wasm32 (see README: trunk serve)"
    );
    std::process::exit(1);
}

#[cfg(target_arch = "wasm32")]
fn main() {
    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window()
            .expect("no window")
            .document()
            .expect("no document");
        let canvas = document
            .get_element_by_id("the_canvas_id")
            .expect("missing #the_canvas_id")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("#the_canvas_id is not a canvas");

        eframe::WebRunner::new()
            .start(
                canvas,
                eframe::WebOptions {
                    wgpu_options: wgpu_config(),
                    ..Default::default()
                },
                Box::new(|cc| Ok(Box::new(App::new(cc)))),
            )
            .await
            .expect("failed to start");

        if let Some(el) = document.get_element_by_id("loading_text") {
            el.remove();
        }
    });
}
