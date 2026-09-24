mod scene;
mod ui;
mod worker;

use std::sync::{Arc, Mutex};

use cubecl::wgpu::{MemoryConfiguration, RuntimeOptions, WgpuRuntime, WgpuSetup, init_device};
use cubecl::{Runtime, client::ComputeClient};
use eframe::egui;
use eframe::wgpu;
use splat_sort::tensor::GpuTensor;
use splatfield::{camera, render, texture};

use scene::{FrameGpu, Loaded};
use worker::SegUi;

/// Selection highlight color: the box overlay and the selected splats' tint.
const SELECT_GREEN: [u8; 3] = [0x30, 0xff, 0x55];

struct App {
    gpu: FrameGpu,
    controller: camera::Controller,
    client: ComputeClient<WgpuRuntime>,
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
    // What the backbuffer currently shows: the frame size and the model it
    // was rendered from (the camera is implied — it only moves through
    // Controller::tick, which reports moves). In-place GPU mutations that a
    // fresh Arc can't cover — the segmentation tint, posterior updates,
    // heatmap/reset toggles — go through paint_dirty instead.
    rendered: Option<(glam::UVec2, Arc<render::Splats>)>,
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
        Self {
            gpu: FrameGpu {
                backbuffer,
                scratch: None,
            },
            controller: camera::Controller::default(),
            client: WgpuRuntime::client(&device),
            splats: Arc::new(Mutex::new(Loaded::default())),
            seg: SegUi::default(),
            next_color: 0,
            sel: Vec::new(),
            removed: Vec::new(),
            undo: Vec::new(),
            rendered: None,
            paint_dirty: false,
        }
    }
}

fn main() -> std::process::ExitCode {
    if std::env::args().nth(1).as_deref() == Some("seg") {
        return splatfield::cli::run(std::env::args().skip(2));
    }
    match eframe::run_native(
        "SplatField",
        eframe::NativeOptions {
            wgpu_options: wgpu_config(),
            ..Default::default()
        },
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    ) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("splatfield: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}
