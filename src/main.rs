use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

#[cfg(target_arch = "wasm32")]
use std::cell::Cell;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

#[cfg(not(target_arch = "wasm32"))]
use anyhow::Context;

use cubecl::prelude::*;
use cubecl::wgpu::{MemoryConfiguration, RuntimeOptions, WgpuRuntime, WgpuSetup, init_device};
use eframe::egui;
use egui::{Color32, Rect};
use glam::{Quat, Vec3};
use splatfield::{camera, ply, render, sog, texture};

const UV_RECT: Rect = Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0));

struct App {
    backbuffer: Rc<RefCell<texture::GpuTexture>>,
    controller: camera::Controller,
    client: ComputeClient<WgpuRuntime>,
    splats: Arc<RwLock<Option<render::Splats>>>,
    reframe: Arc<AtomicBool>,
    #[cfg(target_arch = "wasm32")]
    rendering: Rc<Cell<bool>>,
}

fn device_descriptor(adapter: &eframe::wgpu::Adapter) -> eframe::wgpu::DeviceDescriptor<'static> {
    eframe::wgpu::DeviceDescriptor {
        required_features: adapter
            .features()
            .difference(eframe::wgpu::Features::MAPPABLE_PRIMARY_BUFFERS)
            .difference(eframe::wgpu::Features::all_experimental_mask()),
        required_limits: adapter.limits(),
        memory_hints: eframe::wgpu::MemoryHints::MemoryUsage,
        ..Default::default()
    }
}

fn wgpu_config() -> eframe::egui_wgpu::WgpuConfiguration {
    eframe::egui_wgpu::WgpuConfiguration {
        wgpu_setup: eframe::egui_wgpu::WgpuSetup::CreateNew(
            eframe::egui_wgpu::WgpuSetupCreateNew {
                device_descriptor: std::sync::Arc::new(device_descriptor),
                ..eframe::egui_wgpu::WgpuSetupCreateNew::without_display_handle()
            },
        ),
        ..Default::default()
    }
}

#[derive(Clone, Copy)]
enum SplatFormat {
    Ply,
    Sog,
}

fn splat_format(file: &egui::DroppedFile) -> Option<SplatFormat> {
    let ext = file
        .path
        .as_ref()
        .and_then(|p| p.extension()?.to_str())
        .or_else(|| file.name.rsplit_once('.').map(|(_, ext)| ext))
        .map(str::to_ascii_lowercase);
    match ext.as_deref() {
        Some("ply") => Some(SplatFormat::Ply),
        Some("sog") => Some(SplatFormat::Sog),
        _ => None,
    }
}

impl App {
    fn new(cc: &eframe::CreationContext) -> Self {
        let render_state = cc.wgpu_render_state.as_ref().expect("Must use wgpu");
        let device = init_device(
            WgpuSetup {
                instance: wgpu::Instance::new(
                    wgpu::InstanceDescriptor::new_without_display_handle(),
                ),
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

        Self {
            backbuffer: Rc::new(RefCell::new(texture::GpuTexture::new(
                render_state.renderer.clone(),
                render_state.device.clone(),
                render_state.queue.clone(),
            ))),
            controller: camera::Controller::new(-Vec3::Z * 2.5, Quat::IDENTITY),
            client: WgpuRuntime::client(&device),
            splats: Arc::new(RwLock::new(None)),
            reframe: Arc::new(AtomicBool::new(false)),
            #[cfg(target_arch = "wasm32")]
            rendering: Rc::new(Cell::new(false)),
        }
    }

    fn load_dropped(&self, file: egui::DroppedFile, ctx: egui::Context) {
        let Some(format) = splat_format(&file) else {
            return;
        };

        let client = self.client.clone();
        let splats = Arc::clone(&self.splats);
        let reframe = Arc::clone(&self.reframe);

        let load = move |reader| -> anyhow::Result<render::Splats> {
            match format {
                SplatFormat::Sog => sog::load_sog(reader, &client),
                SplatFormat::Ply => ply::load_ply(reader, &client),
            }
        };

        let on_loaded = move |result: anyhow::Result<render::Splats>| match result {
            Ok(data) => {
                *splats.write().unwrap() = Some(data);
                reframe.store(true, Ordering::Release);
                ctx.request_repaint();
            }
            Err(e) => log::error!("Failed to load splat: {e:?}"),
        };

        #[cfg(not(target_arch = "wasm32"))]
        {
            let Some(path) = file.path else { return };
            std::thread::spawn(move || {
                on_loaded(
                    std::fs::File::open(&path)
                        .with_context(|| format!("Failed to open {path:?}"))
                        .and_then(|f| load(std::io::BufReader::new(f))),
                );
            });
        }

        #[cfg(target_arch = "wasm32")]
        {
            let Some(bytes) = file.bytes else { return };
            wasm_bindgen_futures::spawn_local(async move {
                on_loaded(load(std::io::Cursor::new(bytes)));
            });
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _: &mut eframe::Frame) {
        if let Some(file) = ui
            .input(|i| i.raw.dropped_files.clone())
            .into_iter()
            .find(|f| splat_format(f).is_some())
        {
            self.load_dropped(file, ui.ctx().clone());
        }

        let binding = self.splats.read().unwrap();
        let Some(splats) = binding.as_ref() else {
            ui.centered_and_justified(|ui| ui.heading("Drag and drop a .ply or .sog file"));
            return;
        };

        if self.reframe.swap(false, Ordering::AcqRel) {
            self.controller.frame_bounds(splats.bounds);
        }

        let size = ui.available_size();
        let (rect, response) = ui.allocate_exact_size(size, egui::Sense::drag());
        let pixel = (glam::vec2(size.x, size.y) * ui.pixels_per_point()).as_uvec2();
        self.controller.tick(&response, ui);
        self.controller.camera.fit_fov(pixel);

        #[cfg(not(target_arch = "wasm32"))]
        if pixel.x > 8 && pixel.y > 8 {
            let img = pollster::block_on(splats.render(&self.controller.camera, pixel));
            self.backbuffer.borrow_mut().update_texture(&img, pixel);
        }

        #[cfg(target_arch = "wasm32")]
        if pixel.x > 8 && pixel.y > 8 && !self.rendering.get() {
            self.rendering.set(true);
            let splats = splats.clone();
            let camera = self.controller.camera.clone();
            let backbuffer = self.backbuffer.clone();
            let rendering = self.rendering.clone();
            let ctx = ui.ctx().clone();

            wasm_bindgen_futures::spawn_local(async move {
                let img = splats.render(&camera, pixel).await;
                backbuffer.borrow_mut().update_texture(&img, pixel);
                rendering.set(false);
                ctx.request_repaint();
            });
        }

        if let Some(id) = self.backbuffer.borrow().texture_id() {
            ui.painter().image(id, rect, UV_RECT, Color32::WHITE);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> anyhow::Result<()> {
    env_logger::init();

    eframe::run_native(
        "SplatField",
        eframe::NativeOptions {
            wgpu_options: wgpu_config(),
            ..Default::default()
        },
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
    .map_err(|e| anyhow::anyhow!("Eframe error: {e}"))
}

#[cfg(target_arch = "wasm32")]
fn main() {
    console_error_panic_hook::set_once();

    wasm_bindgen_futures::spawn_local(async {
        let canvas = web_sys::window()
            .unwrap()
            .document()
            .unwrap()
            .get_element_by_id("the_canvas_id")
            .unwrap()
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .unwrap();

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

        if let Some(el) = web_sys::window()
            .unwrap()
            .document()
            .unwrap()
            .get_element_by_id("loading_text")
        {
            el.remove();
        }
    });
}
