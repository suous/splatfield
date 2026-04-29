use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, RwLock};

#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(target_arch = "wasm32")]
use std::cell::Cell;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

use cubecl::wgpu::{MemoryConfiguration, RuntimeOptions, WgpuDevice, WgpuSetup, init_device};
use eframe::egui;
use egui::{Color32, Rect};
use glam::{Quat, Vec3};
use log::error;
use splatfield::{camera, file, render, texture};

const UV_RECT: Rect = Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0));

struct App {
    backbuffer: Rc<RefCell<texture::GpuTexture>>,
    controller: camera::Controller,
    device: WgpuDevice,
    splats: Arc<RwLock<Option<render::Splats>>>,
    #[cfg(not(target_arch = "wasm32"))]
    reframe: Arc<AtomicBool>,
    #[cfg(target_arch = "wasm32")]
    rendering: Rc<Cell<bool>>,
}

fn device_descriptor(adapter: &eframe::wgpu::Adapter) -> eframe::wgpu::DeviceDescriptor<'static> {
    eframe::wgpu::DeviceDescriptor {
        required_features: adapter
            .features()
            .difference(eframe::wgpu::Features::MAPPABLE_PRIMARY_BUFFERS),
        required_limits: adapter.limits(),
        memory_hints: eframe::wgpu::MemoryHints::MemoryUsage,
        experimental_features: unsafe { eframe::wgpu::ExperimentalFeatures::enabled() },
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

impl App {
    fn new(cc: &eframe::CreationContext) -> Self {
        let render_state = cc.wgpu_render_state.as_ref().expect("Must use wgpu");
        cc.egui_ctx.set_visuals(egui::Visuals::dark());

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
            device,
            splats: Arc::new(RwLock::new(None)),
            #[cfg(not(target_arch = "wasm32"))]
            reframe: Arc::new(AtomicBool::new(false)),
            #[cfg(target_arch = "wasm32")]
            rendering: Rc::new(Cell::new(false)),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn load_first_ply(&self, file: egui::DroppedFile, ctx: egui::Context) {
        if let Some(path) = file.path {
            let device = self.device.clone();
            let splats = Arc::clone(&self.splats);
            let reframe = Arc::clone(&self.reframe);

            std::thread::spawn(move || {
                let Ok(reader) = std::fs::File::open(&path) else {
                    error!("Failed to open file: {path:?}");
                    return;
                };
                match file::load_ply(reader, &device) {
                    Ok(data) => {
                        *splats.write().unwrap() = Some(data);
                        reframe.store(true, Ordering::Release);
                        ctx.request_repaint();
                    }
                    Err(e) => error!("Failed to load splat: {e:?}"),
                }
            });
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn load_first_ply(&mut self, file: egui::DroppedFile, _ctx: egui::Context) {
        if let Some(bytes) = file.bytes {
            match file::load_ply(std::io::Cursor::new(bytes), &self.device) {
                Ok(data) => {
                    self.controller.frame_bounds(data.bounds);
                    *self.splats.write().unwrap() = Some(data);
                }
                Err(e) => error!("Failed to load splat: {e:?}"),
            }
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _: &mut eframe::Frame) {
        if let Some(file) = ui
            .input(|i| i.raw.dropped_files.clone())
            .into_iter()
            .find(|f| {
                f.name.ends_with(".ply")
                    || f.path
                        .as_ref()
                        .is_some_and(|p| p.extension().is_some_and(|ext| ext == "ply"))
            })
        {
            self.load_first_ply(file, ui.ctx().clone());
        }

        let binding = self.splats.read().unwrap();
        let Some(splats) = binding.as_ref() else {
            ui.centered_and_justified(|ui| ui.heading("Drag and drop a .ply file"));
            return;
        };

        #[cfg(not(target_arch = "wasm32"))]
        if self.reframe.swap(false, Ordering::AcqRel) {
            self.controller.frame_bounds(splats.bounds);
        }

        let size = ui.available_size();
        let (rect, response) = ui.allocate_exact_size(size, egui::Sense::drag());
        let pixel = (glam::vec2(size.x, size.y) * ui.ctx().pixels_per_point()).as_uvec2();
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
            ui.painter().rect_filled(rect, 0.0, Color32::BLACK);
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
    });
}
