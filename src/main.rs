use cubecl::wgpu::{
    AutoGraphicsApi, GraphicsApi, MemoryConfiguration, RuntimeOptions, WgpuDevice, WgpuSetup,
    init_device,
};
use eframe::{NativeOptions, egui, egui_wgpu::WgpuSetupCreateNew};
use egui::{Color32, Rect};
use glam::{Quat, Vec3};
use log::error;
use splatfield::{camera, file, render, texture};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

const UV_RECT: Rect = Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0));

struct App {
    backbuffer: texture::GpuTexture,
    controller: camera::Controller,
    device: WgpuDevice,
    splats: Arc<RwLock<Option<render::Splats>>>,
    reframe: Arc<AtomicBool>,
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
                backend: AutoGraphicsApi::backend(),
            },
            RuntimeOptions {
                tasks_max: 64,
                memory_config: MemoryConfiguration::ExclusivePages,
            },
        );

        Self {
            backbuffer: texture::GpuTexture::new(
                render_state.renderer.clone(),
                render_state.device.clone(),
                render_state.queue.clone(),
            ),
            controller: camera::Controller::new(-Vec3::Z * 2.5, Quat::IDENTITY),
            device,
            splats: Arc::new(RwLock::new(None)),
            reframe: Arc::new(AtomicBool::new(false)),
        }
    }

    fn load_ply_file(&self, path: std::path::PathBuf, ctx: egui::Context) {
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

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _: &mut eframe::Frame) {
        egui::CentralPanel::default()
            .frame(egui::Frame::new().inner_margin(0.0))
            .show_inside(ui, |ui| {
                for file in ui.input(|i| i.raw.dropped_files.clone()) {
                    if let Some(path) = &file.path
                        && path.extension().is_some_and(|ext| ext == "ply")
                    {
                        self.load_ply_file(path.clone(), ui.ctx().clone());
                    }
                }

                let binding = self.splats.read().unwrap();
                let Some(splats) = binding.as_ref() else {
                    ui.centered_and_justified(|ui| ui.heading("Drag and drop a .ply file"));
                    return;
                };

                if self.reframe.swap(false, Ordering::AcqRel) {
                    self.controller.frame_bounds(splats.bounds);
                }

                let size = ui.available_size();
                let (rect, response) = ui.allocate_exact_size(size, egui::Sense::drag());
                let size = glam::vec2(size.x, size.y);
                self.controller.tick(&response, ui);
                self.controller.camera.fit_fov(size.round().as_uvec2());

                let pixel_size = (size * ui.ctx().pixels_per_point().round()).as_uvec2();
                if pixel_size.x > 8 && pixel_size.y > 8 {
                    let img = splats.render(&self.controller.camera, pixel_size);
                    if let Some(id) = self.backbuffer.update_texture(&img, pixel_size) {
                        ui.painter().rect_filled(rect, 0.0, Color32::BLACK);
                        ui.painter().image(id, rect, UV_RECT, Color32::WHITE);
                    }
                }
            });
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::builder()
        .target(env_logger::Target::Stdout)
        .init();

    let native_options = NativeOptions {
        viewport: egui::ViewportBuilder::default(),
        wgpu_options: eframe::egui_wgpu::WgpuConfiguration {
            wgpu_setup: eframe::egui_wgpu::WgpuSetup::CreateNew(WgpuSetupCreateNew {
                instance_descriptor: wgpu::InstanceDescriptor::new_without_display_handle(),
                display_handle: None,
                native_adapter_selector: None,
                power_preference: eframe::wgpu::PowerPreference::HighPerformance,
                device_descriptor: std::sync::Arc::new(|adapter: &eframe::wgpu::Adapter| {
                    eframe::wgpu::DeviceDescriptor {
                        label: Some("egui+cube"),
                        required_features: adapter
                            .features()
                            .difference(eframe::wgpu::Features::MAPPABLE_PRIMARY_BUFFERS),
                        required_limits: adapter.limits(),
                        memory_hints: eframe::wgpu::MemoryHints::MemoryUsage,
                        trace: eframe::wgpu::Trace::Off,
                        // SAFETY: wgpu requires ExperimentalFeatures::enabled() for experimental device features.
                        experimental_features: unsafe {
                            eframe::wgpu::ExperimentalFeatures::enabled()
                        },
                    }
                }),
            }),
            ..Default::default()
        },
        ..Default::default()
    };

    eframe::run_native(
        "SplatField",
        native_options,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
    .map_err(|e| anyhow::anyhow!("Eframe error: {e}"))
}
