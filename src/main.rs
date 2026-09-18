use std::sync::{Arc, Mutex};

use anyhow::Context;

use cubecl::wgpu::{RuntimeOptions, WgpuRuntime, WgpuSetup, init_device};
use cubecl::{Runtime, client::ComputeClient};
use eframe::egui::{self, Color32, Rect};
use eframe::wgpu;
use splatfield::{camera, ply, render, sog, texture};

const UV_RECT: Rect = Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0));

/// The loaded model plus its "reframe the camera" flag under one lock: the
/// loader callback and the UI thread coordinate through this alone.
#[derive(Default)]
struct Loaded {
    splats: Option<Arc<render::Splats>>,
    reframe: bool,
}

/// GPU frame state: the presentation texture and the per-frame scratch
/// (built on first use, reused every frame after).
struct FrameGpu {
    backbuffer: texture::GpuTexture,
    scratch: Option<render::RenderScratch>,
}

struct App {
    gpu: FrameGpu,
    controller: camera::Controller,
    client: ComputeClient<WgpuRuntime>,
    splats: Arc<Mutex<Loaded>>,
    // What the backbuffer currently shows: the frame size and the model it
    // was rendered from (the camera is implied — it only moves through
    // Controller::tick, which reports moves).
    rendered: Option<(glam::UVec2, Arc<render::Splats>)>,
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

enum SplatFormat {
    Ply,
    Sog,
}

fn splat_format(path: &std::path::Path) -> Option<SplatFormat> {
    match path.extension()?.to_str()?.to_ascii_lowercase().as_str() {
        "ply" => Some(SplatFormat::Ply),
        "sog" => Some(SplatFormat::Sog),
        _ => None,
    }
}

impl App {
    fn new(cc: &eframe::CreationContext) -> Self {
        let render_state = cc.wgpu_render_state.as_ref().expect("Must use wgpu");
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
                ..Default::default()
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
            rendered: None,
        }
    }

    fn load_dropped(&self, file: egui::DroppedFileHandle, format: SplatFormat, ctx: egui::Context) {
        let client = self.client.clone();
        let splats = Arc::clone(&self.splats);

        let load = move |reader| -> anyhow::Result<render::Splats> {
            let cpu = match format {
                SplatFormat::Sog => sog::parse_sog(reader)?,
                SplatFormat::Ply => ply::parse_ply(reader)?,
            };
            Ok(cpu.upload(&client))
        };

        let on_loaded = move |result: anyhow::Result<render::Splats>| match result {
            Ok(data) => {
                let mut slot = splats.lock().unwrap();
                slot.splats = Some(Arc::new(data));
                slot.reframe = true;
                drop(slot);
                ctx.request_repaint();
            }
            Err(e) => eprintln!("Failed to load splat: {e:#}"),
        };

        let path = file.path().to_owned();
        std::thread::spawn(move || {
            on_loaded(
                std::fs::File::open(&path)
                    .with_context(|| format!("Failed to open {path:?}"))
                    .and_then(|f| load(std::io::BufReader::new(f))),
            );
        });
    }
}

/// Render `splats` into the frame's scratch and present the bitmap to the
/// backbuffer; the pipeline syncs on the counters readback, on the UI thread.
fn render_frame(
    client: &ComputeClient<WgpuRuntime>,
    frame: &mut FrameGpu,
    splats: &render::Splats,
    camera: &camera::Camera,
    pixel: glam::UVec2,
) {
    let img = splats.render_with(
        frame.scratch.get_or_insert_with(|| {
            render::RenderScratch::new(client, splats.attributes.shape[0], pixel)
        }),
        camera,
        pixel,
    );
    frame.backbuffer.update_texture(&img, pixel);
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _: &mut eframe::Frame) {
        let dropped = ui.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .find_map(|f| splat_format(f.path()).map(|format| (f.clone(), format)))
        });
        if let Some((file, format)) = dropped {
            self.load_dropped(file, format, ui.ctx().clone());
        }

        let mut slot = self.splats.lock().unwrap();
        if slot.reframe
            && let Some(s) = &slot.splats
        {
            self.controller.frame_bounds(s.bounds);
            slot.reframe = false;
        }
        let Some(splats) = slot.splats.clone() else {
            ui.centered_and_justified(|ui| ui.heading("Drag and drop a .ply or .sog file"));
            return;
        };
        drop(slot);

        let size = ui.available_size();
        let (rect, response) = ui.allocate_exact_size(size, egui::Sense::drag());
        let pixel = (glam::vec2(size.x, size.y) * ui.pixels_per_point()).as_uvec2();

        // Below ~8px the aspect is 0/0: fit_fov would poison the camera with
        // NaN for every later frame, so skip input + render entirely.
        if pixel.x > 8 && pixel.y > 8 {
            let moved = self.controller.tick(&response, ui);
            self.controller.camera.fit_fov(pixel);

            // A camera-neutral event (e.g. a bare click) would re-run the
            // whole ~30-launch pipeline only to repaint an identical bitmap —
            // skip it. Any load brings a fresh Arc, so pointer inequality
            // covers reframes too.
            let stale = moved
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
                );
                self.rendered = Some((pixel, Arc::clone(&splats)));
            }
        }

        ui.painter().image(
            self.gpu.backbuffer.texture_id(),
            rect,
            UV_RECT,
            Color32::WHITE,
        );
    }
}

fn main() -> eframe::Result<()> {
    eframe::run_native(
        "SplatField",
        eframe::NativeOptions {
            wgpu_options: wgpu_config(),
            ..Default::default()
        },
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
}
