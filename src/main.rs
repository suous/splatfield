use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

#[cfg(target_arch = "wasm32")]
use std::cell::Cell;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

#[cfg(not(target_arch = "wasm32"))]
use anyhow::Context;

use cubecl::wgpu::{MemoryConfiguration, RuntimeOptions, WgpuRuntime, WgpuSetup, init_device};
use cubecl::{Runtime, client::ComputeClient};
use eframe::egui;
use eframe::wgpu;
use egui::{Color32, Rect};
use splatfield::{camera, ply, render, sog, texture};

const UV_RECT: Rect = Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0));

/// The loaded model plus its "reframe the camera" flag under one lock: the
/// loader callback and the UI thread coordinate through this alone.
#[derive(Default)]
struct Loaded {
    splats: Option<render::Splats>,
    reframe: bool,
}

struct App {
    backbuffer: Rc<RefCell<texture::GpuTexture>>,
    scratch: Rc<RefCell<Option<render::RenderScratch>>>,
    controller: camera::Controller,
    client: ComputeClient<WgpuRuntime>,
    splats: Arc<Mutex<Loaded>>,
    #[cfg(target_arch = "wasm32")]
    rendering: Rc<Cell<bool>>,
}

fn device_descriptor(adapter: &wgpu::Adapter) -> wgpu::DeviceDescriptor<'static> {
    wgpu::DeviceDescriptor {
        required_features: adapter.features().difference(
            wgpu::Features::MAPPABLE_PRIMARY_BUFFERS | wgpu::Features::all_experimental_mask(),
        ),
        required_limits: adapter.limits(),
        memory_hints: wgpu::MemoryHints::MemoryUsage,
        ..Default::default()
    }
}

fn wgpu_config() -> eframe::egui_wgpu::WgpuConfiguration {
    eframe::egui_wgpu::WgpuConfiguration {
        wgpu_setup: eframe::egui_wgpu::WgpuSetup::CreateNew(
            eframe::egui_wgpu::WgpuSetupCreateNew {
                device_descriptor: Arc::new(device_descriptor),
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

fn splat_format(file: &(impl egui::DroppedFile + ?Sized)) -> Option<SplatFormat> {
    let ext = file
        .path()
        .extension()
        .and_then(|ext| ext.to_str())
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
            controller: camera::Controller::default(),
            client: WgpuRuntime::client(&device),
            scratch: Rc::new(RefCell::new(None)),
            splats: Arc::new(Mutex::new(Loaded::default())),
            #[cfg(target_arch = "wasm32")]
            rendering: Rc::new(Cell::new(false)),
        }
    }

    fn load_dropped(&self, file: egui::DroppedFileHandle, format: SplatFormat, ctx: egui::Context) {
        let client = self.client.clone();
        let splats = Arc::clone(&self.splats);

        let load = move |reader| -> anyhow::Result<render::Splats> {
            match format {
                SplatFormat::Sog => Ok(sog::parse_sog(reader)?.upload(&client)),
                SplatFormat::Ply => Ok(ply::parse_ply(reader)?.upload(&client)),
            }
        };

        let on_loaded = move |result: anyhow::Result<render::Splats>| match result {
            Ok(data) => {
                let mut slot = splats.lock().unwrap();
                slot.splats = Some(data);
                slot.reframe = true;
                drop(slot);
                ctx.request_repaint();
            }
            Err(e) => log::error!("Failed to load splat: {e:?}"),
        };

        #[cfg(not(target_arch = "wasm32"))]
        {
            let path = file.path().to_owned();
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
            wasm_bindgen_futures::spawn_local(async move {
                match file.bytes_async().await {
                    Ok(bytes) => on_loaded(load(std::io::Cursor::new(bytes))),
                    Err(e) => on_loaded(Err(anyhow::anyhow!("Failed to read dropped file: {e}"))),
                }
            });
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _: &mut eframe::Frame) {
        let dropped = ui.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .find_map(|f| splat_format(f.as_ref()).map(|format| (f.clone(), format)))
        });
        if let Some((file, format)) = dropped {
            self.load_dropped(file, format, ui.ctx().clone());
        }

        let mut slot = self.splats.lock().unwrap();
        if slot.reframe {
            if let Some(s) = &slot.splats {
                self.controller.frame_bounds(s.bounds);
            }
            slot.reframe = false;
        }
        let Some(splats) = slot.splats.clone() else {
            ui.centered_and_justified(|ui| ui.heading("Drag and drop a .ply or .sog file"));
            return;
        };
        drop(slot); // release the lock before rendering the frame

        let size = ui.available_size();
        let (rect, response) = ui.allocate_exact_size(size, egui::Sense::drag());
        let pixel = (glam::vec2(size.x, size.y) * ui.pixels_per_point()).as_uvec2();
        self.controller.tick(&response, ui);
        self.controller.camera.fit_fov(pixel);

        if pixel.x > 8 && pixel.y > 8 {
            let total = splats.attributes.shape[0];

            #[cfg(not(target_arch = "wasm32"))]
            {
                let img = pollster::block_on(splats.render_with(
                    self.scratch.borrow_mut().get_or_insert_with(|| {
                        render::RenderScratch::new(&self.client, total, pixel)
                    }),
                    &self.controller.camera,
                    pixel,
                ));
                self.backbuffer.borrow_mut().update_texture(&img, pixel);
            }

            // `rendering` also guards the scratch borrow: an in-flight async
            // render on wasm holds it across the await.
            #[cfg(target_arch = "wasm32")]
            if !self.rendering.get() {
                self.rendering.set(true);
                let camera = self.controller.camera.clone();
                let scratch = self.scratch.clone();
                let backbuffer = self.backbuffer.clone();
                let rendering = self.rendering.clone();
                let client = self.client.clone();
                let ctx = ui.ctx().clone();

                wasm_bindgen_futures::spawn_local(async move {
                    let img = splats
                        .render_with(
                            scratch.borrow_mut().get_or_insert_with(|| {
                                render::RenderScratch::new(&client, total, pixel)
                            }),
                            &camera,
                            pixel,
                        )
                        .await;
                    backbuffer.borrow_mut().update_texture(&img, pixel);
                    rendering.set(false);
                    ctx.request_repaint();
                });
            }
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
