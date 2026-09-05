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

/// GPU frame state shared with the render task: the presentation texture and
/// the per-frame scratch (built on first use, reused every frame after).
struct FrameGpu {
    backbuffer: texture::GpuTexture,
    scratch: Option<render::RenderScratch>,
}

struct App {
    gpu: Rc<RefCell<FrameGpu>>,
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
    match file
        .path()
        .extension()?
        .to_str()?
        .to_ascii_lowercase()
        .as_str()
    {
        "ply" => Some(SplatFormat::Ply),
        "sog" => Some(SplatFormat::Sog),
        _ => None,
    }
}

/// `eprintln!` is a no-op on wasm32, so errors surface through the JS console.
#[cfg(target_arch = "wasm32")]
fn report(msg: String) {
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    unsafe extern "C" {
        #[wasm_bindgen(js_namespace = console)]
        fn error(s: &str);
    }
    error(&msg);
}

#[cfg(not(target_arch = "wasm32"))]
fn report(msg: String) {
    eprintln!("{msg}");
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
            gpu: Rc::new(RefCell::new(FrameGpu {
                backbuffer: texture::GpuTexture::new(
                    render_state.renderer.clone(),
                    render_state.device.clone(),
                    render_state.queue.clone(),
                ),
                scratch: None,
            })),
            controller: camera::Controller::default(),
            client: WgpuRuntime::client(&device),
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
                slot.splats = Some(Arc::new(data));
                slot.reframe = true;
                drop(slot);
                ctx.request_repaint();
            }
            Err(e) => report(format!("Failed to load splat: {e:?}")),
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

/// Render `splats` into the shared scratch and present the bitmap to the
/// backbuffer. The `FrameGpu` borrow lives across the await — wasm's
/// `rendering` flag guards re-entering while one render is in flight, and
/// native `block_on` drives the future on the same thread.
#[allow(clippy::await_holding_refcell_ref)] // render_with borrows the scratch for its duration
async fn render_frame(
    client: &ComputeClient<WgpuRuntime>,
    gpu: &Rc<RefCell<FrameGpu>>,
    splats: &render::Splats,
    camera: &camera::Camera,
    pixel: glam::UVec2,
) {
    let mut gpu = gpu.borrow_mut();
    let img = splats
        .render_with(
            gpu.scratch.get_or_insert_with(|| {
                render::RenderScratch::new(client, splats.attributes.shape[0], pixel)
            }),
            camera,
            pixel,
        )
        .await;
    gpu.backbuffer.update_texture(&img, pixel);
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

        // Below ~8px the aspect is 0/0: fit_fov would poison the camera with
        // NaN for every later frame, so skip input + render entirely.
        if pixel.x > 8 && pixel.y > 8 {
            self.controller.tick(&response, ui);
            self.controller.camera.fit_fov(pixel);

            #[cfg(not(target_arch = "wasm32"))]
            pollster::block_on(render_frame(
                &self.client,
                &self.gpu,
                &splats,
                &self.controller.camera,
                pixel,
            ));

            #[cfg(target_arch = "wasm32")]
            if !self.rendering.get() {
                self.rendering.set(true);
                let camera = self.controller.camera.clone();
                let gpu = self.gpu.clone();
                let rendering = self.rendering.clone();
                let client = self.client.clone();
                let ctx = ui.ctx().clone();

                wasm_bindgen_futures::spawn_local(async move {
                    render_frame(&client, &gpu, &splats, &camera, pixel).await;
                    rendering.set(false);
                    ctx.request_repaint();
                });
            }
        }

        let id = self.gpu.borrow().backbuffer.texture_id();
        ui.painter().image(id, rect, UV_RECT, Color32::WHITE);
    }
}

#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(target_arch = "wasm32")]
fn main() {
    wasm_bindgen_futures::spawn_local(async {
        let canvas = web_sys::window()
            .expect("no window")
            .document()
            .expect("no document")
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
