use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

#[cfg(target_arch = "wasm32")]
use web_sys::wasm_bindgen::JsCast;

#[cfg(not(target_arch = "wasm32"))]
use anyhow::Context;

use cubecl::wgpu::{MemoryConfiguration, RuntimeOptions, WgpuRuntime, WgpuSetup, init_device};
use cubecl::{Runtime, client::ComputeClient};
use eframe::egui::{self, Color32, Rect, TextureId};
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

/// The render task moves the frame state out for the duration of a frame, so
/// no borrow is held across its await; `None` means a render is in flight.
type FrameSlot = Rc<RefCell<Option<FrameGpu>>>;

struct App {
    gpu: FrameSlot,
    // Stable for the texture's lifetime — recreate_texture reuses the id — so
    // the paint path never borrows `gpu`.
    tex_id: TextureId,
    controller: camera::Controller,
    client: ComputeClient<WgpuRuntime>,
    splats: Arc<Mutex<Loaded>>,
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
    let ext = path.extension()?.to_str()?;
    if ext.eq_ignore_ascii_case("ply") {
        Some(SplatFormat::Ply)
    } else if ext.eq_ignore_ascii_case("sog") {
        Some(SplatFormat::Sog)
    } else {
        None
    }
}

/// `eprintln!` is a no-op on wasm32, so errors surface through the JS console.
#[cfg(target_arch = "wasm32")]
fn report(msg: String) {
    web_sys::console::error_1(&msg.as_str().into());
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

        let backbuffer = texture::GpuTexture::new(
            render_state.renderer.clone(),
            render_state.device.clone(),
            render_state.queue.clone(),
        );
        let tex_id = backbuffer.texture_id();
        Self {
            gpu: Rc::new(RefCell::new(Some(FrameGpu {
                backbuffer,
                scratch: None,
            }))),
            tex_id,
            controller: camera::Controller::default(),
            client: WgpuRuntime::client(&device),
            splats: Arc::new(Mutex::new(Loaded::default())),
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
            Err(e) => report(format!("Failed to load splat: {e:#}")),
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
/// backbuffer. The frame state is moved out for the duration so no borrow is
/// held across the await; native `block_on` and wasm's single in-flight task
/// both drive it on one thread.
async fn render_frame(
    client: &ComputeClient<WgpuRuntime>,
    slot: &FrameSlot,
    splats: &render::Splats,
    camera: &camera::Camera,
    pixel: glam::UVec2,
) {
    let Some(mut frame) = slot.borrow_mut().take() else {
        return;
    };
    let img = splats
        .render_with(
            frame.scratch.get_or_insert_with(|| {
                render::RenderScratch::new(client, splats.attributes.shape[0], pixel)
            }),
            camera,
            pixel,
        )
        .await;
    frame.backbuffer.update_texture(&img, pixel);
    *slot.borrow_mut() = Some(frame);
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
        drop(slot);

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
            let idle = self.gpu.borrow().is_some();
            #[cfg(target_arch = "wasm32")]
            if idle {
                let camera = self.controller.camera.clone();
                let gpu = self.gpu.clone();
                let client = self.client.clone();
                let ctx = ui.ctx().clone();

                wasm_bindgen_futures::spawn_local(async move {
                    render_frame(&client, &gpu, &splats, &camera, pixel).await;
                    ctx.request_repaint();
                });
            }
        }

        ui.painter()
            .image(self.tex_id, rect, UV_RECT, Color32::WHITE);
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
