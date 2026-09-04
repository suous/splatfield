use std::sync::Arc;

use crate::tensor::GpuTensor;
// Use eframe's re-export so wgpu types always match the render state's device/queue.
use eframe::egui::{TextureId, epaint::mutex::RwLock};
use eframe::egui_wgpu::Renderer;
use eframe::wgpu;

pub struct GpuTexture {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: Arc<RwLock<Renderer>>,
    texture: (wgpu::Texture, TextureId),
}

impl GpuTexture {
    /// Registers a zero-initialized 1×1 texture: wgpu zeroes new buffers, so
    /// the placeholder paints fully transparent until the first render.
    pub fn new(renderer: Arc<RwLock<Renderer>>, device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let texture = Self::create_texture(&device, glam::UVec2::ONE);
        let view = texture.create_view(&Default::default());
        let id = renderer
            .write()
            .register_native_texture(&device, &view, wgpu::FilterMode::Linear);
        Self {
            device,
            queue,
            renderer,
            texture: (texture, id),
        }
    }

    pub fn texture_id(&self) -> TextureId {
        self.texture.1
    }

    pub fn update_texture(&mut self, img: &GpuTensor, size: glam::UVec2) {
        let _ = img.client.flush();

        if self.texture.0.width() != size.x || self.texture.0.height() != size.y {
            self.recreate_texture(size);
        }

        let (texture, _) = &self.texture;
        self.copy_to_texture(img, texture);
    }

    fn create_texture(device: &wgpu::Device, size: glam::UVec2) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    fn recreate_texture(&mut self, size: glam::UVec2) {
        // Reuse the registered TextureId so egui's texture set doesn't grow.
        let texture = Self::create_texture(&self.device, size);
        let view = texture.create_view(&Default::default());
        let (_, id) = self.texture;
        self.renderer.write().update_egui_texture_from_wgpu_texture(
            &self.device,
            &view,
            wgpu::FilterMode::Linear,
            id,
        );
        self.texture = (texture, id);
    }

    fn copy_to_texture(&self, img: &GpuTensor, texture: &wgpu::Texture) {
        let resource = img.client.get_resource(img.handle.clone()).unwrap();

        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &resource.resource().buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: resource.resource().offset,
                    bytes_per_row: Some(img.shape[1] as u32 * 4),
                    rows_per_image: None,
                },
            },
            texture.as_image_copy(),
            texture.size(),
        );

        self.queue.submit([encoder.finish()]);
    }
}
