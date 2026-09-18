use std::sync::Arc;

use splat_sort::tensor::GpuTensor;
use eframe::egui::{TextureId, epaint::mutex::RwLock};
use eframe::egui_wgpu::Renderer;
use eframe::wgpu;

pub struct GpuTexture {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: Arc<RwLock<Renderer>>,
    texture: wgpu::Texture,
    id: TextureId,
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
            texture,
            id,
        }
    }

    pub fn texture_id(&self) -> TextureId {
        self.id
    }

    pub fn update_texture(&mut self, img: &GpuTensor, size: glam::UVec2) {
        img.client.flush().expect("flush bitmap before copy");

        if self.texture.width() != size.x || self.texture.height() != size.y {
            self.recreate_texture(size);
        }

        self.copy_to_texture(img);
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
        self.texture = Self::create_texture(&self.device, size);
        let view = self.texture.create_view(&Default::default());
        self.renderer.write().update_egui_texture_from_wgpu_texture(
            &self.device,
            &view,
            wgpu::FilterMode::Linear,
            self.id,
        );
    }

    fn copy_to_texture(&self, img: &GpuTensor) {
        let resource = img
            .client
            .get_resource(img.handle.clone())
            .expect("bitmap buffer after flush");

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
            self.texture.as_image_copy(),
            self.texture.size(),
        );

        self.queue.submit([encoder.finish()]);
    }
}
