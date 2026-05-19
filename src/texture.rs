use std::sync::Arc;

use crate::tensor::GpuTensor;
use eframe::egui_wgpu::Renderer;
use egui::{TextureId, epaint::mutex::RwLock};

pub struct GpuTexture {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: Arc<RwLock<Renderer>>,
    texture: Option<(wgpu::Texture, TextureId)>,
}

impl GpuTexture {
    pub fn new(renderer: Arc<RwLock<Renderer>>, device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self {
            device,
            queue,
            renderer,
            texture: None,
        }
    }

    pub fn texture_id(&self) -> Option<TextureId> {
        self.texture.as_ref().map(|(_, id)| *id)
    }

    pub fn update_texture(&mut self, img: &GpuTensor, size: glam::UVec2) {
        let _ = img.client.flush();

        let needs_resize = self
            .texture
            .as_ref()
            .is_none_or(|(t, _)| t.width() != size.x || t.height() != size.y);

        if needs_resize {
            self.texture = Some(self.ensure_texture(size));
        }

        let (texture, _) = self.texture.as_ref().unwrap();
        self.copy_to_texture(img, texture);
    }

    fn ensure_texture(&mut self, size: glam::UVec2) -> (wgpu::Texture, TextureId) {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
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
            view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
        });
        let view = texture.create_view(&Default::default());
        let id = match &mut self.texture {
            Some((_, id)) => {
                self.renderer.write().update_egui_texture_from_wgpu_texture(
                    &self.device,
                    &view,
                    wgpu::FilterMode::Linear,
                    *id,
                );
                *id
            }
            None => self.renderer.write().register_native_texture(
                &self.device,
                &view,
                wgpu::FilterMode::Linear,
            ),
        };
        (texture, id)
    }

    fn copy_to_texture(&self, img: &GpuTensor, texture: &wgpu::Texture) {
        let resource = img.client.get_resource(img.handle.clone()).unwrap();
        let (buffer, offset) = (&resource.resource().buffer, resource.resource().offset);

        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset,
                    bytes_per_row: Some(img.shape[1] as u32 * 4),
                    rows_per_image: None,
                },
            },
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: texture.width(),
                height: texture.height(),
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit([encoder.finish()]);
    }
}
