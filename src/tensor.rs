use cubecl::ir::{ElemType, FloatKind, UIntKind};
use cubecl::prelude::*;
use cubecl::server::{CubeCountSelection, Handle};
use cubecl::wgpu::WgpuRuntime;
use cubecl::zspace::Shape;

pub const F32: StorageType = StorageType::Scalar(ElemType::Float(FloatKind::F32));
pub const U32: StorageType = StorageType::Scalar(ElemType::UInt(UIntKind::U32));

pub fn cube_count_1d(client: &ComputeClient<WgpuRuntime>, length: u32, groups: u32) -> CubeCount {
    CubeCountSelection::new(client, length.div_ceil(groups)).cube_count()
}

#[derive(Clone)]
pub struct GpuTensor {
    pub client: ComputeClient<WgpuRuntime>,
    pub handle: Handle,
    pub shape: Shape,
    pub dtype: StorageType,
}

impl core::fmt::Debug for GpuTensor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GpuTensor")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .finish()
    }
}

impl GpuTensor {
    pub fn new(
        client: ComputeClient<WgpuRuntime>,
        shape: impl Into<Shape>,
        handle: Handle,
        dtype: StorageType,
    ) -> Self {
        Self {
            client,
            handle,
            shape: shape.into(),
            dtype,
        }
    }

    pub fn empty(
        client: &ComputeClient<WgpuRuntime>,
        shape: impl Into<Shape>,
        dtype: StorageType,
    ) -> GpuTensor {
        let shape = shape.into();
        let buffer = client.empty(shape.iter().product::<usize>() * dtype.size());
        Self::new(client.clone(), shape, buffer, dtype)
    }

    pub fn from<T: bytemuck::Pod>(
        client: &ComputeClient<WgpuRuntime>,
        shape: impl Into<Shape>,
        dtype: StorageType,
        data: &[T],
    ) -> Self {
        let buffer = client.create_from_slice(bytemuck::cast_slice(data));
        Self::new(client.clone(), shape, buffer, dtype)
    }

    pub async fn read_pair(&self) -> [u32; 2] {
        let bytes = self.client.read_async(vec![self.handle.clone()]).await;
        bytemuck::cast_slice(&bytes.unwrap()[0]).try_into().unwrap()
    }

    pub fn as_array_arg(&self) -> ArrayArg<WgpuRuntime> {
        // SAFETY: handle originates from a valid GPU allocation with matching dtype and shape.
        unsafe { ArrayArg::from_raw_parts(self.handle.clone(), self.shape.iter().product()) }
    }
}
