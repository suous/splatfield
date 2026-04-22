use cubecl::client::ComputeClient;
use cubecl::frontend::ArrayArg;
use cubecl::ir::{ElemType, FloatKind, StorageType, UIntKind};
use cubecl::server::{CubeCountSelection, Handle};
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use cubecl::zspace::Shape;
use cubecl::{CubeCount, Runtime};

pub const F32: StorageType = StorageType::Scalar(ElemType::Float(FloatKind::F32));
pub const U32: StorageType = StorageType::Scalar(ElemType::UInt(UIntKind::U32));

pub fn cube_count_1d(client: &ComputeClient<WgpuRuntime>, length: u32, groups: u32) -> CubeCount {
    CubeCountSelection::new(client, length.div_ceil(groups)).cube_count()
}

pub fn empty_tensor(shape: impl Into<Shape>, device: &WgpuDevice, dtype: StorageType) -> GpuTensor {
    let shape = shape.into();
    let client = WgpuRuntime::client(device);
    let buffer = client.empty(shape.iter().product::<usize>() * dtype.size());
    GpuTensor::new(client, device.clone(), shape, buffer, dtype)
}

pub fn create_tensor_from_data<T: bytemuck::Pod>(
    shape: impl Into<Shape>,
    device: &WgpuDevice,
    dtype: StorageType,
    data: &[T],
) -> GpuTensor {
    let client = WgpuRuntime::client(device);
    let buffer = client.create_from_slice(bytemuck::cast_slice(data));
    GpuTensor::new(client, device.clone(), shape, buffer, dtype)
}

#[derive(Clone)]
pub struct GpuTensor {
    pub client: ComputeClient<WgpuRuntime>,
    pub handle: Handle,
    pub shape: Shape,
    pub device: WgpuDevice,
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
        device: WgpuDevice,
        shape: impl Into<Shape>,
        handle: Handle,
        dtype: StorageType,
    ) -> Self {
        Self {
            client,
            handle,
            shape: shape.into(),
            device,
            dtype,
        }
    }

    pub fn read_u32_at(&self, index: usize) -> u32 {
        bytemuck::cast_slice(&self.client.read_one_unchecked(self.handle.clone()))[index]
    }

    pub fn as_array_arg(&self) -> ArrayArg<WgpuRuntime> {
        // SAFETY: handle originates from a valid GPU allocation with matching dtype and shape.
        unsafe { ArrayArg::from_raw_parts(self.handle.clone(), self.shape.iter().product()) }
    }
}
