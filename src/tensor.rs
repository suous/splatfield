use cubecl::prelude::*;
use cubecl::server::{CubeCountSelection, Handle};
use cubecl::wgpu::WgpuRuntime;
use cubecl::zspace::Shape;

const STRIDE: usize = 4;

pub fn cube_count_1d(client: &ComputeClient<WgpuRuntime>, length: u32, groups: u32) -> CubeCount {
    CubeCountSelection::new(client, length.div_ceil(groups)).cube_count()
}

#[derive(Clone)]
pub struct GpuTensor {
    pub client: ComputeClient<WgpuRuntime>,
    pub handle: Handle,
    pub shape: Shape,
}

impl core::fmt::Debug for GpuTensor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GpuTensor")
            .field("shape", &self.shape)
            .finish()
    }
}

impl GpuTensor {
    fn new(client: ComputeClient<WgpuRuntime>, shape: impl Into<Shape>, handle: Handle) -> Self {
        Self {
            client,
            handle,
            shape: shape.into(),
        }
    }

    pub fn empty(client: &ComputeClient<WgpuRuntime>, shape: impl Into<Shape>) -> GpuTensor {
        let shape = shape.into();
        let buffer = client.empty(shape.iter().product::<usize>() * STRIDE);
        Self::new(client.clone(), shape, buffer)
    }

    /// Upload to the GPU. An owned `Vec` is moved without copying; a slice is
    /// copied once — on wasm32 each large copy risks the 4 GiB linear-memory ceiling.
    pub fn from<T: bytemuck::NoUninit + Send + Sync>(
        client: &ComputeClient<WgpuRuntime>,
        shape: impl Into<Shape>,
        data: impl Into<Vec<T>>,
    ) -> Self {
        let buffer = client.create(cubecl::bytes::Bytes::from_elems(data.into()));
        Self::new(client.clone(), shape, buffer)
    }

    pub fn read_vec<T: bytemuck::Pod>(&self) -> Vec<T> {
        let bytes = self.client.read_one_unchecked(self.handle.clone());
        bytemuck::cast_slice(&bytes).to_vec()
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
