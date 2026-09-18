use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::wgpu::WgpuRuntime;
use cubecl::zspace::Shape;

/// Serializes GPU-touching tests: concurrent clients share one physical GPU,
/// and the memory pressure makes pool-reclaim–sensitive assertions (memory
/// accounting) flaky. CPU-only tests don't take this lock.
#[cfg(test)]
pub(crate) static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Lock the GPU and hand back a client on the shared test device. Hold the
/// guard for the whole test body.
#[cfg(test)]
pub(crate) fn test_client() -> (
    std::sync::MutexGuard<'static, ()>,
    ComputeClient<WgpuRuntime>,
) {
    (
        GPU_TEST_LOCK.lock().unwrap(),
        WgpuRuntime::client(&cubecl::wgpu::WgpuDevice::default()),
    )
}

#[derive(Clone)]
pub struct GpuTensor {
    pub client: ComputeClient<WgpuRuntime>,
    pub handle: Handle,
    pub shape: Shape,
}

impl GpuTensor {
    pub fn empty(client: &ComputeClient<WgpuRuntime>, shape: impl Into<Shape>) -> GpuTensor {
        let shape = shape.into();
        // All buffers hold 4-byte elements (f32/u32), so f32 sizing covers both.
        let handle = client.empty(shape.iter().product::<usize>() * size_of::<f32>());
        Self {
            client: client.clone(),
            handle,
            shape,
        }
    }

    /// Upload to the GPU. An owned `Vec` is moved without copying; a slice is
    /// copied once.
    pub fn from<T: bytemuck::NoUninit + Send + Sync>(
        client: &ComputeClient<WgpuRuntime>,
        shape: impl Into<Shape>,
        data: impl Into<Vec<T>>,
    ) -> Self {
        let handle = client.create(cubecl::bytes::Bytes::from_elems(data.into()));
        Self {
            client: client.clone(),
            handle,
            shape: shape.into(),
        }
    }

    pub fn read_vec<T: bytemuck::Pod>(&self) -> Vec<T> {
        let bytes = self.client.read_one_unchecked(self.handle.clone());
        bytemuck::cast_slice(&bytes).to_vec()
    }

    /// Overwrite the buffer in place — stream-ordered, non-blocking, no kernel.
    pub fn write<T: bytemuck::NoUninit + Send + Sync>(&self, data: impl Into<Vec<T>>) {
        self.client
            .write(&self.handle, cubecl::bytes::Bytes::from_elems(data.into()));
    }

    pub fn as_buffer_arg(&self) -> BufferArg<WgpuRuntime> {
        // SAFETY: handle originates from a valid GPU allocation with matching dtype and shape.
        unsafe { BufferArg::from_raw_parts(self.handle.clone(), self.shape.iter().product()) }
    }
}
