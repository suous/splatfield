use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::zspace::Shape;

/// Serializes GPU-touching tests: concurrent clients share one physical GPU,
/// and the memory pressure makes pool-reclaim–sensitive assertions (memory
/// accounting) flaky. CPU-only tests don't take this lock.
#[cfg(any(test, feature = "test-utils"))]
pub static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Lock the GPU and hand back a client on the shared test device. Hold the
/// guard for the whole test body.
#[cfg(any(test, feature = "test-utils"))]
pub fn test_client() -> (std::sync::MutexGuard<'static, ()>, Client) {
    (
        // Poison-tolerant: one failing GPU test must not cascade
        // PoisonErrors through every other test's client setup.
        GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()),
        cubecl::Device::default().client(),
    )
}

#[derive(Clone)]
pub struct GpuTensor {
    pub client: Client,
    pub handle: Handle,
    pub shape: Shape,
}

impl GpuTensor {
    pub fn empty(client: &Client, shape: impl Into<Shape>) -> GpuTensor {
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
        client: &Client,
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

    /// Blocking read: cubecl polls once and panics on wasm — test/host paths
    /// only; wasm callers need [`Self::read_vec_async`].
    pub fn read_vec<T: bytemuck::Pod>(&self) -> Vec<T> {
        let bytes = self.client.read_one_unchecked(self.handle.clone());
        bytemuck::cast_slice(&bytes).to_vec()
    }

    /// Overwrite the buffer in place — stream-ordered, non-blocking, no kernel.
    pub fn write<T: bytemuck::NoUninit + Send + Sync>(&self, data: impl Into<Vec<T>>) {
        self.client
            .write(&self.handle, cubecl::bytes::Bytes::from_elems(data.into()));
    }

    /// Read the whole buffer as `T`. Async on every target: cubecl's blocking
    /// reads poll once and panic on wasm, so native callers await this through
    /// `cubecl::future::block_on` instead.
    pub async fn read_vec_async<T: bytemuck::Pod>(&self) -> Vec<T> {
        let bytes = self.client.read_async(vec![self.handle.clone()]).await;
        bytemuck::cast_slice(&bytes.unwrap()[0]).to_vec()
    }

    pub fn as_buffer_arg(&self) -> BufferArg {
        // SAFETY: handle originates from a valid GPU allocation with matching dtype and shape.
        unsafe { BufferArg::from_raw_parts(self.handle.clone(), self.shape.iter().product()) }
    }
}
