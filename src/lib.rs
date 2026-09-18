pub mod camera;
mod layout;
pub mod ply;
mod project;
mod raster;
pub mod render;
pub mod sog;
pub mod texture;

/// GPU test support for the root crate: the sort crate's `test_client` is
/// `#[cfg(test)]` there and invisible across crates, so root GPU tests keep
/// their own copy of the pattern.
#[cfg(test)]
pub(crate) mod gpu_testing {
    use cubecl::prelude::*;
    use cubecl::wgpu::WgpuRuntime;

    /// Serializes GPU-touching tests: concurrent clients share one physical GPU,
    /// and the memory pressure makes pool-reclaim–sensitive assertions (memory
    /// accounting) flaky. CPU-only tests don't take this lock.
    static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Lock the GPU and hand back a client on the shared test device. Hold the
    /// guard for the whole test body.
    pub(crate) fn test_client() -> (
        std::sync::MutexGuard<'static, ()>,
        ComputeClient<WgpuRuntime>,
    ) {
        (
            GPU_TEST_LOCK.lock().unwrap(),
            WgpuRuntime::client(&cubecl::wgpu::WgpuDevice::default()),
        )
    }
}
