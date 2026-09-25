#![deny(unreachable_pub)]

pub mod camera;
/// First-run model fetch (src/fetch.rs): the host half downloads the release
/// zip into the gsam cache; the wasm half streams it, verifies the pins, and
/// hands a [`gsam::ModelStore`] to the OPFS cache.
pub mod fetch;
pub mod layout;
/// OPFS cache of the verified release (src/opfs.rs): persist the files once
/// after a pinned download, then load straight from disk on later page
/// loads. Manifest + checks are pure and host-tested; the I/O half is
/// wasm-only and best-effort — any cache problem reads as a miss.
pub mod opfs;
/// Worker<->main-thread protocol: request/response enums + bincode wire
/// codec, host-tested (src/pipeline.rs).
pub mod pipeline;
mod ply;
mod project;
mod raster;
pub mod render;
pub mod seg;
pub mod sog;
pub mod texture;

// The SH DC coefficient the app's palette→DC conversion divides by.
// `to_dc` is that conversion; the kernel math in `project` uses the same
// constant.
pub use project::to_dc;

/// Route every thread's kernel launches through one ordered stream. cubecl's
/// default policy is per-thread, and per-thread streams don't order against
/// each other — a worker thread's writes (the segmentation tint, the Beta
/// state) would land unordered relative to the UI thread's renders, which
/// then read stale buffers. Idempotent; call once at startup.
pub fn use_single_stream() {
    cubecl_environment::stream::set_policy(cubecl_environment::stream::StreamPolicy::Single);
}

/// The `.sog` archive extension, without the dot: shared by the picker list
/// below and the loader's dispatch so the two can't drift apart.
pub(crate) const SOG_EXTENSION: &str = "sog";

/// Scene file extensions the GUI accepts — `load_scene` dispatches on them.
/// One list so a new format can't land in the parser while the GUI still
/// silently rejects it.
pub const SCENE_EXTENSIONS: [&str; 2] = ["ply", SOG_EXTENSION];

/// Parse a scene file by extension — `.sog` archive, anything else PLY.
/// The loader rule shared by the GUI and any future fetch loader; bytes
/// arrive from drag-and-drop, so the reader (not a path) is the input.
pub fn load_scene(
    ext: &std::ffi::OsStr,
    file: impl std::io::Read + std::io::Seek,
) -> anyhow::Result<render::CpuSplats> {
    if ext.eq_ignore_ascii_case(SOG_EXTENSION) {
        sog::parse_sog(file)
    } else {
        ply::parse_ply(std::io::BufReader::new(file))
    }
}

/// GPU test support: the sort crate's serialized test client, entered only
/// after setting the single-stream policy — the root render/seg tests'
/// cross-thread ordering assertions depend on it, which sort's own
/// single-threaded test callers don't.
#[cfg(test)]
pub(crate) mod gpu_testing {
    pub(crate) fn test_client() -> (std::sync::MutexGuard<'static, ()>, cubecl::client::Client) {
        crate::use_single_stream();
        splat_sort::tensor::test_client()
    }
}
