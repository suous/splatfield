use std::sync::OnceLock;

use anyhow::Result;
use ort::{
    environment::{Environment, GlobalThreadPoolOptions},
    ep::{CPU, ExecutionProvider},
    session::{Session, builder::GraphOptimizationLevel},
};

/// The one process-wide ONNX Runtime environment — ort allows exactly one
/// per process, so it lives in a OnceLock and carries the thread policy for
/// every session the oracle builds, in a single global pool instead of one
/// pool per session.
///
/// The pool leaves a couple of cores for the UI thread: the oracle runs on
/// the worker while the app must stay interactive. Inter-op parallelism
/// barely helps these sequential graphs (and parallel execution is off by
/// default), so only the intra-op pool needs a size.
///
/// The pool is also capped: at full width on many-core hybrid CPUs the
/// quantized encoder graphs access-violate inside ORT's threaded kernels
/// (reproduced on a 24-core Core Ultra 9 275HX: 16 threads pass, 22
/// segfault; 8 is as fast as 16 on that machine, so cap there).
fn env() -> Result<&'static Environment> {
    static ENV: OnceLock<Result<Environment, String>> = OnceLock::new();
    ENV.get_or_init(|| {
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        GlobalThreadPoolOptions::default()
            .with_intra_threads(threads.saturating_sub(2).clamp(1, 8))
            .and_then(|pool| ort::init().with_global_thread_pool(pool).build())
            .map_err(|e| e.to_string())
    })
    .as_ref()
    .map_err(|e| anyhow::anyhow!("ort environment init failed: {e}"))
}

/// Builder config steps return `Error<SessionBuilder>` — the builder rides
/// the error as a recovery payload (`e.recover()`), which makes the error
/// neither Send nor Sync, so it cannot cross `?` into anyhow. The Level1
/// pin below is load-bearing and must fail loud with the runtime's message
/// — never silently via `unwrap_or_else(|e| e.recover())` — so it funnels
/// through this message-carrying rewrap. (The thread policy needs no such
/// handling: it moved to the global pool in [`env`], whose setters return
/// plain `Result`.)
fn step(r: ort::session::builder::BuilderResult) -> Result<ort::session::builder::SessionBuilder> {
    r.map_err(|e| anyhow::anyhow!("session builder step failed: {}", e.message()))
}

/// The CPU arena is OFF: with it on, the arena grew ~1 GB per run on the
/// quantized graphs until jetsam killed the process on a 16 GB machine.
///
/// `level` caps graph optimization - the grounding-dino q4f16 export
/// crashes ORT at the default Level3 (a LayerNorm fusion trips over an
/// inserted precision cast) and must load at Level1.
pub(crate) fn build(path: &std::path::Path, level: GraphOptimizationLevel) -> Result<Session> {
    let mut builder = step(Session::builder(env()?)?.with_optimization_level(level))?;
    CPU::default().with_arena_allocator(false).register(&mut builder)?;
    Ok(builder.commit_from_file(path)?)
}
