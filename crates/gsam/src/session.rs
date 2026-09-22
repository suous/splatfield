use anyhow::Result;
use ort::{
    ep::{CPU, ExecutionProvider},
    session::{Session, builder::GraphOptimizationLevel},
};

/// The CPU arena is OFF: with it on, the arena grew ~1 GB per run on the
/// quantized graphs until jetsam killed the process on a 16 GB machine.
///
/// `level` caps graph optimization - the grounding-dino q4f16 export
/// crashes ORT at the default Level3 (a LayerNorm fusion trips over an
/// inserted precision cast) and must load at Level1.
pub(crate) fn build(path: &std::path::Path, level: GraphOptimizationLevel) -> Result<Session> {
    let mut builder = Session::builder()?.with_optimization_level(level)?;
    CPU::default()
        .with_arena_allocator(false)
        .register(&mut builder)?;
    // Leave a couple of cores for the UI thread: the oracle runs on the
    // worker while the app must stay interactive. Inter-op parallelism
    // barely helps these sequential graphs - one intra-op pool is the
    // documented pattern.
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    builder = builder.with_intra_threads(threads.saturating_sub(2).max(1))?;
    builder = builder.with_inter_threads(1)?;
    Ok(builder.commit_from_file(path)?)
}
