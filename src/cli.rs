//! The `splatfield seg` subcommand: headless text-prompted segmentation to
//! PLY.

use crate::camera::Camera;
use crate::ply;
use crate::seg::active::Config;
use crate::seg::prompted::run_text;
use anyhow::{Context, Result, ensure};
use clap::Parser;
use cubecl::{prelude::*, wgpu::WgpuRuntime};

/// Segment a 3D Gaussian Splat scene by a text prompt (B3-Seg, Base oracle)
/// and write the surviving splats as a PLY.
///
/// Progress and diagnostics go to stderr; only the PLY ever touches stdout.
#[derive(Parser)]
#[command(name = "splatfield seg", no_binary_name = true)]
struct Args {
    /// scene file: .ply, .sog archive, or '-' for a PLY on stdin
    // stdin always needs somewhere to go: the output PLY can't default to
    // <stem>.seg.ply when there is no stem.
    #[arg(requires_if("-", "output"))]
    input: String,
    /// text description of the object to keep
    prompt: String,
    /// output PLY, '-' for stdout (default: <stem>.seg.ply)
    #[arg(short, long)]
    output: Option<String>,
    /// active-loop rounds
    // usize has no ranged factory in clap (platform-dependent width), so the
    // u64-ranged parser is instantiated over it directly.
    #[arg(short, long, default_value_t = Config::default().iterations, value_parser = clap::builder::RangedU64ValueParser::<usize>::from(1..))]
    iterations: usize,
    /// oracle render resolution
    #[arg(
        short,
        long,
        default_value_t = Config::default().resolution,
        value_parser = clap::value_parser!(u32).range(16..)
    )]
    resolution: u32,
    /// candidate views per round
    #[arg(
        short,
        long,
        default_value_t = Config::default().candidates,
        value_parser = clap::builder::RangedU64ValueParser::<usize>::from(1..)
    )]
    candidates: usize,
    /// bootstrap view position x,y,z, looking at the scene center (default:
    /// top-down, framing the whole scene)
    #[arg(long, value_parser = parse_camera)]
    camera: Option<[f32; 3]>,
}

/// `x,y,z` → camera position, for `--camera`.
fn parse_camera(s: &str) -> Result<[f32; 3], String> {
    let n = |part: &str| {
        part.trim()
            .parse::<f32>()
            .map_err(|_| format!("'{s}' is not x,y,z numbers"))
    };
    let mut parts = s.split(',');
    match (parts.next(), parts.next(), parts.next(), parts.next()) {
        (Some(x), Some(y), Some(z), None) => Ok([n(x)?, n(y)?, n(z)?]),
        _ => Err(format!("'{s}' is not x,y,z numbers")),
    }
}

/// Indices whose label is `true` — the object to keep.
fn keep_indices(labels: &[bool]) -> Vec<usize> {
    labels
        .iter()
        .enumerate()
        .filter(|&(_, &label)| label)
        .map(|(i, _)| i)
        .collect()
}

fn segment(a: Args) -> Result<()> {
    crate::use_single_stream();
    eprintln!("splatfield: initializing GPU…");
    let client = WgpuRuntime::client(&cubecl::wgpu::WgpuDevice::default());

    eprintln!("splatfield: loading {}…", a.input);
    let cpu = if a.input == "-" {
        ply::parse_ply(std::io::stdin().lock()).context("reading stdin")?
    } else {
        crate::load_scene_file(std::path::Path::new(&a.input))
            .with_context(|| format!("parsing {}", a.input))?
    };
    let total = cpu.count();
    eprintln!("splatfield: uploading {total} splats to GPU…");
    let splats = std::sync::Arc::new(cpu.clone().upload(&client));

    // The paper's bootstrap observation: one canonical view of the whole
    // scene — top-down by default, or wherever `--camera` aims it.
    let mut bootstrap = Camera::default();
    bootstrap.frame_bounds(splats.bounds);
    if let Some([x, y, z]) = a.camera {
        let (min, max) = splats.bounds;
        let center = (min + max) * 0.5;
        let position = glam::vec3(x, y, z);
        bootstrap = Camera::look_at(position, center);
    }

    let cfg = Config {
        iterations: a.iterations,
        resolution: a.resolution,
        candidates: a.candidates,
    };
    let mut round = 0;
    let segmenter = run_text(
        splats,
        &a.prompt,
        cfg,
        bootstrap,
        &mut |p| eprintln!("splatfield: {p}"),
        &mut |it, _| {
            round += 1;
            eprintln!(
                "splatfield: round {round} eig {:.3} fg px {} cam {:?}",
                it.eig, it.fg_pixels, it.camera.position
            );
            true
        },
    )?;

    let keep = keep_indices(&segmenter.segmentation());
    ensure!(
        !keep.is_empty(),
        "segmentation kept 0 of {total} splats — the oracle never fired"
    );

    // The gathered scene is already CPU-resident; no upload/readback detour.
    let kept = cpu.gather(&keep);

    match a.output.as_deref() {
        Some("-") => kept.write_ply(std::io::stdout().lock())?,
        Some(path) => {
            let out = std::fs::File::create(path).with_context(|| format!("creating {path}"))?;
            kept.write_ply(out)?
        }
        None => {
            let path = std::path::Path::new(&a.input).with_extension("seg.ply");
            let out = std::fs::File::create(&path).with_context(|| format!("creating {path:?}"))?;
            kept.write_ply(out)?;
            eprintln!("splatfield: wrote {}", path.display());
        }
    }
    eprintln!("splatfield: kept {} of {total} splats", keep.len());
    Ok(())
}

/// Entry of the `seg` subcommand; the GUI `main` dispatches on `argv[1]`.
pub fn run(args: impl Iterator<Item = String>) -> std::process::ExitCode {
    match Args::try_parse_from(args) {
        Ok(args) => match segment(args) {
            Ok(()) => std::process::ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("splatfield: {e:#}");
                std::process::ExitCode::FAILURE
            }
        },
        Err(e) => {
            let _ = e.print();
            std::process::ExitCode::from(if e.use_stderr() { 2 } else { 0 })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::ATTR_PLANES;
    use crate::render::CpuSplats;
    use crate::seg::active::Segmenter;
    use crate::seg::views::{cluster_scene, target};

    fn flags(list: &[&str]) -> Result<Args, clap::Error> {
        Args::try_parse_from(list.iter().copied())
    }

    #[test]
    fn test_parse_defaults_and_flags() {
        let a = flags(&["m.ply", "chair"]).unwrap();
        assert_eq!(a.input, "m.ply");
        assert_eq!(a.prompt, "chair");
        assert_eq!(a.output, None);
        assert_eq!(a.iterations, 20);
        assert_eq!(a.resolution, 512);
        assert_eq!(a.candidates, 20);

        let a = flags(&["-", "cat", "-o", "-", "-i", "3", "-r", "256", "-c", "5"]).unwrap();
        assert_eq!(a.output.as_deref(), Some("-"));
        assert_eq!((a.iterations, a.resolution, a.candidates), (3, 256, 5));

        let a = flags(&["m.ply", "cat", "--camera", "1.5,-2,0.25"]).unwrap();
        assert_eq!(a.camera, Some([1.5, -2.0, 0.25]));
        assert!(
            flags(&["m.ply", "cat", "--camera", "1,2"]).is_err(),
            "needs 3 coords"
        );
    }

    #[test]
    fn test_parse_rejects() {
        assert!(flags(&["m.ply"]).is_err(), "prompt missing");
        assert!(flags(&["m.ply", "a", "b"]).is_err(), "extra positional");
        assert!(flags(&["m.ply", "a", "--nope"]).is_err(), "unknown flag");
        assert!(flags(&["m.ply", "a", "-i"]).is_err(), "dangling flag");
        assert!(flags(&["-", "a"]).is_err(), "stdin needs -o");
        assert!(flags(&["m.ply", "a", "-i", "x"]).is_err(), "bad number");
        assert!(flags(&["m.ply", "a", "-i", "0"]).is_err(), "zero rounds");
        assert!(matches!(flags(&["-h"]), Err(e) if !e.use_stderr()), "help");
    }

    /// End-to-end headless run against the perfect synthetic oracle: the
    /// keep-set must isolate the target cluster, and the filtered scene must
    /// round-trip through the PLY writer with exactly the kept vertices.
    #[test]
    fn test_seg_segments_cluster_and_roundtrips_ply() {
        let (attributes, centers) = cluster_scene();
        let n = attributes.len() / ATTR_PLANES;

        let (_gpu, client) = crate::gpu_testing::test_client();
        let cpu = CpuSplats {
            attributes,
            sh_coeffs: vec![0.0; n * 3],
        };
        let splats = std::sync::Arc::new(cpu.clone().upload(&client));

        let mut bootstrap = Camera::default();
        bootstrap.frame_bounds(splats.bounds);
        let mut oracle = target(centers[0].0, centers[0].1);
        let mut segmenter = Segmenter::new(
            splats,
            Config {
                resolution: 256,
                candidates: 12,
                iterations: 10,
            },
            bootstrap,
        )
        .unwrap();
        segmenter.run_with(&mut oracle, |_, _| true).unwrap();

        let keep = keep_indices(&segmenter.segmentation());
        let correct = keep.iter().filter(|&&i| i < 40).count();
        assert!(
            correct >= 34 && keep.len() <= 44,
            "keep must isolate cluster 0 ({correct}/40, len {})",
            keep.len()
        );

        let kept = cpu.gather(&keep);
        let mut bytes = Vec::new();
        kept.write_ply(&mut bytes).unwrap();
        let parsed = ply::parse_ply(&bytes[..]).unwrap();
        assert_eq!(parsed.count(), keep.len());
    }
}
