# SplatField

A 3D Gaussian Splatting viewer with **open-vocabulary segmentation**: describe any
object in text — `teddy bear`, `wooden chair`, `fire hydrant` — and it gets tinted
in the scene. No fixed label set, no training, no reconstruction cameras.

Rendering and GPU compute run on [`cubecl`](https://crates.io/crates/cubecl) +
[`wgpu`](https://wgpu.rs/); the 2D oracle (GroundingDINO + SAM2 on ONNX
Runtime, CPU). Each prompt runs the [B3-Seg](https://sony.github.io/B3-Seg-project/)
loop: pick views by analytic EIG, lift 2D oracle masks into Beta–Bernoulli
evidence, tint on the GPU.

<!-- rumdl-disable MD033 -->
<div align="center">
<video src="https://github.com/user-attachments/assets/7f555fa5-f663-4105-9b79-1fcae418824f" controls="controls" muted="muted" loop="loop" autoplay="autoplay" playsinline="true" width="480">
Should show demo.
</video>
</div>

Built with [egui](https://github.com/emilk/egui), [wgpu](https://github.com/gfx-rs/wgpu), and [cubecl](https://github.com/tracel-ai/cubecl).

Built to learn the details of [3D Gaussian Splatting](https://arxiv.org/abs/2308.04079). Learned a lot from [Brush](https://github.com/ArthurBrussee/brush).

## How to use

1. Grab a native binary from [Releases](https://github.com/suous/splatfield/releases/tag/v0.2.0-rc.1) — Windows, Linux, macOS (Apple Silicon).
2. Run it:

   ```sh
   splatfield
   ```

3. Drag & drop a `.ply`/`.sog` file onto the window. No scene at hand? Try the
   teddy-bear demo (`bear.3d71a266.sog`) from the
   [fixtures release](https://github.com/suous/splatfield/releases/tag/fixtures-v1).
4. Type what you want to segment — `teddy bear` for the demo. Any noun phrase
   works: the label set is open vocabulary, not a fixed list.

**First run** fetches the oracle models as one ~150 MB zip (~180 MB unzipped)
from a pinned, SHA-256-verified release into the platform cache
(`~/Library/Caches/splatfield/models/models-v1/` on macOS).

### Controls

- **Orbit** — drag · **pan** — middle/right-drag or ctrl+drag · **zoom** — scroll/pinch
- **Segment** — type a prompt, press Enter. Progress pill, posterior heatmap
  (default on), stop and reset buttons
- **Box select** — shift+drag a rectangle. Hits highlight green. Delete removes
  them; Cmd/Ctrl+Z undoes

## Segmentation

**GUI.** Type a prompt (e.g. `bear`), press Enter. Each round renders candidate
views, queries the oracle, and lifts the 2D mask into Beta–Bernoulli evidence.

The pill shows round progress and EIG. The viewport overlays the posterior mean
as a toggleable heatmap. **Stop** takes effect at the next round boundary — the
partial posterior still tints and cuts. On finish, the object takes the next
palette color.

- **Cut** — removes the background. Kept splats keep their colors (rebuilt from
  the pristine CPU master).
- **Reset** — restores the loaded model.
- **Save** — writes `<source>.edited.ply`.
- **iterations** — 1–20, default 20. Caps oracle observations.
- Box-select, Delete, and undo work between runs. Edits staged mid-run are
  rejected until it finishes.

**Headless CLI.** Same loop, no GUI (`splatfield seg`). Writes surviving splats
as PLY:

```sh
splatfield seg <model.ply|-> "<prompt>" [options]

  -o, --output <path>   output PLY, '-' for stdout
                        (default: <stem>.seg.ply; required when input is stdin)
  -i, --iterations <n>  active-loop rounds (default: 20)
  -r, --resolution <n>  oracle render resolution (default: 512)
  -c, --candidates <n>  candidate views per round (default: 20)
      --camera x,y,z    bootstrap view position, looking at the scene center
                        (default: top-down, framing the whole scene)
  -h, --help
```

```sh
splatfield seg data/bear.3d71a266.sog "teddy bear"
```

Progress goes to stderr; only the PLY touches stdout.

## Oracle backend

`grounded-sam2`: grounding-dino-tiny (q4f16) grounds the free-form prompt and
detects the object; its best box prompts sam2-tiny (q4f16) for the mask. Models
are HuggingFace [ONNX](https://huggingface.co/onnx-community) exports in the platform cache. First use fetches a pinned,
checksum-verified release zip. Sessions run on CPU.

## Differences from the B3-Seg paper

The loop follows the paper: analytic EIG over a camera-free candidate sphere,
Beta–Bernoulli conjugate updates, MAP label `a_i > b_i`. Engineering deltas:

1. **q4f16 quantized CPU inference** — GroundingDINO at ORT optimization
   Level1 (Level3's LayerNorm fusion crashes); SAM2 at Level3. Intra-op threads
   = cores − 2 to keep the UI interactive.
2. **Entropy lookup table instead of in-kernel H** — Metal lacks fp64 and f32
   entropy cancels at scale; H reads from a log-spaced 445×445 f32 table
   (f64-computed) over [1, 1024] (bilinear in ln a, ln b), < 1e-3 nats error.
3. **CLIP re-ranking skipped** (paper Sec 3.4) — we take the detector's best
   box; multi-instance prompts segment one instance.
4. **No posterior mask fed back to SAM2** (paper Sec 3.4) — the existing ONNX
   export has no mask input.
5. **Fixed-point evidence** — per-Gaussian responsibilities accumulate in u32
   (WGSL atomics are integer-only), scaled to render resolution.
6. **Early stop** — a round returning an empty mask while nothing is foreground
   ends the run.

## Docs

| Doc | Description |
|-----|-------------|
| [Pipeline](docs/00_pipeline.md) | 3DGS forward pipeline overview, data structures, and rendering stages |
| [Projection](docs/01_project.md) | 3D covariance, local affine approximation, and 2D covariance projection |
| [Rasterization](docs/02_rasterization.md) | Tile-based dispatch, per-pixel evaluation, and alpha compositing |

### Interactive Diagrams

| Diagram | Description |
|---------|-------------|
| [Tile Intersection](https://suous.github.io/splatfield/intersects.svg) | Drag to reshape the Gaussian and see tile coverage change |
| [Alpha Compositing](https://suous.github.io/splatfield/compositing.svg) | Drag splats, adjust opacity and depth order, inspect pixel blending |

## Architecture

6 GPU kernel passes per frame (~8.7K lines of Rust across the workspace):

1. **Project** — 3D to 2D projection, covariance, SH color, per-splat tile counts
2. **Depth sort** — radix sort by depth
3. **Scan** — prefix-sum tile counts in depth order
4. **Remap** — emit intersections at scan offsets
5. **Tile sort** — stable radix sort by tile
6. **Rasterize** — front-to-back alpha blend; per-tile ranges come from in-kernel binary search

## Development

```sh
cargo run -r                                        # from source
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test        # GPU tests; serializes access to one device
```

## Benchmarks

```sh
cargo bench --workspace
```

`radix_sort` needs no fixture. `render_frame` needs the bear fixture at
`data/bear.3d71a266_sh2.sog` and silently reports no benchmarks without it.

## TODO

- [ ] **Box-selection segmentation** — feed Shift+drag rect straight to SAM2 (no detector pass); `Oracle` seam unchanged.
- [ ] **WASM support** — viewer side is proven; `ort` has no browser backend,
  model fetch is native-only, wasm memory ceilings need a plan.
- [ ] Multi-instance prompts — palette per instance.
- [ ] CoreML/GPU execution provider for the oracle — CPU oracle is the long pole (~4.3 s/view).

## References

- **3D Gaussian Splatting** — <https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/> · [arXiv:2308.04079](https://arxiv.org/abs/2308.04079)
- **B³-Seg** (CVPR 2026 highlight) — <https://sony.github.io/B3-Seg-project/> · [arXiv:2602.17134](https://arxiv.org/abs/2602.17134)
- **GroundingDINO** — <https://github.com/IDEA-Research/GroundingDINO> · [arXiv:2303.05499](https://arxiv.org/abs/2303.05499)
- **SAM 2** — <https://github.com/facebookresearch/sam2> · [arXiv:2408.00714](https://arxiv.org/abs/2408.00714)
- **ONNX exports** — [grounding-dino-tiny-ONNX](https://huggingface.co/onnx-community/grounding-dino-tiny-ONNX/) · [sam2.1-hiera-tiny-ONNX](https://huggingface.co/onnx-community/sam2.1-hiera-tiny-ONNX/)
- **Bear demo scene** — <https://aholojs.dev/en-US/examples/splatting-basic/>

<details>
<summary><strong>EfficientSAM3 experiment</strong></summary>

A point-in-time swap of the oracle for [EfficientSAM3](https://simonzeng7108.github.io/efficientsam3/)
(distilled SAM3, one fused mask-native invocation per view) on 24 orbit views of
the bear scene.

| | grounded-sam2 (shipped) | EfficientSAM3 (reverted) |
|---|---|---|
| Per-view latency | 4298 ms | 3243 ms (EV-M) / 3345 ms (RV-M) — 20–25 % faster |
| Disk / warm RSS | 177 MB / ~1.0 GB | 136–143 MB / ~2.0 GB |
| Mask stability | consistent whole-object masks | part decomposition + score reordering; whole-bear query lost argmax in 4/10 views |
| Over-capture | box-bound | union merges mat/terrain — kept set ballooned to 919k of 971k splats |
| Prompt sensitivity | low | high — "bear" vs "teddy bear" fire different queries |
| 10-round outcome | stable, no holes | kept set oscillated 273k ↔ 919k |

**Verdict.** The oracle's job is evidence the Beta update can trust; 20–25 %
latency doesn't buy back cross-view mask wobble. Revisit when posterior masking
lands, a distilled variant emits a stable whole-object query, or the 2× RSS is
solved.

</details>
