# SplatField

A simple pure Rust 3D Gaussian Splatting viewer — under 1.3K lines of code.

<!-- rumdl-disable MD033 -->
<div align="center">
<video src="https://github.com/user-attachments/assets/34152477-4633-4d58-be98-b9e36abfb4df" controls="controls" muted="muted" loop="loop" autoplay="autoplay" playsinline="true" width="480">
Should show demo.
</video>
</div>

Built with [egui](https://github.com/emilk/egui), [wgpu](https://github.com/gfx-rs/wgpu), and [cubecl](https://github.com/tracel-ai/cubecl).

Built to learn the details of [3D Gaussian Splatting](https://arxiv.org/abs/2308.04079). Learned a lot from [Brush](https://github.com/ArthurBrussee/brush).

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

6 GPU kernel passes per frame:

1. **Project** — 3D to 2D projection, covariance, SH color, per-splat tile counts
2. **Depth sort** — radix sort by depth
3. **Scan** — prefix-sum tile counts in depth order
4. **Remap** — emit intersections at scan offsets
5. **Tile sort** — stable radix sort by tile
6. **Rasterize** — front-to-back alpha blend; per-tile ranges come from in-kernel binary search

```bash
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Language              Files        Lines         Code     Comments       Blanks
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Rust                     10         1446         1274            7          165
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Total                    10         1465         1274           23          168
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Benchmarks

```bash
cargo bench
```

`radix_sort` runs anywhere. `render_frame` needs the bear fixture at
`data/bear.3d71a266_sh2.sog` and silently reports no benchmarks without it.
