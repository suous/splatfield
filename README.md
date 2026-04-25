# SplatField

A simple pure Rust 3D Gaussian Splatting viewer — under 1.3K lines of code.

<!-- rumdl-disable MD033 -->
<div align="center">
<video src="assets/demo.mp4" controls="controls" muted="muted" loop="loop" autoplay="autoplay" playsinline="true" width="480">
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

## Architecture

6 GPU kernel passes per frame:

1. **Project** — 3D to 2D projection, covariance, SH color, tile intersections
2. **Depth sort** — radix sort by depth
3. **Remap** — apply depth ordering to intersection IDs
4. **Tile sort** — sort intersections by tile
5. **Tile ranges** — build per-tile `[start, end)` ranges
6. **Rasterize** — front-to-back alpha blend to RGBA8 bitmap

```bash
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Language              Files        Lines         Code     Comments       Blanks
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Rust                     10         1446         1274            7          165
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Total                    10         1465         1274           23          168
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
