# CLAUDE.md

## CRITICAL — use relative paths for file tools

Pass **relative paths** (e.g. `src/tensor.rs`, `CLAUDE.md`) to `Read`, `Edit`, `Write`, and any file tool. **Never hand-type the absolute repo path.** The absolute path is often mistyped, causing resolution failures. Relative paths resolve against the session CWD and always work; `Bash` search uses `rg`/`fd` with relative paths too.

## Commands

```bash
cargo run          # Run GUI app
cargo test         # Run tests
cargo bench        # Run radix sort benchmarks
```

Format and lint via the auto_check hook — no manual
`cargo fmt` / `cargo clippy` needed.

## Architecture

GPU-accelerated Gaussian Splatting renderer. egui GUI loads PLY files via drag-and-drop.

### Rendering Pipeline (`render::Splats::render_with()`)

6 GPU kernel passes per frame:

1. **Project** — 3D→2D projection, covariance, SH color, tile intersections
2. **Depth sort** — radix sort by depth (`sort::radix_argsort`)
3. **Remap** — apply depth ordering to intersection IDs
4. **Tile sort** — sort intersections by tile (`sort::radix_argsort`)
5. **Tile ranges** — build per-tile `[start, end)` ranges
6. **Rasterize** — front-to-back alpha blend → RGBA8 bitmap

### Data Layout

| Buffer | Shape | Contents |
|---|---|---|
| `attributes` | `[n, 11]` | field-major planes `x, y, z, qw, qx, qy, qz, sx, sy, sz, opacity` — plane k of splat i at `k * n + i` (warp-coalesced reads) |
| `sh_coeffs` | `[n, channels, 3]` | field-major SH: coefficient k channel c of splat i at `(k * 3 + c) * n + i` |
| `projected` | `[n, 9]` | `mean2d_xy, conic_xyz, rgb, opacity` |
| output | `[h, w]` | packed `u32` RGBA8 |

### GPU Compute

Kernels use `cubecl` with `#[cube(launch)]` targeting `WgpuRuntime`.
Custom types via `#[cube]`/`CubeLaunch` in `helpers`.
Dispatch: `calculate_cube_count_elemwise` + `CubeDim::new_1d(TILE_SIZE)`.
