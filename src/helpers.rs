use cubecl::prelude::*;

#[derive(CubeType, CubeLaunch, Clone, Copy)]
#[expand(derive(Clone, Copy))]
pub(crate) struct Vec2F {
    pub x: f32,
    pub y: f32,
}

#[derive(CubeType, CubeLaunch, Clone, Copy)]
#[expand(derive(Clone, Copy))]
pub(crate) struct Vec3F {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(CubeType, CubeLaunch, Clone, Copy)]
#[expand(derive(Clone, Copy))]
pub(crate) struct Vec4F {
    pub w: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(CubeType, Clone, Copy)]
#[expand(derive(Clone, Copy))]
pub(crate) struct Mat3 {
    pub row0: Vec3F,
    pub row1: Vec3F,
    pub row2: Vec3F,
}

#[derive(CubeType, Clone, Copy)]
pub(crate) struct TileBBox {
    pub min_x: u32,
    pub min_y: u32,
    pub max_x: u32,
    pub max_y: u32,
}

pub(crate) const TILE_WIDTH: u32 = 16;
pub(crate) const TILE_SIZE: u32 = TILE_WIDTH * TILE_WIDTH;

#[cube]
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0f32 / (1.0f32 + (-x).exp())
}

#[cube]
pub(crate) fn quantize_u8(v: f32) -> u32 {
    (v * 255.0).clamp(0.0, 255.0) as u32
}

#[cube]
pub(crate) fn tile_bbox(mean: Vec2F, ext: Vec2F, bounds: Vec2F) -> TileBBox {
    let inv_tile = 1.0f32 / TILE_WIDTH as f32;
    TileBBox {
        min_x: ((mean.x - ext.x) * inv_tile).clamp(0.0, bounds.x) as u32,
        min_y: ((mean.y - ext.y) * inv_tile).clamp(0.0, bounds.y) as u32,
        max_x: ((mean.x + ext.x) * inv_tile + 1.0).clamp(0.0, bounds.x) as u32,
        max_y: ((mean.y + ext.y) * inv_tile + 1.0).clamp(0.0, bounds.y) as u32,
    }
}

const SH_C0: f32 = 0.282_094_8_f32;
const SH_C1: f32 = 0.488_602_52_f32;
const SH_C2_0: f32 = 0.946_174_7_f32;
const SH_C2_1: f32 = 0.315_391_57_f32;
const SH_C2_2: f32 = -1.092_548_5_f32;
const SH_C2_3: f32 = 0.546_274_24_f32;

#[rustfmt::skip]
#[cube]
pub(crate) fn sh_to_rgb(chs: u32, dir: Vec3F, splat: u32, n: u32, shs: &[f32]) -> (f32, f32, f32) {
    // Field-major layout: coefficient k, channel c lives at shs[(k*3+c)*n + splat],
    // so consecutive threads read consecutive addresses (coalesced).
    let s = splat as usize;
    let stride = n as usize;
    let mut r = SH_C0 * shs[s];
    let mut g = SH_C0 * shs[stride + s];
    let mut b = SH_C0 * shs[2 * stride + s];

    if chs >= 4 {
        r += SH_C1 * (-dir.y * shs[3 * stride + s] + dir.z * shs[6 * stride + s] - dir.x * shs[9 * stride + s]);
        g += SH_C1 * (-dir.y * shs[4 * stride + s] + dir.z * shs[7 * stride + s] - dir.x * shs[10 * stride + s]);
        b += SH_C1 * (-dir.y * shs[5 * stride + s] + dir.z * shs[8 * stride + s] - dir.x * shs[11 * stride + s]);
    }

    if chs >= 9 {
        let b4 = SH_C2_2 * dir.z * dir.x;
        let b5 = SH_C2_2 * dir.z * dir.y;
        let b6 = SH_C2_0 * dir.z * dir.z - SH_C2_1;
        let b7 = SH_C2_3 * 2.0 * dir.x * dir.y;
        let b8 = SH_C2_3 * (dir.x * dir.x - dir.y * dir.y);

        r += b6 * shs[18 * stride + s] + b7 * shs[12 * stride + s] + b5 * shs[15 * stride + s] + b4 * shs[21 * stride + s] + b8 * shs[24 * stride + s];
        g += b6 * shs[19 * stride + s] + b7 * shs[13 * stride + s] + b5 * shs[16 * stride + s] + b4 * shs[22 * stride + s] + b8 * shs[25 * stride + s];
        b += b6 * shs[20 * stride + s] + b7 * shs[14 * stride + s] + b5 * shs[17 * stride + s] + b4 * shs[23 * stride + s] + b8 * shs[26 * stride + s];
    }

    if chs >= 16 {
        let sh_c1x = dir.x * dir.x - dir.y * dir.y;
        let sh_c1y = 2.0f32 * dir.x * dir.y;
        let tmp0c = -2.285_229f32 * dir.z * dir.z + 0.457_045_8;
        let tmp1b = 1.445_305_7f32 * dir.z;
        let b9 = -0.590_043_6f32 * dir.x * sh_c1y + dir.y * sh_c1x;
        let b10 = -0.590_043_6f32 * dir.x * sh_c1x - dir.y * sh_c1y;
        let b11 = tmp0c * dir.y;
        let b12 = tmp0c * dir.x;
        let b13 = dir.z * (1.865_881_7f32 * dir.z * dir.z - 1.119_529);
        let b14 = tmp1b * sh_c1y;
        let b15 = tmp1b * sh_c1x;

        r += b9 * shs[27 * stride + s] + b10 * shs[30 * stride + s] + b11 * shs[33 * stride + s] + b12 * shs[36 * stride + s] + b13 * shs[39 * stride + s] + b14 * shs[42 * stride + s] + b15 * shs[45 * stride + s];
        g += b9 * shs[28 * stride + s] + b10 * shs[31 * stride + s] + b11 * shs[34 * stride + s] + b12 * shs[37 * stride + s] + b13 * shs[40 * stride + s] + b14 * shs[43 * stride + s] + b15 * shs[46 * stride + s];
        b += b9 * shs[29 * stride + s] + b10 * shs[32 * stride + s] + b11 * shs[35 * stride + s] + b12 * shs[38 * stride + s] + b13 * shs[41 * stride + s] + b14 * shs[44 * stride + s] + b15 * shs[47 * stride + s];
    }

    (r, g, b)
}
