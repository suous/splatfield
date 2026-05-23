use cubecl::prelude::*;

#[derive(CubeType, CubeLaunch, Clone, Copy)]
pub(crate) struct Vec2F {
    pub x: f32,
    pub y: f32,
}

#[derive(CubeType, CubeLaunch, Clone, Copy)]
pub(crate) struct Vec3F {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(CubeType, CubeLaunch, Clone, Copy)]
pub(crate) struct Vec4F {
    pub w: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(CubeType, Clone, Copy)]
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
    1.0 / (1.0 + (-x).exp())
}

#[cube]
pub(crate) fn quantize_u8(v: f32) -> u32 {
    (v * 255.0).clamp(0.0, 255.0) as u32
}

#[cube]
pub(crate) fn tile_bbox(mean: Vec2F, ext: Vec2F, bounds: Vec2F) -> TileBBox {
    let inv_tile = 1.0 / TILE_WIDTH as f32;
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
pub(crate) fn sh_to_rgb(chs: u32, dir: Vec3F, shs: &Array<f32>) -> (f32, f32, f32) {
    let bi = (ABSOLUTE_POS_X as usize) * chs as usize * 3;
    let mut r = SH_C0 * shs[bi];
    let mut g = SH_C0 * shs[bi + 1];
    let mut b = SH_C0 * shs[bi + 2];

    if chs >= 4 {
        r += SH_C1 * (-dir.y * shs[bi + 3] + dir.z * shs[bi + 6] - dir.x * shs[bi + 9]);
        g += SH_C1 * (-dir.y * shs[bi + 4] + dir.z * shs[bi + 7] - dir.x * shs[bi + 10]);
        b += SH_C1 * (-dir.y * shs[bi + 5] + dir.z * shs[bi + 8] - dir.x * shs[bi + 11]);
    }

    if chs >= 9 {
        let b4 = SH_C2_2 * dir.z * dir.x;
        let b5 = SH_C2_2 * dir.z * dir.y;
        let b6 = SH_C2_0 * dir.z * dir.z - SH_C2_1;
        let b7 = SH_C2_3 * 2.0 * dir.x * dir.y;
        let b8 = SH_C2_3 * (dir.x * dir.x - dir.y * dir.y);

        r += b6 * shs[bi + 18] + b7 * shs[bi + 12] + b5 * shs[bi + 15] + b4 * shs[bi + 21] + b8 * shs[bi + 24];
        g += b6 * shs[bi + 19] + b7 * shs[bi + 13] + b5 * shs[bi + 16] + b4 * shs[bi + 22] + b8 * shs[bi + 25];
        b += b6 * shs[bi + 20] + b7 * shs[bi + 14] + b5 * shs[bi + 17] + b4 * shs[bi + 23] + b8 * shs[bi + 26];
    }

    if chs >= 16 {
        let sh_c1x = dir.x * dir.x - dir.y * dir.y;
        let sh_c1y = 2.0 * dir.x * dir.y;
        let tmp0c = -2.285_229 * dir.z * dir.z + 0.457_045_8;
        let tmp1b = 1.445_305_7 * dir.z;
        let b9 = -0.590_043_6 * dir.x * sh_c1y + dir.y * sh_c1x;
        let b10 = -0.590_043_6 * dir.x * sh_c1x - dir.y * sh_c1y;
        let b11 = tmp0c * dir.y;
        let b12 = tmp0c * dir.x;
        let b13 = dir.z * (1.865_881_7 * dir.z * dir.z - 1.119_529);
        let b14 = tmp1b * sh_c1y;
        let b15 = tmp1b * sh_c1x;

        r += b9 * shs[bi + 27] + b10 * shs[bi + 30] + b11 * shs[bi + 33] + b12 * shs[bi + 36] + b13 * shs[bi + 39] + b14 * shs[bi + 42] + b15 * shs[bi + 45];
        g += b9 * shs[bi + 28] + b10 * shs[bi + 31] + b11 * shs[bi + 34] + b12 * shs[bi + 37] + b13 * shs[bi + 40] + b14 * shs[bi + 43] + b15 * shs[bi + 46];
        b += b9 * shs[bi + 29] + b10 * shs[bi + 32] + b11 * shs[bi + 35] + b12 * shs[bi + 38] + b13 * shs[bi + 41] + b14 * shs[bi + 44] + b15 * shs[bi + 47];
    }

    (r, g, b)
}
