use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

#[derive(CubeType, CubeLaunch, Clone, Copy)]
#[expand(derive(Clone, Copy))]
pub(crate) struct Vec2F {
    pub x: f32,
    pub y: f32,
}

impl From<glam::Vec2> for Vec2FLaunch<WgpuRuntime> {
    fn from(v: glam::Vec2) -> Self {
        Self::new(v.x, v.y)
    }
}

#[derive(CubeType, CubeLaunch, Clone, Copy)]
#[expand(derive(Clone, Copy))]
pub(crate) struct Vec3F {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl From<glam::Vec3> for Vec3FLaunch<WgpuRuntime> {
    fn from(v: glam::Vec3) -> Self {
        Self::new(v.x, v.y, v.z)
    }
}

// Affine3A's matrix3/translation are Vec3A.
impl From<glam::Vec3A> for Vec3FLaunch<WgpuRuntime> {
    fn from(v: glam::Vec3A) -> Self {
        Self::new(v.x, v.y, v.z)
    }
}

pub(crate) const TILE_WIDTH: u32 = 16;
pub(crate) const TILE_SIZE: u32 = TILE_WIDTH * TILE_WIDTH;

// Splat attribute planes, field-major: plane k of splat i lives at
// attributes[k * n + i] so a warp's loads coalesce.
pub(crate) const ATTR_PLANES: usize = 11;
pub(crate) const PLANE_X: usize = 0;
pub(crate) const PLANE_Y: usize = 1;
pub(crate) const PLANE_Z: usize = 2;
pub(crate) const PLANE_QW: usize = 3;
pub(crate) const PLANE_QX: usize = 4;
pub(crate) const PLANE_QY: usize = 5;
pub(crate) const PLANE_QZ: usize = 6;
pub(crate) const PLANE_SX: usize = 7;
pub(crate) const PLANE_SY: usize = 8;
pub(crate) const PLANE_SZ: usize = 9;
pub(crate) const PLANE_OPACITY: usize = 10;

// Projected row: [mean2d_xy, conic_xyz, rgb, opacity].
pub(crate) const PROJ_FLOATS: usize = 9;
