use eframe::egui::{self, CursorIcon, PointerButton, Response};
use glam::{Affine3A, Quat, UVec2, Vec2, Vec3};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub fov: Vec2,
    pub position: Vec3,
    pub rotation: Quat,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            fov: Self::BASE_FOV,
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        }
    }
}

impl Camera {
    /// Reference fov every `fit_fov` derives from. The fit must never evolve
    /// the previous frame's `fov`: both branches only grow one axis, so
    /// in-place fitting ratchets the fov wider on every aspect reversal.
    const BASE_FOV: Vec2 = Vec2::splat(0.8);

    pub fn fit_fov(&mut self, pixel_size: UVec2) {
        let tan = (Self::BASE_FOV * 0.5).map(f32::tan);
        let aspect = pixel_size.x as f32 / pixel_size.y as f32;

        self.fov = if aspect > tan.x / tan.y {
            Vec2::new(2.0 * (aspect * tan.y).atan(), Self::BASE_FOV.y)
        } else {
            Vec2::new(Self::BASE_FOV.x, 2.0 * (tan.x / aspect).atan())
        };
    }

    pub fn focal(&self, img_size: UVec2) -> Vec2 {
        img_size.as_vec2() * 0.5 / (self.fov * 0.5).map(f32::tan)
    }

    pub fn w2c(&self) -> Affine3A {
        Affine3A::from_rotation_translation(self.rotation, self.position).inverse()
    }

    /// Center on `bounds` and pull back along -Y, pitched -90° around X, so
    /// the whole model is in view. Returns the focus distance used.
    pub fn frame_bounds(&mut self, (min, max): (Vec3, Vec3)) -> f32 {
        let d = (max - min).max_element() * 2.0;
        self.position = (min + max) * 0.5 - Vec3::Y * d;
        self.rotation = Quat::from_rotation_x(-core::f32::consts::FRAC_PI_2);
        d
    }
}

pub struct Controller {
    pub camera: Camera,
    focus_distance: f32,
}

impl Default for Controller {
    fn default() -> Self {
        Self {
            camera: Camera::default(),
            focus_distance: 2.5,
        }
    }
}

impl Controller {
    pub fn frame_bounds(&mut self, bounds: (Vec3, Vec3)) {
        self.focus_distance = self.camera.frame_bounds(bounds);
    }

    /// Apply one frame of input. Returns whether the camera actually moved —
    /// the caller skips re-rendering otherwise.
    pub fn tick(&mut self, response: &Response, ui: &egui::Ui) -> bool {
        let (touch, mods, pointer_delta, scroll, translation) = ui.input(|i| {
            (
                i.multi_touch(),
                i.modifiers,
                i.pointer.delta(),
                i.smooth_scroll_delta.y,
                i.translation_delta(),
            )
        });
        let t = touch.is_some();
        let is_pan = !t
            && (response.dragged_by(PointerButton::Middle)
                || response.dragged_by(PointerButton::Secondary)
                || response.dragged_by(PointerButton::Primary) && mods.ctrl);
        let is_orbit = !t && response.dragged_by(PointerButton::Primary) && !is_pan;

        if response.hovered() {
            ui.set_cursor_icon(if mods.ctrl || is_pan {
                CursorIcon::Move
            } else {
                CursorIcon::PointingHand
            });
        }

        let drag = if response.drag_started() {
            Vec2::ZERO
        } else {
            glam::vec2(pointer_delta.x, pointer_delta.y)
        };
        let pivot = self.camera.position + self.camera.rotation * Vec3::Z * self.focus_distance;

        if is_orbit {
            let yaw = Quat::from_rotation_y(drag.x * 0.002);
            let pitch = Quat::from_axis_angle(self.camera.rotation * Vec3::X, -drag.y * 0.002);
            self.camera.rotation = (yaw * pitch * self.camera.rotation).normalize();
        }

        let zoom = scroll * 0.001 + touch.map_or(0.0, |m| (m.zoom_delta - 1.0) * 2.0);
        self.focus_distance = (self.focus_distance * (1.0 - zoom)).clamp(0.1, 10000.0);
        self.camera.position = pivot - self.camera.rotation * Vec3::Z * self.focus_distance;

        let m = self.focus_distance / response.rect.width().max(response.rect.height());
        let pan = if is_pan {
            drag
        } else if translation != egui::Vec2::ZERO && scroll == 0.0 {
            glam::vec2(translation.x, translation.y)
        } else {
            Vec2::ZERO
        };
        if pan != Vec2::ZERO {
            self.camera.position -= (self.camera.rotation * Vec3::X) * pan.x * m;
            self.camera.position += (self.camera.rotation * Vec3::NEG_Y) * pan.y * m;
        }

        is_orbit || zoom != 0.0 || pan != Vec2::ZERO
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Aspect round trips must not drift the fov: deriving from the previous
    /// frame's value ratcheted both axes wider per reversal, shrinking and
    /// radially warping the model with each resize until restart.
    #[test]
    fn test_fit_fov_is_history_free() {
        let mut camera = Camera::default();
        camera.fit_fov(glam::uvec2(1000, 1000));
        let square = camera.fov;

        for _ in 0..10 {
            camera.fit_fov(glam::uvec2(1600, 1000));
            camera.fit_fov(glam::uvec2(1000, 1000));
        }
        assert_eq!(camera.fov, square, "aspect round trips must not drift fov");

        // A fit keeps the projection isotropic: fx == fy.
        camera.fit_fov(glam::uvec2(1600, 1000));
        let focal = camera.focal(glam::uvec2(1600, 1000));
        assert!((focal.x - focal.y).abs() < 1e-3);
    }
}
