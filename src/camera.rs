use egui::{CursorIcon, PointerButton, Response};
use glam::{Affine3A, Quat, UVec2, Vec2, Vec3};

#[derive(Clone)]
pub struct Camera {
    pub fov: Vec2,
    pub position: Vec3,
    pub rotation: Quat,
}

impl Camera {
    pub fn fit_fov(&mut self, pixel_size: UVec2) {
        let tan = (self.fov * 0.5).map(f32::tan);
        let aspect = pixel_size.x as f32 / pixel_size.y as f32;

        if aspect > tan.x / tan.y {
            self.fov.x = 2.0 * (aspect * tan.y).atan();
        } else {
            self.fov.y = 2.0 * (tan.x / aspect).atan();
        }
    }

    pub fn focal(&self, img_size: UVec2) -> Vec2 {
        img_size.as_vec2() * 0.5 / (self.fov * 0.5).map(f32::tan)
    }

    pub fn w2c(&self) -> Affine3A {
        Affine3A::from_rotation_translation(self.rotation, self.position).inverse()
    }
}

pub struct Controller {
    pub camera: Camera,
    pub focus_distance: f32,
}

impl Controller {
    pub fn new(position: Vec3, rotation: Quat) -> Self {
        Self {
            camera: Camera {
                position,
                rotation,
                fov: Vec2::splat(0.8),
            },
            focus_distance: 2.5,
        }
    }

    pub fn frame_bounds(&mut self, (min, max): (Vec3, Vec3)) {
        let d = (max - min).max_element() * 2.0;
        self.camera.position = (min + max) * 0.5 - Vec3::Y * d;
        self.camera.rotation = Quat::from_rotation_x((-90f32).to_radians());
        self.focus_distance = d;
    }

    pub fn tick(&mut self, response: &Response, ui: &egui::Ui) {
        let (touch, mods) = ui.input(|i| (i.multi_touch(), i.modifiers));
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
            ui.input(|i| glam::vec2(i.pointer.delta().x, i.pointer.delta().y))
        };
        let pivot = self.camera.position + self.camera.rotation * Vec3::Z * self.focus_distance;

        if is_orbit {
            let yaw = Quat::from_rotation_y(drag.x * 0.002);
            let pitch = Quat::from_axis_angle(self.camera.rotation * Vec3::X, -drag.y * 0.002);
            self.camera.rotation = (yaw * pitch * self.camera.rotation).normalize();
        }

        let (scroll, td) = ui.input(|i| (i.smooth_scroll_delta.y, i.translation_delta()));
        let zoom = scroll * 0.001 + touch.map_or(0.0, |m| (m.zoom_delta - 1.0) * 2.0);
        self.focus_distance = (self.focus_distance * (1.0 - zoom)).clamp(0.1, 10000.0);
        self.camera.position = pivot - self.camera.rotation * Vec3::Z * self.focus_distance;

        let m = self.focus_distance / response.rect.width().max(response.rect.height());
        let pan = if is_pan {
            drag
        } else if td != egui::Vec2::ZERO && scroll == 0.0 {
            glam::vec2(td.x, td.y)
        } else {
            Vec2::ZERO
        };
        if pan != Vec2::ZERO {
            self.camera.position -= (self.camera.rotation * Vec3::X) * pan.x * m;
            self.camera.position += (self.camera.rotation * Vec3::NEG_Y) * pan.y * m;
        }
    }
}
