use glam::*;
use wgpu::util::DeviceExt;

use crate::core::BufferWrapper;

/// The basic color modifiers buffer for the [`BasicModifiers`](crate::BasicModifiers).
#[derive(Debug)]
pub struct BasicColorModifiersBuffer(wgpu::Buffer);

impl BasicColorModifiersBuffer {
    /// Create a new basic color modifiers buffer.
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Basic Color Modifiers Buffer"),
            contents: bytemuck::bytes_of(&BasicColorModifiersPod::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self(buffer)
    }

    /// Update the basic color modifiers with override RGB color.
    pub fn update_with_override_rgb(
        &self,
        queue: &wgpu::Queue,
        rgb: Vec3,
        alpha: f32,
        contrast: f32,
        exposure: f32,
        gamma: f32,
    ) {
        self.update_with_pod(
            queue,
            &BasicColorModifiersPod::new_with_override_rgb(rgb, alpha, contrast, exposure, gamma),
        );
    }

    /// Update the basic color modifiers buffer with HSV modifications.
    pub fn update_with_hsv_modifiers(
        &self,
        queue: &wgpu::Queue,
        hsv: Vec3,
        alpha: f32,
        contrast: f32,
        exposure: f32,
        gamma: f32,
    ) {
        self.update_with_pod(
            queue,
            &BasicColorModifiersPod::new_with_hsv_modifiers(hsv, alpha, contrast, exposure, gamma),
        );
    }

    /// Update the basic color modifiers buffer with [`BasicColorModifiersPod`].
    pub fn update_with_pod(&self, queue: &wgpu::Queue, pod: &BasicColorModifiersPod) {
        queue.write_buffer(&self.0, 0, bytemuck::bytes_of(pod));
    }
}

impl BufferWrapper for BasicColorModifiersBuffer {
    fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

/// The POD representation of the basic color modifiers buffer.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BasicColorModifiersPod {
    /// If any value is negative, then it is used to override the RGB color,
    /// otherwise it is used to apply HSV modifications.
    pub rgb_or_hsv: Vec3,
    pub alpha: f32,
    pub contrast: f32,
    pub exposure: f32,
    pub gamma: f32,
    _padding: u32,
}

impl BasicColorModifiersPod {
    /// Creates a new basic color modifiers with RGB color override.
    pub fn new_with_override_rgb(
        rgb: Vec3,
        alpha: f32,
        contrast: f32,
        exposure: f32,
        gamma: f32,
    ) -> Self {
        Self {
            rgb_or_hsv: -rgb,
            alpha,
            contrast,
            exposure,
            gamma,
            _padding: 0,
        }
    }

    /// Creates a new basic color modifiers with HSV modifications.
    pub const fn new_with_hsv_modifiers(
        hsv: Vec3,
        alpha: f32,
        contrast: f32,
        exposure: f32,
        gamma: f32,
    ) -> Self {
        Self {
            rgb_or_hsv: hsv,
            alpha,
            contrast,
            exposure,
            gamma,
            _padding: 0,
        }
    }
}

impl Default for BasicColorModifiersPod {
    fn default() -> Self {
        Self {
            rgb_or_hsv: Vec3::new(0.0, 1.0, 1.0),
            alpha: 1.0,
            contrast: 0.0,
            exposure: 0.0,
            gamma: 1.0,
            _padding: 0,
        }
    }
}

/// The transform flags buffer for the [`BasicModifiers`](crate::BasicModifiers).
#[derive(Debug)]
pub struct TransformFlagsBuffer(wgpu::Buffer);

impl TransformFlagsBuffer {
    /// Create a new transform flags buffer.
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Transform Flags Buffer"),
            size: std::mem::size_of::<u32>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self(buffer)
    }

    /// Update the transform flags buffer.
    pub fn update(&self, queue: &wgpu::Queue, flags: TransformFlags) {
        queue.write_buffer(&self.0, 0, bytemuck::bytes_of(&flags.bits()));
    }
}

impl BufferWrapper for TransformFlagsBuffer {
    fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

bitflags::bitflags! {
    /// Flags for whether to apply model and Gaussian transforms.
    #[repr(transparent)]
    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
    pub struct TransformFlags: u32 {
        /// Whether to apply the model transform.
        const MODEL = 0b0001;
        /// Whether to apply the Gaussian transform.
        const GAUSSIAN = 0b0010;
    }
}

/// The scale rotation buffer for the [`BasicModifiers`](crate::BasicModifiers).
#[derive(Debug)]
pub struct ScaleRotationBuffer(wgpu::Buffer);

impl ScaleRotationBuffer {
    /// Create a new scale and rotation buffer.
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Scale Rotation Buffer"),
            contents: bytemuck::bytes_of(&Mat3::IDENTITY),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self(buffer)
    }

    /// Update the scale and rotation buffer.
    pub fn update(&self, queue: &wgpu::Queue, scale_rotation: &Mat3) {
        queue.write_buffer(&self.0, 0, bytemuck::bytes_of(scale_rotation));
    }

    /// Update the scale and rotation buffer with scale and rotation.
    pub fn update_with_scale_rotation(&self, queue: &wgpu::Queue, scale: Vec3, rotation: Quat) {
        self.update(
            queue,
            &Mat3::from_mat4(Mat4::from_scale_rotation_translation(
                scale,
                rotation,
                Vec3::ZERO,
            )),
        );
    }
}

impl BufferWrapper for ScaleRotationBuffer {
    fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}
