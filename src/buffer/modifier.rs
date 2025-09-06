use glam::*;
use wgpu::util::DeviceExt;

use crate::core::{self, BufferWrapper, FixedSizeBufferWrapper};

/// The basic color modifiers buffer for the [`BasicModifiersBundle`](crate::BasicModifiersBundle).
///
/// This buffer holds the data for basic color modifications, including RGB override or
/// HSV modifications, alpha, contrast, exposure, and gamma adjustments.
#[derive(Debug)]
pub struct BasicColorModifiersBuffer(wgpu::Buffer);

impl BasicColorModifiersBuffer {
    /// Create a new basic color modifiers buffer.
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Basic Color Modifiers Buffer"),
            contents: bytemuck::bytes_of(&BasicColorModifiersPod::default()),
            usage: Self::DEFAULT_USAGES,
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

impl From<BasicColorModifiersBuffer> for wgpu::Buffer {
    fn from(wrapper: BasicColorModifiersBuffer) -> Self {
        wrapper.0
    }
}

impl TryFrom<wgpu::Buffer> for BasicColorModifiersBuffer {
    type Error = core::FixedSizeBufferWrapperError;

    fn try_from(buffer: wgpu::Buffer) -> Result<Self, Self::Error> {
        Self::verify_buffer_size(&buffer).map(|()| Self(buffer))
    }
}

impl FixedSizeBufferWrapper for BasicColorModifiersBuffer {
    type Pod = BasicColorModifiersPod;
}

/// The POD representation of the basic color modifiers buffer.
///
/// ## RGB Override or HSV Modifications
///
/// If any value of `rgb_or_hsv` is negative, then it is used to override the RGB color,
/// otherwise it is used to apply HSV modifications.
///
/// For example, to override the color to red, use `rgb_or_hsv = Vec3::new(-1.0, 0.0, 0.0)`,
/// to apply a hue shift of 180 degrees, use `rgb_or_hsv = Vec3::new(0.5, 1.0, 1.0)`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BasicColorModifiersPod {
    /// If any value of this vector is negative, then it is used to override the RGB color,
    /// otherwise it is used to apply HSV modifications.
    pub rgb_or_hsv: Vec3,
    pub alpha: f32,
    pub contrast: f32,
    pub exposure: f32,
    pub gamma: f32,
    pub _padding: u32,
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

/// The transform flags buffer for the [`BasicModifiersBundle`](crate::BasicModifiersBundle).
#[derive(Debug)]
pub struct TransformFlagsBuffer(wgpu::Buffer);

impl TransformFlagsBuffer {
    /// Create a new transform flags buffer.
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Transform Flags Buffer"),
            size: std::mem::size_of::<u32>() as wgpu::BufferAddress,
            usage: Self::DEFAULT_USAGES,
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

impl From<TransformFlagsBuffer> for wgpu::Buffer {
    fn from(wrapper: TransformFlagsBuffer) -> Self {
        wrapper.0
    }
}

impl TryFrom<wgpu::Buffer> for TransformFlagsBuffer {
    type Error = core::FixedSizeBufferWrapperError;

    fn try_from(buffer: wgpu::Buffer) -> Result<Self, Self::Error> {
        Self::verify_buffer_size(&buffer).map(|()| Self(buffer))
    }
}

impl FixedSizeBufferWrapper for TransformFlagsBuffer {
    type Pod = u32;
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

/// The scale rotation buffer for the [`BasicModifiersBundle`](crate::BasicModifiersBundle).
///
/// This buffer holds the per Gaussian rotation and scale modification data.
#[derive(Debug)]
pub struct RotScaleBuffer(wgpu::Buffer);

impl RotScaleBuffer {
    /// Create a new scale and rotation buffer.
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Scale Rot Buffer"),
            contents: bytemuck::bytes_of(&RotScalePod::default()),
            usage: Self::DEFAULT_USAGES,
        });

        Self(buffer)
    }

    /// Update the scale and rotation buffer.
    pub fn update(&self, queue: &wgpu::Queue, rot: Quat, scale: Vec3) {
        self.update_with_pod(queue, &RotScalePod::new(rot, scale));
    }

    /// Update the scale and rotation buffer with [`RotScalePod`].
    pub fn update_with_pod(&self, queue: &wgpu::Queue, pod: &RotScalePod) {
        queue.write_buffer(&self.0, 0, bytemuck::bytes_of(pod));
    }
}

impl BufferWrapper for RotScaleBuffer {
    fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

impl From<RotScaleBuffer> for wgpu::Buffer {
    fn from(wrapper: RotScaleBuffer) -> Self {
        wrapper.0
    }
}

impl TryFrom<wgpu::Buffer> for RotScaleBuffer {
    type Error = core::FixedSizeBufferWrapperError;

    fn try_from(buffer: wgpu::Buffer) -> Result<Self, Self::Error> {
        Self::verify_buffer_size(&buffer).map(|()| Self(buffer))
    }
}

impl FixedSizeBufferWrapper for RotScaleBuffer {
    type Pod = RotScalePod;
}

/// The POD representation of the scale and rotation buffer.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RotScalePod {
    pub rot: Quat,
    pub scale: Vec3,
    pub _padding: u32,
}

impl RotScalePod {
    /// Create a new scale and rotation pod.
    pub fn new(rot: Quat, scale: Vec3) -> Self {
        Self {
            rot,
            scale,
            _padding: 0,
        }
    }
}

impl Default for RotScalePod {
    fn default() -> Self {
        Self {
            rot: Quat::IDENTITY,
            scale: Vec3::ONE,
            _padding: 0,
        }
    }
}
