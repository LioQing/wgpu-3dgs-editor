use glam::*;
use wgpu::util::DeviceExt;

use crate::core::{self, BufferWrapper, FixedSizeBufferWrapper};

/// The selection storage buffer.
///
/// This buffer holds the selection bitmask for Gaussians, where each bit represents whether
/// a Gaussian is selected (1) or not (0).
#[derive(Debug, Clone)]
pub struct SelectionBuffer(wgpu::Buffer);

impl SelectionBuffer {
    /// Create a new selection buffer.
    pub fn new(device: &wgpu::Device, gaussian_count: u32) -> Self {
        Self::new_with_label(device, "", gaussian_count)
    }

    /// Create a new selection buffer with additional label.
    pub fn new_with_label(device: &wgpu::Device, label: &str, gaussian_count: u32) -> Self {
        let size = gaussian_count.div_ceil(32) * std::mem::size_of::<u32>() as u32;

        let data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(format!("{label} Selection Buffer").as_str()),
            size: size as wgpu::BufferAddress,
            usage: Self::DEFAULT_USAGES,
            mapped_at_creation: false,
        });

        Self(data)
    }
}

impl BufferWrapper for SelectionBuffer {
    const DEFAULT_USAGES: wgpu::BufferUsages = wgpu::BufferUsages::from_bits_retain(
        wgpu::BufferUsages::STORAGE.bits()
            | wgpu::BufferUsages::COPY_DST.bits()
            | wgpu::BufferUsages::COPY_SRC.bits(),
    );

    fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

impl From<SelectionBuffer> for wgpu::Buffer {
    fn from(wrapper: SelectionBuffer) -> Self {
        wrapper.0
    }
}

impl From<wgpu::Buffer> for SelectionBuffer {
    fn from(buffer: wgpu::Buffer) -> Self {
        Self(buffer)
    }
}

/// The selection operation uniform buffer for storing selection operations.
///
/// This buffer holds a single u32 value representing the index of the selection operation to be
/// performed, this includes both primitive operations and custom operations, the value is obtained
/// from [`SelectionExpr::as_u32`](crate::SelectionExpr::as_u32), see its documentation for more
/// details.
#[derive(Debug, Clone)]
pub struct SelectionOpBuffer(wgpu::Buffer);

impl SelectionOpBuffer {
    /// Create a new selection operation buffer.
    pub fn new(device: &wgpu::Device, op: u32) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Selection Operation Buffer"),
            contents: bytemuck::bytes_of(&op),
            usage: Self::DEFAULT_USAGES,
        });

        Self(buffer)
    }

    /// Update the selection operation buffer.
    pub fn update(&self, queue: &wgpu::Queue, op: u32) {
        queue.write_buffer(&self.0, 0, bytemuck::bytes_of(&op));
    }
}

impl BufferWrapper for SelectionOpBuffer {
    fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

impl From<SelectionOpBuffer> for wgpu::Buffer {
    fn from(wrapper: SelectionOpBuffer) -> Self {
        wrapper.0
    }
}

impl TryFrom<wgpu::Buffer> for SelectionOpBuffer {
    type Error = core::FixedSizeBufferWrapperError;

    fn try_from(buffer: wgpu::Buffer) -> Result<Self, Self::Error> {
        Self::verify_buffer_size(&buffer).map(|()| Self(buffer))
    }
}

impl FixedSizeBufferWrapper for SelectionOpBuffer {
    type Pod = u32;
}

/// An inverse transform uniform buffer for selection operations.
///
/// This buffer holds the inverse of the combined scale, rotation, and translation transform,
/// which is used for [`SelectionBundle::create_sphere_bundle`](crate::SelectionBundle::create_sphere_bundle)
/// and [`SelectionBundle::create_box_bundle`](crate::SelectionBundle::create_box_bundle), but
/// can also be used for your custom selection operations.
#[derive(Debug, Clone)]
pub struct InvTransformBuffer(wgpu::Buffer);

impl InvTransformBuffer {
    /// Create a new inverse transform buffer.
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Inverse Transform Buffer"),
            size: std::mem::size_of::<Mat4>() as wgpu::BufferAddress,
            usage: Self::DEFAULT_USAGES,
            mapped_at_creation: false,
        });

        Self(buffer)
    }

    /// Update the inverse transform buffer.
    pub fn update(&self, queue: &wgpu::Queue, inv_transform: Mat4) {
        queue.write_buffer(&self.0, 0, bytemuck::bytes_of(&inv_transform));
    }

    /// Update the sphere selection buffer with the scale, rotation, and position.
    pub fn update_with_scale_rot_pos(
        &self,
        queue: &wgpu::Queue,
        scale: Vec3,
        rot: Quat,
        pos: Vec3,
    ) {
        let inv_transform = Mat4::from_scale_rotation_translation(scale, rot, pos).inverse();
        self.update(queue, inv_transform);
    }
}

impl BufferWrapper for InvTransformBuffer {
    fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

impl From<InvTransformBuffer> for wgpu::Buffer {
    fn from(wrapper: InvTransformBuffer) -> Self {
        wrapper.0
    }
}

impl TryFrom<wgpu::Buffer> for InvTransformBuffer {
    type Error = core::FixedSizeBufferWrapperError;

    fn try_from(buffer: wgpu::Buffer) -> Result<Self, Self::Error> {
        Self::verify_buffer_size(&buffer).map(|()| Self(buffer))
    }
}

impl FixedSizeBufferWrapper for InvTransformBuffer {
    type Pod = Mat4;
}
