use glam::*;
use wgpu::util::DeviceExt;

use crate::core::BufferWrapper;

/// The selection storage buffer for storing selected Gaussians as a bitvec.
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
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self(data)
    }
}

impl BufferWrapper for SelectionBuffer {
    fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

/// The selection operation uniform buffer for storing selection operations.
#[derive(Debug, Clone)]
pub struct SelectionOpBuffer(wgpu::Buffer);

impl SelectionOpBuffer {
    /// Create a new selection operation buffer.
    pub fn new(device: &wgpu::Device, op: u32) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mask Operation Buffer"),
            contents: bytemuck::bytes_of(&op),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
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
