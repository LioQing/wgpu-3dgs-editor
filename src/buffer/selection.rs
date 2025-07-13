use glam::*;
use wgpu::util::DeviceExt;

use crate::{Error, core::BufferWrapper};

/// The selection storage buffer for storing selected Gaussians as a bitvec.
#[derive(Debug, Clone)]
pub struct SelectionBuffer {
    data: wgpu::Buffer,
    download: wgpu::Buffer,
}

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

        let download = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gaussians Edit Download Buffer"),
            size: size as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self { data, download }
    }

    /// Download the selections.
    pub async fn download(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<u32>, Error> {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Selection Download Encoder"),
        });
        self.prepare_download(&mut encoder);
        queue.submit(Some(encoder.finish()));

        self.map_download(device).await
    }

    /// Prepare for downloading the Gaussian selections.
    pub fn prepare_download(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_buffer_to_buffer(self.buffer(), 0, &self.download, 0, self.download.size());
    }

    /// Map the download buffer to read the Gaussian selections.
    pub async fn map_download(&self, device: &wgpu::Device) -> Result<Vec<u32>, Error> {
        let (tx, rx) = oneshot::channel();
        let buffer_slice = self.download.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            if let Err(e) = tx.send(result) {
                log::error!("Error occurred while sending Gaussian selection: {e:?}");
            }
        });
        device.poll(wgpu::PollType::Wait)?;
        rx.await??;

        let edits = bytemuck::allocation::pod_collect_to_vec(&buffer_slice.get_mapped_range());
        self.download.unmap();

        Ok(edits)
    }

    /// Get the download buffer.
    pub fn download_buffer(&self) -> &wgpu::Buffer {
        &self.download
    }
}

impl BufferWrapper for SelectionBuffer {
    fn buffer(&self) -> &wgpu::Buffer {
        &self.data
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
