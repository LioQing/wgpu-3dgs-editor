use glam::*;
use wgpu::util::DeviceExt;

use crate::Error;

/// The mask storage buffer for storing masked Gaussians as a bitvec.
#[derive(Debug, Clone)]
pub struct MaskBuffer {
    data: wgpu::Buffer,
    download: wgpu::Buffer,
}

impl MaskBuffer {
    /// Create a new mask buffer.
    pub fn new(device: &wgpu::Device, gaussian_count: u32) -> Self {
        Self::new_with_label(device, "", gaussian_count)
    }

    /// Create a new mask buffer with additional label.
    pub fn new_with_label(device: &wgpu::Device, label: &str, gaussian_count: u32) -> Self {
        let size = gaussian_count.div_ceil(32) * std::mem::size_of::<u32>() as u32;

        let data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(format!("Mask {label} Buffer").as_str()),
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

    /// Download the mask edit.
    pub async fn download(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<u32>, Error> {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Mask Download Encoder"),
        });
        self.prepare_download(&mut encoder);
        queue.submit(Some(encoder.finish()));

        self.map_download(device).await
    }

    /// Prepare for downloading the Gaussian edit.
    pub fn prepare_download(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_buffer_to_buffer(self.buffer(), 0, &self.download, 0, self.download.size());
    }

    /// Map the download buffer to read the Gaussian edit.
    pub async fn map_download(&self, device: &wgpu::Device) -> Result<Vec<u32>, Error> {
        let (tx, rx) = oneshot::channel();
        let buffer_slice = self.download.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            if let Err(e) = tx.send(result) {
                log::error!("Error occurred while sending Gaussian edit: {e:?}");
            }
        });
        device.poll(wgpu::PollType::Wait)?;
        rx.await??;

        let edits = bytemuck::allocation::pod_collect_to_vec(&buffer_slice.get_mapped_range());
        self.download.unmap();

        Ok(edits)
    }

    /// Get the buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.data
    }

    /// Get the download buffer.
    pub fn download_buffer(&self) -> &wgpu::Buffer {
        &self.download
    }
}

/// The mask shape.
#[derive(Debug, Clone)]
pub struct MaskShape {
    /// Kind.
    pub kind: MaskShapeKind,
    /// Position.
    pub pos: Vec3,
    /// Rotation.
    pub rotation: Quat,
    /// Scale.
    pub scale: Vec3,
}

impl MaskShape {
    /// Create a new mask shape.
    pub fn new(kind: MaskShapeKind) -> Self {
        Self {
            kind,
            pos: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }

    /// Convert to [`MaskOpShapePod`].
    pub fn to_mask_op_shape_pod(&self) -> MaskOpShapePod {
        match self.kind {
            MaskShapeKind::Box => MaskOpShapePod::box_shape(self.pos, self.rotation, self.scale),
            MaskShapeKind::Ellipsoid => {
                MaskOpShapePod::ellipsoid_shape(self.pos, self.rotation, self.scale)
            }
        }
    }
}

/// The mask shape uniform buffer for storing mask operation shape.
#[derive(Debug, Clone)]
pub struct MaskOpShapeBuffer(wgpu::Buffer);

impl MaskOpShapeBuffer {
    /// Create a new mask shape buffer.
    pub fn new(device: &wgpu::Device, mask_shape: &MaskOpShapePod) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mask Shape Buffer"),
            contents: bytemuck::bytes_of(mask_shape),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self(buffer)
    }

    /// Update the mask shapes buffer.
    pub fn update(&self, queue: &wgpu::Queue, mask_shapes: &MaskOpShapePod) {
        queue.write_buffer(&self.0, 0, bytemuck::bytes_of(mask_shapes));
    }

    /// Get the buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

/// The mask shape kinds.
#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MaskShapeKind {
    Box = 0,
    Ellipsoid = 1,
}

/// The POD representation of a mask operation shape for evaluation.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MaskOpShapePod {
    /// The mask shape kind.
    pub kind: u32,

    /// The padding.
    _padding: [u32; 3],

    /// The inverse transformation matrix.
    ///
    /// The world is transformed using this matrix,
    /// then according to the mask shape kind,
    /// the mask is applied using unit sphere or box.
    pub inv_transform: Mat4,
}

impl MaskOpShapePod {
    /// Create a new mask shape.
    pub const fn new(kind: MaskShapeKind, inv_transform: Mat4) -> Self {
        Self {
            kind: kind as u32,
            _padding: [0; 3],
            inv_transform,
        }
    }

    /// Create a new box mask shape with transform.
    pub fn box_shape_with_transform(transform: Mat4) -> Self {
        Self::new(MaskShapeKind::Box, transform.inverse())
    }

    /// Create a new ellipsoid mask shape.
    pub fn box_shape(pos: Vec3, rotation: Quat, scale: Vec3) -> Self {
        Self::box_shape_with_transform(Mat4::from_scale_rotation_translation(scale, rotation, pos))
    }

    /// Create a new ellipsoid mask shape with transform.
    pub fn ellipsoid_shape_with_transform(transform: Mat4) -> Self {
        Self::new(MaskShapeKind::Ellipsoid, transform.inverse())
    }

    /// Create a new ellipsoid mask shape.
    pub fn ellipsoid_shape(pos: Vec3, rotation: Quat, scale: Vec3) -> Self {
        Self::ellipsoid_shape_with_transform(Mat4::from_scale_rotation_translation(
            scale, rotation, pos,
        ))
    }
}

/// The mask operation uniform buffer for storing mask operations.
#[derive(Debug, Clone)]
pub struct MaskOpBuffer(wgpu::Buffer);

impl MaskOpBuffer {
    /// Create a new mask operation buffer.
    pub fn new(device: &wgpu::Device, mask_op: SelectionOp) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mask Operation Buffer"),
            contents: bytemuck::bytes_of(&(mask_op.as_u32())),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self(buffer)
    }

    /// Update the mask operation buffer.
    pub fn update(&self, queue: &wgpu::Queue, mask_op: SelectionOp) {
        queue.write_buffer(&self.0, 0, bytemuck::bytes_of(&(mask_op.as_u32())));
    }

    /// Get the buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

/// The selection operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionOp {
    Union,
    Intersection,
    SymmetricDifference,
    Difference,
    Complement,
    /// From either [`SelectionOpExpr::Unary`](crate::SelectionOpExpr::Unary) or
    /// [`SelectionOpExpr::Binary`](crate::SelectionOpExpr::Binary).
    Custom(u32),
}

impl SelectionOp {
    /// Get the selection operation as a u32.
    pub fn as_u32(&self) -> u32 {
        match self {
            SelectionOp::Union => 0,
            SelectionOp::Intersection => 1,
            SelectionOp::SymmetricDifference => 2,
            SelectionOp::Difference => 3,
            SelectionOp::Complement => 4,
            SelectionOp::Custom(op) => *op,
        }
    }
}
