use crate::core::BufferWrapper;

#[derive(Debug)]
pub struct BasicColorModifiersBuffer(wgpu::Buffer);

impl BufferWrapper for BasicColorModifiersBuffer {
    fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

#[derive(Debug)]
pub struct TransformFlagsBuffer(wgpu::Buffer);

impl BufferWrapper for TransformFlagsBuffer {
    fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

#[derive(Debug)]
pub struct ScaleRotationBuffer(wgpu::Buffer);

impl BufferWrapper for ScaleRotationBuffer {
    fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}
