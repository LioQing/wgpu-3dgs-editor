#![doc = include_str!("../README.md")]

mod buffer;
mod error;
mod modifier;
mod selection;
mod selection_modifier;
pub mod shader;

pub use buffer::*;
pub use error::*;
pub use modifier::*;
pub use selection::*;
pub use selection_modifier::*;

pub use wgpu_3dgs_core as core;

use wgpu_3dgs_core::{
    GaussianPod, GaussianTransformBuffer, Gaussians, GaussiansBuffer, ModelTransformBuffer,
};

/// An editor for Gaussians.
///
/// This enables the application of a sequence of [`Modifier`]s to the Gaussians.
pub struct Editor<G: GaussianPod> {
    pub model_transform_buffer: ModelTransformBuffer,
    pub gaussian_transform_buffer: GaussianTransformBuffer,
    pub gaussians_buffer: GaussiansBuffer<G>,

    pub modifiers: Vec<Box<dyn Modifier<G>>>,
}

impl<G: GaussianPod> Editor<G> {
    /// Create a new basic editor.
    pub fn new(
        device: &wgpu::Device,
        gaussians: &Gaussians,
        modifiers: impl IntoIterator<Item = impl Into<Box<dyn Modifier<G>>>>,
    ) -> Self {
        log::debug!("Creating model transform buffer");
        let model_transform_buffer = ModelTransformBuffer::new(device);

        log::debug!("Creating gaussian transform buffer");
        let gaussian_transform_buffer = GaussianTransformBuffer::new(device);

        log::debug!("Creating gaussians buffer");
        let gaussians_buffer = GaussiansBuffer::new(device, &gaussians.gaussians);

        log::debug!("Basic editor created");

        Self {
            model_transform_buffer,
            gaussian_transform_buffer,
            gaussians_buffer,

            modifiers: modifiers.into_iter().map(Into::into).collect(),
        }
    }

    /// Apply the modifiers to the Gaussians.
    pub fn apply(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        for modifier in &self.modifiers {
            modifier.apply(
                device,
                encoder,
                &self.gaussians_buffer,
                &self.model_transform_buffer,
                &self.gaussian_transform_buffer,
            );
        }
    }
}
