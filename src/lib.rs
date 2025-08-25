#![doc = include_str!("../README.md")]

mod buffer;
mod error;
mod modifier;
mod non_destructive_modifier;
mod selection;
mod selection_modifier;
pub mod shader;

pub use buffer::*;
pub use error::*;
pub use modifier::*;
pub use non_destructive_modifier::*;
pub use selection::*;
pub use selection_modifier::*;

pub use wgpu_3dgs_core as core;

use wgpu_3dgs_core::{
    GaussianPod, GaussianTransformBuffer, Gaussians, GaussiansBuffer, ModelTransformBuffer,
};

/// An editor for Gaussians.
///
/// This enables the application of a sequence of [`Modifier`]s to the Gaussians.
///
/// Note that some [`GaussianPod`] configurations may not be able to be downloaded after modification.
pub struct Editor<G: GaussianPod> {
    pub model_transform_buffer: ModelTransformBuffer,
    pub gaussian_transform_buffer: GaussianTransformBuffer,
    pub gaussians_buffer: GaussiansBuffer<G>,
}

impl<G: GaussianPod> Editor<G> {
    /// Create a new basic editor.
    pub fn new(device: &wgpu::Device, gaussians: &Gaussians) -> Self {
        log::debug!("Creating model transform buffer");
        let model_transform_buffer = ModelTransformBuffer::new(device);

        log::debug!("Creating gaussian transform buffer");
        let gaussian_transform_buffer = GaussianTransformBuffer::new(device);

        log::debug!("Creating gaussians buffer");
        let gaussians_buffer = GaussiansBuffer::new_with_usage(
            device,
            &gaussians.gaussians,
            GaussiansBuffer::<G>::DEFAULT_USAGE | wgpu::BufferUsages::COPY_SRC,
        );

        log::debug!("Basic editor created");

        Self {
            model_transform_buffer,
            gaussian_transform_buffer,
            gaussians_buffer,
        }
    }

    /// Apply the modifiers to the Gaussians.
    pub fn apply<'a>(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        modifiers: impl IntoIterator<Item = &'a dyn Modifier<G>>,
    ) {
        for modifier in modifiers.into_iter() {
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
