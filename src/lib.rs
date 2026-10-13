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
    BufferWrapper, GaussianPod, GaussianTransformBuffer, GaussiansBuffer, IterGaussian,
    ModelTransformBuffer, ShDegreeMismatchError,
};

/// An editor for Gaussians.
///
/// This enables the application of a sequence of [`Modifier`]s to the Gaussians by storing
/// the necessary buffers, including:
/// - [`ModelTransformBuffer`]
/// - [`GaussianTransformBuffer`]
/// - [`GaussiansBuffer`]
///
/// If you wish to manage these buffers yourself, you do not need to use this struct.
///
/// Note that some [`GaussianPod`] configurations may not be able to be downloaded after
/// modification, see documentations of [`core::GaussianShConfig`] and [`core::GaussianCov3dConfig`]
/// for details.
pub struct Editor<G: GaussianPod> {
    pub model_transform_buffer: ModelTransformBuffer,
    pub gaussian_transform_buffer: GaussianTransformBuffer,
    pub gaussians_buffer: GaussiansBuffer<G>,
}

impl<G: GaussianPod> Editor<G> {
    /// Create a new editor from typed or runtime-degree Gaussians.
    ///
    /// Returns an error if the source SH degree does not match the storage degree of `G`.
    pub fn new(
        device: &wgpu::Device,
        gaussians: &impl IterGaussian,
    ) -> Result<Self, ShDegreeMismatchError> {
        log::debug!("Creating gaussians buffer");
        let gaussians_buffer = GaussiansBuffer::try_new_with_usage(
            device,
            gaussians,
            GaussiansBuffer::<G>::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
        )?;

        log::debug!("Creating model transform buffer");
        let model_transform_buffer = ModelTransformBuffer::new(device);

        log::debug!("Creating gaussian transform buffer");
        let gaussian_transform_buffer = GaussianTransformBuffer::new(device);

        log::debug!("Basic editor created");

        Ok(Self {
            model_transform_buffer,
            gaussian_transform_buffer,
            gaussians_buffer,
        })
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
