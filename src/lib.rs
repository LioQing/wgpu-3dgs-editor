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
    ComputeBundle, GaussianPod, GaussianTransformBuffer, Gaussians, GaussiansBuffer,
    ModelTransformBuffer,
};

/// The basic editor.
#[derive(Debug)]
pub struct BasicEditor<G: GaussianPod> {
    pub model_transform_buffer: ModelTransformBuffer,
    pub gaussian_transform_buffer: GaussianTransformBuffer,
    pub gaussians_buffer: GaussiansBuffer<G>,
    pub selection_buffer: SelectionBuffer,
    pub transform_flags_buffer: TransformFlagsBuffer,
    pub basic_color_modifiers_buffer: BasicColorModifiersBuffer,
    pub scale_rotation_buffer: ScaleRotationBuffer,

    pub selection: SelectionBundle,
    pub modifiers: BasicModifiersBundle,
}

impl<G: GaussianPod> BasicEditor<G> {
    /// Create a new basic editor.
    pub fn new(
        device: &wgpu::Device,
        selection_ops: Vec<ComputeBundle<()>>,
        gaussians: &Gaussians,
    ) -> Self {
        log::debug!("Creating model transform buffer");
        let model_transform_buffer = ModelTransformBuffer::new(device);

        log::debug!("Creating gaussian transform buffer");
        let gaussian_transform_buffer = GaussianTransformBuffer::new(device);

        log::debug!("Creating gaussians buffer");
        let gaussians_buffer = GaussiansBuffer::new(device, &gaussians.gaussians);

        log::debug!("Creating selection buffer");
        let selection_buffer = SelectionBuffer::new(device, gaussians.gaussians.len() as u32);

        log::debug!("Creating transform flags buffer");
        let transform_flags_buffer = TransformFlagsBuffer::new(device);

        log::debug!("Creating basic color modifiers buffer");
        let basic_color_modifiers_buffer = BasicColorModifiersBuffer::new(device);

        log::debug!("Creating scale rotation buffer");
        let scale_rotation_buffer = ScaleRotationBuffer::new(device);

        log::debug!("Creating selection bundle");
        let selection = SelectionBundle::new::<G>(device, selection_ops);

        log::debug!("Creating basic modifiers bundle");
        let modifiers = BasicModifiersBundle::new(
            device,
            &gaussians_buffer,
            &model_transform_buffer,
            &gaussian_transform_buffer,
            &transform_flags_buffer,
            &basic_color_modifiers_buffer,
            &scale_rotation_buffer,
        );

        log::debug!("Basic editor created");

        Self {
            model_transform_buffer,
            gaussian_transform_buffer,
            gaussians_buffer,
            selection_buffer,
            transform_flags_buffer,
            basic_color_modifiers_buffer,
            scale_rotation_buffer,

            selection,
            modifiers,
        }
    }
}
