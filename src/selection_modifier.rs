use crate::{
    BasicColorModifiersBuffer, BasicModifiersBundle, ScaleRotationBuffer, SelectionBuffer,
    SelectionBundle, SelectionExpr, TransformFlagsBuffer,
    core::{
        ComputeBundle, GaussianPod, GaussianTransformBuffer, GaussiansBuffer, ModelTransformBuffer,
    },
};

/// A struct to handle selection and modification together.
#[derive(Debug)]
pub struct SelectionModifier {
    pub selection_expr: SelectionExpr,
    pub selection_buffer: SelectionBuffer,
    pub transform_flags_buffer: TransformFlagsBuffer,
    pub basic_color_modifiers_buffer: BasicColorModifiersBuffer,
    pub scale_rotation_buffer: ScaleRotationBuffer,

    pub selection: SelectionBundle,
    pub modifiers: BasicModifiersBundle,
}

impl SelectionModifier {
    /// Create a new [`SelectionModifier`].
    ///
    /// `selection_bundles` are used for [`SelectionExpr::Unary`] or [`SelectionExpr::Binary`] and
    /// must have the same bind group 0 as the [`SelectionBundle::GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR`]
    /// (see [`SelectionBundle`] docs for more details).
    pub fn new<G: GaussianPod>(
        device: &wgpu::Device,
        gaussians_buffer: &GaussiansBuffer<G>,
        model_transform_buffer: &ModelTransformBuffer,
        gaussian_transform_buffer: &GaussianTransformBuffer,
        selection_bundles: Vec<ComputeBundle<()>>,
    ) -> Self {
        log::debug!("Creating selection buffer");
        let selection_buffer = SelectionBuffer::new(device, gaussians_buffer.len() as u32);

        log::debug!("Creating transform flags buffer");
        let transform_flags_buffer = TransformFlagsBuffer::new(device);

        log::debug!("Creating basic color modifiers buffer");
        let basic_color_modifiers_buffer = BasicColorModifiersBuffer::new(device);

        log::debug!("Creating scale rotation buffer");
        let scale_rotation_buffer = ScaleRotationBuffer::new(device);

        log::debug!("Creating selection modifier");
        let selection = SelectionBundle::new::<G>(device, selection_bundles);

        log::debug!("Creating basic modifiers bundle");
        let modifiers = BasicModifiersBundle::new(
            device,
            gaussians_buffer,
            model_transform_buffer,
            gaussian_transform_buffer,
            &transform_flags_buffer,
            &basic_color_modifiers_buffer,
            &scale_rotation_buffer,
        );

        log::debug!("Selection modifier created");

        Self {
            selection_expr: SelectionExpr::default(),
            selection_buffer,
            transform_flags_buffer,
            basic_color_modifiers_buffer,
            scale_rotation_buffer,

            selection,
            modifiers,
        }
    }

    /// Apply the selection modifiers to the Gaussians.
    pub fn apply<'a, G: GaussianPod>(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        gaussians: &GaussiansBuffer<G>,
        model_transform: &ModelTransformBuffer,
        gaussian_transform: &GaussianTransformBuffer,
    ) {
        self.selection.evaluate(
            device,
            encoder,
            &self.selection_expr,
            &self.selection_buffer,
            model_transform,
            gaussian_transform,
            gaussians,
        );

        self.modifiers.apply(encoder, gaussians.len() as u32);
    }
}
