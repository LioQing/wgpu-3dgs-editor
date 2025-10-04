use crate::{
    Modifier, NonDestructiveModifierCreateError,
    core::{
        BufferWrapper, GaussianPod, GaussianTransformBuffer, GaussiansBuffer,
        GaussiansBufferUpdateError, ModelTransformBuffer,
    },
};

/// A wrapper for any [`Modifier`] to not modify the original Gaussians.
///
/// This modifier stores a copy of the original Gaussians, whenever it is applied, it first
/// copies original Gaussians to the destination Gaussians buffer, then applies the inner modifier.
#[derive(Debug)]
pub struct NonDestructiveModifier<G: GaussianPod, M: Modifier<G>> {
    pub modifier: M,
    pub original_gaussians: GaussiansBuffer<G>,
}

impl<G: GaussianPod, M: Modifier<G>> NonDestructiveModifier<G, M> {
    /// Create a new non-destructive modifier.
    ///
    /// `gaussians` must have [`wgpu::BufferUsages::COPY_SRC`] in its usage.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        modifier: M,
        gaussians: &GaussiansBuffer<G>,
    ) -> Result<Self, NonDestructiveModifierCreateError> {
        if !gaussians
            .buffer()
            .usage()
            .contains(wgpu::BufferUsages::COPY_SRC)
        {
            return Err(NonDestructiveModifierCreateError::MissingCopySrcBufferUsage);
        }

        let original_gaussians = GaussiansBuffer::new_empty_with_usage(
            device,
            gaussians.len(),
            GaussiansBuffer::<G>::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });

        encoder.copy_buffer_to_buffer(
            gaussians.buffer(),
            0,
            original_gaussians.buffer(),
            0,
            gaussians.buffer().size(),
        );

        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::PollType::Wait)?;

        Ok(Self {
            modifier,
            original_gaussians,
        })
    }

    /// Try apply the modifier.
    ///
    /// Returns [`Err`] if the original Gaussians count does not match the target Gaussians count.
    pub fn try_apply(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        gaussians: &GaussiansBuffer<G>,
        model_transform: &ModelTransformBuffer,
        gaussian_transform: &GaussianTransformBuffer,
    ) -> Result<(), GaussiansBufferUpdateError> {
        self.try_apply_with(encoder, gaussians, |encoder, modifier, gaussians| {
            modifier.apply(
                device,
                encoder,
                gaussians,
                model_transform,
                gaussian_transform,
            );
        })
    }

    /// Apply the modifier with a function.
    ///
    /// This is convenient when you are not using the [`Modifier::apply`] function, and instead
    /// want to use other functions to apply the modifier.
    pub fn try_apply_with(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gaussians: &GaussiansBuffer<G>,
        f: impl FnOnce(&mut wgpu::CommandEncoder, &M, &GaussiansBuffer<G>),
    ) -> Result<(), GaussiansBufferUpdateError> {
        if self.original_gaussians.len() != gaussians.len() {
            return Err(GaussiansBufferUpdateError::CountMismatch {
                count: gaussians.len(),
                expected_count: self.original_gaussians.len(),
            });
        }

        encoder.copy_buffer_to_buffer(
            self.original_gaussians.buffer(),
            0,
            gaussians.buffer(),
            0,
            self.original_gaussians.buffer().size(),
        );

        f(encoder, &self.modifier, gaussians);

        Ok(())
    }
}

impl<G: GaussianPod, M: Modifier<G>> Modifier<G> for NonDestructiveModifier<G, M> {
    fn apply(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        gaussians: &GaussiansBuffer<G>,
        model_transform: &ModelTransformBuffer,
        gaussian_transform: &GaussianTransformBuffer,
    ) {
        self.try_apply(
            device,
            encoder,
            gaussians,
            model_transform,
            gaussian_transform,
        )
        .expect("apply non-destructive modifier")
    }
}
