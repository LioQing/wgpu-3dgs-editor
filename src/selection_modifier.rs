use crate::{
    BasicModifier, Modifier, SelectionBuffer, SelectionBundle, SelectionExpr, WithSelection,
    core::{
        ComputeBundle, GaussianPod, GaussianTransformBuffer, GaussiansBuffer, ModelTransformBuffer,
    },
};

/// A struct to handle custom selection and custom modifier together.
///
/// ## Overview
///
/// This modifier holdes a [`SelectionBundle`] and a [`Modifier`] along with necessary
/// buffers, and applies the selection followed by the basic modifier in order.
///
/// The [`Modifier`] can use the [`SelectionModifier::selection_buffer`] to determine which
/// Gaussians to modify.
///
/// ## Usage
///
/// You can supply your own selection bundles and modifier when creating a
/// [`SelectionModifier`].
///
/// The creation expects a modifier factory function instead of a modifier,
/// so that the modifier can be created with a reference to the selection buffer.
///
/// ```rust no_run
/// # use pollster::FutureExt;
/// #
/// # async {
/// # use wgpu_3dgs_editor::{
/// #     Editor, Modifier, SelectionBuffer, SelectionBundle, SelectionModifier,
/// #     core::{
/// #         self, BufferWrapper, glam::*,
/// #     },
/// # };
/// #
/// # type GaussianPod = core::GaussianPodWithShSingleCov3dSingleConfigs;
/// #
/// # let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
/// #
/// # let adapter = instance
/// #     .request_adapter(&wgpu::RequestAdapterOptions::default())
/// #     .await
/// #     .expect("adapter");
/// #
/// # let (device, _queue) = adapter
/// #     .request_device(&wgpu::DeviceDescriptor {
/// #         label: Some("Device"),
/// #         required_features: wgpu::Features::empty(),
/// #         required_limits: adapter.limits(),
/// #         memory_hints: wgpu::MemoryHints::default(),
/// #         trace: wgpu::Trace::Off,
/// #     })
/// #     .await
/// #     .expect("device");
/// #
/// // Create an editor that holds the buffers for the Gaussians and will apply the modifier
/// let editor = Editor::new(
///     &device,
///     &core::Gaussians {
///         gaussians: vec![core::Gaussian {
///             rot: Quat::IDENTITY,
///             pos: Vec3::ZERO,
///             color: U8Vec4::ZERO,
///             sh: [Vec3::ZERO; 15],
///             scale: Vec3::ONE,
///         }],
///     },
/// );
///
/// // Create your selection bundles
/// let selection_bundles = vec![
///     // The built-in sphere selection bundle as example
///     SelectionBundle::<GaussianPod>::create_sphere_bundle(&device),
/// ];
///
/// struct MyCustomModifier(core::ComputeBundle);
///
/// impl MyCustomModifier {
///     pub fn new(
///         device: &wgpu::Device,
///         // My buffers,
///         selection: &SelectionBuffer,
///     ) -> Self {
///         // Build your compute bundle here,
///         // and include the selection buffer to only modify selected Gaussians
///         let compute_bundle = core::ComputeBundleBuilder::new()
///             // Configure the modifier compute bundle here
///             .build(
///                 &device,
///                 [
///                     [selection.buffer().as_entire_binding()].to_vec(),
///                     [/* Your buffers */].to_vec(),
///                 ],
///             )
///             .unwrap();
///         Self(compute_bundle)
///     }
/// }
///
/// impl Modifier<GaussianPod> for MyCustomModifier {
///     fn apply(
///         &self,
///         _device: &wgpu::Device,
///         encoder: &mut wgpu::CommandEncoder,
///         gaussians: &core::GaussiansBuffer<GaussianPod>,
///         _model_transform: &core::ModelTransformBuffer,
///         _gaussian_transform: &core::GaussianTransformBuffer,
///     ) {
///         self.0.dispatch(encoder, gaussians.len() as u32);
///     }
/// }
///
/// let selection_modifier = SelectionModifier::new(
///     &device,
///     &editor.gaussians_buffer,
///     selection_bundles,
///     |selection_buffer| {
///         // The factory closure
///         MyCustomModifier::new(
///             &device,
///             // Your buffers,
///             selection_buffer,
///         )
///     },
/// );
/// # }
/// # .block_on();
/// ```
///
/// Alternatively, you can use a modifier closure instead of a struct (but be reminded this could
/// harm readability of your code).
///
/// ```rust no_run
/// # use pollster::FutureExt;
/// #
/// # async {
/// # use wgpu_3dgs_editor::{
/// #     Editor, Modifier, SelectionBuffer, SelectionBundle, SelectionModifier,
/// #     core::{
/// #         self, BufferWrapper, glam::*,
/// #     },
/// # };
/// #
/// # type GaussianPod = core::GaussianPodWithShSingleCov3dSingleConfigs;
/// #
/// # let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
/// #
/// # let adapter = instance
/// #     .request_adapter(&wgpu::RequestAdapterOptions::default())
/// #     .await
/// #     .expect("adapter");
/// #
/// # let (device, _queue) = adapter
/// #     .request_device(&wgpu::DeviceDescriptor {
/// #         label: Some("Device"),
/// #         required_features: wgpu::Features::empty(),
/// #         required_limits: adapter.limits(),
/// #         memory_hints: wgpu::MemoryHints::default(),
/// #         trace: wgpu::Trace::Off,
/// #     })
/// #     .await
/// #     .expect("device");
/// #
/// # let editor = Editor::new(
/// #     &device,
/// #     &core::Gaussians {
/// #         gaussians: vec![core::Gaussian {
/// #             rot: Quat::IDENTITY,
/// #             pos: Vec3::ZERO,
/// #             color: U8Vec4::ZERO,
/// #             sh: [Vec3::ZERO; 15],
/// #             scale: Vec3::ONE,
/// #         }],
/// #     },
/// # );
/// #
/// # let selection_bundles = vec![
/// #     // The built-in sphere selection bundle as example
/// #     SelectionBundle::<GaussianPod>::create_sphere_bundle(&device),
/// # ];
/// #
/// let selection_modifier = SelectionModifier::new(
///     &device,
///     &editor.gaussians_buffer,
///     selection_bundles,
///     |selection_buffer| {
///         // The factory closure
///         // Build your compute bundle here,
///         // and include the selection buffer to only modify selected Gaussians
///         let modifier_bundle = core::ComputeBundleBuilder::new()
///             .build(
///                 &device,
///                 [
///                     [selection_buffer.buffer().as_entire_binding()].to_vec(),
///                     [/* Your buffers */].to_vec(),
///                 ],
///             )
///             .unwrap();
///
///         // This function signature has blanket impl of the modifier trait
///         move |_device: &wgpu::Device,
///                 encoder: &mut wgpu::CommandEncoder,
///                 gaussians: &core::GaussiansBuffer<GaussianPod>,
///                 _model_transform: &core::ModelTransformBuffer,
///                 _gaussian_transform: &core::GaussianTransformBuffer| {
///             modifier_bundle.dispatch(encoder, gaussians.len() as u32);
///         }
///     },
/// );
/// # }
/// # .block_on();
/// ``````
#[derive(Debug)]
pub struct SelectionModifier<G: GaussianPod, M: Modifier<G>> {
    pub selection_expr: SelectionExpr,
    pub selection_buffer: SelectionBuffer,
    pub selection: SelectionBundle<G>,
    pub modifier: M,
}

impl<G: GaussianPod, M: Modifier<G>> SelectionModifier<G, M> {
    /// Create a new selection modifier.
    ///
    /// `bundles` are used for [`SelectionExpr::Unary`], [`SelectionExpr::Binary`], or
    /// [`SelectionExpr::Selection`], they must have the same bind group 0 as the
    /// [`SelectionBundle::GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR`], see documentation of
    /// [`SelectionBundle`] for more details.
    pub fn new(
        device: &wgpu::Device,
        gaussians_buffer: &GaussiansBuffer<G>,
        selection_bundles: Vec<ComputeBundle<()>>,
        modifier: impl FnOnce(&SelectionBuffer) -> M,
    ) -> Self {
        log::debug!("Creating selection buffer");
        let selection_buffer = SelectionBuffer::new(device, gaussians_buffer.len() as u32);

        log::debug!("Creating selection bundle");
        let selection = SelectionBundle::<G>::new(device, selection_bundles);

        log::debug!("Creating modifier");
        let modifier = modifier(&selection_buffer);

        log::debug!("Creating selection modifier");

        Self {
            selection_expr: SelectionExpr::default(),
            selection_buffer,
            selection,
            modifier,
        }
    }

    /// Apply the selection and the modifier with the given expression and buffer.
    ///
    /// [`SelectionModifier::apply`] is the equivalent of applying this function with
    /// [`SelectionModifier::selection_expr`] and [`SelectionModifier::selection_buffer`].
    pub fn apply_with(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        gaussians: &GaussiansBuffer<G>,
        model_transform: &ModelTransformBuffer,
        gaussian_transform: &GaussianTransformBuffer,
        selection_expr: &SelectionExpr,
        selection_buffer: &SelectionBuffer,
    ) {
        self.selection.evaluate(
            device,
            encoder,
            selection_expr,
            selection_buffer,
            model_transform,
            gaussian_transform,
            gaussians,
        );

        self.modifier.apply(
            device,
            encoder,
            gaussians,
            model_transform,
            gaussian_transform,
        );
    }
}

impl<G: GaussianPod, M: Modifier<G>> Modifier<G> for SelectionModifier<G, M> {
    fn apply(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        gaussians: &GaussiansBuffer<G>,
        model_transform: &ModelTransformBuffer,
        gaussian_transform: &GaussianTransformBuffer,
    ) {
        self.apply_with(
            device,
            encoder,
            gaussians,
            model_transform,
            gaussian_transform,
            &self.selection_expr,
            &self.selection_buffer,
        );
    }
}

/// A type alias of [`SelectionModifier`] with [`BasicModifier`].
pub type BasicSelectionModifier<G> = SelectionModifier<G, BasicModifier<G, WithSelection>>;

impl<G: GaussianPod> BasicSelectionModifier<G> {
    /// Create a new selection modifier with [`BasicModifier`].
    ///
    /// `bundles` are used for [`SelectionExpr::Unary`], [`SelectionExpr::Binary`], or
    /// [`SelectionExpr::Selection`], they must have the same bind group 0 as the
    /// [`SelectionBundle::GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR`], see documentation of
    /// [`SelectionBundle`] for more details.
    pub fn new_with_basic_modifier(
        device: &wgpu::Device,
        gaussians_buffer: &GaussiansBuffer<G>,
        model_transform: &ModelTransformBuffer,
        gaussian_transform: &GaussianTransformBuffer,
        selection_bundles: Vec<ComputeBundle<()>>,
    ) -> Self {
        Self::new(
            device,
            gaussians_buffer,
            selection_bundles,
            |selection_buffer| {
                BasicModifier::new_with_selection(
                    device,
                    gaussians_buffer,
                    model_transform,
                    gaussian_transform,
                    selection_buffer,
                )
            },
        )
    }
}
