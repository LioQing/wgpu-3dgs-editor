use crate::{
    BasicColorModifiersBuffer, RotScaleBuffer, SelectionBuffer, TransformFlagsBuffer,
    core::{
        self, BufferWrapper, ComputeBundle, ComputeBundleBuilder, GaussianPod,
        GaussianTransformBuffer, GaussiansBuffer, ModelTransformBuffer,
    },
    shader,
};

/// A trait to apply modifier to Gaussians.
///
/// ## Overview
///
/// This trait simply defines a method to apply modifications to a set of Gaussians stored in a
/// [`GaussiansBuffer`]. It makes it convenient for users to apply a sequence of modifications.
///
/// The trait is also blanket implemented for closures with the same signature, allowing users to
/// easily create modifier closures instead of having to define a modifier struct.
///
/// [`Editor`](crate::Editor) also provides an `apply` method which takes a slice of
/// [`Modifier`]s to apply them in sequence to the stored Gaussians.
///
/// ## Usage
///
/// There are many ways to use this but the recommended way is to implement this trait for a closure
/// which dispatch a [`ComputeBundle`].
///
/// ```rust no_run
/// # pollster::block_on(async {
/// # use wgpu_3dgs_editor::{
/// #     Editor, MODIFIER_GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR, Modifier,
/// #     core::{
/// #         self, BufferWrapper, GaussianPod as _, GaussianTransformBuffer,
/// #         GaussiansBuffer, ModelTransformBuffer, glam::*,
/// #     },
/// #     shader,
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
/// # const MY_CUSTOM_BIND_GROUP_LAYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor =
/// #     wgpu::BindGroupLayoutDescriptor {
/// #         label: Some("My Custom Bind Group Layout"),
/// #         entries: &[wgpu::BindGroupLayoutEntry {
/// #             binding: 0,
/// #             visibility: wgpu::ShaderStages::COMPUTE,
/// #             ty: wgpu::BindingType::Buffer {
/// #                 ty: wgpu::BufferBindingType::Uniform,
/// #                 has_dynamic_offset: false,
/// #                 min_binding_size: None,
/// #             },
/// #             count: None,
/// #         }],
/// #     };
/// #
/// # let my_buffer = device.create_buffer(&wgpu::BufferDescriptor {
/// #     label: Some("My Buffer"),
/// #     size: 4,
/// #     usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
/// #     mapped_at_creation: false,
/// # });
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
/// // Create the modifier compute bundle
/// let my_modifier_bundle = core::ComputeBundleBuilder::new()
///     .label("My Modifier")
///     .bind_group_layouts([
///         // For accessing Gaussians and transforms
///         &MODIFIER_GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR,
///         // Your custom bind group layout here
///         &MY_CUSTOM_BIND_GROUP_LAYOUT_DESCRIPTOR,
///     ])
///     .resolver({
///         let mut resolver =
///             wesl::StandardResolver::new("path/to/my/folder/containing/wesl");
///         // Required for using core buffer structs.
///         resolver.add_package(&core::shader::PACKAGE);
///         // Optionally add this for some utility functions.
///         resolver.add_package(&shader::PACKAGE);
///         resolver
///     })
///     .main_shader("package::my_wesl_filename".parse().unwrap())
///     .entry_point("main")
///     .wesl_compile_options(wesl::CompileOptions {
///         // Required for enabling the correct features for core struct.
///         features: GaussianPod::wesl_features(),
///         ..Default::default()
///     })
///     .build(
///         &device,
///         [
///             vec![
///                 editor.gaussians_buffer.buffer().as_entire_binding(),
///                 editor.model_transform_buffer.buffer().as_entire_binding(),
///                 editor.gaussian_transform_buffer.buffer().as_entire_binding(),
///             ],
///             vec![my_buffer.buffer().as_entire_binding()],
///         ],
///     )
///     .map_err(|e| log::error!("{e}"))
///     .expect("my modifier bundle");
///
/// // Create the modifier closure
/// // This function signature implements Modifier by default
/// let my_modifier =
///     |_device: &wgpu::Device,
///         encoder: &mut wgpu::CommandEncoder,
///         gaussians: &GaussiansBuffer<GaussianPod>,
///         _model_transform: &ModelTransformBuffer,
///         _gaussian_transform: &GaussianTransformBuffer| {
///         my_modifier_bundle.dispatch(encoder, gaussians.len() as u32);
///     };
///
/// # let mut encoder =
/// #     device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
///
/// // Apply the modifier using the editor
/// editor.apply(
///     &device,
///     &mut encoder,
///     [&my_modifier as &dyn Modifier<GaussianPod>],
/// );
/// # });
/// ```
///
/// ## Shader Format
///
/// You may copy and paste the following shader bindings for
/// [`MODIFIER_GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR`] into your custom selection operation
/// shader to ensure that the bindings are correct, then add your own bindings after that.
///
/// ```wgsl
/// import wgpu_3dgs_core::{
///     gaussian::Gaussian,
///     gaussian_transform::GaussianTransform,
///     model_transform::ModelTransform,
/// };
///
/// @group(0) @binding(0)
/// var<storage, read_write> gaussians: array<Gaussian>;
///
/// @group(0) @binding(1)
/// var<uniform> model_transform: ModelTransform;
///
/// @group(0) @binding(2)
/// var<uniform> gaussian_transform: GaussianTransform;
///
/// // Your custom bindings here...
/// //
/// // You may also apply modifier to selected gaussians only by adding:
/// // @group(1) @binding(N)
/// // var<storage, read> selection: array<u32>;
///
/// override workgroup_size: u32;
///
/// @compute @workgroup_size(workgroup_size)
/// fn main(@builtin(global_invocation_id) id: vec3<u32>) {
///     let index = id.x;
///
///     if index >= arrayLength(&gaussians) {
///         return;
///     }
///     
///     @if(/* using selection buffer */) {
///         let word_index = index / 32u;
///         let bit_index = index % 32u;
///         let bit_mask = 1u << bit_index;
///         if (selection[word_index] & bit_mask) == 0 {
///             return;
///         }
///     }
///
///     var gaussian = gaussians[index];
///
///     // Your custom modifier operation code here...
///
///     gaussians[index] = gaussian;
/// }
pub trait Modifier<G: GaussianPod> {
    /// Apply the modifier to the Gaussians.
    fn apply(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        gaussians: &GaussiansBuffer<G>,
        model_transform: &ModelTransformBuffer,
        gaussian_transform: &GaussianTransformBuffer,
    );
}

impl<
    G: GaussianPod,
    F: Fn(
        &wgpu::Device,
        &mut wgpu::CommandEncoder,
        &GaussiansBuffer<G>,
        &ModelTransformBuffer,
        &GaussianTransformBuffer,
    ),
> Modifier<G> for F
{
    fn apply(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        gaussians: &GaussiansBuffer<G>,
        model_transform: &ModelTransformBuffer,
        gaussian_transform: &GaussianTransformBuffer,
    ) {
        self(
            device,
            encoder,
            gaussians,
            model_transform,
            gaussian_transform,
        );
    }
}

/// The bind group layout descriptor for the Gaussians buffer, with the
/// model transform and Gaussian transform.
///
/// This bind group layout takes the following buffers:
/// - [`GaussiansBuffer`]
/// - [`ModelTransformBuffer`]
/// - [`GaussianTransformBuffer`]
///
/// This bind group is usually at group 0 for [`Modifier`]s.
pub const MODIFIER_GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor =
    wgpu::BindGroupLayoutDescriptor {
        label: Some("Modifier Gaussians Bind Group Layout"),
        entries: &[
            // Gaussians storage buffer
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Model transform uniform buffer
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Gaussian transform uniform buffer
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    };

/// A marker struct to indicate that a modifier takes a selection buffer.
#[derive(Debug)]
pub struct WithSelection;

/// A marker struct to indicate that a modifier does not take a selection buffer.
#[derive(Debug)]
pub struct NoSelection;

/// A specialized [`ComputeBundle`] for some built-in basic modifier.
///
/// This bundle includes the modifiers for [`BasicColorModifiersBuffer`],
/// [`RotScaleBuffer`], and [`TransformFlagsBuffer`] (which provides flags for applying
/// [`core::ModelTransformBuffer`] and [`core::GaussianTransformBuffer`]).
#[derive(Debug)]
pub struct BasicModifierBundle<G: GaussianPod, S = NoSelection, B = wgpu::BindGroup> {
    bundle: ComputeBundle<B>,
    gaussian_pod_marker: std::marker::PhantomData<G>,
    selection_marker: std::marker::PhantomData<S>,
}

impl<G: GaussianPod, S, B> BasicModifierBundle<G, S, B> {
    /// Gets the inner [`ComputeBundle`].
    pub fn bundle(&self) -> &ComputeBundle<B> {
        &self.bundle
    }
}

impl<G: GaussianPod> BasicModifierBundle<G> {
    /// The bind group layout descriptor for the [`BasicModifierBundle`].
    ///
    /// This bind group layout takes the following buffers:
    /// - [`TransformFlagsBuffer`]
    /// - [`BasicColorModifiersBuffer`]
    /// - [`RotScaleBuffer`]
    ///
    /// This is at group 1, because group 0 is the [`MODIFIER_GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR`].
    pub const BIND_GROUP_LAYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor<'static> =
        wgpu::BindGroupLayoutDescriptor {
            label: Some("Basic Modifier Bind Group Layout"),
            entries: &BasicModifierBundle::<G, WithSelection>::BIND_GROUP_LAYOUT_DESCRIPTOR
                .entries
                .split_at(3)
                .0,
        };

    /// Creates a new [`BasicModifierBundle`] bundle.
    pub fn new(
        device: &wgpu::Device,
        gaussians_buffer: &GaussiansBuffer<G>,
        model_transform_buffer: &ModelTransformBuffer,
        gaussian_transform_buffer: &GaussianTransformBuffer,
        transform_flags_buffer: &TransformFlagsBuffer,
        basic_color_modifiers_buffer: &BasicColorModifiersBuffer,
        rot_scale_buffer: &RotScaleBuffer,
    ) -> Self {
        Self::create_bundle_builder(false)
            .build(
                &device,
                [
                    [
                        gaussians_buffer.buffer().as_entire_binding(),
                        model_transform_buffer.buffer().as_entire_binding(),
                        gaussian_transform_buffer.buffer().as_entire_binding(),
                    ],
                    [
                        transform_flags_buffer.buffer().as_entire_binding(),
                        basic_color_modifiers_buffer.buffer().as_entire_binding(),
                        rot_scale_buffer.buffer().as_entire_binding(),
                    ],
                ],
            )
            .map(|bundle| Self {
                bundle,
                gaussian_pod_marker: std::marker::PhantomData,
                selection_marker: std::marker::PhantomData,
            })
            .map_err(|e| log::error!("{e}"))
            .expect("basic modifier bundle")
    }

    /// Creates a new [`ComputeBundleBuilder`] for the basic modifier.
    fn create_bundle_builder<'a>(
        has_selection: bool,
    ) -> ComputeBundleBuilder<'a, wesl::PkgResolver> {
        ComputeBundleBuilder::new()
            .label("Basic Modifier")
            .bind_group_layouts([
                &MODIFIER_GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR,
                match has_selection {
                    true => &BasicModifierBundle::<G, WithSelection>::BIND_GROUP_LAYOUT_DESCRIPTOR,
                    false => &BasicModifierBundle::<G>::BIND_GROUP_LAYOUT_DESCRIPTOR,
                },
            ])
            .resolver({
                let mut resolver = wesl::PkgResolver::new();
                resolver.add_package(&core::shader::PACKAGE);
                resolver.add_package(&shader::PACKAGE);
                resolver
            })
            .main_shader(
                "wgpu_3dgs_editor::modifier::basic"
                    .parse()
                    .expect("modifier::basic module path"),
            )
            .entry_point("main")
            .wesl_compile_options(wesl::CompileOptions {
                features: wesl::Features {
                    flags: G::features()
                        .into_iter()
                        .chain(std::iter::once(("selection_buffer", has_selection)))
                        .map(|(k, v)| (k.to_string(), v.into()))
                        .collect(),
                    ..Default::default()
                },
                ..Default::default()
            })
    }
}

impl<G: GaussianPod> BasicModifierBundle<G, WithSelection> {
    /// The bind group layout descriptor for the [`BasicModifierBundle`] with a [`SelectionBuffer`].
    ///
    /// Thie bind group layout takes the following buffers:
    /// - [`TransformFlagsBuffer`]
    /// - [`BasicColorModifiersBuffer`]
    /// - [`RotScaleBuffer`]
    /// - [`SelectionBuffer`]
    ///
    /// This is at group 1, because group 0 is the [`MODIFIER_GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR`].
    pub const BIND_GROUP_LAYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor<'static> =
        wgpu::BindGroupLayoutDescriptor {
            label: Some("Basic Modifier Bind Group Layout"),
            entries: &[
                // Transform flags uniform buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Basic color modifiers uniform buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Scale rotation uniform buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Selection buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        };

    /// Creates a new [`BasicModifierBundle`] bundle with [`SelectionBuffer`].
    pub fn new_with_selection(
        device: &wgpu::Device,
        gaussians_buffer: &GaussiansBuffer<G>,
        model_transform_buffer: &ModelTransformBuffer,
        gaussian_transform_buffer: &GaussianTransformBuffer,
        transform_flags_buffer: &TransformFlagsBuffer,
        basic_color_modifiers_buffer: &BasicColorModifiersBuffer,
        rot_scale_buffer: &RotScaleBuffer,
        selection_buffer: &SelectionBuffer,
    ) -> Self {
        BasicModifierBundle::<G>::create_bundle_builder(true)
            .build(
                &device,
                [
                    vec![
                        gaussians_buffer.buffer().as_entire_binding(),
                        model_transform_buffer.buffer().as_entire_binding(),
                        gaussian_transform_buffer.buffer().as_entire_binding(),
                    ],
                    vec![
                        transform_flags_buffer.buffer().as_entire_binding(),
                        basic_color_modifiers_buffer.buffer().as_entire_binding(),
                        rot_scale_buffer.buffer().as_entire_binding(),
                        selection_buffer.buffer().as_entire_binding(),
                    ],
                ],
            )
            .map(|bundle| Self {
                bundle,
                gaussian_pod_marker: std::marker::PhantomData,
                selection_marker: std::marker::PhantomData,
            })
            .map_err(|e| log::error!("{e}"))
            .expect("basic modifier bundle")
    }
}

impl<G: GaussianPod, S> BasicModifierBundle<G, S> {
    /// Apply the basic modifier to the Gaussians.
    pub fn apply_with_count(&self, encoder: &mut wgpu::CommandEncoder, gaussian_count: u32) {
        self.bundle().dispatch(encoder, gaussian_count);
    }
}

impl<G: GaussianPod, S> Modifier<G> for BasicModifierBundle<G, S> {
    fn apply(
        &self,
        _device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        gaussians: &GaussiansBuffer<G>,
        _model_transform: &ModelTransformBuffer,
        _gaussian_transform: &GaussianTransformBuffer,
    ) {
        self.apply_with_count(encoder, gaussians.len() as u32);
    }
}

impl<G: GaussianPod> BasicModifierBundle<G, NoSelection, ()> {
    /// Creates a new [`BasicModifierBundle`] bundle without a bind group.
    pub fn new_without_bind_group(device: &wgpu::Device) -> Self {
        BasicModifierBundle::<G>::create_bundle_builder(false)
            .build_without_bind_groups(&device)
            .map(|bundle| Self {
                bundle,
                gaussian_pod_marker: std::marker::PhantomData,
                selection_marker: std::marker::PhantomData,
            })
            .expect("basic modifier bundle")
    }
}

impl<G: GaussianPod> BasicModifierBundle<G, WithSelection, ()> {
    /// Creates a new [`BasicModifierBundle`] bundle without a bind group with selection buffer.
    pub fn new_without_bind_group_with_selection(device: &wgpu::Device) -> Self {
        BasicModifierBundle::<G>::create_bundle_builder(true)
            .build_without_bind_groups(&device)
            .map(|bundle| Self {
                bundle,
                gaussian_pod_marker: std::marker::PhantomData,
                selection_marker: std::marker::PhantomData,
            })
            .expect("basic modifier bundle")
    }
}

impl<G: GaussianPod, S> BasicModifierBundle<G, S, ()> {
    /// Apply the basic modifier to the Gaussians.
    ///
    /// - `gaussians_bind_group` is the bind group created from [`MODIFIER_GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR`].
    /// - `bind_group` is the bind group created from [`BasicModifierBundle::BIND_GROUP_LAYOUT_DESCRIPTOR`].
    pub fn apply_with_count<'a>(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gaussians_bind_group: &wgpu::BindGroup,
        bind_group: &wgpu::BindGroup,
        gaussian_count: u32,
    ) {
        self.bundle()
            .dispatch(encoder, gaussian_count, [gaussians_bind_group, bind_group]);
    }
}

/// A struct to handle basic modifier.
///
/// This modifier holds a [`BasicModifierBundle`] along with necessary buffers, and applies the
/// basic modifier.
#[derive(Debug)]
pub struct BasicModifier<G: GaussianPod, S = NoSelection> {
    pub transform_flags_buffer: TransformFlagsBuffer,
    pub basic_color_modifiers_buffer: BasicColorModifiersBuffer,
    pub rot_scale_buffer: RotScaleBuffer,
    pub modifier: BasicModifierBundle<G, S>,
}

impl<G: GaussianPod> BasicModifier<G> {
    /// Create a new basic modifier.
    pub fn new(
        device: &wgpu::Device,
        gaussians_buffer: &GaussiansBuffer<G>,
        model_transform_buffer: &ModelTransformBuffer,
        gaussian_transform_buffer: &GaussianTransformBuffer,
    ) -> Self {
        log::debug!("Creating transform flags buffer");
        let transform_flags_buffer = TransformFlagsBuffer::new(device);

        log::debug!("Creating basic color modifiers buffer");
        let basic_color_modifiers_buffer = BasicColorModifiersBuffer::new(device);

        log::debug!("Creating rotation scale buffer");
        let rot_scale_buffer = RotScaleBuffer::new(device);

        log::debug!("Creating basic modifier bundle");
        let modifier = BasicModifierBundle::new(
            device,
            gaussians_buffer,
            model_transform_buffer,
            gaussian_transform_buffer,
            &transform_flags_buffer,
            &basic_color_modifiers_buffer,
            &rot_scale_buffer,
        );

        log::debug!("Basic modifier created");

        Self {
            transform_flags_buffer,
            basic_color_modifiers_buffer,
            rot_scale_buffer,

            modifier,
        }
    }
}

impl<G: GaussianPod> BasicModifier<G, WithSelection> {
    /// Create a new basic modifier with selection.
    pub fn new_with_selection(
        device: &wgpu::Device,
        gaussians_buffer: &GaussiansBuffer<G>,
        model_transform_buffer: &ModelTransformBuffer,
        gaussian_transform_buffer: &GaussianTransformBuffer,
        selection_buffer: &SelectionBuffer,
    ) -> Self {
        log::debug!("Creating transform flags buffer");
        let transform_flags_buffer = TransformFlagsBuffer::new(device);

        log::debug!("Creating basic color modifiers buffer");
        let basic_color_modifiers_buffer = BasicColorModifiersBuffer::new(device);

        log::debug!("Creating rotation scale buffer");
        let rot_scale_buffer = RotScaleBuffer::new(device);

        log::debug!("Creating basic modifier bundle");
        let modifier = BasicModifierBundle::new_with_selection(
            device,
            gaussians_buffer,
            model_transform_buffer,
            gaussian_transform_buffer,
            &transform_flags_buffer,
            &basic_color_modifiers_buffer,
            &rot_scale_buffer,
            selection_buffer,
        );

        log::debug!("Basic modifier created");

        Self {
            transform_flags_buffer,
            basic_color_modifiers_buffer,
            rot_scale_buffer,

            modifier,
        }
    }
}

impl<G: GaussianPod, S> Modifier<G> for BasicModifier<G, S> {
    fn apply(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        gaussians: &GaussiansBuffer<G>,
        model_transform: &ModelTransformBuffer,
        gaussian_transform: &GaussianTransformBuffer,
    ) {
        self.modifier.apply(
            device,
            encoder,
            gaussians,
            model_transform,
            gaussian_transform,
        );
    }
}
