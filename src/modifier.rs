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
/// easily create custom modifiers using closures.
///
/// [`Editor`](crate::Editor) also provides an `apply` method which takes a slice of
/// [`Modifier`]s to apply them in sequence to the stored Gaussians.
///
/// ## Usage
///
/// There are many ways to use this but the recommended way is to implement this trait for a closure
/// which dispatch a [`ComputeBundle`].
///
/// ```rust
/// // Create the modifier compute bundle
/// let my_modifier_bundle = ComputeBundleBuilder::new()
///     .label("My Modifier")
///     .bind_group_layouts([
///         &MODIFIER_GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR,
///         &MY_CUSTOM_BIND_GROUP_LAYOUT_DESCRIPTOR, // Put your custom bind group layout here.
///     ])
///     .resolver({
///         let mut resolver = wesl::PkgResolver::new();
///         resolver.add_package(&core::shader::PACKAGE); // Required for using core buffer structs.
///         resolver.add_package(&shader::PACKAGE); // Optionally add this for some utility functions.
///         resolver
///     })
///     .main_shader("path::to::my::wesl::module".parse().unwrap())
///     .entry_point("main")
///     .wesl_compile_options(wesl::CompileOptions {
///         features: G::wesl_features(), // Required for enabling the correct features for core struct.
///         ..Default::default()
///     })
///     .build(
///         &device,
///         [
///             [
///                 gaussians_buffer.buffer().as_entire_binding(),
///                 model_transform_buffer.buffer().as_entire_binding(),
///                 gaussian_transform_buffer.buffer().as_entire_binding(),
///             ],
///             [ /* Your custom bind group resources */],
///         ],
///     )
///     .map_err(|e| log::error!("{e}"))
///     .expect("my modifier bundle");
///
/// // Create the modifier closure
/// let my_modifier =
///     move |device: &wgpu::Device,
///             encoder: &mut wgpu::CommandEncoder,
///             gaussians: &GaussiansBuffer<G>,
///             model_transform: &ModelTransformBuffer,
///             gaussian_transform: &GaussianTransformBuffer| {
///         my_modifier_bundle.dispatch(encoder, gaussians.len() as u32);
///     };
///
/// // Apply the modifier using an editor as an example
/// let editor = Editor::new(&device, &gaussians);
/// editor.apply(&device, &mut encoder, [&my_modifier as &dyn gs::Modifier<GaussianPod>]);
/// ```
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

/// A specialized [`ComputeBundle`] for some built-in basic modifiers.
///
/// This bundle includes the modifiers for [`BasicColorModifiersBuffer`],
/// [`RotScaleBuffer`], and [`TransformFlagsBuffer`] (which provides flags for applying
/// [`core::ModelTransformBuffer`] and [`core::GaussianTransformBuffer`]).
#[derive(Debug)]
pub struct BasicModifiersBundle<B = wgpu::BindGroup> {
    bundle: ComputeBundle<B>,
    has_selection: bool,
}

impl<B> BasicModifiersBundle<B> {
    /// Gets the inner [`ComputeBundle`].
    pub fn bundle(&self) -> &ComputeBundle<B> {
        &self.bundle
    }

    /// Checks if this bundle takes a selection buffer to selectively apply modifiers.
    pub fn has_selection(&self) -> bool {
        self.has_selection
    }
}

impl BasicModifiersBundle {
    /// The bind group layout descriptor for the basic modifiers with a selection buffer.
    ///
    /// Thie bind group layout takes the following buffers:
    /// - [`TransformFlagsBuffer`]
    /// - [`BasicColorModifiersBuffer`]
    /// - [`RotScaleBuffer`]
    /// - [`SelectionBuffer`]
    ///
    /// This is at group 1, because group 0 is the [`MODIFIER_GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR`].
    pub const BIND_GROUP_LAYOUT_DESCRIPTOR_WITH_SELECTION: wgpu::BindGroupLayoutDescriptor<
        'static,
    > = wgpu::BindGroupLayoutDescriptor {
        label: Some("Basic Modifiers Bind Group Layout"),
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

    /// The bind group layout descriptor for the basic modifiers without a selection buffer.
    ///
    /// This bind group layout takes the following buffers:
    /// - [`TransformFlagsBuffer`]
    /// - [`BasicColorModifiersBuffer`]
    /// - [`RotScaleBuffer`]
    ///
    /// This is at group 1, because group 0 is the [`MODIFIER_GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR`].
    pub const BIND_GROUP_LAYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor<'static> =
        wgpu::BindGroupLayoutDescriptor {
            label: Some("Basic Modifiers Bind Group Layout"),
            entries: &Self::BIND_GROUP_LAYOUT_DESCRIPTOR_WITH_SELECTION
                .entries
                .split_at(3)
                .0,
        };

    /// Creates a new [`BasicModifiersBundle`] bundle.
    pub fn new<G: GaussianPod>(
        device: &wgpu::Device,
        gaussians_buffer: &GaussiansBuffer<G>,
        model_transform_buffer: &ModelTransformBuffer,
        gaussian_transform_buffer: &GaussianTransformBuffer,
        transform_flags_buffer: &TransformFlagsBuffer,
        basic_color_modifiers_buffer: &BasicColorModifiersBuffer,
        rot_scale_buffer: &RotScaleBuffer,
    ) -> Self {
        Self::create_bundle_builder::<G>(false)
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
                has_selection: false,
            })
            .map_err(|e| log::error!("{e}"))
            .expect("basic modifiers bundle")
    }

    /// Creates a new [`BasicModifiersBundle`] bundle with selection buffer.
    pub fn new_with_selection<G: GaussianPod>(
        device: &wgpu::Device,
        gaussians_buffer: &GaussiansBuffer<G>,
        model_transform_buffer: &ModelTransformBuffer,
        gaussian_transform_buffer: &GaussianTransformBuffer,
        transform_flags_buffer: &TransformFlagsBuffer,
        basic_color_modifiers_buffer: &BasicColorModifiersBuffer,
        rot_scale_buffer: &RotScaleBuffer,
        selection_buffer: &SelectionBuffer,
    ) -> Self {
        Self::create_bundle_builder::<G>(true)
            .build(
                &device,
                [
                    [
                        gaussians_buffer.buffer().as_entire_binding(),
                        model_transform_buffer.buffer().as_entire_binding(),
                        gaussian_transform_buffer.buffer().as_entire_binding(),
                    ]
                    .to_vec(),
                    [
                        transform_flags_buffer.buffer().as_entire_binding(),
                        basic_color_modifiers_buffer.buffer().as_entire_binding(),
                        rot_scale_buffer.buffer().as_entire_binding(),
                        selection_buffer.buffer().as_entire_binding(),
                    ]
                    .to_vec(),
                ],
            )
            .map(|bundle| Self {
                bundle,
                has_selection: false,
            })
            .map_err(|e| log::error!("{e}"))
            .expect("basic modifiers bundle")
    }

    /// Apply the basic modifiers to the Gaussians.
    pub fn apply_with_count(&self, encoder: &mut wgpu::CommandEncoder, gaussian_count: u32) {
        self.bundle().dispatch(encoder, gaussian_count);
    }

    /// Creates a new [`ComputeBundleBuilder`] for the basic modifiers.
    ///
    /// This is usually not called directly, but used internally to create the bundle.
    pub fn create_bundle_builder<'a, G: GaussianPod>(
        has_selection: bool,
    ) -> ComputeBundleBuilder<'a, wesl::PkgResolver> {
        ComputeBundleBuilder::new()
            .label("Basic Modifiers")
            .bind_group_layouts([
                &MODIFIER_GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR,
                match has_selection {
                    true => &Self::BIND_GROUP_LAYOUT_DESCRIPTOR_WITH_SELECTION,
                    false => &Self::BIND_GROUP_LAYOUT_DESCRIPTOR,
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
                        .chain(std::iter::once(("selection_buffer", has_selection.into())))
                        .map(|(k, v)| (k.to_string(), v.into()))
                        .collect(),
                    ..Default::default()
                },
                ..Default::default()
            })
    }
}

impl<G: GaussianPod> Modifier<G> for BasicModifiersBundle {
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

impl BasicModifiersBundle<()> {
    /// Creates a new [`BasicModifiersBundle`] bundle without a bind group.
    pub fn new_without_bind_group<G: GaussianPod>(device: &wgpu::Device) -> Self {
        BasicModifiersBundle::create_bundle_builder::<G>(false)
            .build_without_bind_groups(&device)
            .map(|bundle| Self {
                bundle,
                has_selection: false,
            })
            .expect("basic modifiers bundle")
    }

    /// Creates a new [`BasicModifiersBundle`] bundle without a bind group with selection buffer.
    pub fn new_without_bind_group_with_selection<G: GaussianPod>(device: &wgpu::Device) -> Self {
        BasicModifiersBundle::create_bundle_builder::<G>(true)
            .build_without_bind_groups(&device)
            .map(|bundle| Self {
                bundle,
                has_selection: true,
            })
            .expect("basic modifiers bundle")
    }

    /// Apply the basic modifiers to the Gaussians.
    ///
    /// - `gaussians_bind_group` is the bind group created from [`MODIFIER_GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR`].
    /// - `bind_group` is the bind group created from [`BasicModifiersBundle::BIND_GROUP_LAYOUT_DESCRIPTOR`].
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
