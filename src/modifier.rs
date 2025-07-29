use crate::{
    BasicColorModifiersBuffer, ScaleRotationBuffer, SelectionBuffer, TransformFlagsBuffer,
    core::{
        self, ComputeBundle, ComputeBundleBuilder, GaussianPod, GaussianTransformBuffer,
        GaussiansBuffer, ModelTransformBuffer, buffer_wrapper_arr,
    },
    shader,
};

/// The bind group layout descriptor for the Gaussians buffer, with the
/// model transform and Gaussian transform.
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
/// [`ScaleRotationBuffer`], and [`TransformFlagsBuffer`] (which provides flags for applying
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
        scale_rotation_buffer: &ScaleRotationBuffer,
    ) -> Self {
        Self::create_bundle_builder::<G>(false)
            .build(
                &device,
                [
                    buffer_wrapper_arr![
                        gaussians_buffer,
                        model_transform_buffer,
                        gaussian_transform_buffer,
                    ]
                    .to_vec(),
                    buffer_wrapper_arr![
                        transform_flags_buffer,
                        basic_color_modifiers_buffer,
                        scale_rotation_buffer,
                    ]
                    .to_vec(),
                ],
            )
            .map(|bundle| Self {
                bundle,
                has_selection: false,
            })
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
        scale_rotation_buffer: &ScaleRotationBuffer,
        selection_buffer: &SelectionBuffer,
    ) -> Self {
        Self::create_bundle_builder::<G>(true)
            .build(
                &device,
                [
                    buffer_wrapper_arr![
                        gaussians_buffer,
                        model_transform_buffer,
                        gaussian_transform_buffer,
                    ]
                    .to_vec(),
                    buffer_wrapper_arr![
                        transform_flags_buffer,
                        basic_color_modifiers_buffer,
                        scale_rotation_buffer,
                        selection_buffer,
                    ]
                    .to_vec(),
                ],
            )
            .map(|bundle| Self {
                bundle,
                has_selection: false,
            })
            .expect("basic modifiers bundle")
    }

    /// Apply the basic modifiers to the Gaussians.
    pub fn apply<'a>(&self, encoder: &mut wgpu::CommandEncoder, gaussian_count: u32) {
        self.bundle().dispatch(encoder, gaussian_count);
    }

    /// Creates a new [`ComputeBundleBuilder`] for the basic modifiers.
    ///
    /// This is usually not called directly, but used internally to create the bundle.
    pub fn create_bundle_builder<'a, G: GaussianPod>(
        has_selection: bool,
    ) -> ComputeBundleBuilder<'a, wesl::PkgResolver, wesl::ModulePath> {
        ComputeBundleBuilder::new()
            .label("Basic Modifiers")
            .bind_groups([
                &MODIFIER_GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR,
                match has_selection {
                    true => &Self::BIND_GROUP_LAYOUT_DESCRIPTOR_WITH_SELECTION,
                    false => &Self::BIND_GROUP_LAYOUT_DESCRIPTOR,
                },
            ])
            .resolver({
                let mut resolver = wesl::PkgResolver::new();
                resolver.add_package(&core::shader::Mod);
                resolver.add_package(&shader::Mod);
                resolver
            })
            .main_shader(wesl::ModulePath::from_path(
                "wgpu_3dgs_editor/modifiers/basic",
            ))
            .entry_point("main")
            .compile_options(wesl::CompileOptions {
                features: G::features()
                    .into_iter()
                    .chain(std::iter::once(("has_selection", has_selection)))
                    .map(|(k, v)| (k.to_string(), v))
                    .collect(),
                ..Default::default()
            })
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
    pub fn apply<'a>(
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
