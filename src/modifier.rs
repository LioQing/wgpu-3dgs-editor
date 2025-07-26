use crate::{
    BasicColorModifiersBuffer, ScaleRotationBuffer, TransformFlagsBuffer,
    core::{
        self, ComputeBundle, ComputeBundleBuilder, GaussianPod, GaussianTransformBuffer,
        GaussiansBuffer, ModelTransformBuffer, buffer_wrapper_arr,
    },
    shader,
};

/// The bind group layout descriptor for the source and destination Gaussians buffers, with the
/// model transform and Gaussian transform.
pub const MODIFIER_GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor =
    wgpu::BindGroupLayoutDescriptor {
        label: Some("Modifier Gaussians Bind Group Layout"),
        entries: &[
            // Source Gaussian storage buffer
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Destination Gaussian storage buffer
            wgpu::BindGroupLayoutEntry {
                binding: 1,
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
                binding: 2,
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
                binding: 3,
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
pub struct BasicModifiers<B = wgpu::BindGroup>(ComputeBundle<B>);

impl<B> BasicModifiers<B> {
    /// Gets the inner [`ComputeBundle`].
    pub fn bundle(&self) -> &ComputeBundle<B> {
        &self.0
    }
}

impl BasicModifiers {
    /// The bind group layout descriptor for the basic modifiers.
    ///
    /// This is at group 1, because group 0 is the [`MODIFIER_GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR`].
    pub const BIND_GROUP_LAYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor<'static> =
        wgpu::BindGroupLayoutDescriptor {
            label: Some("Basic Modifiers Bind Group Layout"),
            entries: &[
                // Basic color modifiers uniform buffer
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
                // Transform flags uniform buffer
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
            ],
        };

    /// Creates a new [`BasicModifiers`] bundle.
    pub fn new<G: GaussianPod>(
        device: &wgpu::Device,
        source: &GaussiansBuffer<G>,
        dest: &GaussiansBuffer<G>,
        model_transform: &ModelTransformBuffer,
        gaussian_transform: &GaussianTransformBuffer,
        basic_color_modifiers_buffer: &BasicColorModifiersBuffer,
        transform_flags_buffer: &TransformFlagsBuffer,
        scale_rotation_buffer: &ScaleRotationBuffer,
    ) -> Self {
        Self::create_bundle_builder::<G>()
            .build(
                &device,
                [
                    buffer_wrapper_arr![source, dest, model_transform, gaussian_transform].to_vec(),
                    buffer_wrapper_arr![
                        basic_color_modifiers_buffer,
                        transform_flags_buffer,
                        scale_rotation_buffer,
                    ]
                    .to_vec(),
                ],
            )
            .map(Self)
            .expect("basic modifiers bundle")
    }

    fn create_bundle_builder<'a, G: GaussianPod>()
    -> ComputeBundleBuilder<'a, wesl::PkgResolver, wesl::ModulePath> {
        let mut resolver = wesl::PkgResolver::new();
        resolver.add_package(&core::shader::Mod);
        resolver.add_package(&shader::Mod);

        ComputeBundleBuilder::new()
            .label("Basic Modifiers")
            .bind_groups([
                &MODIFIER_GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR,
                &Self::BIND_GROUP_LAYOUT_DESCRIPTOR,
            ])
            .resolver(resolver)
            .main_shader(wesl::ModulePath::from_path(
                "wgpu_3dgs_editor/modifiers/basic",
            ))
            .entry_point("main")
            .compile_options(wesl::CompileOptions {
                features: G::features_map(),
                ..Default::default()
            })
    }
}

impl BasicModifiers<()> {
    /// Creates a new [`BasicModifiers`] bundle without a bind group.
    pub fn new_no_bind_group<G: GaussianPod>(device: &wgpu::Device) -> Self {
        BasicModifiers::create_bundle_builder::<G>()
            .build_without_bind_groups(&device)
            .map(Self)
            .expect("basic modifiers bundle")
    }
}
