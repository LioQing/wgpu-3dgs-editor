use glam::*;

use crate::{
    SelectionBuffer, SelectionOpBuffer,
    core::{
        self, BufferWrapper, ComputeBundle, ComputeBundleBuilder, GaussianPod,
        GaussianTransformBuffer, GaussiansBuffer, ModelTransformBuffer,
    },
    shader,
};

/// A selection expression tree.
///
/// ## Overview
///
/// This can be used to carry out operations on selection buffers, these operations are evaluated
/// by [`SelectionBundle::evaluate`] in a recursive manner (depth-first).
///
/// ## Custom Operations
///
/// [`SelectionExpr::Unary`], [`SelectionExpr::Binary`], and [`SelectionExpr::Selection`] are
/// custom operations that can be defined with additional [`ComputeBundle`]s, so they also
/// carry a vector of bind groups that are used in the operation when dispatched/evaluated.
/// These vectors should correspond to the selection bundle's bind groups starting at index 1,
/// because index 0 must be defined by [`SelectionBundle::GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR`].
#[derive(Debug, Default)]
pub enum SelectionExpr {
    /// Apply an identity operation.
    #[default]
    Identity,
    /// Union of the two selections.
    Union(Box<SelectionExpr>, Box<SelectionExpr>),
    /// Interaction of the two selections.
    Intersection(Box<SelectionExpr>, Box<SelectionExpr>),
    /// Difference of the two selections.
    Difference(Box<SelectionExpr>, Box<SelectionExpr>),
    /// Symmetric difference of the two selections.
    SymmetricDifference(Box<SelectionExpr>, Box<SelectionExpr>),
    /// Complement of the selection.
    Complement(Box<SelectionExpr>),
    /// Apply a custom unary operation.
    Unary(usize, Box<SelectionExpr>, Vec<wgpu::BindGroup>),
    /// Apply a custom binary operation.
    Binary(
        Box<SelectionExpr>,
        usize,
        Box<SelectionExpr>,
        Vec<wgpu::BindGroup>,
    ),
    /// Create a selection.
    Selection(usize, Vec<wgpu::BindGroup>),
    /// Directly use a selection buffer.
    Buffer(SelectionBuffer),
}

impl SelectionExpr {
    /// The first u32 value for a custom operation.
    pub const CUSTOM_OP_START: u32 = 5;

    /// Create a new [`SelectionExpr::Identity`].
    pub fn identity() -> Self {
        Self::Identity
    }

    /// Create a new [`SelectionExpr::Union`].
    pub fn union(self, other: Self) -> Self {
        Self::Union(Box::new(self), Box::new(other))
    }

    /// Create a new [`SelectionExpr::Intersection`].
    pub fn intersection(self, other: Self) -> Self {
        Self::Intersection(Box::new(self), Box::new(other))
    }

    /// Create a new [`SelectionExpr::Difference`].
    pub fn difference(self, other: Self) -> Self {
        Self::Difference(Box::new(self), Box::new(other))
    }

    /// Create a new [`SelectionExpr::SymmetricDifference`].
    pub fn symmetric_difference(self, other: Self) -> Self {
        Self::SymmetricDifference(Box::new(self), Box::new(other))
    }

    /// Create a new [`SelectionExpr::Complement`].
    pub fn complement(self) -> Self {
        Self::Complement(Box::new(self))
    }

    /// Create a new [`SelectionExpr::Unary`].
    pub fn unary(self, op: usize, bind_groups: Vec<wgpu::BindGroup>) -> Self {
        Self::Unary(op, Box::new(self), bind_groups)
    }

    /// Create a new [`SelectionExpr::Binary`].
    pub fn binary(self, op: usize, other: Self, bind_groups: Vec<wgpu::BindGroup>) -> Self {
        Self::Binary(Box::new(self), op, Box::new(other), bind_groups)
    }

    /// Create a new [`SelectionExpr::Selection`].
    pub fn selection(op: usize, bind_groups: Vec<wgpu::BindGroup>) -> Self {
        Self::Selection(op, bind_groups)
    }

    /// Create a new [`SelectionExpr::Buffer`].
    pub fn buffer(buffer: SelectionBuffer) -> Self {
        Self::Buffer(buffer)
    }

    /// Update the expression in place.
    pub fn update_with(&mut self, f: impl FnOnce(Self) -> Self) {
        *self = f(std::mem::take(self));
    }

    /// Get the u32 associated with this expression's operation.
    ///
    /// The value returned is not the same as that returned by [`SelectionExpr::custom_op_index`],
    /// but rather a value that can be used to identify the operation by the compute shader, custom
    /// operation's index are offset by [`SelectionExpr::CUSTOM_OP_START`].
    ///
    /// You usually do not need to use this method, it is used internally for evaluation of the
    /// compute shader.
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            SelectionExpr::Union(_, _) => Some(0),
            SelectionExpr::Intersection(_, _) => Some(1),
            SelectionExpr::SymmetricDifference(_, _) => Some(2),
            SelectionExpr::Difference(_, _) => Some(3),
            SelectionExpr::Complement(_) => Some(4),
            SelectionExpr::Unary(op, _, _) => Some(*op as u32 + Self::CUSTOM_OP_START),
            SelectionExpr::Binary(_, op, _, _) => Some(*op as u32 + Self::CUSTOM_OP_START),
            SelectionExpr::Selection(op, _) => Some(*op as u32 + Self::CUSTOM_OP_START),
            SelectionExpr::Buffer(_) => None,
            SelectionExpr::Identity => None,
        }
    }

    /// Whether this expression is an identity operation.
    pub fn is_identity(&self) -> bool {
        matches!(self, SelectionExpr::Identity)
    }

    /// Whether this expression is a primitive operation.
    pub fn is_primitive(&self) -> bool {
        matches!(
            self,
            SelectionExpr::Union(..)
                | SelectionExpr::Intersection(..)
                | SelectionExpr::Difference(..)
                | SelectionExpr::SymmetricDifference(..)
                | SelectionExpr::Complement(..)
        )
    }

    /// Whether this expression is a custom operation.
    pub fn is_custom(&self) -> bool {
        matches!(
            self,
            SelectionExpr::Unary(..) | SelectionExpr::Binary(..) | SelectionExpr::Selection(..)
        )
    }

    /// Whether this expression is a selection operation.
    pub fn is_operation(&self) -> bool {
        matches!(
            self,
            SelectionExpr::Union(..)
                | SelectionExpr::Intersection(..)
                | SelectionExpr::Difference(..)
                | SelectionExpr::SymmetricDifference(..)
                | SelectionExpr::Complement(..)
                | SelectionExpr::Unary(..)
                | SelectionExpr::Binary(..)
                | SelectionExpr::Selection(..)
        )
    }

    /// Whether this expression is a selection buffer.
    pub fn is_buffer(&self) -> bool {
        matches!(self, SelectionExpr::Buffer(_))
    }

    /// Get the custom operation index.
    ///
    /// This is the index of the custom operation in [`SelectionBundle::bundles`] vector.
    pub fn custom_op_index(&self) -> Option<usize> {
        match self {
            SelectionExpr::Unary(op, _, _)
            | SelectionExpr::Binary(_, op, _, _)
            | SelectionExpr::Selection(op, _) => Some(*op),
            _ => None,
        }
    }

    /// Get the custom operation bind groups for this expression.
    pub fn custom_bind_groups(&self) -> Option<&Vec<wgpu::BindGroup>> {
        match self {
            SelectionExpr::Unary(_, _, bind_groups) => Some(bind_groups),
            SelectionExpr::Binary(_, _, _, bind_groups) => Some(bind_groups),
            SelectionExpr::Selection(_, bind_groups) => Some(bind_groups),
            _ => None,
        }
    }

    /// Get the custom operation index and bind groups for this expression.
    pub fn custom_op_index_and_bind_groups(&self) -> Option<(usize, &Vec<wgpu::BindGroup>)> {
        match self {
            SelectionExpr::Unary(op, _, bind_groups)
            | SelectionExpr::Binary(_, op, _, bind_groups)
            | SelectionExpr::Selection(op, bind_groups) => Some((*op, bind_groups)),
            _ => None,
        }
    }
}

/// A collection of specialized [`ComputeBundle`] for selection operations.
///
/// ## Custom Operations
///
/// All [`ComputeBundle`]s supplied to this bundle as a [`SelectionExpr::Unary`],
/// [`SelectionExpr::Binary`], or [`SelectionExpr::Selection`] custom operation must have the same
/// bind group 0 as the [`SelectionBundle::GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR`]. They must also
/// not have the bind group itself, as it will be supplied automatically during evaluation.
///
/// Note that [`SelectionExpr::Unary`] will also get the source selection buffer, but it will be
/// empty (all zeros), you should operate on the destination selection buffer only.
///
/// It is recommended to use [`ComputeBundleBuilder`] to create the custom operation bundles,
/// and build them using [`ComputeBundleBuilder::build_without_bind_groups`].
///
/// ```rust no_run
/// # pollster::block_on(async {
/// # use wgpu_3dgs_editor::{
/// #     Editor, MODIFIER_GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR, Modifier, SelectionBuffer,
/// #     SelectionBundle, SelectionExpr,
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
/// # let my_existing_selection_buffer = SelectionBuffer::new(&device, 1024);
/// #
/// // Create an editor that holds the buffers for the Gaussians
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
/// // Create the selection custom operation compute bundle
/// let my_selection_custom_op_bundle = core::ComputeBundleBuilder::new()
///     .label("My Selection")
///     .bind_group_layouts([
///         &SelectionBundle::<GaussianPod>::GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR,
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
///     .build_without_bind_groups(&device)
///     .map_err(|e| log::error!("{e}"))
///     .expect("my selection custom op bundle");
///
/// // Create the selection bundle
/// let selection_bundle =
///     SelectionBundle::<GaussianPod>::new(&device, vec![my_selection_custom_op_bundle]);
///
/// // Create the bind group for your custom operation
/// let my_selection_custom_op_bind_group = selection_bundle.bundles[0]
///     .create_bind_group(
///         &device,
///         1, // Index 0 is always the Gaussians buffer
///         [my_buffer.buffer().as_entire_binding()],
///     )
///     .unwrap();
///
/// // Create the selection expression
/// let selection_expr = SelectionExpr::selection(
///     0, // The bundle index for your custom operation in the selection bundle
///     vec![my_selection_custom_op_bind_group],
/// )
/// .union(
///     // Combine with other selection expressions using different functions
///     // Here is an existing selection buffer for example
///     SelectionExpr::Buffer(my_existing_selection_buffer),
/// );
///
/// // Create a selection buffer for the result
/// let dest_selection_buffer =
///     SelectionBuffer::new(&device, editor.gaussians_buffer.len() as u32);
///
/// # let mut encoder =
/// #     device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
///
/// // Evaluate the selection expression
/// selection_bundle.evaluate(
///     &device,
///     &mut encoder,
///     &selection_expr,
///     &dest_selection_buffer,
///     &editor.model_transform_buffer,
///     &editor.gaussian_transform_buffer,
///     &editor.gaussians_buffer,
/// );
/// # });
/// ```
///
/// ## Shader Format
///
/// You may copy and paste the following shader bindings for
/// [`SelectionBundle::GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR`] into your custom selection operation
/// shader to ensure that the bindings are correct, then add your own bindings after that.
///
/// ```wgsl
/// import wgpu_3dgs_core::{
///     gaussian::Gaussian,
///     gaussian_transform::GaussianTransform,
///     model_transform::{model_to_world, ModelTransform},
/// };
///
/// @group(0) @binding(0)
/// var<uniform> op: u32;
///
/// @group(0) @binding(1)
/// var<storage, read> source: array<u32>;
///
/// @group(0) @binding(2)
/// var<storage, read_write> dest: array<atomic<u32>>;
///
/// @group(0) @binding(3)
/// var<uniform> model_transform: ModelTransform;
///
/// @group(0) @binding(4)
/// var<uniform> gaussian_transform: GaussianTransform;
///
/// @group(0) @binding(5)
/// var<storage, read> gaussians: array<Gaussian>;
///
/// // Your custom bindings here...
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
///     let gaussian = gaussians[index];
///
///     let world_pos = model_to_world(model_transform, gaussian.position);
///
///     // Your custom selection operation code here...
///
///     let word_index = index / 32u;
///     let bit_index = index % 32u;
///     let bit_mask = 1u << bit_index;
///     if /* Condition for selecting the Gaussian */ {
///         atomicOr(&dest[word_index], bit_mask);
///     } else {
///         atomicAnd(&dest[word_index], ~bit_mask);
///     }
/// }
/// ```
#[derive(Debug)]
pub struct SelectionBundle<G: GaussianPod> {
    /// The compute bundle for primitive selection operations.
    primitive_bundle: ComputeBundle<()>,
    /// The compute bundles for selection custom operations.
    pub bundles: Vec<ComputeBundle<()>>,
    /// The Gaussian pod marker.
    gaussian_pod_marker: std::marker::PhantomData<G>,
}

impl<G: GaussianPod> SelectionBundle<G> {
    /// The Gaussians bind group layout descriptors.
    ///
    /// This bind group layout takes the following buffers:
    /// - [`SelectionOpBuffer`]
    /// - Source [`SelectionBuffer`]
    /// - Destination [`SelectionBuffer`]
    /// - [`ModelTransformBuffer`]
    /// - [`GaussianTransformBuffer`]
    /// - [`GaussiansBuffer`]
    pub const GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor<'static> =
        wgpu::BindGroupLayoutDescriptor {
            label: Some("Selection Gaussians Bind Group Layout"),
            entries: &[
                // Selection operation buffer
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
                // Source selection buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Destination selection buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Model transform buffer
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
                // Gaussian transform buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Gaussians buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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

    /// Create a new selection bundle.
    ///
    /// `bundles` are used for [`SelectionExpr::Unary`], [`SelectionExpr::Binary`], or
    /// [`SelectionExpr::Selection`] as custom operations, they must have the same bind group 0 as
    /// the [`SelectionBundle::GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR`], see documentation of
    /// [`SelectionBundle`] for more details.
    pub fn new(device: &wgpu::Device, bundles: Vec<ComputeBundle<()>>) -> Self {
        let primitive_bundle = Self::create_primitive_bundle(device);

        Self {
            primitive_bundle,
            bundles,
            gaussian_pod_marker: std::marker::PhantomData,
        }
    }

    /// Get the Gaussians bind group layout.
    pub fn gaussians_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.primitive_bundle.bind_group_layouts()[0]
    }

    /// Evaluate and apply the selection expression.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        expr: &SelectionExpr,
        dest: &SelectionBuffer,
        model_transform: &ModelTransformBuffer,
        gaussian_transform: &GaussianTransformBuffer,
        gaussians: &GaussiansBuffer<G>,
    ) {
        if let SelectionExpr::Identity = expr {
            return;
        } else if let SelectionExpr::Buffer(buffer) = expr {
            encoder.copy_buffer_to_buffer(
                buffer.buffer(),
                0,
                dest.buffer(),
                0,
                dest.buffer().size(),
            );
            return;
        }

        let d = dest;
        let m = model_transform;
        let g = gaussian_transform;
        let gs = gaussians;

        let op = SelectionOpBuffer::new(device, expr.as_u32().expect("operation expression"));
        let source = SelectionBuffer::new(device, gaussians.len() as u32);

        match expr {
            SelectionExpr::Union(l, r) => {
                self.evaluate(device, encoder, l, &source, m, g, gs);
                self.evaluate(device, encoder, r, d, m, g, gs);
            }
            SelectionExpr::Intersection(l, r) => {
                self.evaluate(device, encoder, l, &source, m, g, gs);
                self.evaluate(device, encoder, r, d, m, g, gs);
            }
            SelectionExpr::Difference(l, r) => {
                self.evaluate(device, encoder, l, &source, m, g, gs);
                self.evaluate(device, encoder, r, d, m, g, gs);
            }
            SelectionExpr::SymmetricDifference(l, r) => {
                self.evaluate(device, encoder, l, &source, m, g, gs);
                self.evaluate(device, encoder, r, d, m, g, gs);
            }
            SelectionExpr::Complement(e) => {
                self.evaluate(device, encoder, e, d, m, g, gs);
            }
            SelectionExpr::Unary(_, e, _) => {
                self.evaluate(device, encoder, e, d, m, g, gs);
            }
            SelectionExpr::Binary(l, _, r, _) => {
                self.evaluate(device, encoder, l, &source, m, g, gs);
                self.evaluate(device, encoder, r, d, m, g, gs);
            }
            SelectionExpr::Selection(_, _) => {}
            SelectionExpr::Identity | SelectionExpr::Buffer(_) => {
                unreachable!();
            }
        }

        let gaussians_bind_group = self
            .primitive_bundle
            .create_bind_group(
                device,
                0,
                [
                    op.buffer().as_entire_binding(),
                    source.buffer().as_entire_binding(),
                    d.buffer().as_entire_binding(),
                    m.buffer().as_entire_binding(),
                    g.buffer().as_entire_binding(),
                    gs.buffer().as_entire_binding(),
                ],
            )
            .expect("gaussians bind group");

        match expr.custom_op_index_and_bind_groups() {
            None => self.primitive_bundle.dispatch(
                encoder,
                (gaussians.len() as u32).div_ceil(32),
                [&gaussians_bind_group],
            ),
            Some((i, bind_groups)) => {
                let bind_groups = std::iter::once(&gaussians_bind_group)
                    .chain(bind_groups)
                    .collect::<Vec<_>>();

                let bundle = &self.bundles[i];

                bundle.dispatch(encoder, gaussians.len() as u32, bind_groups);
            }
        }
    }

    /// Create the selection primitive operation [`ComputeBundle`].
    ///
    /// - Bind group 0 is [`SelectionBundle::GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR`].
    ///
    /// You usually do not need to use this method, it is used internally for creating the
    /// primitive operation bundle for evaluation.
    pub fn create_primitive_bundle(device: &wgpu::Device) -> ComputeBundle<()> {
        ComputeBundleBuilder::new()
            .label("Selection Primitive Operations")
            .bind_group_layout(&Self::GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR)
            .resolver({
                let mut resolver = wesl::PkgResolver::new();
                resolver.add_package(&core::shader::PACKAGE);
                resolver.add_package(&shader::PACKAGE);
                resolver
            })
            .main_shader(
                "wgpu_3dgs_editor::selection::primitive"
                    .parse()
                    .expect("selection::primitive module path"),
            )
            .entry_point("main")
            .wesl_compile_options(wesl::CompileOptions {
                features: G::wesl_features(),
                ..Default::default()
            })
            .build_without_bind_groups(device)
            .map_err(|e| log::error!("{e}"))
            .expect("primitive bundle")
    }

    /// The sphere selection bind group layout descriptor.
    ///
    /// This bind group layout takes the following buffers:
    /// - [`InvTransformBuffer`](crate::InvTransformBuffer)
    pub const SPHERE_BIND_GROUP_LAYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor<'static> =
        wgpu::BindGroupLayoutDescriptor {
            label: Some("Sphere Selection Bind Group Layout"),
            entries: &[
                // Inverse transform uniform buffer
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
            ],
        };

    /// Create a sphere selection custom operation.
    ///
    /// - Bind group 0 is [`SelectionBundle::GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR`].
    /// - Bind group 1 is [`SelectionBundle::SPHERE_BIND_GROUP_LAYOUT_DESCRIPTOR`].
    pub fn create_sphere_bundle(device: &wgpu::Device) -> ComputeBundle<()> {
        let mut resolver = wesl::PkgResolver::new();
        resolver.add_package(&core::shader::PACKAGE);
        resolver.add_package(&shader::PACKAGE);

        ComputeBundleBuilder::new()
            .label("Sphere Selection")
            .bind_group_layouts([
                &Self::GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR,
                &Self::SPHERE_BIND_GROUP_LAYOUT_DESCRIPTOR,
            ])
            .main_shader(
                "wgpu_3dgs_editor::selection::sphere"
                    .parse()
                    .expect("selection::sphere module path"),
            )
            .entry_point("main")
            .wesl_compile_options(wesl::CompileOptions {
                features: G::wesl_features(),
                ..Default::default()
            })
            .resolver(resolver)
            .build_without_bind_groups(device)
            .map_err(|e| log::error!("{e}"))
            .expect("sphere selection compute bundle")
    }

    /// The box selection bind group layout descriptor.
    ///
    /// This bind group layout takes the following buffers:
    /// - [`InvTransformBuffer`](crate::InvTransformBuffer)
    pub const BOX_BIND_GROUP_LAYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor<'static> =
        wgpu::BindGroupLayoutDescriptor {
            label: Some("Box Selection Bind Group Layout"),
            entries: &[
                // Inverse transform uniform buffer
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
            ],
        };

    /// Create a box selection custom operation.
    ///
    /// - Bind group 0 is [`SelectionBundle::GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR`].
    /// - Bind group 1 is [`SelectionBundle::BOX_BIND_GROUP_LAYOUT_DESCRIPTOR`].
    pub fn create_box_bundle(device: &wgpu::Device) -> ComputeBundle<()> {
        let mut resolver = wesl::PkgResolver::new();
        resolver.add_package(&core::shader::PACKAGE);
        resolver.add_package(&shader::PACKAGE);

        ComputeBundleBuilder::new()
            .label("Box Selection")
            .bind_group_layouts([
                &Self::GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR,
                &Self::BOX_BIND_GROUP_LAYOUT_DESCRIPTOR,
            ])
            .main_shader(
                "wgpu_3dgs_editor::selection::box"
                    .parse()
                    .expect("selection::box module path"),
            )
            .entry_point("main")
            .wesl_compile_options(wesl::CompileOptions {
                features: G::wesl_features(),
                ..Default::default()
            })
            .resolver(resolver)
            .build_without_bind_groups(device)
            .map_err(|e| log::error!("{e}"))
            .expect("box selection compute bundle")
    }
}
