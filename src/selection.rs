use glam::*;

use crate::{
    SelectionBuffer, SelectionOpBuffer,
    core::{
        self, BufferWrapper, ComputeBundle, ComputeBundleBuilder, GaussianPod,
        GaussianTransformBuffer, GaussiansBuffer, ModelTransformBuffer,
    },
    shader,
};

macro_rules! package_module_path {
    ($($components:ident)::+) => {
        wesl::ModulePath {
            origin: wesl::syntax::PathOrigin::Package,
            components: vec![$(stringify!($components).to_string()),+],
        }
    }
}

/// A selection expression tree.
///
/// This can be used to carry out operations on selection buffers.
///
/// [`SelectionExpr::Unary`], [`SelectionExpr::Binary`], and [`SelectionExpr::Selection`] are
/// custom operations that can be defined with additional [`ComputeBundle`]s, so they also
/// carry a vector of bind groups that are used in the operation when dispatched/evaluated.
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
    Unary(u32, Box<SelectionExpr>, Vec<wgpu::BindGroup>),
    /// Apply a custom binary operation.
    Binary(
        Box<SelectionExpr>,
        u32,
        Box<SelectionExpr>,
        Vec<wgpu::BindGroup>,
    ),
    /// Create a selection.
    Selection(u32, Vec<wgpu::BindGroup>),
    /// Use a selection buffer.
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
    pub fn unary(self, op: u32, bind_groups: Vec<wgpu::BindGroup>) -> Self {
        Self::Unary(op, Box::new(self), bind_groups)
    }

    /// Create a new [`SelectionExpr::Binary`].
    pub fn binary(self, op: u32, other: Self, bind_groups: Vec<wgpu::BindGroup>) -> Self {
        Self::Binary(Box::new(self), op, Box::new(other), bind_groups)
    }

    /// Create a new [`SelectionExpr::Selection`].
    pub fn selection(op: u32, bind_groups: Vec<wgpu::BindGroup>) -> Self {
        Self::Selection(op, bind_groups)
    }

    /// Create a new [`SelectionExpr::Buffer`].
    pub fn buffer(buffer: SelectionBuffer) -> Self {
        Self::Buffer(buffer)
    }

    /// Get the u32 associated with this expression's operation.
    ///
    /// The value returned is not the same as that returned by [`SelectionExpr::custom_op_index`],
    /// but rather a value that can be used to identify the operation in a compute shader, custom
    /// operation's index are offset by [`SelectionExpr::CUSTOM_OP_START`].
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            SelectionExpr::Union(_, _) => Some(0),
            SelectionExpr::Intersection(_, _) => Some(1),
            SelectionExpr::Difference(_, _) => Some(2),
            SelectionExpr::SymmetricDifference(_, _) => Some(3),
            SelectionExpr::Complement(_) => Some(4),
            SelectionExpr::Unary(op, _, _) => Some(*op + Self::CUSTOM_OP_START),
            SelectionExpr::Binary(_, op, _, _) => Some(*op + Self::CUSTOM_OP_START),
            SelectionExpr::Selection(op, _) => Some(*op + Self::CUSTOM_OP_START),
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
    pub fn custom_op_index(&self) -> Option<u32> {
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
    pub fn custom_op_index_and_bind_groups(&self) -> Option<(u32, &Vec<wgpu::BindGroup>)> {
        match self {
            SelectionExpr::Unary(op, _, bind_groups)
            | SelectionExpr::Binary(_, op, _, bind_groups)
            | SelectionExpr::Selection(op, bind_groups) => Some((*op, bind_groups)),
            _ => None,
        }
    }
}

/// A specialized [`ComputeBundle`] for selection operations.
///
/// All [`ComputeBundle`]s supplied to this bundle as a [`SelectionExpr::Unary`] or
/// [`SelectionExpr::Binary`] must have the same bind group 0 as the
/// [`SelectionBundle::GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR`].
///
/// ```wgsl
/// import wgpu_3dgs_core::{
///     gaussian::Gaussian,
///     gaussian_transform::GaussianTransform,
///     model_transform::ModelTransform,
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
/// ```
#[derive(Debug)]
pub struct SelectionBundle {
    /// The compute bundle for primitive selection operations.
    pub primitive_bundle: ComputeBundle<()>,
    /// The compute bundles for selection operations.
    pub bundles: Vec<ComputeBundle<()>>,
}

impl SelectionBundle {
    /// The Gaussians bind group layout descriptors.
    pub const GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor<'static> =
        wgpu::BindGroupLayoutDescriptor {
            label: Some("Mask Evaluator Bind Group Layout"),
            entries: &[
                // Mask operation buffer
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
                // Source mask buffer
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
                // Destination mask buffer
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
                // Gaussian buffer
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
    pub fn new<'a, G: GaussianPod>(device: &wgpu::Device, bundles: Vec<ComputeBundle<()>>) -> Self {
        let primitive_bundle = Self::create_primitive_bundle::<G>(device);

        Self {
            primitive_bundle,
            bundles,
        }
    }

    /// Get the Gaussians bind group layout.
    pub fn gaussians_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.primitive_bundle.bind_group_layouts()[0]
    }

    /// Evaluate the selection expression.
    pub fn evaluate<G: GaussianPod>(
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
                &buffer.buffer(),
                0,
                &dest.buffer(),
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
                    &op as &dyn BufferWrapper,
                    &source as &dyn BufferWrapper,
                    d as &dyn BufferWrapper,
                    m as &dyn BufferWrapper,
                    g as &dyn BufferWrapper,
                    gs as &dyn BufferWrapper,
                ],
            )
            .expect("gaussians bind group");

        match expr.custom_op_index_and_bind_groups() {
            None => self.primitive_bundle.dispatch(
                encoder,
                gaussians.len() as u32,
                [&gaussians_bind_group],
            ),
            Some((i, bind_groups)) => {
                let bind_groups = std::iter::once(&gaussians_bind_group)
                    .chain(bind_groups)
                    .collect::<Vec<_>>();

                let bundle = &self.bundles[i as usize];

                bundle.dispatch(encoder, gaussians.len() as u32, bind_groups);
            }
        }
    }

    /// Create the primitive selection operation [`ComputeBundle`].
    pub fn create_primitive_bundle<G: GaussianPod>(device: &wgpu::Device) -> ComputeBundle<()> {
        let mut resolver = wesl::PkgResolver::new();
        resolver.add_package(&core::shader::Mod);
        resolver.add_package(&shader::Mod);

        ComputeBundleBuilder::new()
            .label("Selection Primitive Operations")
            .bind_group(&SelectionBundle::GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR)
            .resolver(resolver)
            .main_shader(package_module_path!(
                wgpu_3dgs_editor::selection::primitive_ops
            ))
            .entry_point("main")
            .compile_options(wesl::CompileOptions {
                features: G::features_map(),
                ..Default::default()
            })
            .build_without_bind_groups(&device)
            .map_err(|e| log::error!("{e}"))
            .expect("primitive bundle")
    }
}

pub mod ops {
    use super::*;

    /// The sphere selection bind group layout descriptor.
    pub const SPHERE_BIND_GROUP_LAYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor<'static> =
        wgpu::BindGroupLayoutDescriptor {
            label: Some("Sphere Selection Bind Group Layout"),
            entries: &[
                // Sphere uniform buffer
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

    /// Create a sphere selection operation.
    ///
    /// - Bind group 0 is [`SelectionBundle::GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR`].
    /// - Bind group 1 is [`SPHERE_BIND_GROUP_LAYOUT_DESCRIPTOR`].
    pub fn sphere<G: GaussianPod>(device: &wgpu::Device) -> ComputeBundle<()> {
        let mut resolver = wesl::PkgResolver::new();
        resolver.add_package(&core::shader::Mod);
        resolver.add_package(&shader::Mod);

        ComputeBundleBuilder::new()
            .label("Sphere Selection")
            .bind_groups([
                &SelectionBundle::GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR,
                &SPHERE_BIND_GROUP_LAYOUT_DESCRIPTOR,
            ])
            .main_shader(package_module_path!(wgpu_3dgs_editor::selection::sphere))
            .entry_point("main")
            .compile_options(wesl::CompileOptions {
                features: G::features_map(),
                ..Default::default()
            })
            .resolver(resolver)
            .build_without_bind_groups(device)
            .map_err(|e| log::error!("{e}"))
            .expect("sphere selection compute bundle")
    }
}
