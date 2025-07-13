use crate::{
    Error, SelectionBuffer, SelectionOpBuffer,
    core::{
        self, BufferWrapper, ComputeBundle, ComputeBundleBuilder, GaussianPod,
        GaussianTransformBuffer, GaussiansBuffer, ModelTransformBuffer,
    },
    shader,
};

/// A selection operation expression tree.
///
/// This can be used to carry out operations on selection buffers.
#[derive(Debug)]
pub enum SelectionOpExpr {
    /// Union of the two selections.
    Union(Box<SelectionOpExpr>, Box<SelectionOpExpr>),
    /// Interaction of the two selections.
    Intersection(Box<SelectionOpExpr>, Box<SelectionOpExpr>),
    /// Difference of the two selections.
    Difference(Box<SelectionOpExpr>, Box<SelectionOpExpr>),
    /// Symmetric difference of the two selections.
    SymmetricDifference(Box<SelectionOpExpr>, Box<SelectionOpExpr>),
    /// Complement of the selection.
    Complement(Box<SelectionOpExpr>),
    /// Apply a custom unary operation.
    Unary(u32, Box<SelectionOpExpr>),
    /// Apply a custom binary operation.
    Binary(Box<SelectionOpExpr>, u32, Box<SelectionOpExpr>),
}

impl SelectionOpExpr {
    /// The first u32 value for a custom operation.
    pub const CUSTOM_OP_START: u32 = 5;

    /// Create a new [`SelectionOpExpr::Union`].
    pub fn union(self, other: Self) -> Self {
        Self::Union(Box::new(self), Box::new(other))
    }

    /// Create a new [`SelectionOpExpr::Intersection`].
    pub fn intersection(self, other: Self) -> Self {
        Self::Intersection(Box::new(self), Box::new(other))
    }

    /// Create a new [`SelectionOpExpr::Difference`].
    pub fn difference(self, other: Self) -> Self {
        Self::Difference(Box::new(self), Box::new(other))
    }

    /// Create a new [`SelectionOpExpr::SymmetricDifference`].
    pub fn symmetric_difference(self, other: Self) -> Self {
        Self::SymmetricDifference(Box::new(self), Box::new(other))
    }

    /// Create a new [`SelectionOpExpr::Complement`].
    pub fn complement(self) -> Self {
        Self::Complement(Box::new(self))
    }

    /// Create a new [`SelectionOpExpr::Unary`].
    pub fn unary(self, op: u32) -> Self {
        Self::Unary(op, Box::new(self))
    }

    /// Create a new [`SelectionOpExpr::Binary`].
    pub fn binary(self, op: u32, other: Self) -> Self {
        Self::Binary(Box::new(self), op, Box::new(other))
    }

    /// Get the u32 associated with this expression's operation.
    pub fn as_u32(&self) -> u32 {
        match self {
            SelectionOpExpr::Union(_, _) => 0,
            SelectionOpExpr::Intersection(_, _) => 1,
            SelectionOpExpr::Difference(_, _) => 2,
            SelectionOpExpr::SymmetricDifference(_, _) => 3,
            SelectionOpExpr::Complement(_) => 4,
            SelectionOpExpr::Unary(op, _) => *op,
            SelectionOpExpr::Binary(_, op, _) => *op,
        }
    }

    /// Whether this expression is a primitive operation.
    pub fn is_primitive(&self) -> bool {
        matches!(
            self,
            SelectionOpExpr::Union(_, _)
                | SelectionOpExpr::Intersection(_, _)
                | SelectionOpExpr::Difference(_, _)
                | SelectionOpExpr::SymmetricDifference(_, _)
                | SelectionOpExpr::Complement(_)
        )
    }

    /// Whether this expression is a custom operation.
    pub fn is_custom(&self) -> bool {
        matches!(
            self,
            SelectionOpExpr::Unary(_, _) | SelectionOpExpr::Binary(_, _, _)
        )
    }

    /// Get the custom operation index, which is its value minus [`SelectionOpExpr::CUSTOM_OP_START`].
    pub fn custom_op_index(&self) -> Option<u32> {
        match self {
            SelectionOpExpr::Unary(op, _) => Some(*op - Self::CUSTOM_OP_START),
            SelectionOpExpr::Binary(_, op, _) => Some(*op - Self::CUSTOM_OP_START),
            _ => None,
        }
    }
}

/// A specialized [`ComputeBundle`] for selection operations.
///
/// All [`ComputeBundle`]s supplied to this bundle as a [`SelectionOpExpr::Unary`] or
/// [`SelectionOpExpr::Binary`] must have the same bind group 0 as the
/// [`SelectionBundle::GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR`].
///
/// ```wgsl
/// import wgpu_3dgs_core::{
///     compute_bundle,
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
pub struct SelectionBundle<B = wgpu::BindGroup> {
    /// The compute bundle for primitive selection operations.
    pub primitive_bundle: ComputeBundle<()>,
    /// The compute bundles for selection operations.
    pub bundles: Vec<ComputeBundle<()>>,
    /// The selection buffers.
    ///
    /// This is populated after the first evaluation.
    pub selection_bufs: Vec<SelectionBuffer>,
    /// The selection operation buffers.
    ///
    /// This is populated after the first evaluation.
    pub selection_op_bufs: Vec<SelectionOpBuffer>,
    /// The bind groups.
    bind_groups: Vec<Vec<B>>,
}

impl<B> SelectionBundle<B> {
    /// Get the Gaussians bind group layout.
    pub fn gaussians_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.primitive_bundle.bind_group_layouts()[0]
    }

    /// Evaluate the selection operation expression with provided buffers.
    pub fn evaluate_with_buffers<'a, G: GaussianPod>(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        expr: &SelectionOpExpr,
        dest: &SelectionBuffer,
        model_transform: &ModelTransformBuffer,
        gaussian_transform: &GaussianTransformBuffer,
        gaussians: &GaussiansBuffer<G>,
        bind_groups: &[Vec<&'a wgpu::BindGroup>],
    ) {
        let d = dest;
        let m = model_transform;
        let g = gaussian_transform;
        let gs = gaussians;
        let bgs = bind_groups;

        let op = SelectionOpBuffer::new(device, expr.as_u32());
        let source = SelectionBuffer::new(device, gaussians.len() as u32);

        match expr {
            SelectionOpExpr::Union(l, r) => {
                self.evaluate_with_buffers(device, encoder, l, &source, m, g, gs, bgs);
                self.evaluate_with_buffers(device, encoder, r, d, m, g, gs, bgs);
            }
            SelectionOpExpr::Intersection(l, r) => {
                self.evaluate_with_buffers(device, encoder, l, &source, m, g, gs, bgs);
                self.evaluate_with_buffers(device, encoder, r, d, m, g, gs, bgs);
            }
            SelectionOpExpr::Difference(l, r) => {
                self.evaluate_with_buffers(device, encoder, l, &source, m, g, gs, bgs);
                self.evaluate_with_buffers(device, encoder, r, d, m, g, gs, bgs);
            }
            SelectionOpExpr::SymmetricDifference(l, r) => {
                self.evaluate_with_buffers(device, encoder, l, &source, m, g, gs, bgs);
                self.evaluate_with_buffers(device, encoder, r, d, m, g, gs, bgs);
            }
            SelectionOpExpr::Complement(e) => {
                self.evaluate_with_buffers(device, encoder, e, d, m, g, gs, bgs);
            }
            SelectionOpExpr::Unary(_, e) => {
                self.evaluate_with_buffers(device, encoder, e, d, m, g, gs, bgs);
            }
            SelectionOpExpr::Binary(l, _, r) => {
                self.evaluate_with_buffers(device, encoder, l, &source, m, g, gs, bgs);
                self.evaluate_with_buffers(device, encoder, r, d, m, g, gs, bgs);
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

        match expr.custom_op_index() {
            None => self.primitive_bundle.dispatch(
                encoder,
                gaussians.len() as u32,
                [&gaussians_bind_group],
            ),
            Some(i) => {
                let bind_groups = std::iter::once(&gaussians_bind_group).chain(
                    bgs.get(expr.as_u32() as usize - 5)
                        .expect("bind group")
                        .iter()
                        .copied(),
                );

                let bundle = &self.bundles[i as usize];

                bundle.dispatch(encoder, gaussians.len() as u32, bind_groups);
            }
        }
    }
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
    ///
    /// `bundle_buffers` is the buffers (skipping the first one for the Gaussian bind group)
    /// for each bundle in the `bundles` vector.
    pub fn new<'a>(
        device: &wgpu::Device,
        bundles: Vec<ComputeBundle<()>>,
        bundle_buffers: impl IntoIterator<
            Item = impl IntoIterator<Item = impl IntoIterator<Item = &'a dyn BufferWrapper>>,
        >,
    ) -> Result<Self, Error> {
        let primitive_bundle = Self::build_primitive_bundle(device);

        let bind_groups = bundles
            .iter()
            .zip(bundle_buffers.into_iter())
            .map(|(bundle, buffers)| {
                let buffers = buffers.into_iter().collect::<Vec<_>>();

                if buffers.len() == primitive_bundle.bind_group_layouts().len() {
                    return Err(Error::Core(
                        core::Error::BufferBindGroupLayoutCountMismatch {
                            buffer_count: buffers.len(),
                            bind_group_layout_count: primitive_bundle.bind_group_layouts().len(),
                        },
                    ));
                }

                Ok(buffers
                    .into_iter()
                    .enumerate()
                    .map(|(i, buffers)| {
                        bundle
                            .create_bind_group(device, i + 1, buffers)
                            .expect("bind group layout exists for the buffer")
                    })
                    .collect::<Vec<_>>())
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            primitive_bundle,
            bundles,
            selection_bufs: Vec::new(),
            selection_op_bufs: Vec::new(),
            bind_groups,
        })
    }

    /// Evaluate the selection operation expression.
    pub fn evaluate<G: GaussianPod>(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        expr: &SelectionOpExpr,
        dest: &SelectionBuffer,
        model_transform: &ModelTransformBuffer,
        gaussian_transform: &GaussianTransformBuffer,
        gaussians: &GaussiansBuffer<G>,
    ) {
        self.evaluate_with_buffers(
            device,
            encoder,
            expr,
            dest,
            model_transform,
            gaussian_transform,
            gaussians,
            self.bind_groups
                .iter()
                .map(|groups| groups.iter().collect::<Vec<_>>())
                .collect::<Vec<_>>()
                .as_slice(),
        );
    }

    /// Create a primitive selection operation [`ComputeBundle`].
    pub fn build_primitive_bundle(device: &wgpu::Device) -> ComputeBundle<()> {
        let mut resolver = wesl::StandardResolver::new("shader/selection");
        resolver.add_package(&shader::selection::Mod);

        ComputeBundleBuilder::<wesl::StandardResolver>::new()
            .label("Selection Primitive Operations")
            .bind_group(&SelectionBundle::GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR)
            .resolver(resolver)
            .main_shader("primitve_ops")
            .entry_point("main")
            .build_without_bind_groups(&device)
            .expect("primitive bundle")
    }
}

impl SelectionBundle<()> {
    /// Create a new selection bundle without bind groups.
    pub fn new_without_bind_groups(device: &wgpu::Device, bundles: Vec<ComputeBundle<()>>) -> Self {
        Self {
            primitive_bundle: SelectionBundle::build_primitive_bundle(device),
            bundles,
            selection_bufs: Vec::new(),
            selection_op_bufs: Vec::new(),
            bind_groups: Vec::new(),
        }
    }

    /// Evaluate the selection operation expression.
    pub fn evaluate<'a, G: GaussianPod>(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        expr: &SelectionOpExpr,
        dest: &SelectionBuffer,
        model_transform: &ModelTransformBuffer,
        gaussian_transform: &GaussianTransformBuffer,
        gaussians: &GaussiansBuffer<G>,
        bind_groups: &[Vec<&'a wgpu::BindGroup>],
    ) {
        self.evaluate_with_buffers(
            device,
            encoder,
            expr,
            dest,
            model_transform,
            gaussian_transform,
            gaussians,
            bind_groups,
        );
    }
}
