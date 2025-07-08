use std::collections::HashMap;

use wgpu_3dgs_core::{BufferWrapper, ComputeBundleBuilder, wesl::DynResolver};

use crate::{
    Error,
    core::{self, ComputeBundle},
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
}

/// A specialized [`ComputeBundle`] for selection operations.
///
/// It is suggested to use the [`SelectionBundleBuilder`] to create this bundle.
#[derive(Debug)]
pub struct SelectionBundle<B = wgpu::BindGroup> {
    /// The compute bundle for selection operations.
    pub bundle: ComputeBundle<B>,
    /// The custom operations.
    ///
    /// The operations are expected to be in the form of `"my_mod::my_op(arg1, arg2, ...)"`,
    /// where `arg1, arg2, ...` can be one of the following:
    /// - The bindings in group 0
    ///     - `op` - The selection operation.
    ///     - `source` - The source selection buffer.
    ///     - `dest` - The destination selection buffer.
    ///     - `model_transform` - The model transform buffer.
    ///     - `gaussian_transform` - The Gaussian transform buffer.
    ///     - `gaussians` - The Gaussian buffer.
    /// - `index` - The current index.
    pub custom_ops: HashMap<String, u32>,
}

impl SelectionBundle {
    /// Evaluate the selection operation expression.
    pub fn evaluate(&self, expr: SelectionOpExpr) {
        unimplemented!()
    }
}

impl SelectionBundle<()> {
    /// Evaluate the selection operation expression.
    pub fn evaluate(&self, expr: SelectionOpExpr, bind_group: &wgpu::BindGroup) {
        unimplemented!()
    }
}

/// A builder for [`SelectionBundle`].
///
/// This builder append a bind group at the end of the [`ComputeBundleBuilder`] you provide,
/// which contains the selection operation, source and destination buffers,
/// model transform, Gaussian transform, and Gaussian buffers.
pub struct SelectionBundleBuilder<'a, R: wesl::Resolver> {
    /// The compute bundle builder.
    pub builder: ComputeBundleBuilder<'a, R>,
    /// The custom operations.
    pub custom_ops: Vec<String>,
}

impl<'a, R: wesl::Resolver> SelectionBundleBuilder<'a, R> {
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

    /// The template for the selection shader.
    pub const TEMPLATE: &'static str = include_str!("shader/selection/template.wesl");

    /// The entry shader [`wesl::ModulePath`]'s string representation.
    ///
    /// This is a string due to not being able to create a constant [`wesl::ModulePath`],
    /// the origin of the path is [`wesl::syntax::PathOrigin::Absolute`].
    pub const ENTRY: &'static str = "selection";

    /// Create a new [`SelectionBundleBuilder`].
    pub fn new() -> Self {
        Self {
            builder: ComputeBundleBuilder::new(),
            custom_ops: Vec::new(),
        }
    }

    /// Set the builder.
    pub fn builder<S: wesl::Resolver>(
        self,
        builder: ComputeBundleBuilder<'a, S>,
    ) -> SelectionBundleBuilder<'a, S> {
        SelectionBundleBuilder {
            builder,
            custom_ops: self.custom_ops,
        }
    }

    /// Add a custom operation to the bundle.
    pub fn custom_op(&mut self, op: String) -> &mut Self {
        self.custom_ops.push(op);
        self
    }

    /// Add custom operations to the bundle.
    pub fn custom_ops(
        mut self,
        ops: impl IntoIterator<Item = String>,
    ) -> SelectionBundleBuilder<'a, R> {
        self.custom_ops.extend(ops);
        self
    }

    /// Build the selection bundle.
    pub fn build<'b>(
        mut self,
        device: &wgpu::Device,
        buffers: impl IntoIterator<Item = impl IntoIterator<Item = &'b dyn BufferWrapper>>,
    ) -> Result<SelectionBundle<wgpu::BindGroup>, Error> {
        let Some(resolver) = std::mem::take(&mut self.builder.resolver) else {
            return Err(Error::Core(core::Error::MissingResolver));
        };

        let mut builder = self
            .builder
            .resolver(Self::build_dyn_resolver(resolver, &self.custom_ops));

        let constants = [
            &[(
                "gaussians_group_index",
                builder.bind_group_layouts.len() as f64,
            )],
            builder.compilation_options.constants,
        ]
        .concat();

        builder.compilation_options.constants = &constants;

        builder
            .bind_group_layouts
            .push(&Self::GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR);

        let bundle = builder.build(device, buffers)?;
        let custom_ops = self
            .custom_ops
            .into_iter()
            .enumerate()
            .map(|(i, op)| (op, i as u32 + 4)) // Custom ops start at 4
            .collect();
        Ok(SelectionBundle { bundle, custom_ops })
    }

    /// Build the compute bundle without bind groups.
    pub fn build_without_bind_groups(
        mut self,
        device: &wgpu::Device,
    ) -> Result<SelectionBundle<()>, Error> {
        let Some(resolver) = std::mem::take(&mut self.builder.resolver) else {
            return Err(Error::Core(core::Error::MissingResolver));
        };

        let mut builder = self
            .builder
            .resolver(Self::build_dyn_resolver(resolver, &self.custom_ops));

        let constants = [
            &[(
                "gaussians_group_index",
                builder.bind_group_layouts.len() as f64,
            )],
            builder.compilation_options.constants,
        ]
        .concat();

        builder.compilation_options.constants = &constants;

        builder
            .bind_group_layouts
            .push(&Self::GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR);

        let bundle = builder.build_without_bind_groups(device)?;
        let custom_ops = self
            .custom_ops
            .into_iter()
            .enumerate()
            .map(|(i, op)| (op, i as u32 + 4)) // Custom ops start at 4
            .collect();
        Ok(SelectionBundle { bundle, custom_ops })
    }

    /// The entry shader [`wesl::ModulePath`].
    pub fn entry() -> wesl::ModulePath {
        wesl::ModulePath {
            origin: wesl::syntax::PathOrigin::Absolute,
            components: vec![Self::ENTRY.to_string()],
        }
    }

    /// Build the [`DynResolver`].
    fn build_dyn_resolver(resolver: R, custom_ops: &[String]) -> DynResolver<R> {
        let custom_ops_shader = custom_ops
            .iter()
            .enumerate()
            .map(|(i, op)| {
                let op_index = i + 4; // Custom ops start at 4
                format!("else if op == {op_index} {{ {op}; }}")
            })
            .collect::<Vec<_>>()
            .join("\n");

        let resolver = DynResolver::new(resolver).with_shader(
            Self::entry(),
            Self::TEMPLATE.replace("{{custom_ops}}", &custom_ops_shader),
        );

        resolver
    }
}
