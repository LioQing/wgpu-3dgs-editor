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
#[derive(Debug)]
pub struct SelectionBundle<B = wgpu::BindGroup> {
    /// The compute bundle for selection operations.
    pub bundle: ComputeBundle<B>,
}

/// A builder for [`SelectionBundle`].
pub struct SelectionBundleBuilder<'a, R: wesl::Resolver> {
    /// The compute bundle builder.
    pub builder: ComputeBundleBuilder<'a, R>,
    /// The custom operations.
    pub custom_ops: Vec<String>,
}

impl<'a, R: wesl::Resolver> SelectionBundleBuilder<'a, R> {
    /// Create a new [`SelectionBundleBuilder`].
    pub fn new() -> Self {
        Self {
            builder: ComputeBundleBuilder::new(),
            custom_ops: Vec::new(),
        }
    }

    /// Create a new [`SelectionBundleBuilder`] with a [`ComputeBundleBuilder`]
    pub fn new_with_builder(builder: ComputeBundleBuilder<'a, R>) -> Self {
        Self {
            builder,
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
    pub fn build(
        mut self,
        device: &wgpu::Device,
        buffers: impl IntoIterator<Item = impl IntoIterator<Item = &'a dyn BufferWrapper>>,
    ) -> Result<SelectionBundle<wgpu::BindGroup>, Error> {
        let Some(resolver) = std::mem::take(&mut self.builder.resolver) else {
            return Err(Error::Core(core::Error::MissingResolver));
        };

        let builder = self.builder.resolver(DynResolver::new(resolver));

        let bundle = builder.build(device, buffers)?;
        Ok(SelectionBundle { bundle })
    }

    /// Build the compute bundle without bind groups.
    pub fn build_without_bind_groups(
        mut self,
        device: &wgpu::Device,
    ) -> Result<SelectionBundle<()>, Error> {
        let Some(resolver) = std::mem::take(&mut self.builder.resolver) else {
            return Err(Error::Core(core::Error::MissingResolver));
        };

        let builder = self.builder.resolver(DynResolver::new(resolver));

        let bundle = builder.build_without_bind_groups(device)?;
        Ok(SelectionBundle { bundle })
    }
}
