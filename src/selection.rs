use wgpu_3dgs_core::{ComputeBundleBuilder, wesl::DynResolver};

use crate::core::ComputeBundle;

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
    pub fn unary(op: u32, other: Self) -> Self {
        Self::Unary(op, Box::new(other))
    }

    /// Create a new [`SelectionOpExpr::Binary`].
    pub fn binary(left: Self, op: u32, right: Self) -> Self {
        Self::Binary(Box::new(left), op, Box::new(right))
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
    pub builder: ComputeBundleBuilder<'a, DynResolver<R>>,
}
