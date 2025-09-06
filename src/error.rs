use thiserror::Error;

/// The error type for [`NonDestructiveModifier::new`](crate::NonDestructiveModifier::new).
#[derive(Debug, Error)]
pub enum NonDestructiveModifierCreateError {
    #[error("missing COPY_SRC buffer usage")]
    MissingCopySrcBufferUsage,
    #[error("{0}")]
    Poll(#[from] wgpu::PollError),
}
