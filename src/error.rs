use thiserror::Error;

use crate::core;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Core(#[from] core::Error),
    #[error("missing COPY_SRC buffer usage")]
    MissingCopySrcBufferUsage,
    #[error("original Gaussian and target Gaussian counts mismatched: {original} != {target}")]
    OriginalTargetCountMismatch { original: usize, target: usize },
    #[error("{0}")]
    WgpuPoll(#[from] wgpu::PollError),
}
