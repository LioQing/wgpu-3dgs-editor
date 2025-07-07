use thiserror::Error;

use crate::core;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Core(#[from] core::Error),
    #[error("{0}")]
    BufferDownloadOneShotReceive(#[from] oneshot::RecvError),
    #[error("{0}")]
    BufferDownloadAsync(#[from] wgpu::BufferAsyncError),
    #[error("{0}")]
    DeviceFailedToPoll(#[from] wgpu::PollError),
}
