use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    BufferDownloadOneShotReceive(#[from] oneshot::RecvError),
    #[error("{0}")]
    BufferDownloadAsync(#[from] wgpu::BufferAsyncError),
    #[error("{0}")]
    DeviceFailedToPoll(#[from] wgpu::PollError),
}
