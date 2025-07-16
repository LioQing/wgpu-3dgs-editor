use thiserror::Error;

use crate::core;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Core(#[from] core::Error),
}
