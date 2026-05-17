use arrow::error::ArrowError;
use lance::Error as LanceError;
use polars::prelude::PolarsError;

use crate::arrow::ArrowBridgeError;

macro_rules! define_lance_error {
    ($name:ident) => {
        #[derive(Debug)]
        pub enum $name {
            Lance(LanceError),
            Arrow(ArrowError),
            Polars(PolarsError),
        }

        impl From<LanceError> for $name {
            fn from(err: LanceError) -> Self {
                Self::Lance(err)
            }
        }

        impl From<ArrowBridgeError> for $name {
            fn from(err: ArrowBridgeError) -> Self {
                match err {
                    ArrowBridgeError::Arrow(err) => Self::Arrow(err),
                    ArrowBridgeError::Polars(err) => Self::Polars(err),
                }
            }
        }

        impl From<PolarsError> for $name {
            fn from(err: PolarsError) -> Self {
                Self::Polars(err)
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    Self::Lance(err) => err.fmt(f),
                    Self::Arrow(err) => err.fmt(f),
                    Self::Polars(err) => err.fmt(f),
                }
            }
        }

        impl std::error::Error for $name {
            fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
                match self {
                    Self::Lance(err) => Some(err),
                    Self::Arrow(err) => Some(err),
                    Self::Polars(err) => Some(err),
                }
            }
        }
    };
}

define_lance_error!(LanceScannerError);
define_lance_error!(LanceWriterError);
