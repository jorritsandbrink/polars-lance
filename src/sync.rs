use once_cell::sync::Lazy;
use tokio::runtime::{Builder, Runtime};

pub(super) static TOKIO_RUNTIME: Lazy<Runtime> = Lazy::new(|| {
    Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime should initialize")
});
