use std::sync::Arc;

use arrow::error::ArrowError;
use arrow::record_batch::RecordBatchIterator;
use lance::dataset::{Dataset, WriteMode, WriteParams};
use polars::frame::chunk_df_for_writing;
use polars::prelude::{CompatLevel, DataFrame, SchemaExt};

use crate::arrow::{ArrowBridgeError, PolarsArrowRecordBatchExt, PolarsArrowSchemaExt};
use crate::err::LanceWriterError;
use crate::sync::TOKIO_RUNTIME;

const LANCE_ARROW_COMPAT_LEVEL: CompatLevel = CompatLevel::oldest();

fn chunk_df_for_lance_write(mut df: DataFrame) -> Result<DataFrame, LanceWriterError> {
    // 512 * 512 matches chunk size used internally by Polars.
    Ok(chunk_df_for_writing(&mut df, 512 * 512)?.into_owned())
}

pub fn write_lance_dataset(df: DataFrame, uri: &str) -> Result<(), LanceWriterError> {
    let mut df = chunk_df_for_lance_write(df)?;

    let dfs = df.split_chunks().collect::<Vec<_>>();

    let batches = dfs.into_iter().map(|df| {
        let mut batches = df.iter_chunks(LANCE_ARROW_COMPAT_LEVEL, false);

        let batch = batches
            .next()
            .expect("chunk dataframe should yield one record batch");
        assert!(
            batches.next().is_none(),
            "chunk dataframe should yield exactly one record batch"
        );

        batch.to_arrow_record_batch().map_err(|err| match err {
            ArrowBridgeError::Arrow(err) => err,
            ArrowBridgeError::Polars(err) => ArrowError::ExternalError(Box::new(err)),
        })
    });

    let schema = df
        .schema()
        .to_arrow(LANCE_ARROW_COMPAT_LEVEL)
        .to_arrow_schema()?;

    let batch_iterator = RecordBatchIterator::new(batches, Arc::new(schema));

    let write_params = WriteParams {
        mode: WriteMode::Overwrite,
        ..WriteParams::default()
    };

    TOKIO_RUNTIME.block_on(Dataset::write(batch_iterator, uri, Some(write_params)))?;
    Ok(())
}
