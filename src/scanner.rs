use arrow::datatypes::Schema as ArrowSchema;
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;
use futures::StreamExt;
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::scanner::{DatasetRecordBatchStream, Scanner};
use lance::{Dataset, Error as LanceError};
use polars::prelude::{DataFrame, Expr, IntoLazy, PolarsError, Schema, SchemaExt};
use std::collections::HashMap;

use crate::arrow_bridge::{ArrowBridgeError, ArrowRecordBatchExt, ArrowSchemaExt};
use crate::sync::TOKIO_RUNTIME;

#[derive(Debug)]
pub enum LanceScannerError {
    Lance(LanceError),
    Arrow(ArrowError),
    Polars(PolarsError),
}

impl From<LanceError> for LanceScannerError {
    fn from(err: LanceError) -> Self {
        Self::Lance(err)
    }
}

impl From<ArrowBridgeError> for LanceScannerError {
    fn from(err: ArrowBridgeError) -> Self {
        match err {
            ArrowBridgeError::Arrow(err) => Self::Arrow(err),
            ArrowBridgeError::Polars(err) => Self::Polars(err),
        }
    }
}

impl From<PolarsError> for LanceScannerError {
    fn from(err: PolarsError) -> Self {
        Self::Polars(err)
    }
}

impl std::fmt::Display for LanceScannerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Lance(err) => err.fmt(f),
            Self::Arrow(err) => err.fmt(f),
            Self::Polars(err) => err.fmt(f),
        }
    }
}

impl std::error::Error for LanceScannerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Lance(err) => Some(err),
            Self::Arrow(err) => Some(err),
            Self::Polars(err) => Some(err),
        }
    }
}

#[derive(Clone, Default)]
pub struct LanceScannerOptions {
    pub with_columns: Option<Vec<String>>,
    pub predicate: Option<Expr>,
    pub n_rows: Option<usize>,
    pub batch_size: Option<usize>,
    pub storage_options: Option<HashMap<String, String>>,
}

pub struct LanceScanner {
    uri: String,
    options: LanceScannerOptions,
    stream: Option<DatasetRecordBatchStream>,
}

impl LanceScanner {
    pub fn new(uri: String, options: LanceScannerOptions) -> Self {
        Self {
            uri,
            options,
            stream: None,
        }
    }

    pub fn next(&mut self) -> Result<Option<DataFrame>, LanceScannerError> {
        if matches!(self.options.n_rows, Some(0)) {
            return Ok(None);
        }

        let stream = self.get_or_init_stream()?;
        let next_batch = Self::next_batch(stream)?;

        let Some(batch) = next_batch else {
            return Ok(None);
        };

        let mut df = DataFrame::from(batch.to_polars_record_batch()?);

        // TODO: Translate the Polars `Expr` into a `LanceFilter` and push the predicate
        // down into the Lance scanner.
        if let Some(predicate) = self.options.predicate.as_ref() {
            df = df
                .lazy()
                .filter(predicate.clone())
                ._with_eager(true)
                .collect()?;
        }

        Ok(Some(df))
    }

    pub fn schema_for_uri(
        uri: &str,
        storage_options: Option<HashMap<String, String>>,
    ) -> Result<Schema, LanceScannerError> {
        let dataset = Self::open_dataset(uri, storage_options)?;
        let lance_schema = dataset.schema();
        let arrow_schema = ArrowSchema::from(lance_schema);
        let polars_arrow_schema = arrow_schema.to_polars_arrow_schema()?;
        Ok(Schema::from_arrow_schema(&polars_arrow_schema))
    }

    fn open_dataset(
        uri: &str,
        storage_options: Option<HashMap<String, String>>,
    ) -> Result<Dataset, LanceError> {
        TOKIO_RUNTIME.block_on(Self::build_lance_dataset_builder(uri, storage_options).load())
    }

    fn build_lance_dataset_builder(
        uri: &str,
        storage_options: Option<HashMap<String, String>>,
    ) -> DatasetBuilder {
        let mut builder = DatasetBuilder::from_uri(uri);
        if let Some(storage_options) = storage_options {
            builder = builder.with_storage_options(storage_options);
        }
        builder
    }

    fn get_or_init_stream(&mut self) -> Result<&mut DatasetRecordBatchStream, LanceError> {
        if self.stream.is_none() {
            let scanner = self.build_lance_scanner()?;
            let stream = TOKIO_RUNTIME.block_on(scanner.try_into_stream())?;
            self.stream = Some(stream);
        }

        Ok(self
            .stream
            .as_mut()
            .expect("stream should be initialized by get_or_init_stream"))
    }

    fn build_lance_scanner(&self) -> Result<Scanner, LanceError> {
        let dataset = Self::open_dataset(&self.uri, self.options.storage_options.clone())?;
        let mut scanner = dataset.scan();

        if let Some(columns) = self.options.with_columns.as_deref() {
            scanner.project(columns)?;
        }

        if let Some(n_rows) = self.options.n_rows {
            scanner.limit(Some(n_rows as i64), None)?;
        }

        if let Some(batch_size) = self.options.batch_size.filter(|batch_size| *batch_size > 0) {
            scanner.batch_size(batch_size);
        }

        Ok(scanner)
    }

    fn next_batch(
        stream: &mut DatasetRecordBatchStream,
    ) -> Result<Option<RecordBatch>, LanceError> {
        TOKIO_RUNTIME.block_on(stream.next()).transpose()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow::array::{ArrayRef, Int32Array, StringArray};
    use arrow::record_batch::{RecordBatch, RecordBatchIterator};
    use lance::Dataset;
    use polars::prelude::{DataFrame, DataType, NamedFrom, Schema, Series};
    use rstest::{fixture, rstest};
    use tempfile::TempDir;

    use super::{LanceScanner, LanceScannerOptions, TOKIO_RUNTIME};

    struct TestDataset {
        uri: String,
        int32_values: Vec<Option<i32>>,
        utf8_values: Vec<&'static str>,
        _temp_dir: TempDir, // Include to keep temp dir alive for duration of fixture.
    }

    #[fixture]
    fn test_dataset() -> TestDataset {
        // Create batches to write to Lance dataset.
        let int32_values = vec![Some(1), None, Some(3)];
        let utf8_values = vec!["a", "b", "c"];
        let batch = RecordBatch::try_from_iter(vec![
            (
                "my_int32_field",
                Arc::new(Int32Array::from(int32_values.clone())) as ArrayRef,
            ),
            (
                "my_utf8_field",
                Arc::new(StringArray::from(utf8_values.clone())) as ArrayRef,
            ),
        ])
        .unwrap();
        let batch_schema = batch.schema();
        let batches = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), batch_schema);

        // Create temp dir to write Lance dataset to.
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = temp_dir
            .path()
            .join("test_dataset.lance")
            .to_str()
            .unwrap()
            .to_owned();

        // Write Lance dataset to temp dir.
        TOKIO_RUNTIME
            .block_on(Dataset::write(batches, &uri, None))
            .unwrap();

        TestDataset {
            _temp_dir: temp_dir,
            uri,
            int32_values,
            utf8_values,
        }
    }

    #[test]
    fn lance_scanner_new() {
        let uri = "file://my_dataset.lance".to_owned();
        let options = LanceScannerOptions {
            with_columns: Some(vec!["a".to_owned(), "b".to_owned()]),
            predicate: None,
            n_rows: Some(3),
            batch_size: Some(128),
            storage_options: None,
        };

        let scanner = LanceScanner::new(uri.clone(), options.clone());

        assert_eq!(scanner.uri, uri);
        assert_eq!(scanner.options.with_columns, options.with_columns);
        assert!(scanner.options.predicate.is_none());
        assert_eq!(scanner.options.n_rows, options.n_rows);
        assert_eq!(scanner.options.batch_size, options.batch_size);
        assert!(scanner.options.storage_options.is_none());
        assert!(scanner.stream.is_none());
    }

    #[test]
    fn lance_scanner_new_with_storage_options() {
        let uri = "s3://my-bucket/my_dataset.lance".to_owned();
        let storage_options = HashMap::from([
            ("aws_access_key_id".to_owned(), "foo".to_owned()),
            ("aws_secret_access_key".to_owned(), "bar".to_owned()),
            ("aws_region".to_owned(), "us-east-1".to_owned()),
        ]);
        let options = LanceScannerOptions {
            storage_options: Some(storage_options.clone()),
            ..Default::default()
        };

        let scanner = LanceScanner::new(uri.clone(), options);

        assert_eq!(scanner.uri, uri);
        assert_eq!(scanner.options.storage_options, Some(storage_options));
    }

    #[rstest]
    fn lance_scanner_next(test_dataset: TestDataset) {
        fn new_expected_dataframe(int32_values: &[Option<i32>], utf8_values: &[&str]) -> DataFrame {
            DataFrame::new_infer_height(vec![
                Series::new("my_int32_field".into(), int32_values).into(),
                Series::new("my_utf8_field".into(), utf8_values).into(),
            ])
            .unwrap()
        }

        let batch_size = 2;
        let mut scanner = LanceScanner::new(
            test_dataset.uri.clone(),
            LanceScannerOptions {
                batch_size: Some(batch_size),
                ..Default::default()
            },
        );

        // First next().
        let df_0 = scanner.next().unwrap().unwrap();
        assert!(scanner.stream.is_some());
        let expected_dataframe_0 = new_expected_dataframe(
            &test_dataset.int32_values[..batch_size],
            &test_dataset.utf8_values[..batch_size],
        );
        assert_eq!(df_0, expected_dataframe_0);

        // Second next().
        let df_1 = scanner.next().unwrap().unwrap();
        let expected_dataframe_1 = new_expected_dataframe(
            &test_dataset.int32_values[batch_size..],
            &test_dataset.utf8_values[batch_size..],
        );
        assert_eq!(df_1, expected_dataframe_1);

        // Third (and final) next().
        assert_eq!(scanner.next().unwrap(), None);
    }

    #[rstest]
    fn lance_scanner_schema_for_uri(test_dataset: TestDataset) {
        let schema = LanceScanner::schema_for_uri(&test_dataset.uri, None).unwrap();

        let expected_schema = Schema::from_iter([
            ("my_int32_field".into(), DataType::Int32),
            ("my_utf8_field".into(), DataType::String),
        ]);
        assert_eq!(schema, expected_schema);
    }
}
