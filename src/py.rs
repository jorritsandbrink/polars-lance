use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::wrap_pyfunction;
use pyo3_polars::error::PyPolarsErr;
use pyo3_polars::{PyDataFrame, PyExpr, PySchema};

use crate::{
    write_lance_dataset, LanceScanner, LanceScannerError, LanceScannerOptions, LanceWriterError,
    PolarsLanceWriteMode,
};

impl From<LanceScannerError> for PyErr {
    fn from(err: LanceScannerError) -> Self {
        match err {
            LanceScannerError::Polars(err) => PyErr::from(PyPolarsErr::from(err)),
            other => PyRuntimeError::new_err(other.to_string()),
        }
    }
}

impl From<LanceWriterError> for PyErr {
    fn from(err: LanceWriterError) -> Self {
        match err {
            LanceWriterError::Polars(err) => PyErr::from(PyPolarsErr::from(err)),
            other => PyRuntimeError::new_err(other.to_string()),
        }
    }
}

#[pyclass(name = "LanceScanner", unsendable)]
pub struct PyLanceScanner(LanceScanner);

#[pymethods]
impl PyLanceScanner {
    #[new]
    #[pyo3(signature = (uri, with_columns=None, predicate=None, n_rows=None, batch_size=None, storage_options=None))]
    fn new(
        uri: String,
        with_columns: Option<Vec<String>>,
        predicate: Option<PyExpr>,
        n_rows: Option<usize>,
        batch_size: Option<usize>,
        storage_options: Option<HashMap<String, String>>,
    ) -> Self {
        Self(LanceScanner::new(
            uri,
            LanceScannerOptions {
                with_columns,
                predicate: predicate.map(|predicate| predicate.0),
                n_rows,
                batch_size,
                storage_options,
            },
        ))
    }

    fn next(&mut self) -> PyResult<Option<PyDataFrame>> {
        self.0
            .next()
            .map(|df| df.map(PyDataFrame))
            .map_err(PyErr::from)
    }

    #[staticmethod]
    #[pyo3(signature = (uri, storage_options=None))]
    fn schema_for_uri(
        uri: String,
        storage_options: Option<HashMap<String, String>>,
    ) -> PyResult<PySchema> {
        LanceScanner::schema_for_uri(&uri, storage_options)
            .map(|schema| PySchema(Arc::new(schema)))
            .map_err(PyErr::from)
    }
}

#[pyfunction]
#[pyo3(signature = (df, target, *, mode = "error"))]
fn write_lance(df: PyDataFrame, target: String, mode: &str) -> PyResult<()> {
    let mode = match mode {
        "error" => PolarsLanceWriteMode::Error,
        "append" => PolarsLanceWriteMode::Append,
        "overwrite" => PolarsLanceWriteMode::Overwrite,
        _ => {
            return Err(PyValueError::new_err(
                "`mode` must be one of: 'error', 'append', 'overwrite'",
            ));
        }
    };

    write_lance_dataset(df.into(), &target, mode).map_err(PyErr::from)
}

#[pymodule]
fn _polars_lance(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyLanceScanner>()?;
    m.add_function(wrap_pyfunction!(write_lance, m)?)?;
    Ok(())
}
