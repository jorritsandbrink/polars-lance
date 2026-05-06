use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3_polars::error::PyPolarsErr;
use pyo3_polars::{PyDataFrame, PyExpr, PySchema};

use crate::{LanceScanner, LanceScannerError, LanceScannerOptions};

impl From<LanceScannerError> for PyErr {
    fn from(err: LanceScannerError) -> Self {
        match err {
            LanceScannerError::Polars(err) => PyErr::from(PyPolarsErr::from(err)),
            other => PyRuntimeError::new_err(other.to_string()),
        }
    }
}

#[pyclass(name = "LanceScanner", unsendable)]
pub struct PyLanceScanner(LanceScanner);

#[pymethods]
impl PyLanceScanner {
    #[new]
    #[pyo3(signature = (uri, with_columns=None, predicate=None, n_rows=None, batch_size=None))]
    fn new(
        uri: String,
        with_columns: Option<Vec<String>>,
        predicate: Option<PyExpr>,
        n_rows: Option<usize>,
        batch_size: Option<usize>,
    ) -> Self {
        Self(LanceScanner::new(
            uri,
            LanceScannerOptions {
                with_columns,
                predicate: predicate.map(|predicate| predicate.0),
                n_rows,
                batch_size,
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
    fn schema_for_uri(uri: String) -> PyResult<PySchema> {
        LanceScanner::schema_for_uri(&uri)
            .map(|schema| PySchema(Arc::new(schema)))
            .map_err(PyErr::from)
    }
}

#[pymodule]
fn _polars_lance(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyLanceScanner>()?;
    Ok(())
}
