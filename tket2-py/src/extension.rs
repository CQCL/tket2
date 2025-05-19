//! Extra stuff needed for using the extensions

use delegate::delegate;
use hugr::ops::constant::CustomConst;
use hugr::ops::constant::ValueName;
use hugr::types::Type;
use serde::{Deserialize, Serialize};
use tket2_hseries::extension::wasm::ConstWasmModule;

use pyo3::prelude::*;

/// Build the python module
pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "extension")?;
    m.add("ConstWasmModule", py.get_type::<PyConstWasmModule>())?;
    Ok(m)
}

/// A wrapper for tket2's `ConstWasmModule`
#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PyConstWasmModule {
    module: ConstWasmModule,
}

#[pymethods]
impl PyConstWasmModule {
    /// Create a new constant WASM module which refers to a wasm file with a
    /// given name and hash
    #[new]
    pub fn new(file_name: String, file_hash: u64) -> Self {
        PyConstWasmModule {
            module: ConstWasmModule {
                name: file_name,
                hash: file_hash,
            },
        }
    }
}

#[typetag::serde]
impl CustomConst for PyConstWasmModule {
    delegate! {
        to self.module {
            fn name(&self) -> ValueName;
            fn equal_consts(&self, other: &dyn CustomConst) -> bool;
            fn get_type(&self) -> Type;
        }
    }
}
