use delegate::delegate;
use hugr::extension::ExtensionSet;
use hugr::ops::constant::CustomConst;
use hugr::ops::constant::ValueName;
use hugr::types::Type;
use serde::{Deserialize, Serialize};
use tket2_hseries::extension::wasm::ConstWasmModule;

use pyo3::prelude::*;

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "extension")?;
    m.add("ConstWasmModule", py.get_type::<PyConstWasmModule>())?;
    Ok(m)
}

#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PyConstWasmModule {
    module: ConstWasmModule,
}

#[pymethods]
impl PyConstWasmModule {
    #[new]
    pub fn new(name: String, hash: u64) -> Self {
        PyConstWasmModule {
            module: ConstWasmModule { name, hash },
        }
    }
}

#[typetag::serde]
impl CustomConst for PyConstWasmModule {
    delegate! {
        to self.module {
            fn name(&self) -> ValueName;
            fn equal_consts(&self, other: &dyn CustomConst) -> bool;
            fn extension_reqs(&self) -> ExtensionSet;
            fn get_type(&self) -> Type;
        }
    }
}
