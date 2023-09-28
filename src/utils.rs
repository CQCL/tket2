//! Utility functions for the library.

use hugr::extension::PRELUDE_REGISTRY;
use hugr::types::{Type, TypeBound};
use hugr::{
    builder::{BuildError, CircuitBuilder, DFGBuilder, Dataflow, DataflowHugr},
    extension::prelude::QB_T,
    types::FunctionType,
    Hugr,
};

pub(crate) fn type_is_linear(typ: &Type) -> bool {
    !TypeBound::Copyable.contains(typ.least_upper_bound())
}

// Convert a pytket object to HUGR
#[cfg(feature = "pyo3")]
mod pyo3 {
    use hugr::Hugr;
    use pyo3::prelude::*;
    use tket_json_rs::circuit_json::SerialCircuit;

    use crate::json::TKETDecode;

    pub(crate) fn pyobj_as_hugr(circ: PyObject) -> PyResult<Hugr> {
        let ser_c = SerialCircuit::_from_tket1(circ);
        let hugr: Hugr = ser_c.decode()?;
        Ok(hugr)
    }
}
#[cfg(feature = "pyo3")]
pub(crate) use self::pyo3::pyobj_as_hugr;

// utility for building simple qubit-only circuits.
#[allow(unused)]
pub(crate) fn build_simple_circuit(
    num_qubits: usize,
    f: impl FnOnce(&mut CircuitBuilder<DFGBuilder<Hugr>>) -> Result<(), BuildError>,
) -> Result<Hugr, BuildError> {
    let qb_row = vec![QB_T; num_qubits];
    let mut h = DFGBuilder::new(FunctionType::new(qb_row.clone(), qb_row))?;

    let qbs = h.input_wires();

    let mut circ = h.as_circuit(qbs.into_iter().collect());

    f(&mut circ)?;

    let qbs = circ.finish();
    h.finish_hugr_with_outputs(qbs, &PRELUDE_REGISTRY)
}

// Test only utils
#[allow(dead_code)]
#[cfg(test)]
pub(crate) mod test {
    /// Open a browser page to render a dot string graph.
    ///
    /// This can be used directly on the output of `Hugr::dot_string`
    #[cfg(not(ci_run))]
    pub(crate) fn viz_dotstr(dotstr: &str) {
        let mut base: String = "https://dreampuf.github.io/GraphvizOnline/#".into();
        base.push_str(&urlencoding::encode(dotstr));
        webbrowser::open(&base).unwrap();
    }
}
