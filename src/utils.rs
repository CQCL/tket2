//! Utility functions for the library.

use hugr::extension::prelude::BOOL_T;
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

// utility for building simple qubit-only circuits.
#[allow(unused)]
pub(crate) fn build_simple_circuit(
    num_qubits: usize,
    f: impl FnOnce(&mut CircuitBuilder<DFGBuilder<Hugr>>) -> Result<(), BuildError>,
) -> Result<Hugr, BuildError> {
    build_simple_measure_circuit(num_qubits, 0, |c| {
        f(c);
        Ok(vec![])
    })
}

// utility for building simple qubit-only circuits with some measure outputs.
#[allow(unused)]
pub(crate) fn build_simple_measure_circuit(
    num_qubits: usize,
    num_measured_bools: usize,
    f: impl FnOnce(&mut CircuitBuilder<DFGBuilder<Hugr>>) -> Result<Vec<hugr::Wire>, BuildError>,
) -> Result<Hugr, BuildError> {
    let inputs = vec![QB_T; num_qubits];
    let outputs = [inputs.clone(), vec![BOOL_T; num_measured_bools]].concat();

    let mut h = DFGBuilder::new(FunctionType::new(inputs, outputs))?;

    let qbs = h.input_wires();

    let mut circ = h.as_circuit(qbs.into_iter().collect());

    let o = f(&mut circ)?;

    let qbs = circ.finish();
    h.finish_hugr_with_outputs([qbs, o].concat(), &PRELUDE_REGISTRY)
}

// Test only utils
#[allow(dead_code)]
#[cfg(test)]
pub(crate) mod test {
    #[allow(unused_imports)]
    use hugr::HugrView;

    /// Open a browser page to render a dot string graph.
    ///
    /// This can be used directly on the output of `Hugr::dot_string`
    ///
    /// Only for use in local testing. Will fail to compile on CI.
    #[cfg(not(ci_run))]
    pub(crate) fn viz_dotstr(dotstr: impl AsRef<str>) {
        let mut base: String = "https://dreampuf.github.io/GraphvizOnline/#".into();
        base.push_str(&urlencoding::encode(dotstr.as_ref()));
        webbrowser::open(&base).unwrap();
    }

    /// Open a browser page to render a HugrView's dot string graph.
    ///
    /// Only for use in local testing. Will fail to compile on CI.
    #[cfg(not(ci_run))]
    pub(crate) fn viz_hugr(hugr: &impl HugrView) {
        viz_dotstr(hugr.dot_string());
    }
}
