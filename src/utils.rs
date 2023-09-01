//! Utility functions for the library.

use hugr::{
    builder::{BuildError, CircuitBuilder, DFGBuilder, Dataflow, DataflowHugr},
    extension::{prelude::QB_T, prelude_registry},
    types::FunctionType,
    Hugr,
};

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
    h.finish_hugr_with_outputs(qbs, &prelude_registry())
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
