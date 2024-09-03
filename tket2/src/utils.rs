//! Utility functions for the library.

use hugr::builder::{Container, DataflowSubContainer, FunctionBuilder, HugrBuilder, ModuleBuilder};
use hugr::extension::PRELUDE_REGISTRY;
use hugr::ops::handle::NodeHandle;
use hugr::types::{Type, TypeBound};
use hugr::Hugr;
use hugr::{
    builder::{BuildError, CircuitBuilder, Dataflow, DataflowHugr},
    extension::prelude::QB_T,
    types::Signature,
};

use crate::circuit::Circuit;
use crate::extension::{REGISTRY, TKET2_EXTENSION_ID};

pub(crate) fn type_is_linear(typ: &Type) -> bool {
    !TypeBound::Copyable.contains(typ.least_upper_bound())
}

/// Utility for building simple qubit-only circuits.
#[allow(unused)]
pub(crate) fn build_simple_circuit<F>(num_qubits: usize, f: F) -> Result<Circuit, BuildError>
where
    F: FnOnce(&mut CircuitBuilder<FunctionBuilder<Hugr>>) -> Result<(), BuildError>,
{
    let qb_row = vec![QB_T; num_qubits];
    let signature = Signature::new(qb_row.clone(), qb_row).with_extension_delta(TKET2_EXTENSION_ID);
    let mut h = FunctionBuilder::new("main", signature)?;

    let qbs = h.input_wires();

    let mut circ = h.as_circuit(qbs);

    f(&mut circ)?;

    let qbs = circ.finish();

    let hugr = h.finish_hugr_with_outputs(qbs, &REGISTRY)?;
    Ok(hugr.into())
}

/// Utility for building a module with a single circuit definition.
#[allow(unused)]
pub(crate) fn build_module_with_circuit<F>(num_qubits: usize, f: F) -> Result<Circuit, BuildError>
where
    F: FnOnce(&mut CircuitBuilder<FunctionBuilder<&mut Hugr>>) -> Result<(), BuildError>,
{
    let mut builder = ModuleBuilder::new();
    let circ = {
        let qb_row = vec![QB_T; num_qubits];
        let circ_signature = Signature::new(qb_row.clone(), qb_row);
        let mut dfg = builder.define_function("main", circ_signature)?;
        let mut circ = dfg.as_circuit(dfg.input_wires());
        f(&mut circ)?;
        let qbs = circ.finish();
        dfg.finish_with_outputs(qbs)?
    };
    let hugr = builder.finish_hugr(&PRELUDE_REGISTRY)?;
    Ok(Circuit::new(hugr, circ.node()))
}

// Test only utils
#[allow(dead_code)]
#[allow(unused_imports)]
#[cfg(test)]
pub(crate) mod test {
    use crate::Circuit;
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

    /// Open a browser page to render a Circuit's dot string graph.
    ///
    /// Only for use in local testing. Will fail to compile on CI.
    #[cfg(not(ci_run))]
    pub(crate) fn viz_circ(circ: &Circuit) {
        viz_dotstr(circ.dot_string());
    }

    /// Open a browser page to render a HugrView's dot string graph.
    ///
    /// Only for use in local testing. Will fail to compile on CI.
    #[cfg(not(ci_run))]
    pub(crate) fn viz_hugr(hugr: &impl HugrView) {
        viz_dotstr(hugr.dot_string());
    }
}
