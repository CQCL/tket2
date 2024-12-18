//! Utility functions for the library.

use hugr::builder::{Container, DataflowSubContainer, FunctionBuilder, HugrBuilder, ModuleBuilder};
use hugr::ops::handle::NodeHandle;
use hugr::types::{Type, TypeBound};
use hugr::Hugr;
use hugr::{
    builder::{BuildError, CircuitBuilder, Dataflow, DataflowHugr},
    extension::prelude::qb_t,
    types::Signature,
};

use crate::circuit::Circuit;
use crate::extension::TKET2_EXTENSION_ID;

pub(crate) fn type_is_linear(typ: &Type) -> bool {
    !TypeBound::Copyable.contains(typ.least_upper_bound())
}

/// Utility for building simple qubit-only circuits.
#[allow(unused)]
pub(crate) fn build_simple_circuit<F>(num_qubits: usize, f: F) -> Result<Circuit, BuildError>
where
    F: FnOnce(&mut CircuitBuilder<FunctionBuilder<Hugr>>) -> Result<(), BuildError>,
{
    let qb_row = vec![qb_t(); num_qubits];
    let signature = Signature::new(qb_row.clone(), qb_row).with_extension_delta(TKET2_EXTENSION_ID);
    let mut h = FunctionBuilder::new("main", signature)?;

    let qbs = h.input_wires();

    let mut circ = h.as_circuit(qbs);

    f(&mut circ)?;

    let qbs = circ.finish();

    let hugr = h.finish_hugr_with_outputs(qbs)?;
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
        let qb_row = vec![qb_t(); num_qubits];
        let circ_signature = Signature::new(qb_row.clone(), qb_row);
        let mut dfg = builder.define_function("main", circ_signature)?;
        let mut circ = dfg.as_circuit(dfg.input_wires());
        f(&mut circ)?;
        let qbs = circ.finish();
        dfg.finish_with_outputs(qbs)?
    };
    let hugr = builder.finish_hugr()?;
    Ok(Circuit::new(hugr, circ.node()))
}
