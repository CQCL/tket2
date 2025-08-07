//! Circuit replacer protocol implementation.

use std::collections::BTreeSet;

use hugr::{
    builder::{DFGBuilder, Dataflow, DataflowHugr},
    hugr::{
        hugrmut::HugrMut,
        views::{RootCheckable, RootChecked, SiblingSubgraph},
    },
    ops::{constant, handle::DataflowParentID, OpTrait},
    std_extensions::arithmetic::float_types::float64_type,
    types::Signature,
    HugrView, OutgoingPort, Wire,
};
use itertools::Itertools;
use pyo3::{prelude::*, IntoPyObjectExt, PyErr};
use tket::{
    extension::rotation::{rotation_type, RotationOp},
    resource::ResourceScope,
    rewrite::replacer::CircuitReplacer,
    Circuit, Subcircuit,
};

use crate::circuit::Tk2Circuit;

/// Python protocol for [`CircuitReplacer`] trait.
pub trait CircuitReplacerPyProtocol {
    /// Wrapper for the CircuitMatcher protocol's `replace_match` method
    fn py_replace_match(
        &self,
        subcircuit: Tk2Circuit,
        match_info: PyObject,
    ) -> PyResult<Vec<Tk2Circuit>>;
}

/// Implement CircuitReplacer for a type that implements PyCircuitReplacer.
macro_rules! impl_circuit_replacer {
    ($R:ty) => {
        impl<M: for<'py> IntoPyObject<'py>> CircuitReplacer<M> for $R {
            fn replace_match<H: HugrView>(
                &self,
                subcircuit: &Subcircuit<H::Node>,
                circuit: &ResourceScope<H>,
                match_info: M,
            ) -> Vec<Circuit> {
                let const_inputs = subcircuit.get_const_inputs(circuit).collect_vec();
                let mut subcircuit = Circuit::new(
                    subcircuit
                        .try_to_subgraph(circuit)
                        .unwrap()
                        .extract_subgraph(circuit.hugr(), "subgraph"),
                );
                let removed_inputs = replace_inputs_with_const(&mut subcircuit, const_inputs);
                let match_info = Python::with_gil(|py| match_info.into_py_any(py))
                    .map_err(|e| panic!("A python error occurred:\n{e}"))
                    .clone()
                    .unwrap();
                match <$R as CircuitReplacerPyProtocol>::py_replace_match(
                    self,
                    subcircuit.into(),
                    match_info,
                ) {
                    Ok(outcome) => outcome
                        .into_iter()
                        .map(|c| add_dummy_inputs(c.circ, removed_inputs.clone()))
                        .collect_vec(),
                    Err(err) => panic!("A python error occurred:\n{}", err),
                }
            }
        }
    };
}

/// Rust wrapper for Python objects implementing the CircuitReplacer protocol.
#[derive(Debug, Clone)]
pub struct PyImplCircuitReplacer {
    py_obj: PyObject,
}

impl CircuitReplacerPyProtocol for PyImplCircuitReplacer {
    /// Call the Python object's replace_match method.
    ///
    /// Panics if the method is not available (should be checked at construction
    /// time).
    fn py_replace_match(
        &self,
        subcircuit: Tk2Circuit,
        match_info: PyObject,
    ) -> PyResult<Vec<Tk2Circuit>> {
        Python::with_gil(|py| {
            // Call the Python replacement's replace_match method
            let result = self.py_obj.bind(py).call_method(
                "replace_match",
                (subcircuit, match_info),
                None,
            )?;
            // Extract Vec<Tk2Circuit>
            result.extract::<Vec<Tk2Circuit>>()
        })
    }
}

impl_circuit_replacer!(PyImplCircuitReplacer);

impl<'py> FromPyObject<'py> for PyImplCircuitReplacer {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        Self::new(ob.to_owned().unbind())
    }
}

impl PyImplCircuitReplacer {
    /// Create a new wrapper, validating that the Python object implements the
    /// required methods.
    pub fn new(py_obj: PyObject) -> PyResult<Self> {
        Python::with_gil(|py| {
            let obj = py_obj.bind(py);

            // Check that the object implements the required method
            if !obj.hasattr("replace_match")? {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Object must implement replace_match method (CircuitReplacer protocol)",
                ));
            }

            Ok(Self { py_obj })
        })
    }
}

/// For each pair (i, value) in `const_inputs`, try to remove the i-th input
/// of `circuit` and replace it with a constant with value `value`.
///
/// This will add a type conversion between floats and rotations if needed.
///
/// Return the pairs that were successfully replaced.
fn replace_inputs_with_const(
    circuit: &mut Circuit,
    mut const_inputs: Vec<(usize, constant::Value)>,
) -> Vec<(usize, constant::Value)> {
    if const_inputs.is_empty() {
        return vec![];
    }

    // 1. create a DFG Hugr that produces all the desired constants.
    let (const_loads, outputs) = {
        let out_sig = const_inputs
            .iter()
            .map(|(_, val)| val.get_type())
            .collect_vec();
        let mut const_loads = DFGBuilder::new(Signature::new(vec![], out_sig)).unwrap();
        let outputs = const_inputs
            .iter()
            .map(|(_, val)| const_loads.add_load_value(val.clone()))
            .collect_vec();
        (
            const_loads
                .finish_hugr_with_outputs(outputs.clone())
                .unwrap(),
            outputs,
        )
    };

    // 2. insert the DFG Hugr into the circuit.
    let p = circuit.parent();
    let node_map = circuit.hugr_mut().insert_subgraph(
        p,
        &const_loads,
        &SiblingSubgraph::try_new_dataflow_subgraph::<_, DataflowParentID>(&const_loads).unwrap(),
    );
    let mut outputs = outputs
        .into_iter()
        .map(|w| Wire::new(*node_map.get(&w.node()).expect("inserted node"), w.source()));

    // 3. move wires linked to the inputs to the constants
    const_inputs.retain(|&(i, _)| {
        let new_const = outputs.next().expect("outputs.len() == const_inputs.len()");
        let old_input = Wire::new(circuit.input_node(), OutgoingPort::from(i));
        rewire(circuit, old_input, new_const)
    });

    // 4. Remove empty inputs
    let remove_inputs: BTreeSet<_> = const_inputs.iter().map(|(i, _)| *i).collect();
    let new_inputs = (0..circuit.circuit_signature().input_count())
        .filter(|i| !remove_inputs.contains(i))
        .collect_vec();
    let new_outputs = (0..circuit.circuit_signature().output_count()).collect_vec();
    let mut circuit: RootChecked<_, DataflowParentID> =
        circuit.hugr_mut().try_into_checked().unwrap();
    circuit
        .map_function_type(&new_inputs, &new_outputs)
        .expect("valid input transformation");

    const_inputs
}

/// For all incoming ports linked to `old_wire`, disconnect them and connect
/// them to `new_wire` instead.
///
/// If the types do not match, will try to add a type cast (between rotation
/// and float).
///
/// Return whether the rewiring was successful.
fn rewire(circuit: &mut Circuit, old_wire: Wire, new_wire: Wire) -> bool {
    let get_type = |wire: Wire| {
        circuit
            .hugr()
            .get_optype(wire.node())
            .dataflow_signature()
            .and_then(|s| s.out_port_type(wire.source()).cloned())
    };
    let Some(old_type) = get_type(old_wire) else {
        return false;
    };
    let Some(new_type) = get_type(new_wire) else {
        return false;
    };

    // Figure out whether a conversion between the two types is needed.
    let conversion_op = match (old_type, new_type) {
        (ot, nt) if ot == nt => None,
        (ot, nt) if ot == float64_type() && nt == rotation_type() => {
            Some(RotationOp::from_halfturns_unchecked)
        }
        (ot, nt) if ot == rotation_type() && nt == float64_type() => Some(RotationOp::to_halfturns),
        _ => {
            return false;
        }
    };

    // Insert conversion operation
    let conversion_op = conversion_op.map(|op| {
        let p = circuit.parent();
        circuit.hugr_mut().add_node_with_parent(p, op)
    });

    let inputs = circuit
        .hugr()
        .linked_inputs(old_wire.node(), old_wire.source())
        .collect_vec();

    // Disconnect old wire
    circuit
        .hugr_mut()
        .disconnect(old_wire.node(), old_wire.source());

    if let Some(conversion_op) = conversion_op {
        // Connect new output to conversion operation
        circuit
            .hugr_mut()
            .connect(new_wire.node(), new_wire.source(), conversion_op, 0);
        // connect conversion operation to all old inputs
        for (node, port) in inputs {
            circuit.hugr_mut().connect(conversion_op, 0, node, port);
        }
    } else {
        // Connect new output to all old inputs
        for (node, port) in inputs {
            circuit
                .hugr_mut()
                .connect(new_wire.node(), new_wire.source(), node, port);
        }
    }

    true
}

/// Expand the input signature of `circuit` by adding unused inputs of the
/// constant value's type at the given indices.
fn add_dummy_inputs(mut circuit: Circuit, inputs: Vec<(usize, constant::Value)>) -> Circuit {
    let old_num_inputs = circuit.circuit_signature().input_count();
    let num_outputs = circuit.circuit_signature().output_count();

    // 1. Add all new inputs at the end
    let new_inputs = inputs.iter().map(|(_, v)| v.get_type()).collect_vec();
    let mut root_checked: RootChecked<_, DataflowParentID> =
        circuit.hugr_mut().try_into_checked().unwrap();
    root_checked
        .extend_inputs(&new_inputs)
        .expect("all inputs copyable");

    let num_inputs = root_checked
        .hugr()
        .inner_function_type()
        .expect("valid DFG")
        .input_count();

    debug_assert_eq!(num_inputs, old_num_inputs + inputs.len());

    // 2. Update the signature
    let mut new_pos = vec![usize::MAX; num_inputs];
    // Put the new inputs at the right positions
    for (i, new_ind) in inputs.into_iter().map(|(new_ind, _)| new_ind).enumerate() {
        new_pos[new_ind] = i + old_num_inputs;
    }
    // Fill the gaps with the old inputs
    let empty_pos = new_pos.iter_mut().filter(|&&mut pos| pos == usize::MAX);
    for (i, pos) in empty_pos.enumerate() {
        *pos = i;
    }

    let unchanged_outputs = (0..num_outputs).collect_vec();
    root_checked
        .map_function_type(&new_pos, &unchanged_outputs)
        .expect("valid input transformation");

    circuit
}
