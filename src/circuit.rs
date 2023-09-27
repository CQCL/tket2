//! Quantum circuit representation and operations.

pub mod command;
mod hash;
pub mod units;

pub use command::{Command, CommandIterator};
pub use hash::CircuitHash;

use hugr::HugrView;

use derive_more::From;
use hugr::hugr::hugrmut::HugrMut;
use hugr::hugr::{NodeType, PortIndex};
use hugr::ops::dataflow::IOTrait;
pub use hugr::ops::OpType;
use hugr::ops::{Input, Output, DFG};
use hugr::types::FunctionType;
pub use hugr::types::{EdgeKind, Signature, Type, TypeRow};
pub use hugr::{Node, Port, Wire};
use itertools::Itertools;
use thiserror::Error;

use self::units::{filter, FilteredUnits, Units};

/// An object behaving like a quantum circuit.
//
// TODO: More methods:
// - other_{in,out}puts (for non-linear i/o + const inputs)?
// - Vertical slice iterator
// - Depth
pub trait Circuit: HugrView {
    /// Return the name of the circuit
    #[inline]
    fn name(&self) -> Option<&str> {
        let meta = self.get_metadata(self.root()).as_object()?;
        meta.get("name")?.as_str()
    }

    /// Returns the function type of the circuit.
    ///
    /// Equivalent to [`HugrView::get_function_type`].
    #[inline]
    fn circuit_signature(&self) -> &FunctionType {
        self.get_function_type()
            .expect("Circuit has no function type")
    }

    /// Returns the input node to the circuit.
    #[inline]
    fn input(&self) -> Node {
        self.get_io(self.root()).expect("Circuit has no input node")[0]
    }

    /// Returns the output node to the circuit.
    #[inline]
    fn output(&self) -> Node {
        self.get_io(self.root())
            .expect("Circuit has no output node")[1]
    }

    /// The number of quantum gates in the circuit.
    #[inline]
    fn num_gates(&self) -> usize
    where
        Self: Sized,
    {
        // TODO: Implement discern quantum gates in the commands iterator.
        self.children(self.root()).count() - 2
    }

    /// Count the number of qubits in the circuit.
    #[inline]
    fn qubit_count(&self) -> usize
    where
        Self: Sized,
    {
        self.qubits().count()
    }

    /// Get the input units of the circuit and their types.
    #[inline]
    fn units(&self) -> Units
    where
        Self: Sized,
    {
        Units::new_circ_input(self)
    }

    /// Get the linear input units of the circuit and their types.
    #[inline]
    fn linear_units(&self) -> FilteredUnits<filter::Linear>
    where
        Self: Sized,
    {
        self.units().filter_units::<filter::Linear>()
    }

    /// Get the non-linear input units of the circuit and their types.
    #[inline]
    fn nonlinear_units(&self) -> FilteredUnits<filter::NonLinear>
    where
        Self: Sized,
    {
        self.units().filter_units::<filter::NonLinear>()
    }

    /// Returns the units corresponding to qubits inputs to the circuit.
    #[inline]
    fn qubits(&self) -> FilteredUnits<filter::Qubits>
    where
        Self: Sized,
    {
        self.units().filter_units::<filter::Qubits>()
    }

    /// Returns all the commands in the circuit, in some topological order.
    ///
    /// Ignores the Input and Output nodes.
    #[inline]
    fn commands(&self) -> CommandIterator<'_, Self>
    where
        Self: Sized,
    {
        // Traverse the circuit in topological order.
        CommandIterator::new(self)
    }
}

/// A circuit object that can be mutated.
pub trait CircuitMut: Circuit + HugrMut {
    /// Remove an empty wire from the circuit.
    ///
    /// The wire to be removed is identified by the index of the outgoing port
    /// at the circuit input node.
    ///
    /// This will change the circuit signature and will shift all ports after
    /// the removed wire by -1. If the wire is connected to the output node,
    /// this will also change the signature output and shift the ports after
    /// the removed wire by -1.
    ///
    /// This will return an error if the wire is not empty or if a HugrError
    /// occurs.
    fn remove_empty_wire(&mut self, input_port: usize) -> Result<(), CircuitMutError> {
        let inp = self.input();
        if input_port >= self.num_outputs(inp) {
            return Err(CircuitMutError::InvalidPortOffset(input_port));
        }
        let input_port = Port::new_outgoing(input_port);
        let link = self
            .linked_ports(inp, input_port)
            .at_most_one()
            .ok()
            .expect("invalid circuit");
        if link.is_some() && link.unwrap().0 != self.output() {
            return Err(CircuitMutError::DeleteNonEmptyWire(input_port.index()));
        }
        if link.is_some() {
            self.disconnect(inp, input_port)?;
        }

        // Shift ports at input
        shift_ports(self, inp, input_port, self.num_outputs(inp))?;
        // Shift ports at output
        if let Some((out, output_port)) = link {
            shift_ports(self, out, output_port, self.num_inputs(out))?;
        }
        // Update input node, output node (if necessary) and root signatures.
        update_signature(self, input_port.index(), link.map(|(_, p)| p.index()));
        Ok(())
    }
}

/// Errors that can occur when mutating a circuit.
#[derive(Debug, Clone, Error, PartialEq, Eq, From)]
pub enum CircuitMutError {
    /// A Hugr error occurred.
    #[error("Hugr error: {0:?}")]
    HugrError(hugr::hugr::HugrError),
    /// The wire to be deleted is not empty.
    #[from(ignore)]
    #[error("Wire {0} cannot be deleted: not empty")]
    DeleteNonEmptyWire(usize),
    /// The wire does not exist.
    #[from(ignore)]
    #[error("Wire {0} does not exist")]
    InvalidPortOffset(usize),
}

/// Shift ports in range (free_port + 1 .. max_ind) by -1.
fn shift_ports<C: HugrMut + ?Sized>(
    circ: &mut C,
    node: Node,
    mut free_port: Port,
    max_ind: usize,
) -> Result<Port, hugr::hugr::HugrError> {
    let dir = free_port.direction();
    let port_range = (free_port.index() + 1..max_ind).map(|p| Port::new(dir, p));
    for port in port_range {
        if let Some(connected_to) = circ
            .linked_ports(node, port)
            .at_most_one()
            .ok()
            .expect("invalid circuit")
        {
            circ.disconnect(node, port)?;
            circ.connect(node, free_port, connected_to.0, connected_to.1)?;
        }
        free_port = port;
    }
    Ok(free_port)
}

// Update the signature of circ when removing the in_index-th input wire and
// the out_index-th output wire.
fn update_signature<C: HugrMut + Circuit + ?Sized>(
    circ: &mut C,
    in_index: usize,
    out_index: Option<usize>,
) {
    let inp = circ.input();
    // Update input node
    let inp_types: TypeRow = {
        let OpType::Input(Input { types }) = circ.get_optype(inp).clone() else {
            panic!("invalid circuit")
        };
        let mut types = types.into_owned();
        types.remove(in_index);
        types.into()
    };
    let new_inp_op = Input::new(inp_types.clone());
    let inp_exts = circ.get_nodetype(inp).input_extensions().cloned();
    circ.replace_op(inp, NodeType::new(new_inp_op, inp_exts));

    // Update output node if necessary.
    let out_types = out_index.map(|out_index| {
        let out = circ.output();
        let out_types: TypeRow = {
            let OpType::Output(Output { types }) = circ.get_optype(out).clone() else {
                panic!("invalid circuit")
            };
            let mut types = types.into_owned();
            types.remove(out_index);
            types.into()
        };
        let new_out_op = Input::new(out_types.clone());
        let inp_exts = circ.get_nodetype(out).input_extensions().cloned();
        circ.replace_op(out, NodeType::new(new_out_op, inp_exts));
        out_types
    });

    // Update root
    let OpType::DFG(DFG { mut signature, .. }) = circ.get_optype(circ.root()).clone() else {
        panic!("invalid circuit")
    };
    signature.input = inp_types;
    if let Some(out_types) = out_types {
        signature.output = out_types;
    }
    let new_dfg_op = DFG { signature };
    let inp_exts = circ.get_nodetype(circ.root()).input_extensions().cloned();
    circ.replace_op(circ.root(), NodeType::new(new_dfg_op, inp_exts));
}

impl<T> Circuit for T where T: HugrView {}
impl<T> CircuitMut for T where T: Circuit + HugrMut {}

#[cfg(test)]
mod tests {
    use hugr::{
        builder::{DFGBuilder, DataflowHugr},
        extension::{prelude::BOOL_T, PRELUDE_REGISTRY},
        Hugr,
    };

    use super::*;
    use crate::{json::load_tk1_json_str, utils::build_simple_circuit, T2Op};

    fn test_circuit() -> Hugr {
        load_tk1_json_str(
            r#"{ "phase": "0",
            "bits": [["c", [0]]],
            "qubits": [["q", [0]], ["q", [1]]],
            "commands": [
                {"args": [["q", [0]]], "op": {"type": "H"}},
                {"args": [["q", [0]], ["q", [1]]], "op": {"type": "CX"}},
                {"args": [["q", [1]]], "op": {"type": "X"}}
            ],
            "implicit_permutation": [[["q", [0]], ["q", [0]]], [["q", [1]], ["q", [1]]]]
        }"#,
        )
        .unwrap()
    }

    #[test]
    fn test_circuit_properties() {
        let circ = test_circuit();

        assert_eq!(circ.name(), None);
        assert_eq!(circ.circuit_signature().input.len(), 3);
        assert_eq!(circ.circuit_signature().output.len(), 3);
        assert_eq!(circ.qubit_count(), 2);
        assert_eq!(circ.num_gates(), 3);

        assert_eq!(circ.units().count(), 3);
        assert_eq!(circ.nonlinear_units().count(), 0);
        assert_eq!(circ.linear_units().count(), 3);
        assert_eq!(circ.qubits().count(), 2);
    }

    #[test]
    fn remove_qubit() {
        let mut circ = build_simple_circuit(2, |circ| {
            circ.append(T2Op::X, [0])?;
            Ok(())
        })
        .unwrap();

        assert_eq!(circ.qubit_count(), 2);
        assert!(circ.remove_empty_wire(1).is_ok());
        assert_eq!(circ.qubit_count(), 1);
        assert_eq!(
            circ.remove_empty_wire(0).unwrap_err(),
            CircuitMutError::DeleteNonEmptyWire(0)
        );
    }

    #[test]
    fn remove_bit() {
        let h = DFGBuilder::new(FunctionType::new(vec![BOOL_T], vec![])).unwrap();
        let mut circ = h.finish_hugr_with_outputs([], &PRELUDE_REGISTRY).unwrap();

        assert_eq!(circ.units().count(), 1);
        assert!(circ.remove_empty_wire(0).is_ok());
        assert_eq!(circ.units().count(), 0);
        assert_eq!(
            circ.remove_empty_wire(2).unwrap_err(),
            CircuitMutError::InvalidPortOffset(2)
        );
    }
}
