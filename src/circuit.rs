//! Quantum circuit representation and operations.

pub mod command;
mod hash;
mod units;

pub use command::{Command, CommandIterator};
pub use hash::CircuitHash;

use hugr::hugr::NodeType;
use hugr::HugrView;

pub use hugr::ops::OpType;
use hugr::types::FunctionType;
pub use hugr::types::{EdgeKind, Signature, Type, TypeRow};
pub use hugr::{Node, Port, Wire};

use self::units::{UnitType, Units};

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
        return self
            .children(self.root())
            .next()
            .expect("Circuit has no input node");
    }

    /// Returns the output node to the circuit.
    #[inline]
    fn output(&self) -> Node {
        return self
            .children(self.root())
            .nth(1)
            .expect("Circuit has no output node");
    }

    /// The number of gates in the circuit.
    #[inline]
    fn num_gates(&self) -> usize {
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
        Units::new_circ_input(self, UnitType::All)
    }

    /// Get the linear input units of the circuit and their types.
    #[inline]
    fn linear_units(&self) -> Units
    where
        Self: Sized,
    {
        Units::new_circ_input(self, UnitType::Linear)
    }

    /// Get the non-linear input units of the circuit and their types.
    #[inline]
    fn nonlinear_units(&self) -> Units
    where
        Self: Sized,
    {
        Units::new_circ_input(self, UnitType::NonLinear)
    }

    /// Returns the units corresponding to qubits inputs to the circuit.
    #[inline]
    fn qubits(&self) -> Units
    where
        Self: Sized,
    {
        Units::new_circ_input(self, UnitType::Qubits)
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

    /// Returns the [`NodeType`] of a command.
    fn command_nodetype(&self, command: &Command) -> &NodeType {
        self.get_nodetype(command.node())
    }

    /// Returns the [`OpType`] of a command.
    fn command_optype(&self, command: &Command) -> &OpType {
        self.get_optype(command.node())
    }
}

impl<T> Circuit for T where T: HugrView {}

#[cfg(test)]
mod tests {
    use hugr::Hugr;

    use crate::{circuit::Circuit, json::load_tk1_json_str};

    fn test_circuit() -> Hugr {
        load_tk1_json_str(
            r#"{
            "phase": "0",
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

        assert!(circ.linear_units().all(|(unit, _, _)| unit.is_linear()));
        assert!(circ.nonlinear_units().all(|(unit, _, _)| unit.is_wire()));
    }
}
