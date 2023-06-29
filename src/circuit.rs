//! Quantum circuit representation and operations.

use hugr::ops::OpTrait;
pub use hugr::ops::OpType as Op;
pub use hugr::types::{ClassicType, EdgeKind, LinearType, Signature, SimpleType, TypeRow};
use hugr::{Direction, HugrView};
pub use hugr::{Node, Port, Wire};

//#[cfg(feature = "pyo3")]
//pub mod py_circuit;

//#[cfg(feature = "tkcxx")]
//pub mod unitarybox;

// TODO: Move TKET1's custom op definition to tket-rs (or hugr?)
//mod tk1ops;

/// A TKET2 circuit.
//
// TODO: Commands iterator
// TODO: Vertical slice iterator
// TODO: Gate count map
// TODO: Depth
pub trait Circuit: HugrView {
    /// Return the name of the circuit
    fn name(&self) -> Option<&str> {
        let meta = self.get_metadata(self.root()).as_object()?;
        meta.get("name")?.as_str()
    }

    /// Get the boundary types of a circuit in a given direction.
    //
    // TODO: We cannot return an iterator because the Signature is a temporary value.
    //       We could change `Signature::df_types` to return a TypeRow.
    #[inline]
    fn boundary(&self, dir: Direction) -> Vec<(Port, SimpleType)> {
        let root = self.root();
        let optype = self.get_optype(root);
        optype
            .signature()
            .df_types(dir)
            .iter()
            .enumerate()
            .map(|(i, typ)| {
                let port = Port::new(dir, i);
                (port, typ.clone())
            })
            .collect()
    }

    /// Input types to the circuit.
    #[inline]
    fn inputs(&self) -> Vec<(Port, SimpleType)> {
        self.boundary(Direction::Incoming)
    }

    /// Output types from the circuit.
    #[inline]
    fn outputs(&self) -> Vec<(Port, SimpleType)> {
        self.boundary(Direction::Outgoing)
    }

    /// Returns the ports corresponding to qubits inputs to the circuit.
    fn qubits(&self) -> Vec<Port> {
        self.inputs()
            .iter()
            .filter(|(_, typ)| typ == &LinearType::Qubit.into())
            .map(|(port, _)| *port)
            .collect()
    }

    /// Returns the ports corresponding to linear inputs to the circuit.
    fn linear_inputs(&self) -> Vec<(Port, SimpleType)> {
        self.inputs()
            .iter()
            .filter(|(_, typ)| typ.is_linear())
            .cloned()
            .collect()
    }

    /// Returns the ports corresponding to linear inputs to the circuit.
    fn classical_inputs(&self) -> Vec<(Port, SimpleType)> {
        self.inputs()
            .iter()
            .filter(|(_, typ)| typ.is_classical())
            .cloned()
            .collect()
    }
}

impl<T: HugrView> Circuit for T {}
