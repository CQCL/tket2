//! A 2d array-like representation of simple quantum circuits.

mod hash;
mod iter;
// mod match_op;
mod position;
mod rewrite;

pub use hash::UpdatableHash;
use itertools::Itertools;
pub use position::OpPosition;
pub use rewrite::{BoxedStaticRewrite, NonConvexRewriteError, StaticRewrite};

use std::{fmt, mem};

use hugr::{Direction, Hugr, HugrView, Port, PortIndex};

use derive_more::{From, Into};
use thiserror::Error;

use crate::{
    circuit::{units::filter, RemoveEmptyWire, ToTk2OpIter},
    utils::build_simple_circuit,
    Circuit, CircuitMutError, Tk2Op,
};

/// A circuit operation
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    From,
    Into,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct StaticOp {
    /// The operation type
    pub op: Tk2Op,
    /// The positions of the operation (qubit and index)
    pub positions: Vec<OpPosition>,
}

/// An integer identifier for an operation in a `StaticSizeCircuit`.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    From,
    Into,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct OpId(usize);

/// A circuit with a fixed number of qubits numbered from 0 to `num_qubits - 1`.
#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct StaticSizeCircuit {
    /// A 2D array of quantum operations for each qubit.
    qubit_ops: Vec<Vec<OpId>>,
    /// A list of all quantum operations in the circuit.
    all_ops: Vec<StaticOp>,
}

impl PartialEq for StaticSizeCircuit {
    fn eq(&self, other: &Self) -> bool {
        let zip1 = self.qubit_ops.iter().zip(other.qubit_ops.iter());
        let zip2 = zip1.flat_map(|(a, b)| a.iter().zip(b.iter()));

        // Compare ops themselves, not their OpIds.
        zip2.map(|(&a, &b)| (self.get(a), other.get(b)))
            .all(|(a, b)| a == b)
    }
}

impl Eq for StaticSizeCircuit {}

impl StaticSizeCircuit {
    /// Create an empty `StaticSizeCircuit` with the given number of qubits.
    pub fn with_qubit_count(qubit_count: usize) -> Self {
        Self {
            qubit_ops: vec![Vec::new(); qubit_count],
            ..Self::default()
        }
    }

    /// Count the number of CX gates in the circuit.
    pub fn cx_count(&self) -> usize {
        self.ops_iter()
            .filter(|op| self.get(*op).unwrap().op == Tk2Op::CX)
            .count()
    }

    /// Add an input node to the circuit.
    ///
    /// Currently using Tk2Op::QAlloc.
    pub fn add_input(&mut self) {
        // Add the input node
        let qalloc = self.append_op(Tk2Op::QAlloc, []);
        let qubits = self.qubits_iter().collect_vec();
        for &qb in &qubits {
            self.qubit_ops[qb.0].insert(0, qalloc);
        }
        self.remap_positions(|pos| OpPosition {
            index: pos.index + 1,
            ..pos
        });
        for &qb in &qubits {
            self.all_ops[qalloc.0].positions.push(OpPosition {
                qubit: qb,
                index: 0,
            });
        }
    }

    /// Returns the number of qubits in the circuit.
    pub fn qubit_count(&self) -> usize {
        self.qubit_ops.len()
    }

    /// Returns the number of operations in the circuit.
    pub fn n_ops(&self) -> usize {
        self.all_ops.len()
    }

    /// Returns an iterator over the qubits in the circuit.
    pub fn qubits_iter(&self) -> impl ExactSizeIterator<Item = StaticQubitIndex> + '_ {
        (0..self.qubit_count()).map(StaticQubitIndex)
    }

    /// Returns the operations on a given qubit.
    pub fn qubit_ops(&self, qubit: StaticQubitIndex) -> &[OpId] {
        &self.qubit_ops[qubit.0]
    }

    /// Returns the operation with the given id.
    pub fn get(&self, op: OpId) -> Option<&StaticOp> {
        self.all_ops.get(op.0)
    }

    fn get_mut(&mut self, op: OpId) -> Option<&mut StaticOp> {
        self.all_ops.get_mut(op.0)
    }

    /// All positions of a given operation.
    pub fn positions(&self, op: OpId) -> Option<&[OpPosition]> {
        Some(&self.get(op)?.positions)
    }

    /// Returns the location and port of the operation linked to the given
    /// operation at the given port.
    pub fn linked_op(&self, op: OpId, port: Port) -> Option<(Port, OpId)> {
        let pos = self.get_position(op, port.index())?;
        match port.direction() {
            Direction::Outgoing => {
                let next_pos = OpPosition {
                    qubit: pos.qubit,
                    index: pos.index + 1,
                };
                if let Some(next_op) = self.at_position(next_pos) {
                    let offset = self.position_offset(next_pos).unwrap();
                    Some((Port::new(Direction::Incoming, offset), next_op))
                } else {
                    None
                }
            }
            Direction::Incoming => {
                let prev_index = pos.index.checked_sub(1)?;
                let prev_pos = OpPosition {
                    qubit: pos.qubit,
                    index: prev_index,
                };
                let offset = self.position_offset(prev_pos).unwrap();
                let prev_op = self.at_position(prev_pos).unwrap();
                Some((Port::new(Direction::Outgoing, offset), prev_op))
            }
        }
    }

    pub(crate) fn append_op(
        &mut self,
        op: Tk2Op,
        qubits: impl IntoIterator<Item = StaticQubitIndex>,
    ) -> OpId {
        let qubits = qubits.into_iter();
        let id = OpId(self.all_ops.len());
        let mut op = StaticOp {
            op,
            positions: Vec::new(),
        };
        for qubit in qubits {
            if qubit.0 >= self.qubit_count() {
                panic!(
                    "Cannot add op on qubit {qubit:?} to circuit with {} qubits",
                    self.qubit_count()
                );
            }
            let index = self.qubit_ops[qubit.0].len();
            self.qubit_ops[qubit.0].push(id);
            op.positions.push(OpPosition { qubit, index });
        }
        self.all_ops.push(op);
        id
    }

    /// Iterate over all operations in the circuit.
    pub fn ops_iter(&self) -> impl Iterator<Item = OpId> {
        (0..self.all_ops.len()).map(OpId)
    }
}

/// A qubit index within a `StaticSizeCircuit`.
#[repr(transparent)]
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    From,
    Into,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct StaticQubitIndex(pub(crate) usize);

// TODO: this is unsafe but was added for ECCRewriter to work.
impl<H: HugrView> From<H> for StaticSizeCircuit {
    fn from(hugr: H) -> Self {
        let circuit = Circuit::from(hugr);
        (&circuit).try_into().unwrap()
    }
}

impl<H: HugrView> TryFrom<&Circuit<H>> for StaticSizeCircuit {
    type Error = StaticSizeCircuitError;

    fn try_from(circuit: &Circuit<H>) -> Result<Self, Self::Error> {
        let mut res = Self::with_qubit_count(circuit.qubit_count());
        for cmd in circuit.commands() {
            let qubits = cmd
                .units(Direction::Incoming)
                .map(|unit| {
                    let Some((qb, _, _)) = filter::filter_qubit(unit) else {
                        return Err(StaticSizeCircuitError::NonQubitInput);
                    };
                    Ok(qb)
                })
                .collect::<Result<Vec<_>, _>>()?;
            if cmd.units(Direction::Outgoing).count() != qubits.len() {
                return Err(StaticSizeCircuitError::InvalidCircuit);
            }
            let op = cmd.optype().try_into().unwrap();
            res.append_op(op, qubits.into_iter().map(|u| StaticQubitIndex(u.index())));
        }
        Ok(res)
    }
}

impl From<StaticSizeCircuit> for Circuit<Hugr> {
    fn from(value: StaticSizeCircuit) -> Self {
        build_simple_circuit(value.qubit_count(), |circ| {
            for cmd in value.commands() {
                if cmd.op == Tk2Op::QAlloc {
                    // ignore qallocs (used as input)
                    continue;
                }
                circ.append(cmd.op, cmd.qubits.into_iter().map(|qb| qb.0))?;
            }
            Ok(())
        })
        .unwrap()
    }
}

/// Errors that can occur when converting a `Circuit` to a `StaticSizeCircuit`.
#[derive(Debug, Error)]
pub enum StaticSizeCircuitError {
    /// An input to a gate was not a qubit.
    #[error("Only qubits are supported as inputs")]
    NonQubitInput,

    /// The given tket2 circuit cannot be expressed as a StaticSizeCircuit.
    #[error("The given tket2 circuit cannot be expressed as a StaticSizeCircuit")]
    InvalidCircuit,
}

impl fmt::Debug for StaticSizeCircuit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StaticSizeCircuit")
            .field("qubit_ops", &self.qubit_ops)
            .finish()
    }
}

impl fmt::Display for StaticSizeCircuit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for qb in self.qubits_iter() {
            write!(f, "Qubit {}:", qb.0)?;
            for &op in self.qubit_ops(qb) {
                let positions = &self.get(op).unwrap().positions;
                let qubits = positions
                    .iter()
                    .map(|pos| format!("{}", pos.qubit.0))
                    .join(", ");
                write!(f, " {op:?} ({qubits})")?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl ToTk2OpIter for StaticSizeCircuit {
    type Iter<'a> = Box<dyn Iterator<Item = Tk2Op> + 'a> where Self: 'a;

    fn tk2_ops(&self) -> Self::Iter<'_> {
        Box::new(self.ops_iter().map(|op| self.get(op).unwrap().op))
    }
}

impl RemoveEmptyWire for StaticSizeCircuit {
    fn remove_empty_wire(&mut self, input_port: usize) -> Result<(), CircuitMutError> {
        self.qubit_ops.remove(input_port);
        self.remap_positions(|loc| {
            assert_ne!(loc.qubit.0, input_port, "input port is not empty");
            let qubit = if loc.qubit.0 > input_port {
                StaticQubitIndex(loc.qubit.0 - 1)
            } else {
                loc.qubit
            };
            OpPosition { qubit, ..loc }
        });
        Ok(())
    }

    fn empty_wires(&self) -> Vec<usize> {
        self.qubits_iter()
            .filter(|&qb| self.qubit_ops(qb).is_empty())
            .map(|qb| qb.0)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use hugr::{Port, PortIndex};
    use portgraph::PortOffset;
    use rstest::rstest;

    use super::StaticSizeCircuit;
    use crate::ops::Tk2Op;
    use crate::static_circ::OpPosition;
    use crate::utils::build_simple_circuit;

    #[test]
    fn test_convert_to_static_size_circuit() {
        // Create a circuit with 2 qubits, a CX gate, and two H gates
        let circuit = build_simple_circuit(2, |circ| {
            circ.append(Tk2Op::H, [0])?;
            circ.append(Tk2Op::CX, [0, 1])?;
            circ.append(Tk2Op::H, [1])?;
            Ok(())
        })
        .unwrap();

        // Convert the circuit to StaticSizeCircuit
        let static_circuit: StaticSizeCircuit = (&circuit).try_into().unwrap();

        // Check the conversion
        assert_eq!(static_circuit.qubit_count(), 2);
        assert_eq!(static_circuit.qubit_ops(0.into()).len(), 2); // H gate on qubit 0
        assert_eq!(static_circuit.qubit_ops(1.into()).len(), 2); // CX and H gate on qubit 1
    }

    #[rstest]
    #[case(PortOffset::Outgoing(0), None)]
    #[case(PortOffset::Incoming(1), None)]
    #[case(
        PortOffset::Outgoing(1),
        Some((PortOffset::Incoming(0).into(), OpPosition {
            qubit: 1.into(),
            index: 1,
        }))
    )]
    #[case(
        PortOffset::Incoming(0),
        Some((PortOffset::Outgoing(0).into(), OpPosition {
            qubit: 0.into(),
            index: 0,
        }))
    )]
    fn test_linked_op(#[case] port: PortOffset, #[case] expected_loc: Option<(Port, OpPosition)>) {
        let circuit = build_simple_circuit(2, |circ| {
            circ.append(Tk2Op::H, [0])?;
            circ.append(Tk2Op::CX, [0, 1])?;
            circ.append(Tk2Op::H, [1])?;
            Ok(())
        })
        .unwrap();
        // Convert the circuit to StaticSizeCircuit
        let static_circuit: StaticSizeCircuit = (&circuit).try_into().unwrap();

        // Define the location of the CX gate
        let cx_pos = OpPosition {
            qubit: 0.into(),
            index: 1,
        };

        // Define the port for the CX gate
        let cx_port = port.into();
        let cx_op_id = static_circuit.at_position(cx_pos).unwrap();

        // Get the linked operation for the CX gate
        let linked_op_location = static_circuit
            .linked_op(cx_op_id, cx_port)
            .map(|(port, op)| (port, static_circuit.get_position(op, port.index()).unwrap()));

        // Check if the linked operation is correct
        assert_eq!(linked_op_location, expected_loc);
    }

    #[test]
    fn test_add_subcircuit() {
        // Create a main circuit
        let main_circuit = build_simple_circuit(3, |circ| {
            circ.append(Tk2Op::H, [0])?;
            circ.append(Tk2Op::CX, [0, 1])?;
            Ok(())
        })
        .unwrap();
        let mut main_circuit: StaticSizeCircuit = (&main_circuit).try_into().unwrap();

        // Create a subcircuit
        let sub_circuit = build_simple_circuit(2, |circ| {
            circ.append(Tk2Op::X, [0])?;
            circ.append(Tk2Op::X, [0])?;
            circ.append(Tk2Op::X, [0])?;
            circ.append(Tk2Op::Y, [1])?;
            Ok(())
        })
        .unwrap();
        let sub_circuit: StaticSizeCircuit = (&sub_circuit).try_into().unwrap();

        // Define the nodes to add from the subcircuit
        let nodes = BTreeSet::from_iter(
            [
                OpPosition {
                    qubit: 0.into(),
                    index: 0,
                },
                OpPosition {
                    qubit: 0.into(),
                    index: 2,
                },
                OpPosition {
                    qubit: 1.into(),
                    index: 0,
                },
            ]
            .into_iter()
            .map(|pos| sub_circuit.at_position(pos).unwrap()),
        );

        // Add the subcircuit to the main circuit
        let node_map = main_circuit.add_subcircuit(&sub_circuit, &nodes);

        // Check if the nodes were added correctly
        assert_eq!(node_map.len(), 3);

        // Three original qubits + first new qubit split into two + second new qubit
        assert_eq!(main_circuit.qubit_count(), 6);

        // Check if the operations were added to the main circuit
        assert_eq!(main_circuit.qubit_ops(3.into()).len(), 1); // First X gate
        assert_eq!(main_circuit.qubit_ops(4.into()).len(), 1); // Second X gate
        assert_eq!(main_circuit.qubit_ops(5.into()).len(), 1); // Y gate

        insta::assert_debug_snapshot!(main_circuit);
    }

    #[test]
    fn serde_roundtrip() {
        let mut circ = StaticSizeCircuit::with_qubit_count(2);
        circ.append_op(Tk2Op::CX, vec![0.into(), 1.into()]);
        let ser = serde_json::to_string_pretty(&circ).unwrap();
        let circ2: StaticSizeCircuit = serde_json::from_str(&ser).unwrap();
        assert_eq!(circ, circ2);
    }
}
