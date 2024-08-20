//! A 2d array-like representation of simple quantum circuits.

mod match_op;

use std::{collections::BTreeMap, fmt, rc::Rc};

use hugr::{Direction, HugrView, Port, PortIndex};
pub(crate) use match_op::MatchOp;

use derive_more::{From, Into};
use thiserror::Error;

use crate::{circuit::units::filter, Circuit};

/// A circuit with a fixed number of qubits numbered from 0 to `num_qubits - 1`.
#[derive(Clone, Default)]
pub struct StaticSizeCircuit {
    /// All quantum operations on qubits.
    qubit_ops: Vec<Vec<Rc<MatchOp>>>,
    /// Map operations to their locations in `qubit_ops`.
    op_locations: BTreeMap<MatchOpPtr, Vec<OpLocation>>,
}

type MatchOpPtr = *const MatchOp;

/// The location of an operation in a `StaticSizeCircuit`.
///
/// Given by the qubit index and the position within that qubit's op list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpLocation {
    /// The index of the qubit the operation acts on.
    qubit: StaticQubitIndex,
    /// The index of the operation in the qubit's operation list.
    op_idx: usize,
}

impl StaticSizeCircuit {
    /// Returns the number of qubits in the circuit.
    #[allow(unused)]
    pub fn qubit_count(&self) -> usize {
        self.qubit_ops.len()
    }

    /// Returns an iterator over the qubits in the circuit.
    pub fn qubits_iter(&self) -> impl ExactSizeIterator<Item = StaticQubitIndex> + '_ {
        (0..self.qubit_count()).map(StaticQubitIndex)
    }

    /// Returns the operations on a given qubit.
    pub fn qubit_ops(
        &self,
        qubit: StaticQubitIndex,
    ) -> impl ExactSizeIterator<Item = &MatchOp> + '_ {
        self.qubit_ops[qubit.0].iter().map(|op| op.as_ref())
    }

    fn get(&self, loc: OpLocation) -> Option<&Rc<MatchOp>> {
        self.qubit_ops.get(loc.qubit.0)?.get(loc.op_idx)
    }

    fn exists(&self, loc: OpLocation) -> bool {
        self.qubit_ops
            .get(loc.qubit.0)
            .map_or(false, |ops| ops.get(loc.op_idx).is_some())
    }

    /// The port of the operation that `loc` is at.
    fn qubit_port(&self, loc: OpLocation) -> usize {
        let op = self.get(loc).unwrap();
        self.op_location(op).iter().position(|l| l == &loc).unwrap()
    }

    /// Returns the location and port of the operation linked to the given
    /// operation at the given port.
    pub fn linked_op(&self, loc: OpLocation, port: Port) -> Option<(Port, OpLocation)> {
        let op = self.get(loc)?;
        let loc = self.op_location(op).get(port.index())?;
        match port.direction() {
            Direction::Outgoing => {
                let next_loc = OpLocation {
                    qubit: loc.qubit,
                    op_idx: loc.op_idx + 1,
                };
                if self.exists(next_loc) {
                    let index = self.qubit_port(next_loc);
                    Some((Port::new(Direction::Incoming, index), next_loc))
                } else {
                    None
                }
            }
            Direction::Incoming => {
                if loc.op_idx == 0 {
                    None
                } else {
                    let prev_loc = OpLocation {
                        qubit: loc.qubit,
                        op_idx: loc.op_idx - 1,
                    };
                    let index = self.qubit_port(prev_loc);
                    Some((Port::new(Direction::Outgoing, index), prev_loc))
                }
            }
        }
    }

    fn op_location(&self, op: &Rc<MatchOp>) -> &[OpLocation] {
        self.op_locations[&Rc::as_ptr(op)].as_slice()
    }

    fn append_op(&mut self, op: MatchOp, qubits: impl IntoIterator<Item = StaticQubitIndex>) {
        let qubits = qubits.into_iter();
        let op = Rc::new(op);
        let op_ptr = Rc::as_ptr(&op);
        for qubit in qubits {
            if qubit.0 >= self.qubit_count() {
                self.qubit_ops.resize(qubit.0 + 1, Vec::new());
            }
            let op_idx = self.qubit_ops[qubit.0].len();
            self.qubit_ops[qubit.0].push(op.clone());
            self.op_locations
                .entry(op_ptr)
                .or_default()
                .push(OpLocation { qubit, op_idx });
        }
    }

    #[allow(unused)]
    fn all_ops_iter(&self) -> impl Iterator<Item = &Rc<MatchOp>> {
        self.qubit_ops.iter().flat_map(|ops| ops.iter())
    }
}

/// A qubit index within a `StaticSizeCircuit`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, From, Into)]
pub struct StaticQubitIndex(usize);

impl<H: HugrView> TryFrom<&Circuit<H>> for StaticSizeCircuit {
    type Error = StaticSizeCircuitError;

    fn try_from(circuit: &Circuit<H>) -> Result<Self, Self::Error> {
        let mut res = Self::default();
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
            let op = cmd.optype().clone().into();
            res.append_op(op, qubits.into_iter().map(|u| StaticQubitIndex(u.index())));
        }
        Ok(res)
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

impl PartialEq for StaticSizeCircuit {
    fn eq(&self, other: &Self) -> bool {
        self.qubit_ops == other.qubit_ops
    }
}

impl Eq for StaticSizeCircuit {}

#[cfg(test)]
mod tests {
    use hugr::Port;
    use portgraph::PortOffset;
    use rstest::rstest;

    use super::StaticSizeCircuit;
    use crate::ops::Tk2Op;
    use crate::static_circ::OpLocation;
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
        Some((PortOffset::Incoming(0).into(), OpLocation {
            qubit: 1.into(),
            op_idx: 1,
        }))
    )]
    #[case(
        PortOffset::Incoming(0),
        Some((PortOffset::Outgoing(0).into(), OpLocation {
            qubit: 0.into(),
            op_idx: 0,
        }))
    )]
    fn test_linked_op(#[case] port: PortOffset, #[case] expected_loc: Option<(Port, OpLocation)>) {
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
        let cx_location = OpLocation {
            qubit: 0.into(),
            op_idx: 1,
        };

        // Define the port for the CX gate
        let cx_port = port.into();

        // Get the linked operation for the CX gate
        let linked_op_location = static_circuit.linked_op(cx_location, cx_port);

        // Check if the linked operation is correct
        assert_eq!(linked_op_location, expected_loc);
    }
}
