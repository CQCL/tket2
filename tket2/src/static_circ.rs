//! A 2d array-like representation of simple quantum circuits.

mod match_op;

use hugr::{Direction, HugrView};
pub(crate) use match_op::MatchOp;

use derive_more::{From, Into};

use crate::{circuit::units::filter, Circuit};

/// A circuit with a fixed number of qubits numbered from 0 to `num_qubits - 1`.
pub(crate) struct StaticSizeCircuit {
    /// All quantum operations on qubits.
    qubit_ops: Vec<Vec<StaticOp>>,
}

impl StaticSizeCircuit {
    /// Returns the number of qubits in the circuit.
    #[allow(unused)]
    pub fn qubit_count(&self) -> usize {
        self.qubit_ops.len()
    }

    /// Returns the operations on a given qubit.
    #[allow(unused)]
    pub fn qubit_ops(&self, qubit: usize) -> &[StaticOp] {
        &self.qubit_ops[qubit]
    }
}

/// A qubit index within a `StaticSizeCircuit`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, From, Into)]
pub(crate) struct StaticQubitIndex(usize);

/// An operation in a `StaticSizeCircuit`.
///
/// Currently only support quantum operations without any classical IO.
#[derive(Debug, Clone)]
pub(crate) struct StaticOp {
    #[allow(unused)]
    op: MatchOp,
    #[allow(unused)]
    qubits: Vec<StaticQubitIndex>,
    // TODO: clbits
}

impl<H: HugrView> TryFrom<&Circuit<H>> for StaticSizeCircuit {
    type Error = StaticSizeCircuitError;

    fn try_from(circuit: &Circuit<H>) -> Result<Self, Self::Error> {
        let mut qubit_ops = vec![Vec::new(); circuit.qubit_count()];
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
            let op = StaticOp {
                op: cmd.optype().clone().into(),
                qubits: qubits
                    .iter()
                    .copied()
                    .map(|u| StaticQubitIndex(u.index()))
                    .collect(),
            };
            for qb in qubits {
                qubit_ops[qb.index()].push(op.clone());
            }
        }
        Ok(Self { qubit_ops })
    }
}

use thiserror::Error;

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

#[cfg(test)]
mod tests {
    use super::StaticSizeCircuit;
    use crate::ops::Tk2Op;
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
        assert_eq!(static_circuit.qubit_ops(0).len(), 2); // H gate on qubit 0
        dbg!(static_circuit.qubit_ops(0));
        assert_eq!(static_circuit.qubit_ops(1).len(), 2); // CX and H gate on qubit 1
    }
}
