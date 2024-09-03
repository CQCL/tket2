use std::rc::Rc;

use thiserror::Error;

use crate::Tk2Op;

use super::{OpId, OpPosition, StaticQubitIndex, StaticSizeCircuit};

pub struct Command {
    pub op: Tk2Op,
    pub id: OpId,
    pub qubits: Vec<StaticQubitIndex>,
}

impl StaticSizeCircuit {
    /// Iterate over all commands in the circuit in topological order.
    pub fn commands(&self) -> CommandIter {
        CommandIter::new(self)
    }

    /// Check if the circuit is acyclic (i.e. if it is a valid DAG).
    pub fn is_acyclic(&self) -> bool {
        fn as_opt<V, E>(r: Result<Option<V>, E>) -> Option<Result<V, E>> {
            match r {
                Ok(None) => None,
                Ok(Some(v)) => Some(Ok(v)),
                Err(e) => Some(Err(e)),
            }
        }
        let mut cmds = self.commands();
        std::iter::from_fn(|| as_opt(cmds.try_next())).all(|x| x.is_ok())
    }
}

/// Traverse operations in the static circuit in topological order.
pub struct CommandIter<'a> {
    circuit: &'a StaticSizeCircuit,
    // For each qubit, the [0, x) interval that has been traversed
    traversed: Vec<usize>,
    // For each qubit, whether it is blocked (i.e. whether we are waiting for
    // ops on other qubits to be traversed first)
    blocked: Vec<bool>,
}

impl<'a> CommandIter<'a> {
    fn new(circuit: &'a StaticSizeCircuit) -> Self {
        let n = circuit.qubit_ops.len();
        Self {
            circuit,
            traversed: vec![0; n],
            blocked: vec![false; n],
        }
    }

    pub fn try_next(&mut self) -> Result<Option<Command>, NonConvexCircuitError> {
        loop {
            let is_full_len =
                |q: StaticQubitIndex| self.traversed[q.0] == self.circuit.qubit_ops[q.0].len();
            if self.circuit.qubits_iter().all(is_full_len) {
                return Ok(None);
            }
            let Some(curr_qubit) = self
                .circuit
                .qubits_iter()
                .find(|&q| !is_full_len(q) && !self.blocked[q.0])
            else {
                return Err(NonConvexCircuitError);
            };
            let curr_op_ind = self.traversed[curr_qubit.0];
            let id = self
                .circuit
                .at_position(OpPosition {
                    qubit: curr_qubit,
                    index: curr_op_ind,
                })
                .expect("checked above that we have not reached the end of the qubit");
            let positions = &self.circuit.get(id).unwrap().positions;
            let block = positions
                .iter()
                .any(|pos| self.traversed[pos.qubit.0] < pos.index);
            if block {
                self.blocked[curr_qubit.0] = true;
            } else {
                for OpPosition { qubit, index } in positions {
                    assert_eq!(self.traversed[qubit.0], *index);
                    self.traversed[qubit.0] += 1;
                    self.blocked[qubit.0] = false;
                }
                let qubits = positions
                    .iter()
                    .map(|OpPosition { qubit, .. }| *qubit)
                    .collect();
                let op = self.circuit.get(id).unwrap().op;
                return Ok(Some(Command { op, id, qubits }));
            }
        }
    }
}

impl<'a> Iterator for CommandIter<'a> {
    type Item = Command;

    fn next(&mut self) -> Option<Self::Item> {
        self.try_next().unwrap()
    }
}

#[derive(Debug, Error)]
#[error("Invalid circuit: non-convex")]
pub struct NonConvexCircuitError;
