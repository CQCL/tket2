use std::rc::Rc;

use thiserror::Error;

use crate::Tk2Op;

use super::{OpLocation, StaticQubitIndex, StaticSizeCircuit};

pub struct Command {
    pub op: Tk2Op,
    pub qubits: Vec<StaticQubitIndex>,
}

impl StaticSizeCircuit {
    pub fn commands(&self) -> impl Iterator<Item = Command> + '_ {
        CommandIter::new(self)
    }
}

/// Traverse operations in the static circuit in topological order.
struct CommandIter<'a> {
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

    pub(crate) fn try_next(&mut self) -> Result<Option<Command>, NonConvexCircuitError> {
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
            let op_ptr = Rc::as_ptr(&self.circuit.qubit_ops[curr_qubit.0][curr_op_ind]);
            let op_locations = &self.circuit.op_locations[&op_ptr];
            let block = op_locations.iter().any(|loc| {
                let OpLocation { qubit, op_idx } = loc;
                self.traversed[qubit.0] < *op_idx
            });
            if block {
                self.blocked[curr_qubit.0] = true;
            } else {
                for OpLocation { qubit, op_idx } in op_locations {
                    assert_eq!(self.traversed[qubit.0], *op_idx);
                    self.traversed[qubit.0] += 1;
                    self.blocked[qubit.0] = false;
                }
                return Ok(Some(Command {
                    op: *self.circuit.qubit_ops[curr_qubit.0][curr_op_ind],
                    qubits: op_locations
                        .iter()
                        .map(|OpLocation { qubit, .. }| *qubit)
                        .collect(),
                }));
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
struct NonConvexCircuitError;
