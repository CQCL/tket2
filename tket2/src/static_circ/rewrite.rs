use std::{collections::BTreeMap, ops::Range, rc::Rc};

use derive_more::{From, Into};
use thiserror::Error;

use super::{OpLocation, StaticQubitIndex, StaticSizeCircuit};

/// An interval of operation indices.
#[derive(Debug, Clone, PartialEq, Eq, From, Into)]
pub(super) struct OpInterval(pub Range<usize>);

/// A subcircuit of a static circuit.
#[derive(Debug, Clone, PartialEq, Eq, From, Into)]
pub struct StaticSubcircuit {
    /// Maps qubit indices to the intervals of operations on that qubit.
    pub(super) op_indices: BTreeMap<StaticQubitIndex, OpInterval>,
}

impl StaticSubcircuit {
    /// The subcircuit before `self`.
    fn before(&self, circuit: &StaticSizeCircuit) -> Self {
        let mut op_indices = BTreeMap::new();
        for qb in circuit.qubits_iter() {
            if let Some(interval) = self.op_indices.get(&qb) {
                let start = interval.0.start;
                op_indices.insert(qb, OpInterval(0..start));
            } else {
                op_indices.insert(qb, OpInterval(0..circuit.qubit_ops(qb).len()));
            }
        }
        StaticSubcircuit { op_indices }
    }

    /// The subcircuit after `self`.
    fn after(&self, circuit: &StaticSizeCircuit) -> Self {
        let op_indices = self
            .op_indices
            .iter()
            .map(|(&qb, interval)| (qb, OpInterval(interval.0.end..circuit.qubit_ops(qb).len())))
            .collect();
        StaticSubcircuit { op_indices }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("invalid subcircuit")]
pub struct InvalidSubcircuitError;

impl StaticSizeCircuit {
    fn subcircuit(&self, subcircuit: &StaticSubcircuit) -> Result<Self, InvalidSubcircuitError> {
        let Self {
            mut qubit_ops,
            mut op_locations,
        } = self.clone();
        for (qb, interval) in subcircuit.op_indices.iter() {
            for op in qubit_ops[qb.0].drain(interval.0.end..) {
                op_locations.remove(&Rc::as_ptr(&op));
            }
            for op in qubit_ops[qb.0].drain(..interval.0.start) {
                op_locations.remove(&Rc::as_ptr(&op));
            }
        }
        let ret = Self {
            qubit_ops,
            op_locations,
        };
        ret.check_valid()?;
        Ok(ret)
    }

    fn append(
        &mut self,
        other: &StaticSizeCircuit,
        qubit_map: impl Fn(StaticQubitIndex) -> StaticQubitIndex,
    ) {
        for (qb, ops) in other.qubit_ops.iter().enumerate() {
            let new_qb = qubit_map(StaticQubitIndex(qb));
            for op in ops.iter() {
                let op_idx = self.qubit_ops[new_qb.0].len();
                self.qubit_ops[new_qb.0].push(op.clone());
                self.op_locations
                    .entry(Rc::as_ptr(op))
                    .or_default()
                    .push(OpLocation {
                        qubit: new_qb,
                        op_idx,
                    });
            }
        }
    }

    fn check_valid(&self) -> Result<(), InvalidSubcircuitError> {
        for op in self.all_ops_iter() {
            if self.op_locations.get(&Rc::as_ptr(op)).is_none() {
                return Err(InvalidSubcircuitError);
            }
        }
        Ok(())
    }
}

/// A rewrite that applies on a static circuit.
pub struct StaticRewrite<F> {
    /// The subcircuit to be replaced.
    pub subcircuit: StaticSubcircuit,
    /// The replacement circuit.
    pub replacement: StaticSizeCircuit,
    /// The qubit map.
    pub qubit_map: F,
}

impl StaticSizeCircuit {
    /// Rewrite a subcircuit in the circuit with a replacement circuit.
    pub fn apply_rewrite<F>(
        &self,
        rewrite: &StaticRewrite<F>,
    ) -> Result<StaticSizeCircuit, InvalidSubcircuitError>
    where
        F: Fn(StaticQubitIndex) -> StaticQubitIndex,
    {
        let mut new_circ = self.subcircuit(&rewrite.subcircuit.before(self))?;
        new_circ.append(&rewrite.replacement, &rewrite.qubit_map);
        let after = self.subcircuit(&rewrite.subcircuit.after(self))?;
        new_circ.append(&after, |qb| qb);
        Ok(new_circ)
    }
}

#[cfg(test)]
mod tests {
    use crate::{utils::build_simple_circuit, Tk2Op};

    use super::*;

    #[test]
    fn test_rewrite_circuit() {
        // Create initial circuit
        let circuit = build_simple_circuit(2, |circ| {
            circ.append(Tk2Op::H, [0])?;
            circ.append(Tk2Op::CX, [0, 1])?;
            circ.append(Tk2Op::CX, [0, 1])?;
            circ.append(Tk2Op::CX, [0, 1])?;
            circ.append(Tk2Op::H, [1])?;
            Ok(())
        })
        .unwrap();

        let initial_circuit: StaticSizeCircuit = (&circuit).try_into().unwrap();

        // Create subcircuit to be replaced
        let subcircuit = StaticSubcircuit {
            op_indices: vec![
                (StaticQubitIndex(0), OpInterval(0..2)),
                (StaticQubitIndex(1), OpInterval(0..1)),
            ]
            .into_iter()
            .collect(),
        };

        let circuit = build_simple_circuit(2, |circ| {
            circ.append(Tk2Op::CX, [0, 1])?;
            circ.append(Tk2Op::H, [0])?;
            Ok(())
        })
        .unwrap();

        let replacement_circuit: StaticSizeCircuit = (&circuit).try_into().unwrap();

        // Define qubit mapping
        let qubit_map = |qb: StaticQubitIndex| qb;

        let rewrite = StaticRewrite {
            subcircuit,
            replacement: replacement_circuit,
            qubit_map,
        };

        // Perform rewrite
        let rewritten_circuit = initial_circuit.apply_rewrite(&rewrite).unwrap();

        // Expected circuit after rewrite
        let circuit = build_simple_circuit(2, |circ| {
            circ.append(Tk2Op::CX, [0, 1])?;
            circ.append(Tk2Op::H, [0])?;
            circ.append(Tk2Op::CX, [0, 1])?;
            circ.append(Tk2Op::CX, [0, 1])?;
            circ.append(Tk2Op::H, [1])?;
            Ok(())
        })
        .unwrap();
        let expected_circuit: StaticSizeCircuit = (&circuit).try_into().unwrap();

        // Assert the rewritten circuit matches the expected circuit
        assert_eq!(rewritten_circuit, expected_circuit);
    }
}
