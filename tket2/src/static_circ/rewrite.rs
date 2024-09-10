use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
    mem,
    ops::Range,
};

use derive_more::{From, Into};
use portmatching::{IndexingScheme, PatternMatch};
use thiserror::Error;

use crate::portmatching::indexing::{
    CircuitPath, OpLocationMap, PatternOpPosition, StaticIndexScheme,
};

use super::{iter::Command, OpId, OpPosition, StaticQubitIndex, StaticSizeCircuit};

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
    /// Instantiate the subcircuit of `self` given by `subcircuit`.
    ///
    /// Keep the same number of qubits (leaving them empty if they are not
    /// present in the subcircuit).
    ///
    /// TODO: Cloning everything and then removing might be too slow.
    pub fn subcircuit(
        &self,
        subcircuit: &StaticSubcircuit,
    ) -> Result<Self, InvalidSubcircuitError> {
        let Self {
            mut qubit_ops,
            mut all_ops,
            ..
        } = self.clone();
        // Update qubit_ops
        let mut remove_op_ids = BTreeSet::new();
        for qb in self.qubits_iter() {
            if let Some(interval) = subcircuit.op_indices.get(&qb) {
                for op in qubit_ops[qb.0].drain(interval.0.end..) {
                    remove_op_ids.insert(op.0);
                }
                for op in qubit_ops[qb.0].drain(..interval.0.start) {
                    remove_op_ids.insert(op.0);
                }
            } else {
                for op in qubit_ops[qb.0].drain(..) {
                    remove_op_ids.insert(op.0);
                }
            }
        }
        let mut i = 0;
        all_ops.retain(|_| {
            i += 1;
            !remove_op_ids.contains(&i)
        });

        let mut ret = Self {
            qubit_ops,
            all_ops,
            ..Default::default()
        };

        // Update op_locations
        ret.remap_positions(|mut loc| {
            if let Some(interval) = subcircuit.op_indices.get(&loc.qubit) {
                loc.index -= interval.0.start;
            }
            loc
        });

        ret.check_valid()?;
        Ok(ret)
    }

    fn append(
        &mut self,
        other: &StaticSizeCircuit,
        qubit_map: impl Fn(StaticQubitIndex) -> StaticQubitIndex,
    ) {
        for Command { op, qubits, .. } in other.commands() {
            let qubits = qubits.into_iter().map(&qubit_map);
            self.append_op(op, qubits);
        }
    }

    /// Add `other` on a new set of qubits.
    ///
    /// Not contiguous intervals of ops will be added to different qubits.
    pub fn add_subcircuit(
        &mut self,
        other: &Self,
        ops: &BTreeSet<OpId>,
    ) -> BTreeMap<OpPosition, OpPosition> {
        // Add all the new ops to `self`
        let mut op_id_map = BTreeMap::new();
        for &id in ops {
            let op = other.get(id).unwrap();
            let new_id = OpId(self.all_ops.len());
            self.all_ops.push(op.clone()); // Note: all positions will be overwritten
            op_id_map.insert(id, new_id);
        }

        // Now let us find out at which positions they must be added
        let mut position_map = BTreeMap::new();
        for qb in other.qubits_iter() {
            let mut new_qubit_ops = vec![];
            for pos in other.qubit_positions(qb) {
                let id = other.at_position(pos).unwrap();
                let offset = other.position_offset(pos).unwrap();
                if ops.contains(&id) {
                    let new_id = op_id_map[&id];
                    let new_index = new_qubit_ops.len();
                    let new_qb = StaticQubitIndex(self.qubit_count());
                    let new_pos = OpPosition {
                        qubit: new_qb,
                        index: new_index,
                    };
                    // Add the op to the new qubit
                    new_qubit_ops.push(new_id);
                    // Update the position of the op
                    self.get_mut(new_id).unwrap().positions[offset] = new_pos;
                    position_map.insert(pos, new_pos);
                } else {
                    // The op is not in `ops`, so we do not need to add it
                    // but we need to clear the qubit and start on a new one
                    if !new_qubit_ops.is_empty() {
                        self.qubit_ops.push(mem::take(&mut new_qubit_ops));
                    }
                }
            }
            if !new_qubit_ops.is_empty() {
                self.qubit_ops.push(new_qubit_ops);
            }
        }

        position_map
    }

    /// Adds the operations on `right` to `left`.
    ///
    /// The right_qubit is removed, and its operations are added to the left
    /// qubit. The resulting circuit has one less qubit than the input.
    pub(crate) fn merge_qubits(
        &mut self,
        StaticQubitIndex(mut left_qubit): StaticQubitIndex,
        StaticQubitIndex(right_qubit): StaticQubitIndex,
    ) {
        let left_offset = self.qubit_ops[left_qubit].len();

        // Remove right qubit
        let right_ops = self.qubit_ops.remove(right_qubit);

        if left_qubit > right_qubit {
            left_qubit -= 1;
        }

        // Append right qubit ops to left qubit
        self.qubit_ops[left_qubit].extend(right_ops);

        // Update op positions
        // TODO: check if this preserves convexity
        self.remap_positions(|pos| match pos.qubit.0.cmp(&right_qubit) {
            Ordering::Equal => OpPosition {
                qubit: StaticQubitIndex(left_qubit),
                index: pos.index + left_offset,
            },
            Ordering::Greater => OpPosition {
                qubit: StaticQubitIndex(pos.qubit.0 - 1),
                ..pos
            },
            _ => pos,
        });
    }

    pub(super) fn remap_positions(&mut self, map: impl Fn(OpPosition) -> OpPosition) {
        for id in self.ops_iter() {
            let op = self.get_mut(id).unwrap();
            let positions = &mut op.positions;
            positions.iter_mut().for_each(|loc| *loc = map(*loc));
        }
    }

    fn check_valid(&self) -> Result<(), InvalidSubcircuitError> {
        for op in self.ops_iter() {
            if self.get(op).is_none() {
                return Err(InvalidSubcircuitError);
            }
        }
        Ok(())
    }
}

pub type BoxedStaticRewrite = StaticRewrite<Box<dyn Fn(StaticQubitIndex) -> StaticQubitIndex>>;

/// A rewrite that applies on a static circuit.
pub struct StaticRewrite<F> {
    /// The subcircuit to be replaced.
    pub subcircuit: StaticSubcircuit,
    /// The replacement circuit.
    pub replacement: StaticSizeCircuit,
    /// The qubit map from the replacement to the subject subcircuit.
    pub qubit_map: F,
}

fn compute_op_intervals(
    pattern: &StaticSizeCircuit,
    match_map: &OpLocationMap<CircuitPath, OpPosition>,
    starts: &[(CircuitPath, usize)],
) -> BTreeMap<StaticQubitIndex, OpInterval> {
    let mut op_indices = BTreeMap::new();
    for qb in pattern.qubits_iter() {
        let fst_loc = PatternOpPosition::from_position(
            OpPosition {
                qubit: qb,
                index: 0,
            },
            starts,
        );
        let last_loc = PatternOpPosition::from_position(
            OpPosition {
                qubit: qb,
                index: pattern.qubit_ops(qb).len() - 1,
            },
            starts,
        );
        let &fst_loc = match_map
            .get_val(&fst_loc.qubit, fst_loc.op_idx as isize)
            .unwrap();
        let &last_loc = match_map
            .get_val(&last_loc.qubit, last_loc.op_idx as isize)
            .unwrap();
        assert_eq!(fst_loc.qubit, last_loc.qubit);
        op_indices.insert(
            fst_loc.qubit,
            OpInterval(fst_loc.index..(last_loc.index + 1)),
        );
    }
    op_indices
}

fn compute_qubit_map(
    pattern: &StaticSizeCircuit,
    match_map: &OpLocationMap<CircuitPath, OpPosition>,
    starts: &[(CircuitPath, usize)],
) -> BTreeMap<StaticQubitIndex, StaticQubitIndex> {
    let mut qubit_map = BTreeMap::new();
    for qb in pattern.qubits_iter() {
        let pattern_qb = starts[qb.0].0;
        let subj_qb = match_map.get_val(&pattern_qb, 0_isize).unwrap().qubit;
        qubit_map.insert(qb, subj_qb);
    }
    qubit_map
}

impl StaticRewrite<Box<dyn Fn(StaticQubitIndex) -> StaticQubitIndex>> {
    /// Create a rewrite from a pattern match.
    pub fn from_pattern_match(
        match_map: &PatternMatch<<StaticIndexScheme as IndexingScheme<StaticSizeCircuit>>::Map>,
        pattern: &StaticSizeCircuit,
        replacement: StaticSizeCircuit,
    ) -> Self {
        let starts = pattern.find_qubit_starts().unwrap();
        let subcircuit = StaticSubcircuit {
            op_indices: compute_op_intervals(pattern, &match_map.match_data, &starts),
        };
        assert_eq!(subcircuit.op_indices.len(), pattern.qubit_count());
        let qubit_map = compute_qubit_map(pattern, &match_map.match_data, &starts);
        let qubit_map = move |qb: StaticQubitIndex| qubit_map[&qb];
        Self {
            subcircuit,
            replacement,
            qubit_map: Box::new(qubit_map),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("rewrite is non-convex")]
pub struct NonConvexRewriteError;

impl StaticSizeCircuit {
    /// Rewrite a subcircuit in the circuit with a replacement circuit.
    pub fn apply_rewrite<F>(
        &self,
        rewrite: &StaticRewrite<F>,
    ) -> Result<StaticSizeCircuit, NonConvexRewriteError>
    where
        F: Fn(StaticQubitIndex) -> StaticQubitIndex,
    {
        let mut new_circ = self
            .subcircuit(&rewrite.subcircuit.before(self))
            .map_err(|_| NonConvexRewriteError)?;
        new_circ.append(&rewrite.replacement, &rewrite.qubit_map);
        let after = self
            .subcircuit(&rewrite.subcircuit.after(self))
            .map_err(|_| NonConvexRewriteError)?;
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
