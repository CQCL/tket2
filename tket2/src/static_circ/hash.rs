use std::{
    hash::{Hash, Hasher},
    ops::Range,
};

use cgmath::num_traits::{WrappingAdd, WrappingShl};

use crate::{
    circuit::{CircuitHash, HashError},
    Tk2Op,
};

use super::{
    rewrite::{OpInterval, StaticRewrite},
    StaticQubitIndex, StaticSizeCircuit,
};

pub struct UpdatableHash {
    cum_hash: Vec<Vec<u64>>,
}

impl UpdatableHash {
    pub fn with_static(circuit: &StaticSizeCircuit) -> Self {
        let num_qubits = circuit.qubit_count();
        let mut cum_hash = Vec::with_capacity(num_qubits);

        for row in circuit.qubit_ops.iter() {
            let mut prev_hash = 0;
            let mut row_hash = Vec::with_capacity(row.len());
            for op in row.iter() {
                let hash = Self::hash_op(op);
                let combined_hash = prev_hash.wrapping_shl(5).wrapping_add(&hash);
                row_hash.push(combined_hash);
                prev_hash = combined_hash;
            }
            cum_hash.push(row_hash);
        }

        Self { cum_hash }
    }

    /// Compute the hash of the circuit that results from applying the given rewrite.
    pub fn hash_rewrite<F>(&self, rewrite: &StaticRewrite<F>) -> Result<u64, ()>
    where
        F: Fn(StaticQubitIndex) -> StaticQubitIndex,
    {
        let new_hash = Self::with_static(&rewrite.replacement);
        Ok(hash_iter((0..self.cum_hash.len()).map(|i| {
            if let Some(interval) = rewrite.subcircuit.op_indices.get(&StaticQubitIndex(i)) {
                splice(&self.cum_hash[i], interval, &new_hash.cum_hash[i])
            } else {
                *self.cum_hash[i].last().unwrap()
            }
        })))
    }

    fn hash_op(op: &Tk2Op) -> u64 {
        let mut hasher = fxhash::FxHasher::default();
        op.hash(&mut hasher);
        hasher.finish()
    }
}

/// Compute the hash that results from replacing the ops in the range [start, end)
/// with the new ops (given by `new_cum_hashes`).
fn splice(cum_hashes: &[u64], interval: &OpInterval, new_cum_hashes: &[u64]) -> u64 {
    let Range { start, end } = interval.0;
    let mut hash = 0;
    if start > 0 {
        hash = hash.wrapping_add(&cum_hashes[start - 1]);
    }
    if !new_cum_hashes.is_empty() {
        hash = hash.wrapping_shl(5 * (new_cum_hashes.len() as u32));
        hash = hash.wrapping_add(new_cum_hashes[new_cum_hashes.len() - 1]);
    }
    if end < cum_hashes.len() {
        hash = hash.wrapping_shl(5 * (cum_hashes.len() - end) as u32);
        hash = hash.wrapping_add(hash_delta(cum_hashes, end..cum_hashes.len()));
    }
    hash
}

/// The hash "contribution" that comes from within the range [start, end).
fn hash_delta(cum_hashes: &[u64], Range { start, end }: Range<usize>) -> u64 {
    if start >= end {
        return 0;
    }
    let end_hash = if end > 0 { cum_hashes[end - 1] } else { 0 };
    let start_hash = if start > 0 { cum_hashes[start - 1] } else { 0 };
    let start_hash_shifted = start_hash.wrapping_shl(5 * (end - start) as u32);
    end_hash.wrapping_sub(start_hash_shifted)
}

fn hash_iter<T: Hash>(iter: impl Iterator<Item = T>) -> u64 {
    let mut hasher = fxhash::FxHasher::default();
    for item in iter {
        item.hash(&mut hasher);
    }
    hasher.finish()
}
impl CircuitHash for StaticSizeCircuit {
    fn circuit_hash(&self) -> Result<u64, HashError> {
        let hash_updater = UpdatableHash::with_static(self);
        Ok(hash_iter(
            hash_updater
                .cum_hash
                .iter()
                .map(|row| row.last().unwrap_or(&0)),
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::{static_circ::rewrite::StaticSubcircuit, utils::build_simple_circuit, Tk2Op};

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

        // Assert the hash of the rewritten circuit matches the spliced hash
        let hash_updater = UpdatableHash::with_static(&initial_circuit);
        let rewritten_hash = hash_updater.hash_rewrite(&rewrite).unwrap();
        let expected_hash = rewritten_circuit.circuit_hash().unwrap();
        assert_eq!(rewritten_hash, expected_hash);
    }
}
