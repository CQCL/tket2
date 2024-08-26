//! Index into patterns.
//!
//! In principle, as patterns are `StaticSizeCircuit`s, we could
//! just use `OpLocation`s, but by using a more tailored type we can
//! make indexing more efficient.

use std::collections::VecDeque;

use crate::static_circ::{OpLocation, StaticQubitIndex, StaticSizeCircuit};

use itertools::Itertools;
use thiserror::Error;

/// To address gates in patterns we use positive as well as negative indices.
///
/// This allows us to shift the indices such that index 0 is always the first
/// to be discovered when traversing the pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct PatternOpLocation {
    pub(super) qubit: CircuitPath,
    pub(super) op_idx: i8,
}

impl PartialOrd for PatternOpLocation {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PatternOpLocation {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let key = |v: &Self| (v.qubit, v.op_idx.abs(), v.op_idx.signum());
        key(self).cmp(&key(other))
    }
}

impl PatternOpLocation {
    pub fn new(qubit: CircuitPath, op_idx: i8) -> Self {
        Self { qubit, op_idx }
    }

    pub fn with_op_idx(self, op_idx: i8) -> Self {
        Self { op_idx, ..self }
    }

    pub fn root() -> Self {
        Self {
            qubit: CircuitPath([0; MAX_PATH_LEN * 2]),
            op_idx: 0,
        }
    }

    pub(super) fn resolve(&self, circ: &StaticSizeCircuit, root: OpLocation) -> Option<OpLocation> {
        let Self { qubit, op_idx } = *self;
        let new_root = get_qubit_root(circ, &qubit.0, root)?;
        let loc = new_root.try_add_op_idx(op_idx as isize)?;
        circ.get(loc).map(|_| loc)
    }
}

#[derive(Debug, Error)]
#[error("Circuit is disconnected")]
pub struct DisconnectedCircuit;

impl StaticSizeCircuit {
    /// For each qubit find the first operation to be reached from the root (0, 0).
    /// (according to some fixed traversal order)
    ///
    /// Errors if the circuit is disconnected.
    pub(crate) fn find_qubit_starts(
        &self,
    ) -> Result<Vec<(CircuitPath, usize)>, DisconnectedCircuit> {
        let mut qubit_starts = vec![None; self.qubit_count()];
        qubit_starts[0] = Some((CircuitPath::root(), 0));
        let mut next_qubits = VecDeque::from_iter([StaticQubitIndex(0)]);

        while let Some(qubit) = next_qubits.pop_front() {
            let (path, start) = qubit_starts[qubit.0].unwrap();
            let ops = self.qubit_ops(qubit);
            let indices = (0..=start).rev().chain((start + 1)..ops.len());
            for i in indices {
                let op = &ops[i];
                let offset = (i as i8) - (start as i8);
                for (port, loc) in self.op_locations(op).iter().enumerate() {
                    let &OpLocation { qubit, op_idx } = loc;
                    if qubit_starts[qubit.0].is_none() {
                        next_qubits.push_back(qubit);
                        let new_path = path.append(offset, port as i8);
                        qubit_starts[qubit.0] = Some((new_path, op_idx));
                    }
                }
            }
        }
        qubit_starts
            .into_iter()
            .map(|opt| opt.ok_or(DisconnectedCircuit))
            .collect()
    }
}

const MAX_PATH_LEN: usize = 8;

/// We identify qubits by a path from the root of the pattern.
///
/// The path is given by a sequence of pairs (op_offset, port),
/// corresponding to moving op_offset along the current qubit and then changing
/// the current qubit to the qubit at the given port.
///
/// Odd items are op_offsets, even items are ports. Ports are always positive.
#[derive(
    Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, serde::Serialize, serde::Deserialize,
)]
pub(crate) struct CircuitPath([i8; MAX_PATH_LEN * 2]);

impl CircuitPath {
    fn resolve(&self, circ: &StaticSizeCircuit, root: OpLocation) -> Option<OpLocation> {
        get_qubit_root(circ, &self.0, root)
    }

    pub(super) fn is_root(&self) -> bool {
        self.len() == 0
    }

    fn root() -> Self {
        Self([0; MAX_PATH_LEN * 2])
    }

    fn len(&self) -> usize {
        let mut ind = 0;
        while self.0[ind] != 0 || self.0[ind + 1] != 0 {
            ind += 2;
        }
        ind / 2
    }

    fn append(&self, op_offset: i8, port: i8) -> Self {
        let mut new_path = *self;
        let ind = self.len() * 2;
        new_path.0[ind] = op_offset;
        new_path.0[ind + 1] = port;
        new_path
    }
}

impl std::fmt::Debug for CircuitPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self.0[..(2 * self.len())]
            .iter()
            .map(|x| x.to_string())
            .join("");
        write!(f, "CircuitPath({})", s)
    }
}

fn get_qubit_root(circ: &StaticSizeCircuit, path: &[i8], root: OpLocation) -> Option<OpLocation> {
    if path.is_empty() {
        return Some(root);
    }
    assert!(path.len() >= 2);
    let [op_offset, port] = path[..2] else {
        unreachable!()
    };
    if op_offset == 0 && port == 0 {
        return Some(root);
    }

    let Some(new_op_idx) = root.op_idx.checked_add_signed(op_offset as isize) else {
        return None;
    };
    let loc = OpLocation {
        qubit: root.qubit,
        op_idx: new_op_idx,
    };
    // Now find the loc for the same op but on `port`
    let new_root = circ.equivalent_location(loc, port as usize)?;
    get_qubit_root(circ, &path[2..], new_root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::Tk2Op;
    use crate::static_circ::{OpLocation, StaticQubitIndex, StaticSizeCircuit};
    use crate::utils::build_simple_circuit;

    use rstest::rstest;

    #[rstest]
    #[case(vec![], Some(OpLocation { qubit: StaticQubitIndex(0), op_idx: 0 }))]
    #[case(vec![1, 1], Some(OpLocation { qubit: StaticQubitIndex(1), op_idx: 0 }))]
    #[case(vec![1, 1, 2, 1], Some(OpLocation { qubit: StaticQubitIndex(2), op_idx: 0 }))]
    #[case(vec![5, 1], None)]
    fn test_circuit_path_resolve(
        #[case] path_elements: Vec<i8>,
        #[case] expected: Option<OpLocation>,
    ) {
        let root = OpLocation {
            qubit: StaticQubitIndex(0),
            op_idx: 0,
        };
        // Create a circuit using build_simple_circuit
        let circuit = build_simple_circuit(3, |circ| {
            circ.append(Tk2Op::H, [0])?;
            circ.append(Tk2Op::CX, [0, 1])?;
            circ.append(Tk2Op::T, [1])?;
            circ.append(Tk2Op::CX, [1, 2])?;
            circ.append(Tk2Op::H, [2])?;
            Ok(())
        })
        .unwrap();

        // Convert the circuit to StaticSizeCircuit
        let static_circuit: StaticSizeCircuit = (&circuit).try_into().unwrap();

        let mut path = CircuitPath::default();
        path.0[..path_elements.len()].copy_from_slice(&path_elements);

        assert_eq!(path.resolve(&static_circuit, root), expected);
    }

    #[test]
    fn test_find_qubit_starts() {
        let circuit = build_simple_circuit(3, |circ| {
            circ.append(Tk2Op::H, [0])?;
            circ.append(Tk2Op::CX, [0, 1])?;
            circ.append(Tk2Op::T, [1])?;
            circ.append(Tk2Op::H, [2])?;
            circ.append(Tk2Op::CX, [2, 1])?;
            circ.append(Tk2Op::H, [2])?;
            Ok(())
        })
        .unwrap();
        let static_circuit: StaticSizeCircuit = (&circuit).try_into().unwrap();
        let starts = static_circuit.find_qubit_starts().unwrap();

        let path = CircuitPath::root();
        assert_eq!(starts.len(), 3);
        assert_eq!(starts[0], (CircuitPath::root(), 0));
        let path = path.append(1, 1);
        assert_eq!(starts[1], (path, 0));
        let path = path.append(2, 0);
        assert_eq!(starts[2], (path, 1));
    }
}
