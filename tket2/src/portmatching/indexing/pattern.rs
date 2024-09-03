//! Index into patterns.
//!
//! In principle, as patterns are `StaticSizeCircuit`s, we could
//! just use `OpLocation`s, but by using a more tailored type we can
//! make indexing more efficient.

use std::collections::VecDeque;

use crate::static_circ::{OpPosition, StaticQubitIndex, StaticSizeCircuit};

use itertools::Itertools;
use thiserror::Error;

/// To address gates in patterns we use positive as well as negative indices.
///
/// This allows us to shift the indices such that index 0 is always the first
/// to be discovered when traversing the pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct PatternOpPosition {
    pub(crate) qubit: CircuitPath,
    pub(crate) op_idx: i8,
}

impl PartialOrd for PatternOpPosition {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PatternOpPosition {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let key = |v: &Self| (v.qubit, v.op_idx.abs(), v.op_idx.signum());
        key(self).cmp(&key(other))
    }
}

impl PatternOpPosition {
    pub fn new(qubit: CircuitPath, op_idx: i8) -> Self {
        Self { qubit, op_idx }
    }

    pub fn with_op_idx(self, op_idx: i8) -> Self {
        Self { op_idx, ..self }
    }

    pub fn from_position(loc: OpPosition, starts: &[(CircuitPath, usize)]) -> Self {
        let (qubit_path, start) = starts[loc.qubit.0];
        let offset = (loc.index as i8) - (start as i8);
        PatternOpPosition::new(qubit_path, offset)
    }

    pub fn root() -> Self {
        Self {
            qubit: CircuitPath([0; MAX_PATH_LEN * 2]),
            op_idx: 0,
        }
    }

    pub(super) fn resolve(&self, circ: &StaticSizeCircuit, root: OpPosition) -> Option<OpPosition> {
        let Self { qubit, op_idx } = *self;
        let new_root = get_qubit_root(circ, &qubit.0, root)?;
        let pos = new_root.try_add_op_idx(op_idx as isize)?;
        circ.exists(pos).then_some(pos)
    }

    pub(crate) fn all_locations_on_path(&self) -> Vec<PatternOpPosition> {
        let prefix = if self.op_idx == 0 {
            if self.qubit.is_empty() {
                return vec![PatternOpPosition::new(CircuitPath::empty_path(), 0)];
            }
            Self {
                qubit: self.qubit.truncate(self.qubit.len() - 1),
                op_idx: self.qubit.op_offset(self.qubit.len() - 1),
            }
        } else {
            let step = if self.op_idx > 0 { 1 } else { -1 };
            Self {
                op_idx: self.op_idx - step,
                ..*self
            }
        };
        let mut res = prefix.all_locations_on_path();
        res.push(*self);
        res
    }
}

/// Circuit is disconnected.
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
        if self.qubit_count() == 0 {
            return Ok(vec![]);
        }
        let mut qubit_starts = vec![None; self.qubit_count()];
        let fst_qubit = {
            let mut loc = OpPosition::start();
            // Find a qubit such that it is on port 0 of the first operation
            // That way qubit = CircuitPath(00) = CircuitPath()
            loop {
                loc = self.equivalent_position(loc, 0).unwrap();
                if loc.index == 0 {
                    break;
                }
                loc = OpPosition { index: 0, ..loc };
            }
            loc.qubit
        };
        qubit_starts[fst_qubit.0] = Some((CircuitPath::empty_path(), 0));
        let mut next_qubits = VecDeque::from_iter([fst_qubit]);

        while let Some(qubit) = next_qubits.pop_front() {
            let (path, start) = qubit_starts[qubit.0].unwrap();
            let ops = self.qubit_ops(qubit);
            let indices = (0..=start).rev().chain((start + 1)..ops.len());
            for i in indices {
                let op = ops[i];
                let offset = (i as i8) - (start as i8);
                let positions = &self.get(op).unwrap().positions;
                for (port, loc) in positions.iter().enumerate() {
                    let &OpPosition { qubit, index } = loc;
                    if qubit_starts[qubit.0].is_none() {
                        next_qubits.push_back(qubit);
                        let new_path = path.append(offset, port as i8);
                        qubit_starts[qubit.0] = Some((new_path, index));
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
/// Even items are op_offsets, odd items are ports. Ports are always positive.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize)]
pub struct CircuitPath([i8; MAX_PATH_LEN * 2]);

impl CircuitPath {
    fn new(path: &[i8]) -> Self {
        let mut new_path = Self::empty_path();
        new_path.0[..path.len()].copy_from_slice(path);
        new_path
    }

    fn resolve(&self, circ: &StaticSizeCircuit, root: OpPosition) -> Option<OpPosition> {
        get_qubit_root(circ, &self.0, root)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn empty_path() -> Self {
        Self([0; MAX_PATH_LEN * 2])
    }

    pub(crate) fn len(&self) -> usize {
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

    fn truncate(&self, n: usize) -> Self {
        let mut new_path = Self::empty_path();
        new_path.0[..(2 * n)].copy_from_slice(&self.0[..(2 * n)]);
        new_path
    }

    pub(crate) fn op_offset(&self, n: usize) -> i8 {
        self.0[2 * n]
    }

    pub(crate) fn port(&self, n: usize) -> i8 {
        self.0[2 * n + 1]
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

impl Ord for CircuitPath {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let key = |v: &Self| {
            (0..v.len())
                .map(|i| (v.op_offset(i).abs(), v.op_offset(i).signum(), v.port(i)))
                .collect_vec()
        };
        key(self).cmp(&key(other))
    }
}

impl PartialOrd for CircuitPath {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

fn get_qubit_root(circ: &StaticSizeCircuit, path: &[i8], root: OpPosition) -> Option<OpPosition> {
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

    let Some(new_index) = root.index.checked_add_signed(op_offset as isize) else {
        return None;
    };
    let loc = OpPosition {
        qubit: root.qubit,
        index: new_index,
    };
    // Now find the loc for the same op but on `port`
    let new_root = circ.equivalent_position(loc, port as usize)?;
    get_qubit_root(circ, &path[2..], new_root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::Tk2Op;
    use crate::static_circ::{OpPosition, StaticQubitIndex, StaticSizeCircuit};
    use crate::utils::build_simple_circuit;

    use rstest::rstest;

    #[rstest]
    #[case(vec![], Some(OpPosition { qubit: StaticQubitIndex(0), index: 0 }))]
    #[case(vec![1, 1], Some(OpPosition { qubit: StaticQubitIndex(1), index: 0 }))]
    #[case(vec![1, 1, 2, 1], Some(OpPosition { qubit: StaticQubitIndex(2), index: 0 }))]
    #[case(vec![5, 1], None)]
    fn test_circuit_path_resolve(
        #[case] path_elements: Vec<i8>,
        #[case] expected: Option<OpPosition>,
    ) {
        let root = OpPosition {
            qubit: StaticQubitIndex(0),
            index: 0,
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

        let path = CircuitPath::new(&path_elements);

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

        let path = CircuitPath::empty_path();
        assert_eq!(starts.len(), 3);
        assert_eq!(starts[0], (CircuitPath::empty_path(), 0));
        let path = path.append(1, 1);
        assert_eq!(starts[1], (path, 0));
        let path = path.append(2, 0);
        assert_eq!(starts[2], (path, 1));
    }

    #[rstest]
    #[case(
        PatternOpPosition::new(CircuitPath::empty_path(), 0),
        vec![PatternOpPosition::new(CircuitPath::empty_path(), 0)]
    )]
    #[case(
        PatternOpPosition::new(CircuitPath::empty_path(), 1),
        vec![
            PatternOpPosition::new(CircuitPath::empty_path(), 0),
            PatternOpPosition::new(CircuitPath::empty_path(), 1)
        ]
    )]
    #[case(
        PatternOpPosition::new(CircuitPath::new(&[1, 0]), 0),
        vec![
            PatternOpPosition::new(CircuitPath::empty_path(), 0),
            PatternOpPosition::new(CircuitPath::empty_path(), 1),
            PatternOpPosition::new(CircuitPath::new(&[1, 0]), 0)
        ]
    )]
    #[case(
        PatternOpPosition::new(CircuitPath::new(&[-1, 1]), 2),
        vec![
            PatternOpPosition::new(CircuitPath::empty_path(), 0),
            PatternOpPosition::new(CircuitPath::empty_path(), -1),
            PatternOpPosition::new(CircuitPath::new(&[-1, 1]), 0),
            PatternOpPosition::new(CircuitPath::new(&[-1, 1]), 1),
            PatternOpPosition::new(CircuitPath::new(&[-1, 1]), 2)
        ]
    )]
    #[case(
        PatternOpPosition::new(CircuitPath::new(&[1, 0, 2, 1]), 2),
        vec![
            PatternOpPosition::new(CircuitPath::empty_path(), 0),
            PatternOpPosition::new(CircuitPath::empty_path(), 1),
            PatternOpPosition::new(CircuitPath::new(&[1, 0]), 0),
            PatternOpPosition::new(CircuitPath::new(&[1, 0]), 1),
            PatternOpPosition::new(CircuitPath::new(&[1, 0]), 2),
            PatternOpPosition::new(CircuitPath::new(&[1, 0, 2, 1]), 0),
            PatternOpPosition::new(CircuitPath::new(&[1, 0, 2, 1]), 1),
            PatternOpPosition::new(CircuitPath::new(&[1, 0, 2, 1]), 2)
        ]
    )]
    fn test_all_locations_on_path(
        #[case] input: PatternOpPosition,
        #[case] expected: Vec<PatternOpPosition>,
    ) {
        let result = input.all_locations_on_path();
        assert_eq!(result, expected, "Failed for input: {:?}", input);
    }
}
