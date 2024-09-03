use super::{OpId, StaticQubitIndex, StaticSizeCircuit};

/// The position of an operation in a `StaticSizeCircuit`.
///
/// Given by the qubit index and the position within that qubit's op list.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct OpPosition {
    /// The index of the qubit the operation acts on.
    pub qubit: StaticQubitIndex,
    /// The index of the operation in the qubit's operation list.
    pub index: usize,
}

impl OpPosition {
    pub(crate) fn try_add_op_idx(self, op_idx: isize) -> Option<Self> {
        Some(Self {
            index: self.index.checked_add_signed(op_idx)?,
            ..self
        })
    }

    pub(crate) fn start() -> Self {
        Self {
            qubit: StaticQubitIndex(0),
            index: 0,
        }
    }
}

impl StaticSizeCircuit {
    /// Get the id of the operation at the given position.
    pub fn at_position(&self, pos: OpPosition) -> Option<OpId> {
        self.qubit_ops[pos.qubit.0].get(pos.index).copied()
    }

    /// Check if the given position exists.
    pub(crate) fn exists(&self, pos: OpPosition) -> bool {
        self.qubit_ops
            .get(pos.qubit.0)
            .map_or(false, |ops| ops.get(pos.index).is_some())
    }

    /// The offset that pos is at in its op.
    pub(crate) fn position_offset(&self, pos: OpPosition) -> Option<usize> {
        let op = self.at_position(pos)?;
        self.get(op)
            .unwrap()
            .positions
            .iter()
            .position(|l| l == &pos)
    }

    pub fn get_position(&self, op: OpId, offset: usize) -> Option<OpPosition> {
        self.get(op)?.positions.get(offset).copied()
    }

    /// Get an equivalent position for the op at `pos` but at `offset`.
    ///
    /// Every op corresponds to as many positions as it has qubits. This
    /// function returns the position of the op at `pos` but at `offset`.
    pub fn equivalent_position(&self, pos: OpPosition, offset: usize) -> Option<OpPosition> {
        let id = self.at_position(pos)?;
        let op = self.get(id).unwrap();
        op.positions.get(offset).copied()
    }

    pub fn are_same_op(&self, pos1: OpPosition, pos2: OpPosition) -> bool {
        let Some(id) = self.at_position(pos1) else {
            return false;
        };
        let op = self.get(id).unwrap();
        op.positions.contains(&pos2)
    }

    pub fn positions_iter(&self) -> impl Iterator<Item = OpPosition> + '_ {
        self.qubits_iter().flat_map(|qb| self.qubit_positions(qb))
    }

    pub fn qubit_positions(&self, qb: StaticQubitIndex) -> impl Iterator<Item = OpPosition> {
        (0..self.qubit_ops(qb).len()).map(move |index| OpPosition { qubit: qb, index })
    }
}
