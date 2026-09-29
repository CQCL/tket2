use super::{PackedPGSlice, SliceIndex};
use crate::CliffordAction;
use crate::backend::ScalarBackend;
use pg_bitpacked::{I_U8, X_U8, Y_U8, Z_U8};
use pg_core::{MeasureData, Op, Pauli};

/// Packed signed observables for an external synthesis policy.
///
/// Indices are stable for the lifetime of the state. Applying a Clifford
/// conjugates every active observable; retiring one excludes it from subsequent
/// updates. This is not a unitary circuit representation: no residual Clifford
/// is accumulated or synthesised. The caller owns ordering and emission policy.
pub struct ObservableState {
    packed: PackedPGSlice,
    indices: Vec<SliceIndex>,
    active: Vec<bool>,
}

impl ObservableState {
    /// Packs observables of the given width, with `true` denoting a minus sign.
    ///
    /// # Panics
    /// Panics if a Pauli string has the wrong width.
    pub fn new(width: usize, observables: impl IntoIterator<Item = (Vec<Pauli>, bool)>) -> Self {
        let mut packed = PackedPGSlice::new(width);
        for (paulis, sign) in observables {
            assert_eq!(paulis.len(), width, "observable width differs from state");
            packed.push_op(
                &Op::Measure {
                    data: MeasureData::new(paulis, sign, 0),
                },
                &ScalarBackend,
            );
        }
        let indices = packed
            .op_sets()
            .flatten()
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        let active = vec![true; indices.len()];
        Self {
            packed,
            indices,
            active,
        }
    }

    /// Returns the number of qubits.
    pub fn width(&self) -> usize {
        self.packed.n_qubits()
    }

    /// Returns the number of observables, including retired ones.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Returns whether no observables were supplied.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Returns whether an observable is still active.
    pub fn is_active(&self, index: usize) -> bool {
        self.active[index]
    }

    /// Returns a Pauli string. A retired string retains its value at retirement.
    pub fn paulis(&self, index: usize) -> Vec<Pauli> {
        (0..self.width())
            .map(|q| match self.packed.pauli(self.indices[index], 0, q) {
                I_U8 => Pauli::I,
                X_U8 => Pauli::X,
                Y_U8 => Pauli::Y,
                Z_U8 => Pauli::Z,
                _ => unreachable!("packed Pauli"),
            })
            .collect()
    }

    /// Returns the sign of an observable (`true` means negative).
    pub fn sign(&self, index: usize) -> bool {
        self.packed.sign_bit(self.indices[index], 0)
    }

    /// Tests commutation without unpacking either observable.
    pub fn commutes(&self, a: usize, b: usize) -> bool {
        let mut parity = false;
        for q in 0..self.width() {
            let a = self.packed.pauli(self.indices[a], 0, q);
            let b = self.packed.pauli(self.indices[b], 0, q);
            parity ^= a != I_U8 && b != I_U8 && a != b;
        }
        !parity
    }

    /// Excludes an observable from future updates. Repeated retirement is harmless.
    pub fn retire(&mut self, index: usize) {
        self.packed.clear_op_mask(self.indices[index]);
        self.active[index] = false;
    }

    /// Returns the weighted change in occupied qubit groups after an action sequence.
    ///
    /// Each group contributes one unit when any of its qubits is nonidentity.
    /// Singleton groups therefore measure Pauli weight. Weights are supplied per
    /// observable; retired observables contribute nothing. Groups may overlap.
    /// The sequence is evaluated without changing this state or constructing a
    /// residual Clifford. Signs do not affect support.
    ///
    /// # Panics
    /// Panics for out-of-range observable or qubit indices, or weights that are
    /// not finite and nonnegative.
    pub fn support_delta(
        &self,
        actions: &[&CliffordAction],
        groups: &[Vec<usize>],
        observables: &[(usize, f64)],
    ) -> f64 {
        assert!(groups.iter().flatten().all(|&q| q < self.width()));
        assert!(
            actions
                .iter()
                .flat_map(|a| a.qubits())
                .all(|&q| q < self.width())
        );
        assert!(
            observables
                .iter()
                .all(|&(r, w)| r < self.len() && w.is_finite() && w >= 0.0)
        );
        let mut touched = vec![false; self.width()];
        for &q in actions.iter().flat_map(|a| a.qubits()) {
            touched[q] = true;
        }
        let groups = groups
            .iter()
            .filter(|group| group.iter().any(|&q| touched[q]))
            .collect::<Vec<_>>();
        if groups.is_empty() || observables.is_empty() {
            return 0.0;
        }
        let weights = observables.iter().map(|&(_, w)| w).collect::<Vec<_>>();
        let words = observables.len().div_ceil(64);
        let mut increases = vec![vec![0u64; words]; groups.len()];
        let mut decreases = increases.clone();
        for (position, &(r, weight)) in observables.iter().enumerate() {
            if !self.is_active(r) || weight == 0.0 {
                continue;
            }
            let mut row = (0..self.width())
                .map(|q| self.packed.pauli(self.indices[r], 0, q))
                .collect::<Vec<_>>();
            let before = groups
                .iter()
                .map(|group| group.iter().any(|&q| row[q] != I_U8))
                .collect::<Vec<_>>();
            for action in actions {
                let local = action.qubits().iter().map(|&q| row[q]).collect::<Vec<_>>();
                let (transformed, _) = action.transform(&local);
                for (&q, p) in action.qubits().iter().zip(transformed) {
                    row[q] = p;
                }
            }
            for (g, group) in groups.iter().enumerate() {
                let after = group.iter().any(|&q| row[q] != I_U8);
                if after && !before[g] {
                    increases[g][position / 64] |= 1 << (position % 64);
                } else if before[g] && !after {
                    decreases[g][position / 64] |= 1 << (position % 64);
                }
            }
        }
        // Sum each group's masks separately: arbitrary actions may change an
        // observable's cost by more than one, unlike the specialised TQE kernel.
        increases
            .iter()
            .zip(&decreases)
            .map(|(inc, dec)| {
                crate::backend::sparse_weight_sum(inc, &weights)
                    - crate::backend::sparse_weight_sum(dec, &weights)
            })
            .sum()
    }

    /// Conjugates all active observables by a placed Clifford, including signs.
    ///
    /// # Panics
    /// Panics if an action qubit lies outside the state.
    pub fn apply(&mut self, action: &CliffordAction) {
        assert!(
            action.qubits().iter().all(|&q| q < self.width()),
            "action outside state"
        );
        for (&index, &active) in self.indices.iter().zip(&self.active) {
            if active {
                let (start, end) = self.packed.op(index).bit_range();
                self.packed.apply_action_range(action, start, end);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pg_core::TableauData;

    #[test]
    fn support_preview_handles_sequences_weights_and_retirement() {
        use Pauli::{I, X, Z};
        let fanout = CliffordAction::new(
            TableauData::new(
                vec![
                    (vec![Z, I, I], false),
                    (vec![Z, Z, I], false),
                    (vec![Z, I, Z], false),
                ],
                vec![
                    (vec![X, X, X], false),
                    (vec![I, X, I], false),
                    (vec![I, I, X], false),
                ],
            ),
            vec![0, 1, 64],
            1.0,
            1,
        );
        let mut state = ObservableState::new(
            65,
            (0..130).map(|r| {
                let mut row = vec![I; 65];
                row[0] = X;
                (row, r % 2 == 0)
            }),
        );
        state.retire(64);
        let weights = (0..130)
            .map(|r| (r, if r % 2 == 0 { 0.5 } else { 1.0 }))
            .collect::<Vec<_>>();
        let groups = vec![vec![0], vec![1], vec![64]];
        let expected = 2.0
            * weights
                .iter()
                .filter(|(r, _)| *r != 64)
                .map(|(_, w)| w)
                .sum::<f64>();
        assert_eq!(state.support_delta(&[&fanout], &groups, &weights), expected);
        assert_eq!(
            state.support_delta(&[&fanout, &fanout], &groups, &weights),
            0.0
        );
        assert_eq!(
            state.support_delta(&[&fanout], &[vec![0, 1, 64]], &weights),
            0.0
        );
        assert_eq!(state.support_delta(&[], &groups, &weights), 0.0);
        assert_eq!(state.support_delta(&[&fanout], &groups, &[]), 0.0);
        for r in 0..130 {
            assert_eq!(state.paulis(r)[64], I);
            assert_eq!(state.sign(r), r % 2 == 0);
        }
        state.apply(&fanout);
        assert_eq!(
            state.support_delta(&[&fanout], &groups, &weights),
            -expected
        );
        assert_eq!(state.paulis(64)[64], I);
    }

    #[test]
    fn signed_updates_and_retirement_cross_packed_boundaries() {
        let mut state = ObservableState::new(
            65,
            (0..130).map(|i| {
                let mut row = vec![Pauli::I; 65];
                row[64] = Pauli::Y;
                (row, i % 2 == 0)
            }),
        );
        state.retire(63);
        state.retire(64);
        let h = CliffordAction::new(
            TableauData::new(vec![(vec![Pauli::X], false)], vec![(vec![Pauli::Z], false)]),
            vec![64],
            1.0,
            1,
        );
        state.apply(&h);
        for i in 0..130 {
            assert_eq!(state.sign(i), (i % 2 == 0) ^ state.is_active(i));
            assert_eq!(state.paulis(i)[64], Pauli::Y);
        }
    }

    #[test]
    fn commutation_and_non_self_inverse_action() {
        let mut state = ObservableState::new(1, [(vec![Pauli::X], false), (vec![Pauli::Y], false)]);
        assert!(!state.commutes(0, 1));
        let s = CliffordAction::new(
            TableauData::new(vec![(vec![Pauli::Z], false)], vec![(vec![Pauli::Y], false)]),
            vec![0],
            1.0,
            1,
        );
        state.apply(&s);
        assert_eq!(state.paulis(0), vec![Pauli::Y]);
        assert_eq!(state.paulis(1), vec![Pauli::X]);
        assert!(!state.sign(0));
        assert!(state.sign(1));
        state.retire(0);
        state.retire(0);
        assert!(!state.is_active(0));
    }
}
