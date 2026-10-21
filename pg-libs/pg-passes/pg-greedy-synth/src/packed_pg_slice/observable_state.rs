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
        // Observable storage is contiguous. Batch active runs so small placed
        // Cliffords update up to 64 signed rows per packed kernel invocation.
        // Retired rows retain their exact value, including their sign.
        let mut run = None;
        for (&index, &active) in self.indices.iter().zip(&self.active) {
            let (start, end) = self.packed.op(index).bit_range();
            if let Some((first, last)) = run
                && (!active || last != start)
            {
                self.packed.apply_action_range(action, first, last);
                run = None;
            }
            if active {
                run = Some((run.map_or(start, |(first, _)| first), end));
            }
        }
        if let Some((start, end)) = run {
            self.packed.apply_action_range(action, start, end);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pg_core::TableauData;

    #[test]
    fn small_packed_actions_match_general_tableaux_exhaustively() {
        use Pauli::{I, X, Y, Z};
        let actions = [
            CliffordAction::new(
                TableauData::new(
                    vec![(vec![Z, I], false), (vec![Z, Z], false)],
                    vec![(vec![X, X], false), (vec![I, X], false)],
                ),
                vec![4, 1],
                1.0,
                1,
            ),
            CliffordAction::new(
                TableauData::new(
                    vec![(vec![I, Z], false), (vec![Z, I], false)],
                    vec![(vec![I, X], false), (vec![X, I], false)],
                ),
                vec![1, 4],
                1.0,
                1,
            ),
            CliffordAction::new(
                TableauData::new(
                    vec![(vec![X, I], true), (vec![I, Z], false)],
                    vec![(vec![Z, I], false), (vec![I, Y], true)],
                ),
                vec![4, 1],
                1.0,
                1,
            ),
            CliffordAction::new(
                TableauData::new(vec![(vec![X], true)], vec![(vec![Z], false)]),
                vec![4],
                1.0,
                1,
            ),
        ];
        let encode = |p| match p {
            I => I_U8,
            X => X_U8,
            Y => Y_U8,
            _ => Z_U8,
        };
        let decode = |p| match p {
            I_U8 => I,
            X_U8 => X,
            Y_U8 => Y,
            _ => Z,
        };
        for n in [0, 1, 63, 64, 65, 128, 257] {
            for action in &actions {
                let mut expected = (0..n)
                    .map(|r| {
                        let mut word = vec![I; 5];
                        word[4] = [I, X, Y, Z][r % 4];
                        word[1] = [I, X, Y, Z][(r / 4) % 4];
                        word[2] = [X, Y, Z, I][r % 4];
                        (word, r % 3 == 0)
                    })
                    .collect::<Vec<_>>();
                let mut state = ObservableState::new(5, expected.clone());
                for (r, expected_row) in expected.iter_mut().enumerate() {
                    if r % 67 == 63 || r % 67 == 64 {
                        state.retire(r);
                    }
                    if state.is_active(r) {
                        let local = action
                            .qubits()
                            .iter()
                            .map(|&q| encode(expected_row.0[q]))
                            .collect::<Vec<_>>();
                        let (output, flip) = action.transform(&local);
                        let mut small = local.iter().copied().map(decode).collect::<Vec<_>>();
                        assert_eq!(action.transform_local(&mut small), flip);
                        assert_eq!(
                            small,
                            output.iter().copied().map(decode).collect::<Vec<_>>()
                        );
                        for (&q, p) in action.qubits().iter().zip(output) {
                            expected_row.0[q] = decode(p);
                        }
                        expected_row.1 ^= flip;
                    }
                }
                state.apply(action);
                for (r, (word, sign)) in expected.into_iter().enumerate() {
                    assert_eq!(state.paulis(r), word, "observable {r}");
                    assert_eq!(state.sign(r), sign, "sign {r}");
                }
            }
        }
    }

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
