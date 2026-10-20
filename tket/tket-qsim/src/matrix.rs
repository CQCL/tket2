use nalgebra::DMatrix;
use num_complex::Complex64;

pub(crate) fn identity(num_qubits: usize) -> DMatrix<Complex64> {
    let dimension = 1 << num_qubits;
    DMatrix::identity(dimension, dimension)
}

pub(crate) fn from_row_major(
    num_qubits: usize,
    data: impl AsRef<[Complex64]>,
) -> DMatrix<Complex64> {
    let dimension = 1 << num_qubits;
    DMatrix::from_row_slice(dimension, dimension, data.as_ref())
}

/// Return `matrix` with its global phase aligned to `reference`.
///
/// The largest-magnitude entry in `reference` is used to determine the phase,
/// avoiding unstable phase estimates from entries close to zero. This
/// operation does not perform an approximate comparison; use an `approx`
/// assertion on the returned matrix.
pub fn align_global_phase_to(
    matrix: &DMatrix<Complex64>,
    reference: &DMatrix<Complex64>,
) -> DMatrix<Complex64> {
    assert_eq!(matrix.shape(), reference.shape());

    let Some((anchor, reference_value)) = reference
        .iter()
        .enumerate()
        .max_by(|(_, lhs), (_, rhs)| lhs.norm_sqr().total_cmp(&rhs.norm_sqr()))
    else {
        return matrix.clone();
    };

    let value = matrix.as_slice()[anchor];
    if value == Complex64::new(0.0, 0.0) || *reference_value == Complex64::new(0.0, 0.0) {
        return matrix.clone();
    }

    let phase = reference_value / value;
    let phase = phase / Complex64::new(phase.norm(), 0.0);
    matrix.map(|value| value * phase)
}

/// Expand a gate acting on `target_qubits` into a full-system matrix.
pub(crate) fn expand_to_system(
    gate: &DMatrix<Complex64>,
    target_qubits: &[usize],
    total_qubits: usize,
) -> DMatrix<Complex64> {
    assert_eq!(gate.nrows(), gate.ncols());
    assert!(gate.nrows().is_power_of_two());
    let num_gate_qubits = gate.nrows().ilog2() as usize;
    assert_eq!(target_qubits.len(), num_gate_qubits);

    let full_dimension = 1usize << total_qubits;
    let gate_dimension = gate.nrows();
    let mut result = DMatrix::zeros(full_dimension, full_dimension);

    for column in 0..full_dimension {
        for gate_row in 0..gate_dimension {
            // Determine the full row index by replacing target qubit bits in
            // `column` with the bits from `gate_row`.
            let mut full_row = column;
            for (gate_bit_position, &system_qubit) in target_qubits.iter().enumerate() {
                let bit = (gate_row >> (num_gate_qubits - 1 - gate_bit_position)) & 1;
                if bit == 1 {
                    full_row |= 1 << (total_qubits - 1 - system_qubit);
                } else {
                    full_row &= !(1 << (total_qubits - 1 - system_qubit));
                }
            }

            // Extract the target qubit bits from `column` to get the gate
            // column index.
            let mut gate_column = 0usize;
            for (gate_bit_position, &system_qubit) in target_qubits.iter().enumerate() {
                let bit = (column >> (total_qubits - 1 - system_qubit)) & 1;
                gate_column |= bit << (num_gate_qubits - 1 - gate_bit_position);
            }

            result[(full_row, column)] += gate[(gate_row, gate_column)];
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use approx::{assert_abs_diff_eq, assert_abs_diff_ne};

    use super::*;

    #[test]
    fn identity_has_the_expected_dimension() {
        let identity = identity(2);
        assert_eq!(identity.shape(), (4, 4));
        assert_eq!(identity, DMatrix::identity(4, 4));
    }

    #[test]
    fn expands_single_qubit_gate() {
        let x = from_row_major(
            1,
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        );
        let expanded = expand_to_system(&x, &[1], 2);
        let expected = identity(1).kronecker(&x);
        assert_abs_diff_eq!(expanded, expected, epsilon = 1e-10);
    }

    #[test]
    fn global_phase_alignment_rejects_disjoint_support() {
        let identity = identity(1);
        let x = from_row_major(
            1,
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        );

        let identity = align_global_phase_to(&identity, &x);
        assert_abs_diff_ne!(identity, x, epsilon = 1e-10);
    }
}
