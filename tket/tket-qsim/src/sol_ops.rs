use nalgebra::DMatrix;
use num_complex::Complex64;
use tket_qsystem::extension::qsystem::sol::SolOp;

use crate::{Simulatable, matrix::from_row_major};

impl Simulatable for SolOp {
    fn num_qubits(&self) -> usize {
        match self {
            SolOp::PhasedX | SolOp::Rz => 1,
            SolOp::PhasedXX => 2,
            // Non-unitary ops
            SolOp::LazyMeasure
            | SolOp::LazyMeasureReset
            | SolOp::TryQAlloc
            | SolOp::QFree
            | SolOp::Reset
            | SolOp::LazyMeasureLeaked
            | SolOp::FutureToMeasurement => {
                panic!("Non-unitary Sol op {self:?} cannot be simulated as a unitary");
            }
            &_ => {
                panic!("Unknown Sol op {self:?}");
            }
        }
    }

    fn unitary(&self, params: &[f64]) -> DMatrix<Complex64> {
        match self {
            SolOp::PhasedX => gate_phased_x(params[0], params[1]),
            SolOp::Rz => gate_rz_radians(params[0]),
            SolOp::PhasedXX => gate_phased_xx(params[0], params[1]),
            _ => panic!("Non-unitary Sol op {self:?} cannot be simulated"),
        }
    }
}

/// PhasedX(θ, φ) — same definition as Helios PhasedX.
/// Parameters are in radians.
fn gate_phased_x(theta: f64, phi: f64) -> DMatrix<Complex64> {
    let ct = (theta / 2.0).cos();
    let st = (theta / 2.0).sin();
    let e_neg_phi = Complex64::from_polar(1.0, -phi);
    let e_pos_phi = Complex64::from_polar(1.0, phi);

    let a = Complex64::new(ct, 0.0);
    let b = Complex64::new(0.0, -1.0) * e_neg_phi * Complex64::new(st, 0.0);
    let c = Complex64::new(0.0, -1.0) * e_pos_phi * Complex64::new(st, 0.0);
    let d = Complex64::new(ct, 0.0);

    from_row_major(1, [a, b, c, d])
}

/// Rz(θ) where θ is in radians.
fn gate_rz_radians(theta: f64) -> DMatrix<Complex64> {
    let e_neg = Complex64::from_polar(1.0, -theta / 2.0);
    let e_pos = Complex64::from_polar(1.0, theta / 2.0);
    let zero = Complex64::new(0.0, 0.0);
    from_row_major(1, [e_neg, zero, zero, e_pos])
}

/// PhasedXX(θ, φ) gate — the native 2-qubit gate for Sol.
///
/// PhasedXX(θ, φ) = exp(-i·θ/2·(cos(φ)X⊗X + sin(φ)Y⊗Y))
///
/// In the computational basis this becomes:
/// ```text
/// [[cos(θ/2),  0,                    0,                    -i·e^{-2iφ}·sin(θ/2)],
///  [0,         cos(θ/2),             -i·sin(θ/2),          0                   ],
///  [0,         -i·sin(θ/2),           cos(θ/2),            0                   ],
///  [-i·e^{2iφ}·sin(θ/2), 0,          0,                    cos(θ/2)           ]]
/// ```
fn gate_phased_xx(theta: f64, phi: f64) -> DMatrix<Complex64> {
    let ct = Complex64::new((theta / 2.0).cos(), 0.0);
    let st = (theta / 2.0).sin();
    let o = Complex64::new(0.0, 0.0);
    let mi = Complex64::new(0.0, -1.0);
    let e_neg2phi = Complex64::from_polar(1.0, -2.0 * phi);
    let e_pos2phi = Complex64::from_polar(1.0, 2.0 * phi);

    let off_diag_01 = mi * Complex64::new(st, 0.0); // for |01⟩↔|10⟩
    let off_diag_00 = mi * e_neg2phi * Complex64::new(st, 0.0); // for |00⟩↔|11⟩
    let off_diag_11 = mi * e_pos2phi * Complex64::new(st, 0.0); // for |11⟩↔|00⟩

    #[rustfmt::skip]
    let data = vec![
        ct,           o,            o,            off_diag_00,
        o,            ct,           off_diag_01,  o,
        o,            off_diag_01,  ct,           o,
        off_diag_11,  o,            o,            ct,
    ];
    from_row_major(2, data)
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;
    use crate::matrix::identity;

    #[test]
    fn test_phased_x_theta_zero_is_identity() {
        let u = SolOp::PhasedX.unitary(&[0.0, 2.34]);
        let id = identity(1);
        assert_abs_diff_eq!(u, id, epsilon = 1e-10);
    }

    #[test]
    fn test_rz_zero_is_identity() {
        let u = SolOp::Rz.unitary(&[0.0]);
        let id = identity(1);
        assert_abs_diff_eq!(u, id, epsilon = 1e-10);
    }

    #[test]
    fn test_phased_xx_zero_is_identity() {
        let u = SolOp::PhasedXX.unitary(&[0.0, 1.23]);
        let id = identity(2);
        assert_abs_diff_eq!(u, id, epsilon = 1e-10);
    }

    #[test]
    fn test_phased_xx_is_unitary() {
        // U†U = I for an arbitrary angle
        let u = SolOp::PhasedXX.unitary(&[1.5, 0.7]);
        let product = u.adjoint() * &u;
        let id = identity(2);
        assert_abs_diff_eq!(product, id, epsilon = 1e-10);
    }
}
