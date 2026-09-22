use pg_bitpacked::{I_U8, X_U8, Y_U8, Z_U8, paulis_to_u64s, u64s_to_paulis};
use pg_core::{Pauli, TableauData};
use pg_qm_tableau::Tableau;

/// A Clifford action together with one concrete placement.
///
/// The tableau acts on the ordered qubits in `qubits`. The cost and duration
/// are target-specific values used to rank and schedule the action.
#[derive(Clone, Debug, PartialEq)]
pub struct CliffordAction {
    tableau: TableauData,
    compiled: Tableau,
    qubits: Vec<usize>,
    cost: f64,
    duration: u64,
}

impl CliffordAction {
    /// Creates a placed Clifford action.
    ///
    /// # Panics
    ///
    /// Panics if the placement is empty or contains duplicate qubits, if its
    /// width differs from the tableau width, or if `cost` is not finite and
    /// nonnegative, or if the images do not define a Clifford tableau.
    pub fn new(tableau: TableauData, qubits: Vec<usize>, cost: f64, duration: u64) -> Self {
        assert!(!qubits.is_empty(), "a Clifford action must act on a qubit");
        assert!(
            cost.is_finite() && cost >= 0.0,
            "Clifford action cost must be finite and nonnegative"
        );
        let width = tableau.get_z_images().len();
        assert_eq!(width, tableau.get_x_images().len());
        assert_eq!(width, qubits.len(), "action and placement widths differ");
        let mut distinct = qubits.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(
            distinct.len(),
            qubits.len(),
            "action qubits must be distinct"
        );
        let images = tableau
            .get_z_images()
            .iter()
            .chain(tableau.get_x_images())
            .collect::<Vec<_>>();
        assert!(
            images.iter().all(|(image, _)| image.len() == width),
            "Clifford image width differs from the placement"
        );
        for (i, (left, _)) in images.iter().enumerate() {
            for (j, (right, _)) in images.iter().enumerate().take(i) {
                let anticommutes = left
                    .iter()
                    .zip(right.iter())
                    .filter(|(a, b)| **a != Pauli::I && **b != Pauli::I && a != b)
                    .count()
                    % 2
                    == 1;
                assert_eq!(
                    anticommutes,
                    i == j + width,
                    "Clifford images must preserve generator commutation"
                );
            }
        }
        let compiled = Tableau::from(tableau.clone());
        Self {
            tableau,
            compiled,
            qubits,
            cost,
            duration,
        }
    }

    /// Returns the local Clifford tableau.
    pub fn tableau(&self) -> &TableauData {
        &self.tableau
    }

    /// Returns the ordered qubits on which the tableau acts.
    pub fn qubits(&self) -> &[usize] {
        &self.qubits
    }

    /// Returns the target-specific action cost.
    pub fn cost(&self) -> f64 {
        self.cost
    }

    /// Returns the target-specific action duration.
    pub fn duration(&self) -> u64 {
        self.duration
    }

    pub(crate) fn transform(&self, paulis: &[u8]) -> (Vec<u8>, bool) {
        assert_eq!(paulis.len(), self.qubits.len());
        let paulis = paulis
            .iter()
            .map(|pauli| match *pauli {
                X_U8 => Pauli::X,
                Y_U8 => Pauli::Y,
                Z_U8 => Pauli::Z,
                I_U8 => Pauli::I,
                _ => unreachable!("invalid packed Pauli"),
            })
            .collect::<Vec<_>>();
        let (z_bits, x_bits) = paulis_to_u64s(&paulis);
        let (z_bits, x_bits, sign) = self.compiled.apply_to_pauli(&z_bits, &x_bits);
        let paulis = u64s_to_paulis(&z_bits, &x_bits, self.qubits.len())
            .into_iter()
            .map(|pauli| match pauli {
                Pauli::X => X_U8,
                Pauli::Y => Y_U8,
                Pauli::Z => Z_U8,
                Pauli::I => I_U8,
            })
            .collect();
        (paulis, sign)
    }

    pub(crate) fn embedded(&self, width: usize) -> Tableau {
        assert!(self.qubits.iter().all(|&qubit| qubit < width));
        let identity_image = |qubit: usize, pauli: Pauli| {
            let mut image = vec![Pauli::I; width];
            image[qubit] = pauli;
            (image, false)
        };
        let mut z_images = (0..width)
            .map(|qubit| identity_image(qubit, Pauli::Z))
            .collect::<Vec<_>>();
        let mut x_images = (0..width)
            .map(|qubit| identity_image(qubit, Pauli::X))
            .collect::<Vec<_>>();
        let embed = |(local, sign): &(Vec<Pauli>, bool)| {
            let mut image = vec![Pauli::I; width];
            for (&qubit, pauli) in self.qubits.iter().zip(local) {
                image[qubit] = *pauli;
            }
            (image, *sign)
        };
        for (local, &qubit) in self.qubits.iter().enumerate() {
            z_images[qubit] = embed(&self.tableau.get_z_images()[local]);
            x_images[qubit] = embed(&self.tableau.get_x_images()[local]);
        }
        Tableau::from(TableauData::new(z_images, x_images))
    }
}
