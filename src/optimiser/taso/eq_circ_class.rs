use std::path::Path;

use hugr::Hugr;
use itertools::Itertools;

use crate::circuit::Circuit;

use super::qtz_circuit::load_ecc_set;

#[derive(Debug, Clone)]
pub enum EqCircClassError {
    NoRepresentative,
}

/// A set of circuits forming an Equivalence Circuit Class (ECC).
///
/// The set contains a distinguished circuit called the representative circuit,
/// typically chosen to be the smallest circuit in the set.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct EqCircClass {
    rep_circ: Hugr,
    /// Other equivalent circuits to the representative.
    other_circs: Vec<Hugr>,
}

impl EqCircClass {
    /// Create a new equivalence class with a representative circuit.
    pub fn new(rep_circ: Hugr, other_circs: Vec<Hugr>) -> Self {
        Self {
            rep_circ,
            other_circs,
        }
    }

    /// The representative circuit of the equivalence class.
    pub fn rep_circ(&self) -> &Hugr {
        &self.rep_circ
    }

    /// The other circuits in the equivalence class.
    pub fn others(&self) -> &[Hugr] {
        &self.other_circs
    }

    /// All circuits in the equivalence class.
    pub fn circuits(&self) -> impl Iterator<Item = &Hugr> {
        std::iter::once(&self.rep_circ).chain(self.other_circs.iter())
    }

    /// Consume into circuits of the equivalence class.
    pub fn into_circuits(self) -> impl Iterator<Item = Hugr> {
        std::iter::once(self.rep_circ).chain(self.other_circs)
    }

    /// The number of circuits in the equivalence class.
    ///
    /// An ECC always has a representative circuit, so this method will always
    /// return an integer strictly greater than 0.
    pub fn n_circuits(&self) -> usize {
        self.other_circs.len() + 1
    }

    /// Create an equivalence class from a set of circuits.
    ///
    /// The smallest circuit is chosen as the representative.
    pub fn from_circuits(circs: impl Into<Vec<Hugr>>) -> Result<Self, EqCircClassError> {
        let mut circs: Vec<_> = circs.into();
        if circs.is_empty() {
            return Err(EqCircClassError::NoRepresentative);
        };

        // Find the index for the smallest circuit
        let min_index = circs.iter().position_min_by_key(|c| c.num_gates()).unwrap();
        let representative = circs.swap_remove(min_index);
        Ok(Self::new(representative, circs))
    }
}

/// Load a set of equivalence classes from a JSON file.
pub fn load_eccs_json_file(path: impl AsRef<Path>) -> Vec<EqCircClass> {
    let all_circs = load_ecc_set(path);

    all_circs
        .into_values()
        .map(EqCircClass::from_circuits)
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
}
