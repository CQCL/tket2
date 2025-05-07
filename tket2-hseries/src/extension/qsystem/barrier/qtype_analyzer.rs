use hugr::extension::prelude::qb_t;
use hugr::std_extensions::collections::array::{self};
use hugr::types::{CustomType, SumType, Type, TypeArg};
use itertools::Itertools;
use std::collections::HashMap;

/// Helper struct to record how a type is unpacked.
#[derive(Clone, Hash, PartialEq, Eq)]
pub(crate) enum UnpackedRow {
    /// No internal qubits, so not unpacked.
    Other(Type),
    /// Qubit container, unpacked to a row of types.
    QbContainer(Vec<Type>),
}

impl UnpackedRow {
    /// Number of wires in the unpacked row.
    pub fn num_wires(&self) -> usize {
        match self {
            UnpackedRow::Other(_) => 1,
            UnpackedRow::QbContainer(row) => row.len(),
        }
    }

    /// Row produced when unpacked.
    pub fn into_row(self) -> Vec<Type> {
        match self {
            UnpackedRow::Other(ty) => vec![ty],
            UnpackedRow::QbContainer(row) => row,
        }
    }

    /// Returns `true` if the unpacked row is [`QbContainer`].
    ///
    /// [`QbContainer`]: UnpackedRow::QbContainer
    #[must_use]
    pub fn is_qb_container(&self) -> bool {
        matches!(self, Self::QbContainer(..))
    }
}

/// If a type is an option of qubit.
pub(crate) fn is_opt_qb(ty: &Type) -> bool {
    if let Some(sum) = ty.as_sum() {
        if let Some(inner) = sum.as_unary_option() {
            return inner == &qb_t();
        }
    }
    false
}

/// If a custom type is an array, return size and element type.
pub(crate) fn array_args(ext: &CustomType) -> Option<(u64, &Type)> {
    array::array_type_def()
        .check_custom(ext)
        .ok()
        .and_then(|_| match ext.args() {
            [TypeArg::BoundedNat { n }, TypeArg::Type { ty: elem_ty }] => Some((*n, elem_ty)),
            _ => None,
        })
}

/// Analyzes quantum types to determine how they should be unpacked
/// for barrier insertion.
pub(crate) struct QTypeAnalyzer {
    /// Cache of unpacked types.
    qubit_ports: HashMap<Type, UnpackedRow>,
}

impl QTypeAnalyzer {
    /// Create a new instance of the [QTypeAnalyzer].
    pub fn new() -> Self {
        Self {
            qubit_ports: HashMap::new(),
        }
    }

    /// Compute the row produced when a type is unpacked.
    /// Uses memoization to avoid recomputing the same type.
    pub fn unpack_type(&mut self, ty: &Type) -> UnpackedRow {
        match self.qubit_ports.get(ty) {
            Some(unpacked) => unpacked.clone(),
            None => {
                let unpacked = self._new_unpack_type(ty);
                self.qubit_ports.insert(ty.clone(), unpacked.clone());
                unpacked
            }
        }
    }

    /// Compute the row produced when a type is unpacked for the first time.
    fn _new_unpack_type(&mut self, ty: &Type) -> UnpackedRow {
        if ty == &qb_t() {
            return UnpackedRow::QbContainer(vec![qb_t()]);
        }

        if let Some(row) = ty.as_sum().and_then(SumType::as_tuple) {
            let inner_unpacked = row
                .iter()
                .map(|t| {
                    self.unpack_type(&t.clone().try_into_type().expect("unexpected row variable."))
                })
                .collect::<Vec<_>>();
            if inner_unpacked.iter().any(UnpackedRow::is_qb_container) {
                let unpacked_row: Vec<_> = inner_unpacked
                    .into_iter()
                    .map(UnpackedRow::into_row)
                    .collect_vec()
                    .concat();
                return UnpackedRow::QbContainer(unpacked_row);
            }

            // other sums containing qubits are ignored.
        }

        if let Some((size, elem_ty)) = ty.as_extension().and_then(array_args) {
            // Special case for Option[Qubit] since it is used in guppy qubit arrays.
            // Fragile - would be better with dedicated guppy array type.
            // Not sure how this can be improved without runtime barrier being able to
            // take a compile time unknown number of qubits.

            if is_opt_qb(elem_ty) {
                return UnpackedRow::QbContainer(vec![qb_t(); size as usize]);
            } else {
                let elem_wc = self.unpack_type(elem_ty);
                return match elem_wc {
                    UnpackedRow::Other(_) => UnpackedRow::Other(ty.clone()),
                    UnpackedRow::QbContainer(inner) => {
                        return UnpackedRow::QbContainer(vec![inner; size as usize].concat());
                    }
                };
            };
        }

        UnpackedRow::Other(ty.clone())
    }

    /// Check if a type is an array with the given element type
    pub fn is_array_of(&self, typ: &Type, elem_type: &Type) -> Option<u64> {
        typ.as_extension()
            .and_then(array_args)
            .and_then(|(size, e_ty)| (e_ty == elem_type).then_some(size))
    }

    /// Check if a type is specifically an array of qubits
    pub fn is_qubit_array(&self, typ: &Type) -> Option<u64> {
        self.is_array_of(typ, &qb_t())
    }
}

impl Default for QTypeAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
