use hugr::extension::prelude::qb_t;
use hugr::std_extensions::collections::array::{self};
use hugr::types::{CustomType, SumType, Type, TypeArg};
use itertools::Itertools;
use std::collections::HashMap;

/// Helper struct to record how a type is unpacked.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
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

#[cfg(test)]
mod test {
    use super::*;
    use hugr::extension::prelude::{bool_t, option_type, usize_t};
    use hugr::std_extensions::collections::array::array_type;

    #[test]
    fn test_primitive_types() {
        let mut analyzer = QTypeAnalyzer::new();

        // Qubit should be a container
        let qubit_result = analyzer.unpack_type(&qb_t());
        assert!(qubit_result.is_qb_container());
        match qubit_result {
            UnpackedRow::QbContainer(types) => {
                assert_eq!(types.len(), 1);
                assert_eq!(types[0], qb_t());
            }
            _ => panic!("Expected QbContainer"),
        }

        // Non-quantum types should not be containers
        let bool_result = analyzer.unpack_type(&bool_t());
        assert!(!bool_result.is_qb_container());
    }

    #[test]
    fn test_array_types() {
        let mut analyzer = QTypeAnalyzer::new();

        // Array of qubits should be a container with that many qubits
        let qubit_array = array_type(3, qb_t());
        let result = analyzer.unpack_type(&qubit_array);
        assert!(result.is_qb_container());
        match result {
            UnpackedRow::QbContainer(types) => {
                assert_eq!(types.len(), 3);
                assert!(types.iter().all(|t| t == &qb_t()));
            }
            _ => panic!("Expected QbContainer"),
        }

        // Array of non-quantum types should not be a container
        let bool_array = array_type(5, bool_t());
        let result = analyzer.unpack_type(&bool_array);
        assert!(!result.is_qb_container());

        // Nested arrays of qubits
        let nested_array = array_type(2, array_type(3, qb_t()));
        let result = analyzer.unpack_type(&nested_array);
        assert!(result.is_qb_container());
        match result {
            UnpackedRow::QbContainer(types) => {
                assert_eq!(types.len(), 6); // 2 arrays of 3 qubits each
                assert!(types.iter().all(|t| t == &qb_t()));
            }
            _ => panic!("Expected QbContainer"),
        }
    }

    #[test]
    fn test_option_types() {
        let mut analyzer = QTypeAnalyzer::new();

        // Option<qubit> by itself should NOT be a container
        let opt_qubit = option_type(qb_t()).into();
        let result = analyzer.unpack_type(&opt_qubit);
        assert!(!result.is_qb_container());

        // Array of Option<qubit> should be a special case (a container with that many qubits)
        let opt_qubit_array = array_type(4, option_type(qb_t()).into());
        let result = analyzer.unpack_type(&opt_qubit_array);
        assert!(result.is_qb_container());
        match result {
            UnpackedRow::QbContainer(types) => {
                assert_eq!(types.len(), 4);
                assert!(types.iter().all(|t| t == &qb_t()));
            }
            _ => panic!("Expected QbContainer"),
        }

        // Option of non-quantum types should not be a container
        let opt_bool = option_type(bool_t()).into();
        let result = analyzer.unpack_type(&opt_bool);
        assert!(!result.is_qb_container());
    }

    #[test]
    fn test_tuple_types() {
        let mut analyzer = QTypeAnalyzer::new();

        // Tuple with no qubits
        let no_qubit_tuple = Type::new_tuple(vec![bool_t(), usize_t()]);
        let result = analyzer.unpack_type(&no_qubit_tuple);
        assert!(!result.is_qb_container());

        // Tuple with qubits
        let qubit_tuple = Type::new_tuple(vec![bool_t(), qb_t(), usize_t()]);
        let result = analyzer.unpack_type(&qubit_tuple);
        assert!(result.is_qb_container());
        match result {
            UnpackedRow::QbContainer(types) => {
                assert_eq!(types.len(), 3); // Only one qubit in the tuple
                assert_eq!(types[1], qb_t());
            }
            _ => panic!("Expected QbContainer"),
        }

        // Tuple with multiple qubits
        let multi_qubit_tuple = Type::new_tuple(vec![qb_t(), bool_t(), qb_t()]);
        let result = analyzer.unpack_type(&multi_qubit_tuple);
        assert!(result.is_qb_container());
        match result {
            UnpackedRow::QbContainer(types) => {
                assert_eq!(types.len(), 3);
                assert_eq!(types[0], qb_t());
                assert_eq!(types[2], qb_t());
            }
            _ => panic!("Expected QbContainer"),
        }

        // Nested tuple with qubits
        let nested_tuple = Type::new_tuple(vec![
            bool_t(),
            Type::new_tuple(vec![usize_t(), qb_t()]),
            usize_t(),
        ]);
        let result = analyzer.unpack_type(&nested_tuple);
        assert!(result.is_qb_container());
        match result {
            UnpackedRow::QbContainer(types) => {
                assert_eq!(types.len(), 4);
                assert_eq!(types[2], qb_t());
            }
            _ => panic!("Expected QbContainer"),
        }
    }

    #[test]
    fn test_complex_types() {
        let mut analyzer = QTypeAnalyzer::new();

        // Array of tuples containing qubits
        let complex_type = array_type(2, Type::new_tuple(vec![bool_t(), qb_t()]));
        let result = analyzer.unpack_type(&complex_type);
        assert!(result.is_qb_container());
        match result {
            UnpackedRow::QbContainer(types) => {
                assert_eq!(types.len(), 4); // 2 arrays, each with one qubit in the tuple
            }
            _ => panic!("Expected QbContainer"),
        }

        // Tuple containing array of qubits and standalone qubit
        let complex_type = Type::new_tuple(vec![array_type(3, qb_t()), qb_t()]);
        let result = analyzer.unpack_type(&complex_type);
        assert!(result.is_qb_container());
        match result {
            UnpackedRow::QbContainer(types) => {
                assert_eq!(types.len(), 4); // 3 from array + 1 standalone
                assert!(types.iter().all(|t| t == &qb_t()));
            }
            _ => panic!("Expected QbContainer"),
        }

        // Super nested complex example
        let complex_type = Type::new_tuple(vec![
            bool_t(),
            array_type(
                2,
                Type::new_tuple(vec![qb_t(), array_type(3, option_type(qb_t()).into())]),
            ),
            qb_t(),
        ]);
        let result = analyzer.unpack_type(&complex_type);
        assert!(result.is_qb_container());
        match result {
            UnpackedRow::QbContainer(types) => {
                assert_eq!(types.len(), 10); // 1 + 2*(1 + 3) + 1 = 10 wires total
            }
            _ => panic!("Expected QbContainer"),
        }
    }

    #[test]
    fn test_helper_methods() {
        let mut analyzer = QTypeAnalyzer::new();

        // Test contains_qubits
        assert!(analyzer.unpack_type(&qb_t()).is_qb_container());
        assert!(analyzer
            .unpack_type(&array_type(2, qb_t()))
            .is_qb_container());
        assert!(!analyzer.unpack_type(&bool_t()).is_qb_container());

        // Test is_qubit_array
        assert_eq!(analyzer.is_qubit_array(&array_type(5, qb_t())), Some(5));
        assert_eq!(analyzer.is_qubit_array(&qb_t()), None);
        assert_eq!(analyzer.is_qubit_array(&array_type(3, bool_t())), None);
    }

    #[test]
    fn test_caching() {
        let mut analyzer = QTypeAnalyzer::new();

        // Create a complex type that will take some processing
        let complex_type = Type::new_tuple(vec![
            array_type(10, qb_t()),
            Type::new_tuple(vec![qb_t(), qb_t(), qb_t()]),
        ]);

        // First call should process everything
        let result1 = analyzer.unpack_type(&complex_type);

        // Second call should use the cache
        let result2 = analyzer.unpack_type(&complex_type);

        assert_eq!(result1, result2);

        // Both should be correct
        match result1 {
            UnpackedRow::QbContainer(types) => {
                assert_eq!(types.len(), 13); // 10 from array + 3 from tuple
                assert!(types.iter().all(|t| t == &qb_t()));
            }
            _ => panic!("Expected QbContainer"),
        }
    }
}
