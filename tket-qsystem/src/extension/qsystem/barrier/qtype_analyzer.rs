use hugr::extension::prelude::qb_t;
use hugr::std_extensions::collections::array::{self};
use hugr::std_extensions::collections::value_array;
use hugr::types::{CustomType, SumType, Type, TypeArg};
use std::collections::HashMap;

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
            [TypeArg::BoundedNat(n), TypeArg::Runtime(elem_ty)] => Some((*n, elem_ty)),
            _ => None,
        })
}

/// Extract the size and element type from a value array type  
pub fn varray_args(ext: &CustomType) -> Option<(u64, &Type)> {
    value_array::value_array_type_def()
        .check_custom(ext)
        .ok()
        .and_then(|_| match ext.args() {
            [TypeArg::BoundedNat(n), TypeArg::Runtime(elem_ty)] => Some((*n, elem_ty)),
            _ => None,
        })
}

/// Analyzes quantum types to determine how they should be unpacked
/// for barrier insertion.
pub(crate) struct QTypeAnalyzer {
    /// Cache of unpacked types.
    qubit_ports: HashMap<Type, Option<Vec<Type>>>,
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
    /// `None` if the type does not contain qubits.
    pub fn unpack_type(&mut self, ty: &Type) -> Option<Vec<Type>> {
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
    fn _new_unpack_type(&mut self, ty: &Type) -> Option<Vec<Type>> {
        if ty == &qb_t() {
            return Some(vec![qb_t()]);
        }

        if let Some(row) = ty.as_sum().and_then(SumType::as_tuple) {
            let mut any_qb = false;
            let unpacked_row = row
                .iter()
                .flat_map(|t| {
                    let t = &t.clone().try_into_type().expect("unexpected row variable.");
                    match self.unpack_type(t) {
                        Some(inner) => {
                            any_qb = true;
                            inner
                        }
                        None => vec![t.clone()],
                    }
                })
                .collect::<Vec<_>>();
            any_qb.then_some(unpacked_row)

            // other sums containing qubits are ignored.
        } else if let Some((size, elem_ty)) = ty
            .as_extension()
            .and_then(|ext| array_args(ext).or_else(|| varray_args(ext)))
        {
            // Special case for Option[Qubit] since it is used in guppy qubit arrays.
            // Fragile - would be better with dedicated guppy array type.
            // Not sure how this can be improved without runtime barrier being able to
            // take a compile time unknown number of qubits.
            if is_opt_qb(elem_ty) {
                Some(vec![qb_t(); size as usize])
            } else {
                self.unpack_type(elem_ty).map(|inner| {
                    let total_size = size as usize * inner.len();
                    let mut result = Vec::with_capacity(total_size);
                    for _ in 0..size {
                        result.extend_from_slice(&inner);
                    }
                    result
                })
            }
        } else {
            None
        }
    }

    /// Count the number of wires in a row in an unpacked type.
    pub fn num_unpacked_wires(&mut self, ty: &Type) -> usize {
        self.unpack_type(ty).as_ref().map_or(1, Vec::len)
    }

    /// Report if a type contains qubits.
    pub fn is_qubit_container(&mut self, ty: &Type) -> bool {
        self.unpack_type(ty).is_some()
    }
}

/// Check if a type is an array with the given element type
pub(crate) fn is_array_of(typ: &Type, elem_type: &Type) -> Option<u64> {
    typ.as_extension()
        .and_then(array_args)
        .and_then(|(size, e_ty)| (e_ty == elem_type).then_some(size))
}

/// Check if a type is specifically an array of qubits
pub(crate) fn is_qubit_array(typ: &Type) -> Option<u64> {
    is_array_of(typ, &qb_t())
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

        let qubit_result = analyzer.unpack_type(&qb_t());
        let types = qubit_result.unwrap();
        assert_eq!(types.len(), 1);
        assert_eq!(types[0], qb_t());

        // Non-quantum types should not be containers
        assert!(!analyzer.is_qubit_container(&bool_t()));
    }

    #[test]
    fn test_array_types() {
        // TODO add value arrays
        let mut analyzer = QTypeAnalyzer::new();

        // Array of qubits should be a container with that many qubits
        let qubit_array = array_type(3, qb_t());
        let result = analyzer.unpack_type(&qubit_array);

        let types = result.unwrap();
        assert_eq!(types.len(), 3);
        assert!(types.iter().all(|t| t == &qb_t()));

        // Array of non-quantum types should not be a container
        let bool_array = array_type(5, bool_t());
        assert!(!analyzer.is_qubit_container(&bool_array));

        // Nested arrays of qubits
        let nested_array = array_type(2, array_type(3, qb_t()));
        let result = analyzer.unpack_type(&nested_array);

        let types = result.unwrap();
        assert_eq!(types.len(), 6); // 2 arrays of 3 qubits each
        assert!(types.iter().all(|t| t == &qb_t()));
    }

    #[test]
    fn test_option_types() {
        let mut analyzer = QTypeAnalyzer::new();

        // Option<qubit> by itself should NOT be a container
        let opt_qubit = option_type(qb_t()).into();
        assert!(!analyzer.is_qubit_container(&opt_qubit));

        // Array of Option<qubit> should be a special case (a container with that many qubits)
        let opt_qubit_array = array_type(4, option_type(qb_t()).into());
        let result = analyzer.unpack_type(&opt_qubit_array);

        let types = result.unwrap();
        assert_eq!(types.len(), 4);
        assert!(types.iter().all(|t| t == &qb_t()));

        // Option of non-quantum types should not be a container
        let opt_bool = option_type(bool_t()).into();
        assert!(!analyzer.is_qubit_container(&opt_bool));
    }

    #[test]
    fn test_tuple_types() {
        let mut analyzer = QTypeAnalyzer::new();

        // Tuple with no qubits
        let no_qubit_tuple = Type::new_tuple(vec![bool_t(), usize_t()]);
        assert!(!analyzer.is_qubit_container(&no_qubit_tuple));

        // Tuple with qubits
        let qubit_tuple = Type::new_tuple(vec![bool_t(), qb_t(), usize_t()]);
        let result = analyzer.unpack_type(&qubit_tuple);

        let types = result.unwrap();
        assert_eq!(types.len(), 3); // Only one qubit in the tuple
        assert_eq!(types[1], qb_t());

        // Tuple with multiple qubits
        let multi_qubit_tuple = Type::new_tuple(vec![qb_t(), bool_t(), qb_t()]);
        let result = analyzer.unpack_type(&multi_qubit_tuple);

        let types = result.unwrap();
        assert_eq!(types.len(), 3);
        assert_eq!(types[0], qb_t());
        assert_eq!(types[2], qb_t());

        // Nested tuple with qubits
        let nested_tuple = Type::new_tuple(vec![
            bool_t(),
            Type::new_tuple(vec![usize_t(), qb_t()]),
            usize_t(),
        ]);
        let result = analyzer.unpack_type(&nested_tuple);

        let types = result.unwrap();
        assert_eq!(types.len(), 4);
        assert_eq!(types[2], qb_t());
    }

    #[test]
    fn test_complex_types() {
        let mut analyzer = QTypeAnalyzer::new();

        // Array of tuples containing qubits
        let complex_type = array_type(2, Type::new_tuple(vec![bool_t(), qb_t()]));
        let result = analyzer.unpack_type(&complex_type);

        let types = result.unwrap();
        assert_eq!(types.len(), 4); // 2 arrays, each with one qubit in the tuple

        // Tuple containing array of qubits and standalone qubit
        let complex_type = Type::new_tuple(vec![array_type(3, qb_t()), qb_t()]);
        let result = analyzer.unpack_type(&complex_type);

        let types = result.unwrap();
        assert_eq!(types.len(), 4); // 3 from array + 1 standalone
        assert!(types.iter().all(|t| t == &qb_t()));

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

        let types = result.unwrap();
        assert_eq!(types.len(), 10); // 1 + 2*(1 + 3) + 1 = 10 wires total
    }

    #[test]
    fn test_helper_functions() {
        let mut analyzer = QTypeAnalyzer::new();
        // Test unpacked_wires
        assert_eq!(analyzer.num_unpacked_wires(&bool_t()), 1);
        assert_eq!(
            analyzer.num_unpacked_wires(&Type::new_tuple(vec![qb_t(), bool_t(), qb_t()])),
            3
        );

        // Test is_qubit_array
        assert_eq!(is_qubit_array(&array_type(5, qb_t())), Some(5));
        assert_eq!(is_qubit_array(&qb_t()), None);
        assert_eq!(is_qubit_array(&array_type(3, bool_t())), None);
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
        let types = result1.unwrap();
        assert_eq!(types.len(), 13); // 10 from array + 3 from tuple
        assert!(types.iter().all(|t| t == &qb_t()));
    }
}
