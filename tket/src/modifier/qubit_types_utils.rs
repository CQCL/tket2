//! Utilities for qubit types.
/// This is needed for dager modifier, for example,
/// to decide whether to flip the input and output wires.
use hugr::{
    extension::prelude::qb_t,
    std_extensions::collections::{
        array::array_type_def, borrow_array::borrow_array_type_def,
        value_array::value_array_type_def,
    },
    types::{CustomType, SumType, Term, Type, TypeEnum, TypeRV},
};

/// Checks if a type is quantum or an array/list of quantum types.
pub fn contain_qubits(ty: &Type) -> bool {
    contain_qubits_rv(&ty.clone().into())
}

fn contain_qubits_in_array(custom_type: &CustomType) -> bool {
    let name = custom_type.name();
    let array_type_defs = [
        array_type_def(),
        value_array_type_def(),
        borrow_array_type_def(),
    ];
    if array_type_defs.iter().any(|def| *def.name() == *name) {
        if let Some(arg) = custom_type.args().get(1) {
            return contain_qubit_term(arg);
        }
    }
    false
}

fn contain_qubits_rv(ty: &TypeRV) -> bool {
    if *ty == qb_t() {
        return true;
    } else if let TypeEnum::Extension(custom_type) = ty.as_type_enum() {
        if contain_qubits_in_array(custom_type) {
            return true;
        }
    } else if let TypeEnum::Sum(sub_type) = ty.as_type_enum() {
        match sub_type {
            // we consider unit type as a quantum type
            SumType::Unit { .. } => return false,
            SumType::General { rows } => {
                return rows.iter().any(|tys| tys.iter().any(contain_qubits_rv))
            }
            _ => {}
        }
    }
    false
}

/// Checks if a term is quantum or an array/list/tuple of quantum types.
pub fn contain_qubit_term(term: &Term) -> bool {
    // TODO: A lot of other cases are just ignored here.
    match term {
        Term::Runtime(ty) => contain_qubits(ty),
        Term::ListType(term) | Term::TupleType(term) => contain_qubit_term(term),
        // Note that we are not checking parameters of types here, but rather whether the term itself is quantum type.
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use hugr::{
        std_extensions::collections::array::array_type,
        types::{Signature, SumType},
    };

    use crate::extension::bool::bool_type;

    use super::*;
    #[test]
    fn test_is_quantum_type() {
        // has qubits
        assert!(contain_qubits(&qb_t()));
        assert!(contain_qubits(&array_type(3, qb_t())));
        assert!(contain_qubits(
            &SumType::new(vec![qb_t(), array_type(2, qb_t())]).into()
        ));
        assert!(contain_qubits(
            &SumType::new(vec![hugr::type_row![], qb_t().into()]).into()
        ));

        // No qubuits
        assert!(!contain_qubits(&bool_type()));
        assert!(!contain_qubits(
            &SumType::new(vec![hugr::type_row![]]).into()
        ));
        assert!(!contain_qubits(&Type::new_function(Signature::new_endo(
            qb_t()
        ))))
    }
}
