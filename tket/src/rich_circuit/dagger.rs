//! Utilities for dagger modifiers
use std::str::FromStr;

use hugr::{
    extension::{prelude::qb_t, SignatureFunc},
    std_extensions::collections::{
        array::array_type_def, borrow_array::borrow_array_type_def,
        value_array::value_array_type_def,
    },
    types::{
        type_param::TypeParam, CustomType, FuncValueType, PolyFuncTypeRV, SumType, Term, Type,
        TypeBound, TypeEnum, TypeRV,
    },
};

#[allow(missing_docs)]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct ModifierDagger;

impl ModifierDagger {
    /// Create a new ModifierDagger.
    pub fn new() -> Self {
        ModifierDagger
    }
}
impl Default for ModifierDagger {
    fn default() -> Self {
        Self::new()
    }
}
impl FromStr for ModifierDagger {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "ModifierDagger" {
            Ok(Self::new())
        } else {
            Err(())
        }
    }
}
impl ModifierDagger {
    /// Signature for the dagger modifier.
    /// FIXME: this should be chaged to `parameterized` endo-morphism,
    /// that is, Q x C -> Q where Q is quantum and C is classical.
    pub fn signature() -> SignatureFunc {
        PolyFuncTypeRV::new(
            [
                TypeParam::new_list_type(TypeBound::Linear),
                TypeParam::new_list_type(TypeBound::Linear),
            ],
            FuncValueType::new(
                TypeRV::new_function(FuncValueType::new(
                    vec![
                        TypeRV::new_row_var_use(0, TypeBound::Linear),
                        TypeRV::new_row_var_use(1, TypeBound::Linear),
                    ],
                    vec![TypeRV::new_row_var_use(0, TypeBound::Linear)],
                )),
                TypeRV::new_function(FuncValueType::new(
                    vec![
                        TypeRV::new_row_var_use(0, TypeBound::Linear),
                        TypeRV::new_row_var_use(1, TypeBound::Linear),
                    ],
                    TypeRV::new_row_var_use(0, TypeBound::Linear),
                )),
            ),
        )
        .into()
    }
}

/// Checks if a type is quantum or an array/list of quantum types.
///  Ideally, this should be a method in hugr::types::TypeRV.
pub fn contain_quantum_type(ty: &Type) -> bool {
    contain_quantum_type_rv(&ty.clone().into())
}

fn contain_quantum_array_type(custom_type: &CustomType) -> bool {
    let name = custom_type.name();
    let array_type_defs = [
        array_type_def(),
        value_array_type_def(),
        borrow_array_type_def(),
    ];
    if array_type_defs.iter().any(|def| *def.name() == *name) {
        if let Some(arg) = custom_type.args().get(1) {
            return contain_quantum_type_term(arg);
        }
    }
    false
}

fn contain_quantum_type_rv(ty: &TypeRV) -> bool {
    if *ty == qb_t() {
        return true;
    } else if let TypeEnum::Extension(custom_type) = ty.as_type_enum() {
        if contain_quantum_array_type(custom_type) {
            return true;
        }
    } else if let TypeEnum::Sum(sub_type) = ty.as_type_enum() {
        match sub_type {
            // we consider unit type as a quantum type
            SumType::Unit { .. } => return false,
            SumType::General { rows } => {
                return rows
                    .into_iter()
                    .any(|tys| tys.iter().any(contain_quantum_type_rv))
            }
            _ => {}
        }
    }
    false
}

/// Checks if a term is quantum or an array/list/tuple of quantum types.
pub fn contain_quantum_type_term(term: &Term) -> bool {
    match term {
        Term::RuntimeType(_) => false,
        Term::StaticType => false,
        Term::BoundedNatType(_) => false,
        Term::ListType(term) | Term::TupleType(term) => contain_quantum_type_term(term),
        Term::Runtime(ty) => contain_quantum_type(ty),
        // TODO: Is this correct?
        // We are not checking parameters of types here, but rather whether the term itself is quantum type.
        Term::List(_) | Term::ListConcat(_) | Term::Tuple(_) | Term::TupleConcat(_) => false,
        Term::ConstType(_) => false,
        _ => false,
        // TODO: A lot of other cases are just ignored here.
    }
}

#[cfg(test)]
mod tests {
    use hugr::{std_extensions::collections::array::array_type, types::SumType};

    use crate::extension::bool::bool_type;

    use super::*;
    #[test]
    fn test_is_quantum_type() {
        // Quantum types
        assert!(contain_quantum_type(&qb_t()));
        assert!(contain_quantum_type(&array_type(3, qb_t())));
        assert!(contain_quantum_type(
            &SumType::new(vec![qb_t(), array_type(2, qb_t())]).into()
        ));
        assert!(contain_quantum_type(
            &SumType::new(vec![hugr::type_row![], qb_t().into()]).into()
        ));

        // Non quantum types
        assert!(!contain_quantum_type(&bool_type()));
        assert!(!contain_quantum_type(
            &SumType::new(vec![hugr::type_row![]]).into()
        ));
    }
}
