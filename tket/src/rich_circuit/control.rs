//! Utilities for Control modifiers
use hugr::{
    extension::{prelude::qb_t, SignatureFunc},
    std_extensions::collections::array::array_type_parametric,
    types::{
        type_param::TypeParam, FuncValueType, PolyFuncTypeRV, Term, TypeArg, TypeBound, TypeRV,
    },
};

#[allow(missing_docs)]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct ModifierControl(usize);

impl ModifierControl {
    /// Create a new ModifierControl with a specific number of controls.
    pub fn new(num: usize) -> Self {
        ModifierControl(num)
    }
}
impl Default for ModifierControl {
    fn default() -> Self {
        Self::new(0)
    }
}
impl ModifierControl {
    /// Signature for the control modifier.
    pub fn signature() -> SignatureFunc {
        PolyFuncTypeRV::new(
            [
                TypeParam::max_nat_type(),
                TypeParam::new_list_type(TypeBound::Linear),
                TypeParam::new_list_type(TypeBound::Linear),
            ],
            FuncValueType::new(
                TypeRV::new_function(FuncValueType::new(
                    vec![
                        TypeRV::new_row_var_use(1, TypeBound::Linear),
                        TypeRV::new_row_var_use(2, TypeBound::Copyable),
                    ],
                    vec![TypeRV::new_row_var_use(1, TypeBound::Linear)],
                )),
                TypeRV::new_function(FuncValueType::new(
                    vec![
                        array_type_parametric(
                            TypeArg::new_var_use(0, TypeParam::max_nat_type()),
                            qb_t(),
                        )
                        .unwrap()
                        .into(),
                        TypeRV::new_row_var_use(1, TypeBound::Linear),
                        TypeRV::new_row_var_use(2, TypeBound::Copyable),
                    ],
                    vec![
                        array_type_parametric(
                            TypeArg::new_var_use(0, TypeParam::max_nat_type()),
                            qb_t(),
                        )
                        .unwrap()
                        .into(),
                        TypeRV::new_row_var_use(1, TypeBound::Linear),
                    ],
                )),
            ),
        )
        .into()
    }
}
