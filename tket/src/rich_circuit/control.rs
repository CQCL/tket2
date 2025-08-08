use std::{marker::PhantomData, str::FromStr, sync::Weak};

use hugr::{
    extension::{prelude::qb_t, SignatureFunc},
    types::{
        type_param::{TermVar, TypeParam},
        FuncValueType, PolyFuncTypeRV, TypeBound, TypeRV,
    },
    Extension,
};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct ModifierControl;

impl ModifierControl {
    pub fn new() -> Self {
        ModifierControl
    }
}
impl Default for ModifierControl {
    fn default() -> Self {
        Self::new()
    }
}
impl FromStr for ModifierControl {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "ModifierControl" {
            Ok(Self::new())
        } else {
            Err(())
        }
    }
}
impl ModifierControl {
    pub fn signature() -> SignatureFunc {
        PolyFuncTypeRV::new(
            [
                TypeParam::new_list_type(TypeBound::Linear),
                TypeParam::new_list_type(TypeBound::Linear),
            ],
            FuncValueType::new(
                TypeRV::new_function(FuncValueType::new(
                    vec![TypeRV::new_row_var_use(0, TypeBound::Linear)],
                    vec![TypeRV::new_row_var_use(1, TypeBound::Linear)],
                )),
                TypeRV::new_function(FuncValueType::new(
                    vec![qb_t().into(), TypeRV::new_row_var_use(0, TypeBound::Linear)],
                    vec![qb_t().into(), TypeRV::new_row_var_use(1, TypeBound::Linear)],
                )),
            ),
        )
        .into()
    }
}
