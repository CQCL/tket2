/// WIP
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
pub struct ModifierPower;

impl ModifierPower {
    pub fn new() -> Self {
        ModifierPower
    }
}
impl Default for ModifierPower {
    fn default() -> Self {
        Self::new()
    }
}
impl FromStr for ModifierPower {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "ModifierPower" {
            Ok(Self::new())
        } else {
            Err(())
        }
    }
}
impl ModifierPower {
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
                    vec![TypeRV::new_row_var_use(0, TypeBound::Linear)],
                    vec![TypeRV::new_row_var_use(1, TypeBound::Linear)],
                )),
            ),
        )
        .into()
    }
}
