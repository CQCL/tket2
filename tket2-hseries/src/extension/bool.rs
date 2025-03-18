//! This module defines the Hugr extension used to represent bools in Guppy.
use std::sync::{Arc, Weak};

use hugr::{
    extension::{simple_op::{try_from_name, MakeOpDef}, ExtensionBuildError, ExtensionId, ExtensionSet, SignatureFunc, TypeDef, Version},
    ops::constant::{CustomConst, ValueName},
    types::{CustomType, PolyFuncType, Signature, Type, TypeBound, TypeRV},
    Extension,
};
use strum::{EnumIter, EnumString, IntoStaticStr};
use lazy_static::lazy_static;
use smol_str::SmolStr;

/// The ID of the `tket2.bool` extension.
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("tket2.bool");
/// The "tket2.bool" extension version
pub const EXTENSION_VERSION: Version = Version::new(0, 1, 0);

lazy_static! {
    /// The "tket2.futures" extension.
    pub static ref EXTENSION: Arc<Extension>  = {
        Extension::new_arc(EXTENSION_ID, EXTENSION_VERSION, |ext, ext_ref| {
            let _ = add_bool_type_def(ext, ext_ref.clone()).unwrap();
        })
    };

    /// The name of the `bool` type.
    pub static ref BOOL_TYPE_NAME: SmolStr = SmolStr::new_inline("bool");
}

fn add_bool_type_def(
    ext: &mut Extension,
    extension_ref: Weak<Extension>,
) -> Result<&TypeDef, ExtensionBuildError> {
    ext.add_type(
        BOOL_TYPE_NAME.to_owned(),
        vec![TypeBound::Copyable.into()],
        "The Guppy bool type".into(),
        TypeBound::Copyable.into(),
        &extension_ref,
    )
}

/// Returns a `bool` [CustomType].
pub fn bool_custom_type(extension_ref: &Weak<Extension>) -> CustomType {
    CustomType::new(
        BOOL_TYPE_NAME.to_owned(),
        vec![],
        EXTENSION_ID,
        TypeBound::Copyable,
        extension_ref,
    )
}

/// Returns a `bool` [Type].
pub fn bool_type() -> Type {
    bool_custom_type(&Arc::downgrade(&EXTENSION)).into()
}

#[derive(Debug, Clone, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
/// Structure for holding constant bool values.
pub struct ConstBool(bool);

impl ConstBool {
    /// Creates a new [`ConstBool`].
    pub fn new(value: bool) -> Self {
        Self(value)
    }

    /// Returns the value of the constant.
    pub fn value(&self) -> &bool {
        &self.0
    }
}

#[typetag::serde]
impl CustomConst for ConstBool {
    fn name(&self) -> ValueName {
        format!("ConstBool({:?})", self.0).into()
    }

    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        hugr::ops::constant::downcast_equal_consts(self, other)
    }

    fn extension_reqs(&self) -> ExtensionSet {
        ExtensionSet::singleton(EXTENSION_ID)
    }

    fn get_type(&self) -> Type {
        bool_type()
    }
}

#[derive(
    Clone,
    Copy,
    Debug,
    serde::Serialize,
    serde::Deserialize,
    Hash,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    EnumIter,
    IntoStaticStr,
    EnumString,
)]
#[allow(missing_docs)]
#[non_exhaustive]
/// Simple enum of "tket2.bool" operations.
pub enum BoolOpDef {
	BoolToSum,
	SumToBool,
    //Eq,
    //Not,
    //And,
    //Or,
    //Xor,
}

impl MakeOpDef for BoolOpDef {
    fn init_signature(&self, extension_ref: &Weak<Extension>) -> SignatureFunc {
        let bool_type = Type::new_extension(bool_custom_type(extension_ref));
        let sum_type = Type::new_unit_sum(2);
        match self {
            BoolOpDef::BoolToSum => {
                PolyFuncType::new(
                    vec![],
                    Signature::new(bool_type, sum_type),
                )
                .into()
            }
            BoolOpDef::SumToBool => {
                PolyFuncType::new(
                    vec![],
                    Signature::new(sum_type, bool_type),
                )
                .into()
            }
        }
    }

    fn from_def(op_def: &hugr::extension::OpDef) -> Result<Self, hugr::extension::simple_op::OpLoadError> {
        try_from_name(op_def.name(), op_def.extension_id())
    }

    fn extension(&self) -> ExtensionId {
        EXTENSION_ID
    }

    fn description(&self) -> String {
        match self {
            BoolOpDef::BoolToSum => "Convert a Guppy bool into a Hugr unit sum.".into(),
            BoolOpDef::SumToBool => "Convert a Hugr unit sum into a Guppy bool.".into(),
        }
    }

    fn extension_ref(&self) -> Weak<Extension> {
        Arc::downgrade(&EXTENSION)
    }

}
