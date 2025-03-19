//! This module defines the Hugr extension used to represent bools in Guppy.
use std::sync::{Arc, Weak};

use hugr::{
    extension::{
        simple_op::{
            try_from_name, HasConcrete, HasDef, MakeExtensionOp, MakeOpDef, MakeRegisteredOp,
        },
        ExtensionBuildError, ExtensionId, ExtensionSet, SignatureFunc, TypeDef, Version,
    },
    ops::{
        constant::{CustomConst, ValueName},
        ExtensionOp, NamedOp, OpName,
    },
    types::{CustomType, PolyFuncType, Signature, Type, TypeBound},
    Extension,
};
use lazy_static::lazy_static;
use smol_str::SmolStr;
use strum::{EnumIter, EnumString, IntoStaticStr};

/// The ID of the `tket2.bool` extension.
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("tket2.bool");
/// The "tket2.bool" extension version
pub const EXTENSION_VERSION: Version = Version::new(0, 1, 0);

lazy_static! {
    /// The "tket2.futures" extension.
    pub static ref EXTENSION: Arc<Extension>  = {
        Extension::new_arc(EXTENSION_ID, EXTENSION_VERSION, |ext, ext_ref| {
            let _ = add_bool_type_def(ext, ext_ref.clone()).unwrap();
            BoolOpDef::load_all_ops(ext, ext_ref).unwrap();
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
    Eq,
    Not,
    And,
    Or,
    Xor,
}

impl MakeOpDef for BoolOpDef {
    fn init_signature(&self, extension_ref: &Weak<Extension>) -> SignatureFunc {
        let bool_type = Type::new_extension(bool_custom_type(extension_ref));
        let sum_type = Type::new_unit_sum(2);
        match self {
            BoolOpDef::BoolToSum => {
                PolyFuncType::new(vec![], Signature::new(bool_type, sum_type)).into()
            }
            BoolOpDef::SumToBool => {
                PolyFuncType::new(vec![], Signature::new(sum_type, bool_type)).into()
            }
            BoolOpDef::Not => PolyFuncType::new(
                vec![],
                Signature::new(bool_type.clone(), vec![bool_type.clone()]),
            )
            .into(),
            BoolOpDef::Eq | BoolOpDef::And | BoolOpDef::Or | BoolOpDef::Xor => PolyFuncType::new(
                vec![],
                Signature::new(bool_type.clone(), vec![bool_type.clone(), bool_type]),
            )
            .into(),
        }
    }

    fn from_def(
        op_def: &hugr::extension::OpDef,
    ) -> Result<Self, hugr::extension::simple_op::OpLoadError> {
        try_from_name(op_def.name(), op_def.extension_id())
    }

    fn extension(&self) -> ExtensionId {
        EXTENSION_ID
    }

    fn description(&self) -> String {
        match self {
            BoolOpDef::BoolToSum => "Convert a Guppy bool into a Hugr unit sum.".into(),
            BoolOpDef::SumToBool => "Convert a Hugr unit sum into a Guppy bool.".into(),
            BoolOpDef::Eq => "Equality between two Guppy bools.".into(),
            BoolOpDef::Not => "Negation of a Guppy bool.".into(),
            BoolOpDef::And => "Logical AND between two Guppy bools.".into(),
            BoolOpDef::Or => "Logical OR between two Guppy bools.".into(),
            BoolOpDef::Xor => "Logical XOR between two Guppy bools.".into(),
        }
    }

    fn extension_ref(&self) -> Weak<Extension> {
        Arc::downgrade(&EXTENSION)
    }
}

impl HasConcrete for BoolOpDef {
    type Concrete = BoolOp;

    fn instantiate(
        &self,
        _type_args: &[hugr::types::TypeArg],
    ) -> Result<Self::Concrete, hugr::extension::simple_op::OpLoadError> {
        Ok(match self {
            BoolOpDef::BoolToSum => BoolOp::BoolToSum,
            BoolOpDef::SumToBool => BoolOp::SumToBool,
            BoolOpDef::Eq => BoolOp::Eq,
            BoolOpDef::Not => BoolOp::Not,
            BoolOpDef::And => BoolOp::And,
            BoolOpDef::Or => BoolOp::Or,
            BoolOpDef::Xor => BoolOp::Xor,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Concrete instances of a "tket2.bool" operations.
pub enum BoolOp {
    /// A `tket2.bool.bool_to_sum` op.
    BoolToSum,
    /// A `tket2.bool.sum_to_bool` op.
    SumToBool,
    /// A `tket2.bool.eq` op.
    Eq,
    /// A `tket2.bool.not` op.
    Not,
    /// A `tket2.bool.and` op.
    And,
    /// A `tket2.bool.or` op.
    Or,
    /// A `tket2.bool.xor` op.
    Xor,
}

impl BoolOp {
    /// Returns the `BoolOpDef` for this operation.
    pub fn bool_op_def(&self) -> BoolOpDef {
        match self {
            BoolOp::BoolToSum => BoolOpDef::BoolToSum,
            BoolOp::SumToBool => BoolOpDef::SumToBool,
            BoolOp::Eq => BoolOpDef::Eq,
            BoolOp::Not => BoolOpDef::Not,
            BoolOp::And => BoolOpDef::And,
            BoolOp::Or => BoolOpDef::Or,
            BoolOp::Xor => BoolOpDef::Xor,
        }
    }
}

impl NamedOp for BoolOp {
    fn name(&self) -> OpName {
        let n: &'static str = self.bool_op_def().into();
        n.into()
    }
}

impl HasDef for BoolOp {
    type Def = BoolOpDef;
}

impl MakeExtensionOp for BoolOp {
    fn from_extension_op(
        ext_op: &ExtensionOp,
    ) -> Result<Self, hugr::extension::simple_op::OpLoadError> {
        BoolOpDef::from_def(ext_op.def())?.instantiate(ext_op.args())
    }

    fn type_args(&self) -> Vec<hugr::types::TypeArg> {
        vec![]
    }
}

impl MakeRegisteredOp for BoolOp {
    fn extension_id(&self) -> ExtensionId {
        EXTENSION_ID
    }

    fn extension_ref(&self) -> Weak<Extension> {
        Arc::downgrade(&EXTENSION)
    }
}

#[cfg(test)]
pub(crate) mod test {
    use super::*;
    use hugr::{builder::{Dataflow, DataflowHugr, FunctionBuilder}, extension::OpDef};
    use strum::IntoEnumIterator;

    fn get_opdef(op: impl NamedOp) -> Option<&'static Arc<OpDef>> {
        EXTENSION.get_op(&op.name())
    }

    #[test]
    fn create_extension() {
        assert_eq!(EXTENSION.name(), &EXTENSION_ID);

        for o in BoolOpDef::iter() {
            assert_eq!(BoolOpDef::from_def(get_opdef(o).unwrap()), Ok(o));
        }
    }

    #[test]
    fn bool_op_from_def() {
        assert_eq!(Ok(BoolOp::Not), BoolOpDef::Not.instantiate(&[]))
    }

    #[test]
    fn test_bool_to_sum() {
        let bool_type = bool_type();
        let sum_type = Type::new_unit_sum(2);

        let op = BoolOp::BoolToSum;

        let hugr = {
            let mut func_builder = FunctionBuilder::new(
                "bool_to_sum",
                Signature::new(bool_type, sum_type),
            )
            .unwrap();
            let [input] = func_builder.input_wires_arr();
            let output = func_builder.add_dataflow_op(op.clone(), [input]).unwrap();
            func_builder
                .finish_hugr_with_outputs(output.outputs())
                .unwrap()
        };
        hugr.validate().unwrap();
    }
}
