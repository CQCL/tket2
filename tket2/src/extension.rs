//! This module defines the Hugr extensions used to represent circuits.
//!
//! This includes a extension for the opaque TKET1 operations.

use std::collections::HashMap;

use super::json::op::JsonOp;
use crate::Tk2Op;
use hugr::extension::prelude::PRELUDE;
use hugr::extension::simple_op::MakeOpDef;
use hugr::extension::{CustomSignatureFunc, ExtensionId, ExtensionRegistry, SignatureError};
use hugr::hugr::IdentList;
use hugr::ops::custom::{ExternalOp, OpaqueOp};
use hugr::std_extensions::arithmetic::float_types::{extension as float_extension, FLOAT64_TYPE};
use hugr::types::type_param::{CustomTypeArg, TypeArg, TypeParam};
use hugr::types::{CustomType, FunctionType, PolyFuncType, Type, TypeBound};
use hugr::{type_row, Extension};
use lazy_static::lazy_static;
use smol_str::SmolStr;

/// Definition for Angle ops and types.
pub mod angle;

/// The ID of the TKET1 extension.
pub const TKET1_EXTENSION_ID: ExtensionId = IdentList::new_unchecked("TKET1");

/// The name for the linear bit custom type.
pub const LINEAR_BIT_NAME: SmolStr = SmolStr::new_inline("LBit");

/// The name for opaque TKET1 operations.
pub const JSON_OP_NAME: SmolStr = SmolStr::new_inline("TKET1 Json Op");

/// The ID of an opaque TKET1 operation metadata.
pub const JSON_PAYLOAD_NAME: SmolStr = SmolStr::new_inline("TKET1 Json Payload");

lazy_static! {
/// A custom type for the encoded TKET1 operation
static ref TKET1_OP_PAYLOAD : CustomType =
    TKET1_EXTENSION.get_type(&JSON_PAYLOAD_NAME).unwrap().instantiate([]).unwrap();

/// The TKET1 extension, containing the opaque TKET1 operations.
pub static ref TKET1_EXTENSION: Extension = {
    let mut res = Extension::new(TKET1_EXTENSION_ID);

    res.add_type(LINEAR_BIT_NAME, vec![], "A linear bit.".into(), TypeBound::Any.into()).unwrap();

    let json_op_payload_def = res.add_type(JSON_PAYLOAD_NAME, vec![], "Opaque TKET1 operation metadata.".into(), TypeBound::Eq.into()).unwrap();
    let json_op_payload = TypeParam::Opaque{ty:json_op_payload_def.instantiate([]).unwrap()};
    res.add_op(
        JSON_OP_NAME,
        "An opaque TKET1 operation.".into(),
        JsonOpSignature(json_op_payload)
    ).unwrap();

    res
};

/// The type for linear bits. Part of the TKET1 extension.
pub static ref LINEAR_BIT: Type = {
    Type::new_extension(TKET1_EXTENSION
        .get_type(&LINEAR_BIT_NAME)
        .unwrap()
        .instantiate([])
        .unwrap())
    };

/// Extension registry including the prelude, TKET1 and Tk2Ops extensions.
pub static ref REGISTRY: ExtensionRegistry = ExtensionRegistry::try_new([
    TKET1_EXTENSION.clone(),
    PRELUDE.clone(),
    TKET2_EXTENSION.clone(),
    float_extension(),
]).unwrap();


}
/// Create a new opaque operation
pub(crate) fn wrap_json_op(op: &JsonOp) -> ExternalOp {
    // TODO: This throws an error
    //let op = serde_yaml::to_value(op).unwrap();
    //let payload = TypeArg::Opaque(CustomTypeArg::new(TKET1_OP_PAYLOAD.clone(), op).unwrap());
    //TKET1_EXTENSION
    //    .get_op(&JSON_OP_NAME)
    //    .unwrap()
    //    .instantiate_opaque([payload])
    //    .unwrap()
    //    .into()
    let sig = op.signature();
    let op = serde_yaml::to_value(op).unwrap();
    let payload = TypeArg::Opaque {
        arg: CustomTypeArg::new(TKET1_OP_PAYLOAD.clone(), op).unwrap(),
    };
    OpaqueOp::new(
        TKET1_EXTENSION_ID,
        JSON_OP_NAME,
        "".into(),
        vec![payload],
        sig,
    )
    .into()
}

/// Extract a json-encoded TKET1 operation from an opaque operation, if
/// possible.
pub(crate) fn try_unwrap_json_op(ext: &ExternalOp) -> Option<JsonOp> {
    // TODO: Check `extensions.contains(&TKET1_EXTENSION_ID)`
    // (but the ext op extensions are an empty set?)
    if ext.name() != format!("{TKET1_EXTENSION_ID}.{JSON_OP_NAME}") {
        return None;
    }
    let Some(TypeArg::Opaque { arg }) = ext.args().get(0) else {
        // TODO: Throw an error? We should never get here if the name matches.
        return None;
    };
    let op = serde_yaml::from_value(arg.value.clone()).ok()?;
    Some(op)
}

struct JsonOpSignature(TypeParam);

impl CustomSignatureFunc for JsonOpSignature {
    fn compute_signature<'o, 'a: 'o>(
        &'a self,
        arg_values: &[TypeArg],
        def: &'o hugr::extension::OpDef,
        extension_registry: &ExtensionRegistry,
    ) -> Result<PolyFuncType, SignatureError> {
        let [TypeArg::Opaque { arg }] = arg_values else {
            // This should have already been checked.
            panic!("Wrong number of arguments");
        };
        let op: JsonOp = serde_yaml::from_value(arg.value.clone()).unwrap(); // TODO Errors!
        Ok(op.signature().into())
    }

    fn static_params(&self) -> &[TypeParam] {
        &[self.0]
    }
}
/// Compute the signature of a json-encoded TKET1 operation.
fn json_op_signature(args: &[TypeArg]) -> Result<FunctionType, SignatureError> {
    let [TypeArg::Opaque { arg }] = args else {
        // This should have already been checked.
        panic!("Wrong number of arguments");
    };
    let op: JsonOp = serde_yaml::from_value(arg.value.clone()).unwrap(); // TODO Errors!
    Ok(op.signature())
}

/// Angle type with given log denominator.
pub fn angle_custom_type(log_denom: u8) -> CustomType {
    angle::angle_custom_type(&TKET2_EXTENSION, angle::type_arg(log_denom))
}

/// Name of tket 2 extension.
pub const TKET2_EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("quantum.tket2");

/// The name of the symbolic expression opaque type arg.
pub const SYM_EXPR_NAME: SmolStr = SmolStr::new_inline("SymExpr");

/// The name of the symbolic expression opaque type arg.
pub const SYM_OP_ID: SmolStr = SmolStr::new_inline("symbolic_float");

lazy_static! {
/// The type of the symbolic expression opaque type arg.
pub static ref SYM_EXPR_T: CustomType =
    TKET2_EXTENSION.get_type(&SYM_EXPR_NAME).unwrap().instantiate([]).unwrap();

/// The extension definition for TKET2 ops and types.
pub static ref TKET2_EXTENSION: Extension = {
    let mut e = Extension::new(TKET2_EXTENSION_ID);
    Tk2Op::load_all_ops(&mut e).expect("add fail");

    let sym_expr_opdef = e.add_type(
        SYM_EXPR_NAME,
        vec![],
        "Symbolic expression.".into(),
        TypeBound::Eq.into(),
    )
    .unwrap();
    let sym_expr_param = TypeParam::Opaque{ty:sym_expr_opdef.instantiate([]).unwrap()};

    e.add_op(
        SYM_OP_ID,
        "Store a sympy expression that can be evaluated to a float.".to_string(),
        PolyFuncType::new(vec![sym_expr_param], FunctionType::new(type_row![], type_row![FLOAT64_TYPE])),
    )
    .unwrap();

    angle::add_to_extension(&mut e);
    e
};
}
