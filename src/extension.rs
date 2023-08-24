//! This module defines the Hugr extensions used to represent circuits.
//!
//! This includes a extension for the opaque TKET1 operations.

use std::collections::HashMap;

use hugr::extension::{ExtensionId, ExtensionSet, SignatureError};
use hugr::ops::custom::{ExternalOp, OpaqueOp};
use hugr::ops::OpName;
use hugr::types::type_param::{CustomTypeArg, TypeArg, TypeParam};
use hugr::types::{CustomType, Type, TypeBound, TypeRow};
use hugr::Extension;
use lazy_static::lazy_static;
use smol_str::SmolStr;

use super::json::op::JsonOp;

/// The ID of the TKET1 extension.
pub const TKET1_EXTENSION_ID: ExtensionId = SmolStr::new_inline("TKET1");

/// The name for the linear bit custom type.
pub const LINEAR_BIT_NAME: SmolStr = SmolStr::new_inline("LBit");

/// The name for opaque TKET1 operations.
pub const JSON_OP_NAME: SmolStr = SmolStr::new_inline("TKET1 Json Op");

lazy_static! {

    /// A custom type for the encoded TKET1 operation
    static ref TKET1_OP_PAYLOAD : CustomType = CustomType::new("TKET1 Json Op", vec![], TKET1_EXTENSION_ID, TypeBound::Eq);

    /// The TKET1 extension, containing the opaque TKET1 operations.
    pub static ref TKET1_EXTENSION: Extension = {
        let mut res = Extension::new(TKET1_EXTENSION_ID);

        res.add_type(LINEAR_BIT_NAME, vec![], "A linear bit.".into(), TypeBound::Any.into()).unwrap();

        let json_op_payload = TypeParam::Opaque(TKET1_OP_PAYLOAD.clone());
        res.add_op_custom_sig(
            JSON_OP_NAME,
            "An opaque TKET1 operation.".into(),
            vec![json_op_payload],
            HashMap::new(),
            vec![],
            json_op_signature,
        ).unwrap();

        res
    };

    /// The type for linear bits. Part of the TKET1 extension.
    pub static ref LINEAR_BIT: Type = {
        Type::new_extension(TKET1_EXTENSION
            .get_type(&LINEAR_BIT_NAME)
            .unwrap()
            .instantiate_concrete([])
            .unwrap())
    };
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
    let payload = TypeArg::Opaque(CustomTypeArg::new(TKET1_OP_PAYLOAD.clone(), op).unwrap());
    OpaqueOp::new(
        TKET1_EXTENSION_ID,
        JSON_OP_NAME,
        "".into(),
        vec![payload],
        Some(sig.into()),
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
    let Some(TypeArg::Opaque(op)) = ext.args().get(0) else {
        // TODO: Throw an error? We should never get here if the name matches.
        return None;
    };
    let op = serde_yaml::from_value(op.value.clone()).ok()?;
    Some(op)
}

/// Compute the signature of a json-encoded TKET1 operation.
fn json_op_signature(args: &[TypeArg]) -> Result<(TypeRow, TypeRow, ExtensionSet), SignatureError> {
    let [TypeArg::Opaque(arg)] = args else {
        // This should have already been checked.
        panic!("Wrong number of arguments");
    };
    let op: JsonOp = serde_yaml::from_value(arg.value.clone()).unwrap(); // TODO Errors!
    let sig = op.signature();
    Ok((sig.input, sig.output, sig.extension_reqs))
}
