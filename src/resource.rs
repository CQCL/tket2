//! This module defines the Hugr resources used to represent circuits.
//!
//! This includes a resource for the opaque TKET1 operations.

use std::collections::HashMap;

use hugr::ops::custom::{ExternalOp, OpaqueOp};
use hugr::ops::{OpName, OpTrait};
use hugr::resource::{OpDef, ResourceId, ResourceSet, SignatureError};
use hugr::types::type_param::{TypeArg, TypeParam};
use hugr::types::TypeRow;
use hugr::Resource;
use lazy_static::lazy_static;
use smol_str::SmolStr;

use super::json::op::JsonOp;

/// The ID of the TKET1 resource.
pub const TKET1_RESOURCE_ID: ResourceId = SmolStr::new_inline("TKET1");

/// The name for opaque TKET1 operations.
pub const JSON_OP_NAME: SmolStr = SmolStr::new_inline("TKET1 Json Op");

lazy_static! {
    /// The TKET1 resource, containing the opaque TKET1 operations.
    pub static ref TKET1_RESOURCE: Resource = {
        let json_op = OpDef::new_with_custom_sig(
            JSON_OP_NAME,
            "An opaque TKET1 operation.".into(),
            vec![TypeParam::Value],
            HashMap::new(),
            json_op_signature,
        );

        let mut res = Resource::new(TKET1_RESOURCE_ID);
        res.add_op(json_op).unwrap();
        res
    };
}

/// Create a new opaque operation
pub(crate) fn wrap_json_op(op: &JsonOp) -> ExternalOp {
    let sig = op.signature();
    let op = serde_yaml::to_value(op).unwrap();
    OpaqueOp::new(
        TKET1_RESOURCE_ID,
        JSON_OP_NAME,
        "".into(),
        vec![TypeArg::Value(op)],
        Some(sig),
    )
    .into()
}

/// Extract a json-encoded TKET1 operation from an opaque operation, if
/// possible.
pub(crate) fn try_unwrap_json_op(ext: &ExternalOp) -> Option<JsonOp> {
    // TODO: Is this enough to ensure no OpDef collisions?
    let _resources = ext.signature().output_resources;
    // TODO: Check `resources.contains(&TKET1_RESOURCE_ID)`
    // (but the ext op resources are an empty set?)
    if ext.name() != format!("{TKET1_RESOURCE_ID}.{JSON_OP_NAME}") {
        return None;
    }
    let Some(TypeArg::Value(op)) = ext.args().get(0) else {
        // TODO: Throw an error? We should never get here if the name matches.
        return None;
    };
    let op = serde_yaml::from_value(op.clone()).ok()?;
    Some(op)
}

/// Compute the signature of a json-encoded TKET1 operation.
fn json_op_signature(args: &[TypeArg]) -> Result<(TypeRow, TypeRow, ResourceSet), SignatureError> {
    let [TypeArg::Value(arg)] = args else {
        panic!("Wrong number of arguments");
        // TODO: Add more Signature Errors
        //return Err(SignatureError::WrongNumArgs(1, args.len()));
    };
    let op: JsonOp = serde_yaml::from_value(arg.clone()).unwrap(); // TODO Errors!
    let sig = op.signature();
    Ok((sig.input, sig.output, sig.output_resources))
}
