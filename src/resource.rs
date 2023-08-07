//! This module defines the Hugr resources used to represent circuits.
//!
//! This includes a resource for the opaque TKET1 operations.

use std::collections::HashMap;

use hugr::ops::custom::{ExternalOp, OpaqueOp};
use hugr::ops::OpName;
use hugr::resource::{ResourceId, ResourceSet, SignatureError};
use hugr::types::type_param::{CustomTypeArg, TypeArg, TypeParam};
use hugr::types::{CustomType, SimpleType, TypeRow, TypeTag};
use hugr::Resource;
use lazy_static::lazy_static;
use smol_str::SmolStr;

use super::json::op::JsonOp;

/// The ID of the TKET1 resource.
pub const TKET1_RESOURCE_ID: ResourceId = SmolStr::new_inline("TKET1");

/// The name for the linear bit custom type.
pub const LINEAR_BIT_NAME: SmolStr = SmolStr::new_inline("LBit");

/// The name for opaque TKET1 operations.
pub const JSON_OP_NAME: SmolStr = SmolStr::new_inline("TKET1 Json Op");

lazy_static! {

    /// A custom type for the encoded TKET1 operation
    static ref TKET1_OP_PAYLOAD : CustomType = CustomType::new("TKET1 Json Op", vec![], TKET1_RESOURCE_ID, TypeTag::Simple);

    /// The TKET1 resource, containing the opaque TKET1 operations.
    pub static ref TKET1_RESOURCE: Resource = {
        let mut res = Resource::new(TKET1_RESOURCE_ID);

        res.add_type(LINEAR_BIT_NAME, vec![], "A linear bit.".into(), TypeTag::Simple.into()).unwrap();

        let json_op_param = TypeParam::Opaque(TKET1_OP_PAYLOAD.clone());
        res.add_op_custom_sig(
            JSON_OP_NAME,
            "An opaque TKET1 operation.".into(),
            vec![json_op_param],
            HashMap::new(),
            vec![],
            json_op_signature,
        ).unwrap();

        res
    };

    /// The type for linear bits. Part of the TKET1 resource.
    pub static ref LINEAR_BIT: SimpleType = {
        TKET1_RESOURCE
            .get_type(&LINEAR_BIT_NAME)
            .unwrap()
            .instantiate_concrete([])
            .unwrap()
            .into()
    };
}

/// Create a new opaque operation
pub(crate) fn wrap_json_op(op: &JsonOp) -> ExternalOp {
    // TODO: This throws an error
    //let op = serde_yaml::to_value(op).unwrap();
    //TKET1_RESOURCE
    //    .operations()
    //    .get(&JSON_OP_NAME)
    //    .unwrap()
    //    .instantiate_opaque([TypeArg::CustomValue(op)])
    //    .unwrap()
    //    .into()
    let sig = op.signature();
    let op = serde_yaml::to_value(op).unwrap();
    OpaqueOp::new(
        TKET1_RESOURCE_ID,
        JSON_OP_NAME,
        "".into(),
        vec![TypeArg::Opaque(
            CustomTypeArg::new(TKET1_OP_PAYLOAD.clone(), op).unwrap(),
        )],
        Some(sig.into()),
    )
    .into()
}

/// Extract a json-encoded TKET1 operation from an opaque operation, if
/// possible.
pub(crate) fn try_unwrap_json_op(ext: &ExternalOp) -> Option<JsonOp> {
    // TODO: Check `resources.contains(&TKET1_RESOURCE_ID)`
    // (but the ext op resources are an empty set?)
    if ext.name() != format!("{TKET1_RESOURCE_ID}.{JSON_OP_NAME}") {
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
fn json_op_signature(
    args: &[TypeArg],
) -> Result<(TypeRow<SimpleType>, TypeRow<SimpleType>, ResourceSet), SignatureError> {
    let [TypeArg::Opaque(arg)] = args else {
        // This should have already been checked.
        panic!("Wrong number of arguments");
    };
    let op: JsonOp = serde_yaml::from_value(arg.value.clone()).unwrap(); // TODO Errors!
    let sig = op.signature();
    Ok((sig.input, sig.output, sig.resource_reqs))
}
