//! This module defines the Hugr resources used to represent circuits.
//!
//! This includes a resource for the opaque TKET1 operations.

use std::collections::HashMap;

use hugr::ops::custom::{ExternalOp, OpaqueOp};
use hugr::ops::OpName;
use hugr::resource::{OpDef, ResourceId, ResourceSet, SignatureError, TypeDef};
use hugr::types::type_param::{TypeArg, TypeParam};
use hugr::types::{Container, CustomType, HashableType, Signature, SimpleType, TypeRow, TypeTag};
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

    /// The type for linear bits. Part of the TKET1 resource.
    pub static ref LINEAR_BIT: SimpleType = SimpleType::Qontainer(Container::Opaque(CustomType::new(
        LINEAR_BIT_NAME,
        [],
        TKET1_RESOURCE_ID,
        TypeTag::Simple,
    )));

    /// The TKET1 resource, containing the opaque TKET1 operations.
    pub static ref TKET1_RESOURCE: Resource = {
        let mut res = Resource::new(TKET1_RESOURCE_ID);

        let linear_type = TypeDef {
            name: LINEAR_BIT_NAME,
            params: vec![],
            description: "A linear bit.".into(),
            resource: None,
            tag: TypeTag::Simple.into(),
        };
        res.add_type(linear_type).unwrap();

        let json_op_param = TypeParam::Value(HashableType::Container(Container::Opaque(
            CustomType::new("TKET1 Json Op", vec![], TKET1_RESOURCE_ID, TypeTag::Simple),
        )));
        let json_op = OpDef::new_with_custom_sig(
            JSON_OP_NAME,
            "An opaque TKET1 operation.".into(),
            vec![json_op_param],
            HashMap::new(),
            json_op_signature,
        );
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
        vec![TypeArg::CustomValue(op)],
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
    let Some(TypeArg::CustomValue(op)) = ext.args().get(0) else {
        // TODO: Throw an error? We should never get here if the name matches.
        return None;
    };
    let op = serde_yaml::from_value(op.clone()).ok()?;
    Some(op)
}

/// Compute the signature of a json-encoded TKET1 operation.
fn json_op_signature(
    args: &[TypeArg],
) -> Result<(TypeRow<SimpleType>, TypeRow<SimpleType>, ResourceSet), SignatureError> {
    let [TypeArg::CustomValue(arg)] = args else {
        panic!("Wrong number of arguments");
        // TODO: Add more Signature Errors
        //return Err(SignatureError::WrongNumArgs(1, args.len()));
    };
    let op: JsonOp = serde_yaml::from_value(arg.clone()).unwrap(); // TODO Errors!
    let Signature {
        signature,
        input_resources,
    } = op.signature();
    Ok((signature.input, signature.output, input_resources))
}
