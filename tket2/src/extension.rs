//! This module defines the Hugr extensions used to represent circuits.
//!
//! This includes a extension for the opaque TKET1 operations.

use crate::serialize::pytket::OpaqueTk1Op;
use crate::Tk2Op;
use angle::ANGLE_TYPE;
use hugr::extension::prelude::PRELUDE;
use hugr::extension::simple_op::MakeOpDef;
use hugr::extension::{
    CustomSignatureFunc, ExtensionId, ExtensionRegistry, SignatureError, Version,
};
use hugr::hugr::IdentList;
use hugr::std_extensions::arithmetic::float_types::EXTENSION as FLOAT_TYPES;
use hugr::types::type_param::{TypeArg, TypeParam};
use hugr::types::{CustomType, PolyFuncType, PolyFuncTypeRV, Signature};
use hugr::{type_row, Extension};
use lazy_static::lazy_static;
use smol_str::SmolStr;

/// Definition for Angle ops and types.
pub mod angle;

/// The ID of the TKET1 extension.
pub const TKET1_EXTENSION_ID: ExtensionId = IdentList::new_unchecked("TKET1");

/// The name for opaque TKET1 operations.
pub const TKET1_OP_NAME: SmolStr = SmolStr::new_inline("TKET1 Json Op");

/// The ID of an opaque TKET1 operation metadata.
pub const TKET1_PAYLOAD_NAME: SmolStr = SmolStr::new_inline("TKET1 Json Payload");

/// Current version of the TKET 1 extension
pub const TKET1_EXTENSION_VERSION: Version = Version::new(0, 1, 0);

lazy_static! {
/// A custom type for the encoded TKET1 operation
pub static ref TKET1_OP_PAYLOAD : CustomType =
    TKET1_EXTENSION.get_type(&TKET1_PAYLOAD_NAME).unwrap().instantiate([]).unwrap();

/// The TKET1 extension, containing the opaque TKET1 operations.
pub static ref TKET1_EXTENSION: Extension = {
    let mut res = Extension::new(TKET1_EXTENSION_ID, TKET1_EXTENSION_VERSION);

    let tket1_op_payload = TypeParam::String;
    res.add_op(
        TKET1_OP_NAME,
        "An opaque TKET1 operation.".into(),
        Tk1Signature([tket1_op_payload])
    ).unwrap();

    res
};

/// Extension registry including the prelude, TKET1 and Tk2Ops extensions.
pub static ref REGISTRY: ExtensionRegistry = ExtensionRegistry::try_new([
    TKET1_EXTENSION.to_owned(),
    PRELUDE.to_owned(),
    TKET2_EXTENSION.to_owned(),
    FLOAT_TYPES.to_owned(),
    angle::ANGLE_EXTENSION.to_owned()
]).unwrap();


}

struct Tk1Signature([TypeParam; 1]);

impl CustomSignatureFunc for Tk1Signature {
    fn compute_signature<'o, 'a: 'o>(
        &'a self,
        arg_values: &[TypeArg],
        _def: &'o hugr::extension::OpDef,
        _extension_registry: &ExtensionRegistry,
    ) -> Result<PolyFuncTypeRV, SignatureError> {
        let [TypeArg::String { arg }] = arg_values else {
            // This should have already been checked.
            panic!("Wrong number of arguments");
        };
        let op: OpaqueTk1Op = serde_json::from_str(arg).unwrap(); // TODO Errors!
        let poly_func: PolyFuncType = op.signature().into();
        Ok(poly_func.into())
    }

    fn static_params(&self) -> &[TypeParam] {
        &self.0
    }
}

/// Name of tket 2 extension.
pub const TKET2_EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("tket2.quantum");

/// The name of the symbolic expression opaque type arg.
pub const SYM_EXPR_NAME: SmolStr = SmolStr::new_inline("SymExpr");

/// The name of the symbolic expression opaque type arg.
pub const SYM_OP_ID: SmolStr = SmolStr::new_inline("symbolic_angle");

/// Current version of the TKET 2 extension
pub const TKET2_EXTENSION_VERSION: Version = Version::new(0, 1, 0);

lazy_static! {
/// The type of the symbolic expression opaque type arg.
pub static ref SYM_EXPR_T: CustomType =
    TKET2_EXTENSION.get_type(&SYM_EXPR_NAME).unwrap().instantiate([]).unwrap();

/// The extension definition for TKET2 ops and types.
pub static ref TKET2_EXTENSION: Extension = {
    let mut e = Extension::new(TKET2_EXTENSION_ID, TKET2_EXTENSION_VERSION);
    Tk2Op::load_all_ops(&mut e).expect("add fail");

    e.add_op(
        SYM_OP_ID,
        "Store a sympy expression that can be evaluated to an angle.".to_string(),
        PolyFuncType::new(vec![TypeParam::String], Signature::new(type_row![], type_row![ANGLE_TYPE])),
    )
    .unwrap();

    angle::add_to_extension(&mut e);
    e
};
}
