//! This module defines the Hugr extensions used to represent circuits.
//!
//! This includes a extension for the opaque TKET1 operations.

use std::sync::Arc;

use crate::serialize::pytket::OpaqueTk1Op;
use crate::Tk2Op;
use hugr::extension::simple_op::MakeOpDef;
use hugr::extension::{
    CustomSignatureFunc, ExtensionId, ExtensionRegistry, SignatureError, Version,
};
use hugr::hugr::IdentList;
use hugr::std_extensions::STD_REG;
use hugr::types::type_param::{TypeArg, TypeParam};
use hugr::types::{CustomType, PolyFuncType, PolyFuncTypeRV};
use hugr::Extension;
use lazy_static::lazy_static;
use smol_str::SmolStr;

/// Definition for Angle ops and types.
pub mod rotation;
pub mod sympy;

use sympy::SympyOpDef;

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
pub static ref TKET1_EXTENSION: Arc<Extension>  = {
    Extension::new_arc(TKET1_EXTENSION_ID, TKET1_EXTENSION_VERSION, |res, ext_ref| {
        res.add_op(
            TKET1_OP_NAME,
            "An opaque TKET1 operation.".into(),
            Tk1Signature([TypeParam::String]),
            ext_ref
        ).unwrap();
    })
};

/// Extension registry including the prelude, std, TKET1, and Tk2Ops extensions.
pub(crate) static ref REGISTRY: ExtensionRegistry = ExtensionRegistry::new(
    STD_REG.iter().map(|e| e.to_owned()).chain([
    TKET1_EXTENSION.to_owned(),
    TKET2_EXTENSION.to_owned(),
    rotation::ROTATION_EXTENSION.to_owned()
]));

}

struct Tk1Signature([TypeParam; 1]);

impl CustomSignatureFunc for Tk1Signature {
    fn compute_signature<'o, 'a: 'o>(
        &'a self,
        arg_values: &[TypeArg],
        _def: &'o hugr::extension::OpDef,
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

/// Current version of the TKET 2 extension
pub const TKET2_EXTENSION_VERSION: Version = Version::new(0, 1, 1);

lazy_static! {
    /// The extension definition for TKET2 ops and types.
    pub static ref TKET2_EXTENSION: Arc<Extension> = {
        Extension::new_arc(TKET2_EXTENSION_ID, TKET2_EXTENSION_VERSION, |res, ext_ref| {
            Tk2Op::load_all_ops(res, ext_ref).expect("add_fail");
            SympyOpDef.add_to_extension(res, ext_ref).unwrap();
        })
    };
}
