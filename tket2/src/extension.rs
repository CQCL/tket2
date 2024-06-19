//! This module defines the Hugr extensions used to represent circuits.
//!
//! This includes a extension for the opaque TKET1 operations.

use crate::serialize::pytket::OpaqueTk1Op;
use crate::Tk2Op;
use hugr::extension::prelude::PRELUDE;
use hugr::extension::simple_op::MakeOpDef;
use hugr::extension::{CustomSignatureFunc, ExtensionId, ExtensionRegistry, SignatureError};
use hugr::hugr::IdentList;
use hugr::std_extensions::arithmetic::float_types::{EXTENSION as FLOAT_EXTENSION, FLOAT64_TYPE};
use hugr::types::type_param::{TypeArg, TypeParam};
use hugr::types::{CustomType, FunctionType, PolyFuncType, TypeBound};
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

lazy_static! {
/// A custom type for the encoded TKET1 operation
pub static ref TKET1_OP_PAYLOAD : CustomType =
    TKET1_EXTENSION.get_type(&TKET1_PAYLOAD_NAME).unwrap().instantiate([]).unwrap();

/// The TKET1 extension, containing the opaque TKET1 operations.
pub static ref TKET1_EXTENSION: Extension = {
    let mut res = Extension::new(TKET1_EXTENSION_ID);

    let tket1_op_payload_def = res.add_type(TKET1_PAYLOAD_NAME, vec![], "Opaque TKET1 operation metadata.".into(), TypeBound::Eq.into()).unwrap();
    let tket1_op_payload = TypeParam::Opaque{ty:tket1_op_payload_def.instantiate([]).unwrap()};
    res.add_op(
        TKET1_OP_NAME,
        "An opaque TKET1 operation.".into(),
        Tk1Signature([tket1_op_payload])
    ).unwrap();

    res
};

/// Extension registry including the prelude, TKET1 and Tk2Ops extensions.
pub static ref REGISTRY: ExtensionRegistry = ExtensionRegistry::try_new([
    TKET1_EXTENSION.clone(),
    PRELUDE.clone(),
    TKET2_EXTENSION.clone(),
    FLOAT_EXTENSION.clone(),
]).unwrap();


}

struct Tk1Signature([TypeParam; 1]);

impl CustomSignatureFunc for Tk1Signature {
    fn compute_signature<'o, 'a: 'o>(
        &'a self,
        arg_values: &[TypeArg],
        _def: &'o hugr::extension::OpDef,
        _extension_registry: &ExtensionRegistry,
    ) -> Result<PolyFuncType, SignatureError> {
        let [TypeArg::Opaque { arg }] = arg_values else {
            // This should have already been checked.
            panic!("Wrong number of arguments");
        };
        let op: OpaqueTk1Op = serde_yaml::from_value(arg.value.clone()).unwrap(); // TODO Errors!
        Ok(op.signature().into())
    }

    fn static_params(&self) -> &[TypeParam] {
        &self.0
    }
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
