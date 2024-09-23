//! Opaque operations encoding sympy expressions.
//!
//! Part of the TKET2 extension.

use std::str::FromStr;

use hugr::extension::simple_op::{
    try_from_name, HasConcrete, HasDef, MakeExtensionOp, MakeOpDef, MakeRegisteredOp, OpLoadError,
};
use hugr::extension::{ExtensionId, ExtensionRegistry, SignatureError};
use hugr::ops::{ExtensionOp, NamedOp, OpName};
use hugr::type_row;
use hugr::types::type_param::TypeParam;
use hugr::types::{CustomType, PolyFuncType, Signature, TypeArg};
use lazy_static::lazy_static;
use smol_str::SmolStr;

use crate::extension::TKET2_EXTENSION;

use super::rotation::ROTATION_TYPE;
use super::{REGISTRY, TKET2_EXTENSION_ID};

/// The name of the symbolic expression opaque type arg.
pub const SYM_EXPR_NAME: SmolStr = SmolStr::new_inline("SymExpr");

/// The name of the symbolic expression opaque type arg.
pub const SYM_OP_ID: SmolStr = SmolStr::new_inline("symbolic_angle");

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
/// An operation hardcoding a Sympy expression in its parameter.
///
/// Returns the expression as an angle.
pub struct SympyOpDef;

impl SympyOpDef {
    /// Create a new concrete sympy definition using the given sympy expression.
    pub fn with_expr(self, expr: String) -> SympyOp {
        SympyOp { expr }
    }
}

impl NamedOp for SympyOpDef {
    fn name(&self) -> hugr::ops::OpName {
        SYM_OP_ID.to_owned()
    }
}

impl FromStr for SympyOpDef {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == SYM_OP_ID {
            Ok(Self)
        } else {
            Err(())
        }
    }
}

impl MakeOpDef for SympyOpDef {
    fn from_def(
        op_def: &hugr::extension::OpDef,
    ) -> Result<Self, hugr::extension::simple_op::OpLoadError>
    where
        Self: Sized,
    {
        try_from_name(op_def.name(), op_def.extension())
    }

    fn signature(&self) -> hugr::extension::SignatureFunc {
        PolyFuncType::new(
            vec![TypeParam::String],
            Signature::new(type_row![], type_row![ROTATION_TYPE]),
        )
        .into()
    }

    fn description(&self) -> String {
        "Store a sympy expression that can be evaluated to an angle.".to_string()
    }

    fn extension(&self) -> hugr::extension::ExtensionId {
        TKET2_EXTENSION_ID
    }
}

/// A concrete operation hardcoding a Sympy expression in its parameter.
///
/// Returns the expression as an angle.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct SympyOp {
    /// The expression to evaluate.
    pub expr: String,
}

impl NamedOp for SympyOp {
    fn name(&self) -> OpName {
        SYM_OP_ID.to_owned()
    }
}

impl MakeExtensionOp for SympyOp {
    fn from_extension_op(ext_op: &ExtensionOp) -> Result<Self, OpLoadError> {
        let def = SympyOpDef::from_def(ext_op.def())?;
        def.instantiate(ext_op.args())
    }

    fn type_args(&self) -> Vec<TypeArg> {
        vec![self.expr.clone().into()]
    }
}

impl MakeRegisteredOp for SympyOp {
    fn extension_id(&self) -> ExtensionId {
        TKET2_EXTENSION_ID.to_owned()
    }

    fn registry<'s, 'r: 's>(&'s self) -> &'r ExtensionRegistry {
        &REGISTRY
    }
}

impl HasConcrete for SympyOpDef {
    type Concrete = SympyOp;

    fn instantiate(&self, type_args: &[TypeArg]) -> Result<Self::Concrete, OpLoadError> {
        let ty = match type_args {
            [TypeArg::String { arg }] => arg.clone(),
            _ => return Err(SignatureError::InvalidTypeArgs.into()),
        };

        Ok(self.with_expr(ty))
    }
}

impl HasDef for SympyOp {
    type Def = SympyOpDef;
}

lazy_static! {

/// The type of the symbolic expression opaque type arg.
pub static ref SYM_EXPR_T: CustomType =
    TKET2_EXTENSION.get_type(&SYM_EXPR_NAME).unwrap().instantiate([]).unwrap();

}

#[cfg(test)]
mod tests {
    use hugr::extension::simple_op::MakeOpDef;
    use hugr::ops::NamedOp;

    use super::*;
    use crate::extension::TKET2_EXTENSION;

    #[test]
    fn test_extension() {
        assert_eq!(TKET2_EXTENSION.name(), &SympyOpDef.extension());

        let opdef = TKET2_EXTENSION.get_op(&SympyOpDef.name());
        assert_eq!(SympyOpDef::from_def(opdef.unwrap()), Ok(SympyOpDef));
    }

    #[test]
    fn test_op() {
        let op = SympyOp {
            expr: "cos(pi/2)".to_string(),
        };

        let op_t: ExtensionOp = op.clone().to_extension_op().unwrap();
        assert!(SympyOpDef::from_op(&op_t).is_ok());

        let new_op = SympyOp::from_op(&op_t).unwrap();
        assert_eq!(new_op, op);
    }
}
