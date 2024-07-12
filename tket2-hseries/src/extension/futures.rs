//! This module defines the Hugr extension used to represent futures.
//!
//! `Future<t>` is a linear type representing a value that will be available in
//! the future.  It can be consumed by `Read`, returning a `t`.  It can be
//! duplicated by `Dup`, and discarded with `Free`.
use hugr::{
    builder::{BuildError, Dataflow},
    extension::{
        simple_op::{try_from_name, MakeExtensionOp, MakeOpDef, MakeRegisteredOp, OpLoadError},
        ExtensionBuildError, ExtensionId, ExtensionRegistry, OpDef, SignatureFunc, TypeDef,
    },
    ops::{custom::ExtensionOp, CustomOp, OpType},
    types::{
        type_param::TypeParam, CustomType, FunctionType, PolyFuncType, Type, TypeArg, TypeBound,
    },
    Extension, Wire,
};
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use smol_str::SmolStr;
use strum_macros::{EnumIter, EnumString, IntoStaticStr};

/// The ID of the `tket2.futures` extension.
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("tket2.futures");

lazy_static! {
    /// The "tket2.futures" extension.
    pub static ref EXTENSION: Extension = {
        let mut ext = Extension::new(EXTENSION_ID);
        let _ = add_future_type_def(&mut ext).unwrap();

        FutureOp::load_all_ops(&mut ext).unwrap();
        ext
    };

    /// Extension registry including the "tket2.futures" extension.
    pub static ref REGISTRY: ExtensionRegistry = ExtensionRegistry::try_new([
        EXTENSION.to_owned()
    ]).unwrap();

    /// The name of the `Future` type.
    pub static ref FUTURE_TYPE_NAME: SmolStr = SmolStr::new_inline("Future");
}

fn add_future_type_def(ext: &mut Extension) -> Result<&TypeDef, ExtensionBuildError> {
    ext.add_type(
        FUTURE_TYPE_NAME.to_owned(),
        vec![TypeBound::Any.into()],
        "A value that is computed asynchronously".into(),
        TypeBound::Any.into(),
    )
}

/// Returns a `Future<t>` [CustomType].
pub fn future_custom_type(t: Type) -> CustomType {
    CustomType::new(
        FUTURE_TYPE_NAME.to_owned(),
        vec![t.into()],
        EXTENSION_ID,
        TypeBound::Any,
    )
}

/// Returns a `Future<t>` [Type].
pub fn future_type(t: Type) -> Type {
    future_custom_type(t).into()
}

#[derive(
    Clone,
    Copy,
    Debug,
    Serialize,
    Deserialize,
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
/// Simple enum of "tket2.futures" operations.
pub enum FutureOp {
    Read,
    Dup,
    Free,
}

impl MakeOpDef for FutureOp {
    fn signature(&self) -> SignatureFunc {
        let t_param = TypeParam::from(TypeBound::Any);
        let t_type = Type::new_var_use(0, TypeBound::Any);
        let future_type = future_type(t_type.clone());
        match self {
            FutureOp::Read => {
                PolyFuncType::new([t_param], FunctionType::new(future_type, t_type)).into()
            }
            FutureOp::Dup => PolyFuncType::new(
                [t_param],
                FunctionType::new(future_type.clone(), vec![future_type.clone(), future_type]),
            )
            .into(),
            FutureOp::Free => {
                PolyFuncType::new([t_param], FunctionType::new(future_type.clone(), vec![])).into()
            }
        }
    }

    fn extension(&self) -> ExtensionId {
        EXTENSION_ID
    }

    fn from_def(op_def: &OpDef) -> Result<Self, hugr::extension::simple_op::OpLoadError> {
        try_from_name(op_def.name(), &EXTENSION_ID)
    }

    fn description(&self) -> String {
        match self {
            FutureOp::Read => "Read a value from a Future, consuming it".into(),
            FutureOp::Dup => {
                "Duplicate a Future. The original Future is consumed and two Futures are returned"
                    .into()
            }
            FutureOp::Free => "Consume a future without reading it.".into(),
        }
    }
}

impl<'a> From<&'a ConcreteFutureOp> for &'static str {
    fn from(value: &ConcreteFutureOp) -> Self {
        value.op.into()
    }
}

/// Concrete "tket2.futures" operations with type set.
struct ConcreteFutureOp {
    op: FutureOp,
    typ: Type,
}

fn concrete_future_op_type_args(
    args: &[TypeArg],
) -> Result<Type, hugr::extension::simple_op::OpLoadError> {
    match args {
        [TypeArg::Type { ty }] => Ok(ty.clone()),
        _ => Err(OpLoadError::InvalidArgs(
            hugr::extension::SignatureError::InvalidTypeArgs,
        )),
    }
}

impl MakeExtensionOp for ConcreteFutureOp {
    fn from_extension_op(
        ext_op: &ExtensionOp,
    ) -> Result<Self, hugr::extension::simple_op::OpLoadError>
    where
        Self: Sized,
    {
        let op = FutureOp::from_def(ext_op.def())?;
        let typ = concrete_future_op_type_args(ext_op.args())?;
        Ok(Self { op, typ })
    }

    fn type_args(&self) -> Vec<hugr::types::TypeArg> {
        vec![self.typ.clone().into()]
    }
}

impl MakeRegisteredOp for ConcreteFutureOp {
    fn extension_id(&self) -> ExtensionId {
        EXTENSION_ID
    }

    fn registry<'s, 'r: 's>(&'s self) -> &'r ExtensionRegistry {
        &REGISTRY
    }
}

impl TryFrom<&OpType> for FutureOp {
    type Error = ();

    fn try_from(value: &OpType) -> Result<Self, Self::Error> {
        let Some(custom_op) = value.as_custom_op() else {
            Err(())?
        };
        match custom_op {
            CustomOp::Extension(ext) => Self::from_extension_op(ext).ok(),
            CustomOp::Opaque(opaque) => try_from_name(opaque.name(), &EXTENSION_ID).ok(),
        }
        .ok_or(())
    }
}

impl TryFrom<&OpType> for ConcreteFutureOp {
    type Error = ();

    fn try_from(value: &OpType) -> Result<Self, Self::Error> {
        (|| {
            let op = value.try_into().ok()?;
            let typ = concrete_future_op_type_args(value.as_custom_op()?.args()).ok()?;
            Some(Self { op, typ })
        })()
        .ok_or(())
    }
}

/// An extension trait for [Dataflow] providing methods to add "tket2.futures"
/// operations.
pub trait FutureOpBuilder: Dataflow {
    /// Add a "tket2.futures.Read" op.
    fn add_read(&mut self, lifted: Wire, typ: Type) -> Result<[Wire; 1], BuildError> {
        Ok(self
            .add_dataflow_op(
                ConcreteFutureOp {
                    op: FutureOp::Read,
                    typ,
                },
                [lifted],
            )?
            .outputs_arr())
    }

    /// Add a "tket2.futures.Dup" op.
    fn add_dup(&mut self, lifted: Wire, typ: Type) -> Result<[Wire; 2], BuildError> {
        Ok(self
            .add_dataflow_op(
                ConcreteFutureOp {
                    op: FutureOp::Dup,
                    typ,
                },
                [lifted],
            )?
            .outputs_arr())
    }

    /// Add a "tket2.futures.Free" op.
    fn add_free(&mut self, lifted: Wire, typ: Type) -> Result<(), BuildError> {
        let op = self.add_dataflow_op(
            ConcreteFutureOp {
                op: FutureOp::Free,
                typ,
            },
            [lifted],
        )?;
        assert!(op.outputs().len() == 0);
        Ok(())
    }
}

impl<D: Dataflow> FutureOpBuilder for D {}

#[cfg(test)]
pub(crate) mod test {
    use cool_asserts::assert_matches;
    use hugr::{
        builder::{Dataflow, DataflowHugr, FunctionBuilder},
        ops::NamedOp,
    };
    use std::sync::Arc;
    use strum::IntoEnumIterator;

    use super::*;

    fn get_opdef(op: impl NamedOp) -> Option<&'static Arc<OpDef>> {
        EXTENSION.get_op(&op.name())
    }

    #[test]
    fn create_extension() {
        assert_eq!(EXTENSION.name(), &EXTENSION_ID);

        for o in FutureOp::iter() {
            assert_eq!(FutureOp::from_def(get_opdef(o).unwrap()), Ok(o));
        }
    }

    #[test]
    fn circuit() {
        let t_param = TypeParam::from(TypeBound::Any);
        let t = Type::new_var_use(0, TypeBound::Any);
        let future_type = future_type(t.clone());

        let hugr = {
            let mut func_builder = FunctionBuilder::new(
                "circuit",
                PolyFuncType::new(vec![t_param], FunctionType::new(future_type, t.clone())),
            )
            .unwrap();
            let [future_w] = func_builder.input_wires_arr();
            let [future_w, lazy_dup_w] = func_builder.add_dup(future_w, t.clone()).unwrap();
            func_builder.add_free(future_w, t.clone()).unwrap();
            let [t_w] = func_builder.add_read(lazy_dup_w, t).unwrap();
            func_builder
                .finish_hugr_with_outputs([t_w], &REGISTRY)
                .unwrap()
        };
        assert_matches!(hugr.validate(&REGISTRY), Ok(_));
    }
}
