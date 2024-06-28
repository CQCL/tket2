//! TODO docs
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
use smol_str::SmolStr;
use strum_macros::{EnumIter, EnumString, IntoStaticStr};

/// TODO docs
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("tket2.lazy");

lazy_static! {
    /// The "tket2.lazy" extension
    pub static ref EXTENSION: Extension = {
        let mut ext = Extension::new(EXTENSION_ID);
        let _ = add_lazy_type_def(&mut ext).unwrap();

        LazyOp::load_all_ops(&mut ext).unwrap();
        ext
    };

    /// TODO docs
    pub static ref REGISTRY: ExtensionRegistry = ExtensionRegistry::try_new([
        EXTENSION.to_owned()
    ]).unwrap();

    /// TODO docs
    pub static ref LAZY_TYPE_NAME: SmolStr = SmolStr::new_inline("Lazy");
}

/// TODO docs
pub fn add_lazy_type_def(ext: &mut Extension) -> Result<&TypeDef, ExtensionBuildError> {
    ext.add_type(
        LAZY_TYPE_NAME.to_owned(),
        vec![TypeBound::Any.into()],
        "A value that is only computed when needed".into(),
        TypeBound::Any.into(),
    )
}

/// TODO docs
pub fn lazy_custom_type(t: Type) -> CustomType {
    CustomType::new(
        LAZY_TYPE_NAME.to_owned(),
        vec![t.into()],
        EXTENSION_ID,
        TypeBound::Any,
    )
}

/// TODO docs
pub fn lazy_type(t: Type) -> Type {
    lazy_custom_type(t).into()
}

#[derive(
    Clone,
    Copy,
    Debug,
    serde::Serialize,
    serde::Deserialize,
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
enum LazyOp {
    Lift,
    Read,
    Dup,
    Free,
}

// impl LazyOp {
//     /// Expose the operation names directly in Tk2Op
//     pub fn exposed_name(&self) -> smol_str::SmolStr {
//         <LazyOp as Into<OpType>>::into(*self).name()
//     }

//     /// Wraps the operation in an [`ExtensionOp`]
//     pub fn into_extension_op(self) -> ExtensionOp {
//         <Self as MakeRegisteredOp>::to_extension_op(self)
//             .expect("Failed to convert to extension op.")
//     }
// }

impl MakeOpDef for LazyOp {
    fn signature(&self) -> SignatureFunc {
        let t_param = TypeParam::from(TypeBound::Any);
        let t_type = Type::new_var_use(0, TypeBound::Any);
        let lazy_type = lazy_type(t_type.clone());
        match self {
            LazyOp::Lift => {
                PolyFuncType::new([t_param], FunctionType::new(t_type, lazy_type)).into()
            }
            LazyOp::Read => {
                PolyFuncType::new([t_param], FunctionType::new(lazy_type, t_type)).into()
            }
            LazyOp::Dup => PolyFuncType::new(
                [t_param],
                FunctionType::new(lazy_type.clone(), vec![lazy_type.clone(), lazy_type]),
            )
            .into(),
            LazyOp::Free => {
                PolyFuncType::new([t_param], FunctionType::new(lazy_type.clone(), vec![])).into()
            }
        }
    }

    fn from_def(op_def: &OpDef) -> Result<Self, hugr::extension::simple_op::OpLoadError> {
        try_from_name(op_def.name())
    }
}

struct ConcreteLazyOp {
    op: LazyOp,
    typ: Type,
}

impl<'a> From<&'a ConcreteLazyOp> for &'static str {
    fn from(value: &ConcreteLazyOp) -> Self {
        value.op.into()
    }
}

fn concrete_lazy_op_type_args(
    args: &[TypeArg],
) -> Result<Type, hugr::extension::simple_op::OpLoadError> {
    match args {
        [TypeArg::Type { ty }] => Ok(ty.clone()),
        _ => Err(OpLoadError::InvalidArgs(
            hugr::extension::SignatureError::InvalidTypeArgs,
        )),
    }
}

impl MakeExtensionOp for ConcreteLazyOp {
    fn from_extension_op(
        ext_op: &ExtensionOp,
    ) -> Result<Self, hugr::extension::simple_op::OpLoadError>
    where
        Self: Sized,
    {
        let op = LazyOp::from_def(ext_op.def())?;
        let typ = concrete_lazy_op_type_args(ext_op.args())?;
        Ok(Self { op, typ })
    }

    fn type_args(&self) -> Vec<hugr::types::TypeArg> {
        vec![self.typ.clone().into()]
    }
}

impl MakeRegisteredOp for ConcreteLazyOp {
    fn extension_id(&self) -> ExtensionId {
        EXTENSION_ID
    }

    fn registry<'s, 'r: 's>(&'s self) -> &'r ExtensionRegistry {
        &REGISTRY
    }
}

impl TryFrom<&OpType> for ConcreteLazyOp {
    type Error = ();

    fn try_from(value: &OpType) -> Result<Self, Self::Error> {
        let Some(custom_op) = value.as_custom_op() else {
            Err(())?
        };
        match custom_op {
            CustomOp::Extension(ext) => ConcreteLazyOp::from_extension_op(ext).ok(),
            CustomOp::Opaque(opaque) if opaque.extension() == &EXTENSION_ID => (|| {
                let op = try_from_name(opaque.name()).ok()?;
                let typ = concrete_lazy_op_type_args(opaque.args()).ok()?;
                Some(Self { op, typ })
            })(),
            _ => None,
        }
        .ok_or(())
    }
}

/// TODO docs
pub trait LazyOpBuilder: Dataflow {
    /// TODO docs
    fn add_lift(&mut self, unlifted: Wire, typ: Type) -> Result<[Wire; 1], BuildError> {
        Ok(self
            .add_dataflow_op(
                ConcreteLazyOp {
                    op: LazyOp::Lift,
                    typ,
                },
                [unlifted],
            )?
            .outputs_arr())
    }

    /// TODO docs
    fn add_read(&mut self, lifted: Wire, typ: Type) -> Result<[Wire; 1], BuildError> {
        Ok(self
            .add_dataflow_op(
                ConcreteLazyOp {
                    op: LazyOp::Read,
                    typ,
                },
                [lifted],
            )?
            .outputs_arr())
    }

    /// TODO docs
    fn add_dup(&mut self, lifted: Wire, typ: Type) -> Result<[Wire; 2], BuildError> {
        Ok(self
            .add_dataflow_op(
                ConcreteLazyOp {
                    op: LazyOp::Dup,
                    typ,
                },
                [lifted],
            )?
            .outputs_arr())
    }

    /// TODO docs
    fn add_free(&mut self, lifted: Wire, typ: Type) -> Result<(), BuildError> {
        let op = self.add_dataflow_op(
            ConcreteLazyOp {
                op: LazyOp::Free,
                typ,
            },
            [lifted],
        )?;
        assert!(op.outputs().len() == 0);
        Ok(())
    }
}

impl<D: Dataflow> LazyOpBuilder for D {}

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

        for o in LazyOp::iter() {
            assert_eq!(LazyOp::from_def(get_opdef(o).unwrap()), Ok(o));
        }
    }

    #[test]
    fn circuit() {
        let t_param = TypeParam::from(TypeBound::Any);
        let t = Type::new_var_use(0, TypeBound::Any);
        let lazy_type = lazy_type(t.clone());

        let hugr = {
            let mut func_builder = FunctionBuilder::new(
                "circuit",
                PolyFuncType::new(
                    vec![t_param],
                    FunctionType::new_endo(vec![t.clone(), lazy_type]),
                ),
            )
            .unwrap();
            let [t_w, lazy_w] = func_builder.input_wires_arr();
            func_builder.add_free(lazy_w, t.clone()).unwrap();
            let [lazy_w] = func_builder.add_lift(t_w, t.clone()).unwrap();
            let [lazy_w, lazy_dup_w] = func_builder.add_dup(lazy_w, t.clone()).unwrap();
            let [t_w] = func_builder.add_read(lazy_dup_w, t).unwrap();
            func_builder
                .finish_hugr_with_outputs([t_w, lazy_w], &REGISTRY)
                .unwrap()
        };
        assert_matches!(hugr.validate(&REGISTRY), Ok(_));
    }
}
