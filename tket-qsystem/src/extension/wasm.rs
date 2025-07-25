//! This module defines the `tket2.wasm` Hugr extension used to model calling
//! into WebAssembly.
//!
//! It depends on the `tket2.futures` extension for handling async calls into
//! WebAssembly.
//!
//! 'tket2.wasm' provides the following types:
//!  - `tket2.wasm.module`: A WebAssembly module.
//!  - `tket2.wasm.context`: A WebAssembly context.
//!  - `tket2.wasm.func`: A WebAssembly function.
//!
//!  Each of which can be constructed in rust via [WasmType].
//!
//!  `tket2.wasm.context` is a linear type that orders runtime effects. It is
//!  obtained via the `tket2.wasm.get_context` operation and destroyed via
//!  `tket2.wasm.dispose_context`.
//!
//!  A `tket2.wasm.module` is obtained by loading a `ConstWasmModule` constant.
//!  We assume that all modules are available in all contexts.
//!
//!  `tket2.wasm.get_context` takes a `prelude.usize`, allowing multiple independent
//!  contexts to exist simultaneously. `get_context` is fallible, returning
//!  `None` if the specified context has already been obtained via an earlier
//!  `get_context` op and not `dispose_context`ed.
//!
//!  `tket2.wasm.func` is a type representing a handle to a function in a
//!  `tket2.wasm.module`. It carries type args defining its signature, but not
//!  its name.
//!
//!  A `tket2.wasm.func` is obtained from a `tket2.wasm.lookup` op, which takes
//!  a compile-time name and signature, and a runtime module. TODO Likely the
//!  module should be compile time here, but I think we need
//!  extension-op-static-edges to do this properly.
//!
//!  `tket2.wasm.func`s are called via the `tket2.wasm.call` op. This op takes:
//!   - a `tket2.wasm.context` identifying where the call will execute;
//!   - a `tket2.wasm.func` identifying the function to call;
//!   - Input arguments as specified by the type of the `tket2.wasm.func`.
//!
//!   It returns a `tket2.futures.future` holding a tuple of results as
//!   specified by the type of the `tket2.wasm.func`.
//!
//!   We provide [WasmType] to assist in constructing and interpreting [Type]s.
//!
//!   We provide [WasmOp] to assist in constructing and interpreting [ExtensionOp]s.
//!
//!   We provide [WasmOpBuilder] to assist in building [hugr::Hugr]s using the
//!   `tket2.wasm` extension.

use std::sync::{Arc, Weak};

use hugr::{
    builder::{BuildError, Dataflow},
    extension::{
        prelude::{option_type, usize_t},
        simple_op::{
            try_from_name, HasConcrete, HasDef, MakeExtensionOp, MakeOpDef, MakeRegisteredOp,
            OpLoadError,
        },
        ExtensionBuildError, ExtensionId, ExtensionRegistry, SignatureError, SignatureFunc,
        TypeDefBound, Version, PRELUDE,
    },
    ops::{
        constant::{downcast_equal_consts, CustomConst, ValueName},
        ExtensionOp, OpName, OpType,
    },
    type_row,
    types::{
        type_param::{TermTypeError, TypeParam},
        CustomType, FuncValueType, PolyFuncTypeRV, Signature, SumType, Type, TypeArg, TypeBound,
        TypeEnum, TypeRV, TypeRow, TypeRowRV,
    },
    Extension, Wire,
};
use itertools::Itertools as _;
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use smol_str::{format_smolstr, SmolStr};
use strum::{EnumIter, EnumString, IntoStaticStr};

use crate::extension::futures;

use super::futures::future_type;

/// The "tket2.wasm" extension id.
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("tket2.wasm");
/// The "tket2.wasm" extension version.
pub const EXTENSION_VERSION: Version = Version::new(0, 2, 0);

lazy_static! {
    /// The `tket2.wasm` extension.
    pub static ref EXTENSION: Arc<Extension> =
        Extension::new_arc(EXTENSION_ID, EXTENSION_VERSION, |ext, ext_ref| {
        add_wasm_type_defs(ext, ext_ref).unwrap();
        WasmOpDef::load_all_ops(ext, ext_ref, ).unwrap();
    });

    /// A [Weak] reference to the `tket2.wasm` op.
    pub static ref EXTENSION_REF: Weak<Extension> = Arc::downgrade(&EXTENSION);

    /// Extension registry including the "tket2.wasm" extension and
    /// dependencies.
    pub static ref REGISTRY: ExtensionRegistry = ExtensionRegistry::new([
        EXTENSION.to_owned(),
        futures::EXTENSION.to_owned(),
        PRELUDE.to_owned()
    ]);

    /// The name of the `tket2.wasm.module` type.
    pub static ref MODULE_TYPE_NAME: SmolStr = SmolStr::new_inline("module");
    /// The name of the `tket2.wasm.context` type.
    pub static ref CONTEXT_TYPE_NAME: SmolStr = SmolStr::new_inline("context");
    /// The name of the `tket2.wasm.func` type.
    pub static ref FUNC_TYPE_NAME: SmolStr = SmolStr::new_inline("func");

    /// The [TypeParam] of `tket2.wasm.lookup` specifying the name of the function.
    pub static ref NAME_PARAM: TypeParam = TypeParam::StringType;
    /// The [TypeParam] of various types and ops specifying the input signature of a function.
    pub static ref INPUTS_PARAM: TypeParam =
        TypeParam::ListType(Box::new(TypeBound::Linear.into()));
    /// The [TypeParam] of various types and ops specifying the output signature of a function.
    pub static ref OUTPUTS_PARAM: TypeParam = TypeParam::ListType(Box::new(TypeBound::Linear.into()));
}

fn add_wasm_type_defs(
    extension: &mut Extension,
    extension_ref: &Weak<Extension>,
) -> Result<(), ExtensionBuildError> {
    extension.add_type(
        MODULE_TYPE_NAME.to_owned(),
        vec![],
        "wasm module".to_owned(),
        TypeDefBound::copyable(),
        extension_ref,
    )?;
    extension.add_type(
        CONTEXT_TYPE_NAME.to_owned(),
        vec![],
        "wasm context".into(),
        TypeDefBound::any(),
        extension_ref,
    )?;
    extension.add_type(
        FUNC_TYPE_NAME.to_owned(),
        vec![INPUTS_PARAM.to_owned(), OUTPUTS_PARAM.to_owned()],
        "wasm func".into(),
        TypeDefBound::copyable(),
        extension_ref,
    )?;
    Ok(())
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
#[allow(missing_docs, non_camel_case_types)]
#[non_exhaustive]
/// Simple enum of ops defined by the `tket2.wasm` extension.
pub enum WasmOpDef {
    get_context,
    dispose_context,
    lookup,
    call,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
/// An enum of types defined by the `tket2.wasm` extension.
///
/// We provide `impl From<WasmType>` for [CustomType] and [Type], and `impl
/// TryFrom<CustomType>` and `impl TryFrom<CustomType>` for [WasmType].
pub enum WasmType {
    /// `tket2.wasm.module`
    Module,
    /// `tket2.wasm.context`
    Context,
    /// `tket2.wasm.func`
    Func {
        /// The input signature of the function. Note that row variables are
        /// allowed.
        inputs: TypeRowRV,
        /// The output signature of the function. Note that row variables are
        /// allowed.
        outputs: TypeRowRV,
    },
}

impl WasmType {
    /// Construct a new `tket2.wasm.func` type.
    pub fn new_func(inputs: impl Into<TypeRowRV>, outputs: impl Into<TypeRowRV>) -> Self {
        Self::Func {
            inputs: inputs.into(),
            outputs: outputs.into(),
        }
    }

    fn get_type(&self, extension_ref: &Weak<Extension>) -> Type {
        self.custom_type(extension_ref).into()
    }

    fn func_custom_type(
        inputs: impl Into<TypeRowRV>,
        outputs: impl Into<TypeRowRV>,
        extension_ref: &Weak<Extension>,
    ) -> CustomType {
        let row_to_arg =
            |row: TypeRowRV| TypeArg::List(row.into_owned().into_iter().map_into().collect());
        CustomType::new(
            FUNC_TYPE_NAME.to_owned(),
            [row_to_arg(inputs.into()), row_to_arg(outputs.into())],
            EXTENSION_ID,
            TypeBound::Copyable,
            extension_ref,
        )
    }

    fn custom_type(&self, extension_ref: &Weak<Extension>) -> CustomType {
        match self {
            Self::Module => CustomType::new(
                MODULE_TYPE_NAME.to_owned(),
                [],
                EXTENSION_ID,
                TypeBound::Copyable,
                extension_ref,
            ),
            Self::Context => CustomType::new(
                CONTEXT_TYPE_NAME.to_owned(),
                [],
                EXTENSION_ID,
                TypeBound::Linear,
                extension_ref,
            ),
            s @ Self::Func { .. } => s.clone().into_custom_type(extension_ref),
        }
    }

    fn into_custom_type(self, extension_ref: &Weak<Extension>) -> CustomType {
        match self {
            Self::Module | Self::Context => self.custom_type(extension_ref),
            Self::Func { inputs, outputs } => {
                Self::func_custom_type(inputs, outputs, extension_ref)
            }
        }
    }

    fn into_type(self, extension_ref: &Weak<Extension>) -> Type {
        self.into_custom_type(extension_ref).into()
    }
}

impl From<WasmType> for CustomType {
    fn from(value: WasmType) -> Self {
        value.into_custom_type(&EXTENSION_REF)
    }
}

impl From<WasmType> for Type {
    fn from(value: WasmType) -> Self {
        value.into_type(&EXTENSION_REF)
    }
}

impl TryFrom<Type> for WasmType {
    type Error = ();

    fn try_from(value: Type) -> Result<Self, Self::Error> {
        let TypeEnum::Extension(custom_type) = value.as_type_enum() else {
            Err(())?
        };

        custom_type.to_owned().try_into().map_err(|_| ())
    }
}

impl TryFrom<CustomType> for WasmType {
    type Error = SignatureError;
    fn try_from(value: CustomType) -> Result<Self, Self::Error> {
        if value.extension() != &EXTENSION_ID {
            Err(SignatureError::ExtensionMismatch(
                EXTENSION_ID,
                value.extension().to_owned(),
            ))?
        }

        match value.name() {
            n if *n == *MODULE_TYPE_NAME => Ok(WasmType::Module),
            n if *n == *CONTEXT_TYPE_NAME => Ok(WasmType::Context),
            n if *n == *FUNC_TYPE_NAME => {
                let [inputs, outputs] = value.args() else {
                    Err(SignatureError::InvalidTypeArgs)?
                };
                let inputs = TypeRowRV::try_from(inputs.clone())?;
                let outputs = TypeRowRV::try_from(outputs.clone())?;

                Ok(WasmType::Func { inputs, outputs })
            }
            n => Err(SignatureError::NameMismatch(
                format_smolstr!(
                    "{}, {} or {}",
                    MODULE_TYPE_NAME.as_str(),
                    CONTEXT_TYPE_NAME.as_str(),
                    FUNC_TYPE_NAME.as_str()
                ),
                n.to_owned(),
            )),
        }
    }
}

impl MakeOpDef for WasmOpDef {
    fn opdef_id(&self) -> hugr::ops::OpName {
        <&'static str>::from(self).into()
    }

    fn init_signature(&self, extension_ref: &Weak<Extension>) -> SignatureFunc {
        let context_type = WasmType::Context.get_type(extension_ref);
        let module_type = WasmType::Module.get_type(extension_ref);
        match self {
            // [usize] -> [Context]
            Self::get_context => Signature::new(
                usize_t(),
                Type::from(WasmOp::get_context_return_type(extension_ref)),
            )
            .into(),
            // [Context] -> []
            Self::dispose_context => Signature::new(context_type, type_row![]).into(),
            // <name: String, inputs: TypeRow, outputs: TypeRow> [Module] -> [WasmType::Func { inputs, outputs }]
            Self::lookup => {
                let inputs = TypeRV::new_row_var_use(1, TypeBound::Copyable);
                let outputs = TypeRV::new_row_var_use(2, TypeBound::Copyable);

                let func_type = WasmType::func_custom_type(inputs, outputs, extension_ref).into();
                PolyFuncTypeRV::new(
                    [
                        NAME_PARAM.to_owned(),
                        INPUTS_PARAM.to_owned(),
                        OUTPUTS_PARAM.to_owned(),
                    ],
                    Signature::new(vec![module_type], vec![func_type]),
                )
                .into()
            }
            // <inputs: TypeRow, outputs: TypeRow> [Context, WasmType::Func { inputs, outputs }, inputs] -> [Context, future<tuple<outputs>>>]
            Self::call => {
                let context_type: TypeRV = context_type.into();
                let inputs = TypeRV::new_row_var_use(0, TypeBound::Copyable);
                let outputs = TypeRV::new_row_var_use(1, TypeBound::Copyable);
                let func_type = Type::new_extension(WasmType::func_custom_type(
                    inputs.clone(),
                    outputs.clone(),
                    extension_ref,
                ));
                let future_type: TypeRV = future_type(Type::new_tuple(outputs)).into();

                PolyFuncTypeRV::new(
                    [INPUTS_PARAM.to_owned(), OUTPUTS_PARAM.to_owned()],
                    FuncValueType::new(
                        vec![context_type.clone(), func_type.into(), inputs],
                        vec![context_type, future_type],
                    ),
                )
                .into()
            }
        }
    }

    fn extension_ref(&self) -> Weak<Extension> {
        EXTENSION_REF.to_owned()
    }

    fn from_def(op_def: &hugr::extension::OpDef) -> Result<Self, OpLoadError>
    where
        Self: Sized,
    {
        try_from_name(op_def.name().as_ref(), op_def.extension_id())
    }

    fn extension(&self) -> ExtensionId {
        EXTENSION_ID
    }
}

impl HasConcrete for WasmOpDef {
    type Concrete = WasmOp;

    fn instantiate(&self, type_args: &[TypeArg]) -> Result<Self::Concrete, OpLoadError> {
        match self {
            WasmOpDef::get_context => {
                let [] = type_args else {
                    Err(SignatureError::from(TermTypeError::WrongNumberArgs(
                        type_args.len(),
                        0,
                    )))?
                };
                Ok(WasmOp::GetContext)
            }
            WasmOpDef::dispose_context => {
                let [] = type_args else {
                    Err(SignatureError::from(TermTypeError::WrongNumberArgs(
                        type_args.len(),
                        0,
                    )))?
                };
                Ok(WasmOp::DisposeContext)
            }
            // <String,in_row,out_row> [] -> []
            WasmOpDef::lookup => {
                let Some([name_arg, inputs_arg, outputs_arg]): Option<[_; 3]> =
                    type_args.to_vec().try_into().ok()
                else {
                    Err(SignatureError::from(TermTypeError::WrongNumberArgs(
                        type_args.len(),
                        3,
                    )))?
                };

                let Some(name) = name_arg.as_string() else {
                    Err(SignatureError::from(TermTypeError::TypeMismatch {
                        term: Box::new(name_arg),
                        type_: Box::new(NAME_PARAM.to_owned()),
                    }))?
                };

                let Ok(inputs) = TypeRowRV::try_from(inputs_arg.clone()) else {
                    Err(SignatureError::from(TermTypeError::TypeMismatch {
                        term: Box::new(inputs_arg),
                        type_: Box::new(INPUTS_PARAM.to_owned()),
                    }))?
                };

                let Ok(outputs) = TypeRowRV::try_from(outputs_arg.clone()) else {
                    Err(SignatureError::from(TermTypeError::TypeMismatch {
                        term: Box::new(outputs_arg),
                        type_: Box::new(OUTPUTS_PARAM.to_owned()),
                    }))?
                };
                Ok(WasmOp::Lookup {
                    name,
                    inputs,
                    outputs,
                })
            }
            WasmOpDef::call => {
                let Some([inputs_arg, outputs_arg]): Option<[_; 2]> =
                    type_args.to_vec().try_into().ok()
                else {
                    Err(SignatureError::from(TermTypeError::WrongNumberArgs(
                        type_args.len(),
                        2,
                    )))?
                };

                let Ok(inputs) = TypeRowRV::try_from(inputs_arg.clone()) else {
                    Err(SignatureError::from(TermTypeError::TypeMismatch {
                        term: Box::new(inputs_arg),
                        type_: Box::new(INPUTS_PARAM.to_owned()),
                    }))?
                };

                let Ok(outputs) = TypeRowRV::try_from(outputs_arg.clone()) else {
                    Err(SignatureError::from(TermTypeError::TypeMismatch {
                        term: Box::new(outputs_arg),
                        type_: Box::new(OUTPUTS_PARAM.to_owned()),
                    }))?
                };

                Ok(WasmOp::Call {
                    inputs: inputs.try_into()?,
                    outputs: outputs.try_into()?,
                })
            }
        }
    }
}

impl TryFrom<&OpType> for WasmOpDef {
    type Error = OpLoadError;

    fn try_from(value: &OpType) -> Result<Self, Self::Error> {
        Self::from_op(
            value
                .as_extension_op()
                .ok_or(OpLoadError::NotMember(value.to_string()))?,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Concrete instantiation(i.e. with type args applied) of a "tket2.wasm" operation.
pub enum WasmOp {
    /// A `tket2.wasm.get_context` op.
    GetContext,
    /// A `tket2.wasm.dispose_context` op.
    DisposeContext,
    /// A `tket2.wasm.lookup` op.
    Lookup {
        /// The name of the function to be looked up.
        name: String,
        /// The input signature of the function to be looked up.
        /// Note that row variables are allowed here.
        inputs: TypeRowRV,
        /// The output signature of the function to be looked up.
        /// Note that row variables are allowed here.
        outputs: TypeRowRV,
    },
    /// A `tket2.wasm.call` op.
    Call {
        /// The input signature of the function to be called
        /// Note that row variables are not allowed here.
        inputs: TypeRow,
        /// The output signature of the function to be called
        /// Note that row variables are not allowed here.
        outputs: TypeRow,
    },
}

impl WasmOp {
    fn wasm_op_def(&self) -> WasmOpDef {
        match self {
            Self::GetContext => WasmOpDef::get_context,
            Self::DisposeContext => WasmOpDef::dispose_context,
            Self::Lookup { .. } => WasmOpDef::lookup,
            Self::Call { .. } => WasmOpDef::call,
        }
    }

    fn get_context_return_type(extension_ref: &Weak<Extension>) -> SumType {
        option_type(Type::new_extension(
            WasmType::Context.into_custom_type(extension_ref),
        ))
    }
}

impl HasDef for WasmOp {
    type Def = WasmOpDef;
}

impl MakeExtensionOp for WasmOp {
    fn op_id(&self) -> OpName {
        self.wasm_op_def().opdef_id()
    }

    fn from_extension_op(ext_op: &ExtensionOp) -> Result<Self, OpLoadError>
    where
        Self: Sized,
    {
        WasmOpDef::from_op(ext_op)?.instantiate(ext_op.args())
    }

    fn type_args(&self) -> Vec<TypeArg> {
        match self {
            WasmOp::GetContext => vec![],
            WasmOp::DisposeContext => vec![],
            WasmOp::Lookup {
                name,
                inputs,
                outputs,
            } => {
                let inputs = TypeArg::from(inputs.clone());
                let outputs = TypeArg::from(outputs.clone());
                vec![name.clone().into(), inputs, outputs]
            }
            WasmOp::Call { inputs, outputs } => {
                let inputs = TypeArg::from(inputs.clone());
                let outputs = TypeArg::from(outputs.clone());
                vec![inputs, outputs]
            }
        }
    }
}

impl MakeRegisteredOp for WasmOp {
    fn extension_id(&self) -> ExtensionId {
        EXTENSION_ID
    }

    fn extension_ref(&self) -> Weak<Extension> {
        Arc::downgrade(&EXTENSION)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
/// A Constant identifying a WebAssembly module.
/// Loading this is the only way to obtain a value of `tket2.wasm.module` type.
pub struct ConstWasmModule {
    /// The name of the module.
    pub name: String,
    /// The hash of the module.
    pub hash: u64,
}

#[typetag::serde]
impl CustomConst for ConstWasmModule {
    fn name(&self) -> ValueName {
        format!("wasm:{}", self.name).into()
    }
    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        downcast_equal_consts(self, other)
    }

    fn get_type(&self) -> Type {
        WasmType::Module.get_type(&EXTENSION_REF)
    }
}

/// An extension trait for [Dataflow] providing methods to add "tket2.wasm"
/// operations and constants.
pub trait WasmOpBuilder: Dataflow {
    /// Add a `tket2.wasm.get_context` op.
    fn add_get_context(&mut self, id: Wire) -> Result<Wire, BuildError> {
        let op = self.add_dataflow_op(WasmOp::GetContext, vec![id])?;
        Ok(op.out_wire(0))
    }

    /// Add a `tket2.wasm.dispose_context` op.
    fn add_dispose_context(&mut self, id: Wire) -> Result<(), BuildError> {
        let _ = self.add_dataflow_op(WasmOp::DisposeContext, vec![id])?;
        Ok(())
    }

    /// Add a `tket2.wasm.lookup` op.
    fn add_lookup(
        &mut self,
        name: impl Into<String>,
        inputs: impl Into<TypeRowRV>,
        outputs: impl Into<TypeRowRV>,
        module: Wire,
    ) -> Result<Wire, BuildError> {
        Ok(self
            .add_dataflow_op(
                WasmOp::Lookup {
                    name: name.into(),
                    inputs: inputs.into(),
                    outputs: outputs.into(),
                },
                [module],
            )?
            .out_wire(0))
    }

    /// Add a `tket2.wasm.call` op.
    ///
    /// We infer the signature from the type of the `func` wire.
    fn add_call(
        &mut self,
        context: Wire,
        func: Wire,
        inputs: impl IntoIterator<Item = Wire>,
    ) -> Result<[Wire; 2], BuildError> {
        let func_wire_type = self.get_wire_type(func)?;
        let Some(WasmType::Func {
            inputs: in_types,
            outputs: out_types,
        }) = func_wire_type.clone().try_into().ok()
        else {
            // TODO Add an Error variant to BuildError for: Input wire has wrong type
            panic!("func wire is not a func type: {func_wire_type}")
        };
        let (in_types, out_types) = (TypeRow::try_from(in_types)?, TypeRow::try_from(out_types)?);

        Ok(self
            .add_dataflow_op(
                WasmOp::Call {
                    inputs: in_types,
                    outputs: out_types,
                },
                [context, func].into_iter().chain(inputs),
            )?
            .outputs_arr())
    }

    /// Add a [ConstWasmModule] and load it.
    fn add_const_module(&mut self, name: impl Into<String>, hash: u64) -> Result<Wire, BuildError> {
        Ok(self.add_load_value(ConstWasmModule {
            name: name.into(),
            hash,
        }))
    }
}

impl<T: Dataflow> WasmOpBuilder for T {}

#[cfg(test)]
mod test {
    use hugr::{
        builder::DFGBuilder, extension::prelude::bool_t, ops::DataflowOpTrait as _, types::Term,
    };
    use rstest::rstest;

    use super::*;

    #[test]
    fn const_wasm_module() {
        let m1 = ConstWasmModule {
            name: "test_mod".to_string(),
            hash: 1,
        };
        let m2 = ConstWasmModule {
            name: "test_mod".to_string(),
            hash: 1,
        };
        assert_eq!(m1.name(), "wasm:test_mod");
        assert!(m1.equal_consts(&m2));
    }

    #[rstest]
    #[case(WasmType::Module)]
    #[case(WasmType::Context)]
    #[case(WasmType::new_func(type_row![], type_row![]))]
    #[case(WasmType::new_func(vec![TypeRV::new_row_var_use(0, TypeBound::Linear)], vec![bool_t()]))]
    fn wasm_type(#[case] wasm_t: WasmType) {
        let hugr_t: Type = wasm_t.clone().into();
        let roundtripped_t = hugr_t.try_into().unwrap();
        assert_eq!(wasm_t, roundtripped_t);
    }

    #[test]
    fn wasm_op_def_instantiate() {
        assert_eq!(
            WasmOpDef::get_context.instantiate(&[]),
            Ok(WasmOp::GetContext)
        );
        assert_eq!(
            WasmOpDef::dispose_context.instantiate(&[]),
            Ok(WasmOp::DisposeContext)
        );
        assert_eq!(
            WasmOpDef::lookup.instantiate(&[
                "lookup_name".into(),
                TypeArg::new_var_use(0, TypeParam::ListType(Box::new(TypeBound::Linear.into()))),
                vec![].into()
            ]),
            Ok(WasmOp::Lookup {
                name: "lookup_name".to_string(),
                inputs: vec![TypeRV::new_row_var_use(0, TypeBound::Linear)].into(),
                outputs: TypeRowRV::from(Vec::<TypeRV>::new())
            })
        );
        assert_eq!(
            WasmOpDef::call.instantiate(&[vec![Type::UNIT.into()].into(), vec![].into()]),
            Ok(WasmOp::Call {
                inputs: vec![Type::UNIT].into(),
                outputs: vec![].into()
            })
        );
    }

    #[rstest]
    #[case::concrete(type_row![], type_row![])]
    #[case::row_vars1(
        vec![
            TypeRV::UNIT,
            TypeRV::try_from(
                TypeArg::new_var_use(
                    0,
                    TypeParam::ListType (Box::new(TypeBound::Copyable.into()))
                )
            ).unwrap()
        ], TypeRowRV::try_from(
            Term::from(vec![TypeArg::from(Type::UNIT), TypeArg::from(usize_t())])
        ).unwrap()
    )]
    fn lookup_signature(
        #[case] inputs: impl Into<TypeRowRV>,
        #[case] outputs: impl Into<TypeRowRV>,
    ) {
        let (inputs, outputs) = (inputs.into(), outputs.into());
        let op = WasmOp::Lookup {
            name: "test".into(),
            inputs: inputs.clone(),
            outputs: outputs.clone(),
        };
        let module_ty = Type::new_extension(WasmType::Module.into_custom_type(&op.extension_ref()));
        let func_ty = Type::new_extension(WasmType::func_custom_type(
            inputs.clone(),
            outputs.clone(),
            &op.extension_ref(),
        ));
        assert_eq!(
            op.to_extension_op().unwrap().signature(),
            Signature::new(module_ty, func_ty)
        );
    }

    #[rstest]
    #[case(type_row![], type_row![])]
    #[case(vec![Type::UNIT], TypeRow::try_from(Term::from(vec![TypeArg::from(Type::UNIT),TypeArg::from(usize_t())])).unwrap())]
    fn build_all(#[case] in_types: impl Into<TypeRow>, #[case] out_types: impl Into<TypeRow>) {
        use futures::FutureOpBuilder as _;
        use hugr::{
            builder::DataflowHugr as _,
            extension::prelude::{ConstUsize, UnpackTuple, UnwrapBuilder as _},
        };

        let (in_types, out_types) = (in_types.into(), out_types.into());
        let _hugr = {
            let mut builder =
                DFGBuilder::new(Signature::new(in_types.clone(), out_types.clone())).unwrap();
            let context_id = builder.add_load_value(ConstUsize::new(0));
            let [context] = {
                let mb_c = builder.add_get_context(context_id).unwrap();
                builder
                    .build_unwrap_sum(1, WasmOp::get_context_return_type(&EXTENSION_REF), mb_c)
                    .unwrap()
            };
            let module = builder.add_const_module("test_module", 0).unwrap();
            let func = builder
                .add_lookup("test_func", in_types, out_types.clone(), module)
                .unwrap();

            let [context, results_future] = builder
                .add_call(context, func, builder.input_wires())
                .unwrap();

            let results = {
                let tuple_ty = Type::new_tuple(out_types.clone());
                let [results_tuple] = builder.add_read(results_future, tuple_ty).unwrap();
                builder
                    .add_dataflow_op(UnpackTuple::new(out_types), [results_tuple])
                    .unwrap()
                    .outputs()
            };

            builder.add_dispose_context(context).unwrap();

            builder.finish_hugr_with_outputs(results).unwrap()
        };
    }
}
