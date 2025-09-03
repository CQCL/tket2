use hugr::{
    extension::{Extension, ExtensionBuildError, ExtensionId, TypeDefBound},
    types::{type_param::TypeParam, CustomType, Type, TypeArg, TypeBound, TypeRow, TypeRowRV},
};
use itertools::Itertools;
use lazy_static::lazy_static;
use smol_str::SmolStr;
use std::marker::PhantomData;
use std::sync::Weak;

lazy_static! {
    /// The name of the `tket.wasm.module` type.
    pub static ref MODULE_TYPE_NAME: SmolStr = SmolStr::new_inline("module");
    /// The name of the `tket.wasm.context` type.
    pub static ref CONTEXT_TYPE_NAME: SmolStr = SmolStr::new_inline("context");
    /// The name of the `tket.wasm.func` type.
    pub static ref FUNC_TYPE_NAME: SmolStr = SmolStr::new_inline("func");

    /// The name of the `tket.wasm.result` type.
    pub static ref RESULT_TYPE_NAME: SmolStr = SmolStr::new_inline("result");

    /// The [TypeParam] of `tket.wasm.lookup_by_id` specifying the id of the function.
    pub static ref ID_PARAM: TypeParam = TypeParam::max_nat_type();
    /// The [TypeParam] of `tket.wasm.lookup_by_name` specifying the name of the function.
    pub static ref NAME_PARAM: TypeParam = TypeParam::StringType;
    /// The [TypeParam] of various types and ops specifying the input signature of a function.
    pub static ref INPUTS_PARAM: TypeParam =
        TypeParam::ListType(Box::new(TypeBound::Linear.into()));
    /// The [TypeParam] of various types and ops specifying the output signature of a function.
    pub static ref OUTPUTS_PARAM: TypeParam = TypeParam::ListType(Box::new(TypeBound::Linear.into()));
}

pub(crate) fn add_compute_type_defs(
    extension: &mut Extension,
    extension_ref: &Weak<Extension>,
) -> Result<(), ExtensionBuildError> {
    extension.add_type(
        MODULE_TYPE_NAME.to_owned(),
        vec![],
        format!("{} module", extension.name),
        TypeDefBound::copyable(),
        extension_ref,
    )?;
    extension.add_type(
        CONTEXT_TYPE_NAME.to_owned(),
        vec![],
        format!("{} context", extension.name),
        TypeDefBound::any(),
        extension_ref,
    )?;
    extension.add_type(
        FUNC_TYPE_NAME.to_owned(),
        vec![INPUTS_PARAM.to_owned(), OUTPUTS_PARAM.to_owned()],
        format!("{} func", extension.name),
        TypeDefBound::copyable(),
        extension_ref,
    )?;
    extension.add_type(
        RESULT_TYPE_NAME.to_owned(),
        vec![OUTPUTS_PARAM.to_owned()],
        format!("{} result", extension.name),
        TypeDefBound::any(),
        extension_ref,
    )?;
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
/// An enum of types defined by compute extensions.
///
/// Instances will provide `impl From<ComputeType<T>>` for [CustomType] and [Type], and `impl
/// TryFrom<CustomType>` and `impl TryFrom<CustomType>` for [ComputeType<T>].
pub enum ComputeType<T> {
    /// `module`
    Module,
    /// `context`
    Context,
    /// `func`
    Func {
        /// The input signature of the function. Note that row variables are
        /// allowed.
        inputs: TypeRowRV,
        /// The output signature of the function. Note that row variables are
        /// allowed.
        outputs: TypeRowRV,
    },
    /// `result`
    Result {
        /// The output signature of the function. Note that row variables are
        /// allowed.
        outputs: TypeRowRV,
    },
    _Unreachable(std::convert::Infallible, PhantomData<T>),
}

impl<T> ComputeType<T> {
    /// Construct a new `tket.wasm.func` type.
    pub fn new_func(inputs: impl Into<TypeRowRV>, outputs: impl Into<TypeRowRV>) -> Self {
        Self::Func {
            inputs: inputs.into(),
            outputs: outputs.into(),
        }
    }

    pub(crate) fn get_type(
        &self,
        extension_id: ExtensionId,
        extension_ref: &Weak<Extension>,
    ) -> Type {
        self.custom_type(extension_id, extension_ref).into()
    }

    pub(crate) fn func_custom_type(
        inputs: impl Into<TypeRowRV>,
        outputs: impl Into<TypeRowRV>,
        extension_id: ExtensionId,
        extension_ref: &Weak<Extension>,
    ) -> CustomType {
        let row_to_arg =
            |row: TypeRowRV| TypeArg::List(row.into_owned().into_iter().map_into().collect());
        CustomType::new(
            FUNC_TYPE_NAME.to_owned(),
            [row_to_arg(inputs.into()), row_to_arg(outputs.into())],
            extension_id,
            TypeBound::Copyable,
            extension_ref,
        )
    }

    pub(crate) fn result_custom_type(
        outputs: impl Into<TypeRowRV>,
        extension_id: ExtensionId,
        extension_ref: &Weak<Extension>,
    ) -> CustomType {
        let row_to_arg =
            |row: TypeRowRV| TypeArg::List(row.into_owned().into_iter().map_into().collect());
        CustomType::new(
            RESULT_TYPE_NAME.to_owned(),
            [row_to_arg(outputs.into())],
            extension_id,
            TypeBound::Linear,
            extension_ref,
        )
    }

    pub(crate) fn custom_type(
        &self,
        extension_id: ExtensionId,
        extension_ref: &Weak<Extension>,
    ) -> CustomType {
        match self {
            Self::Module => CustomType::new(
                MODULE_TYPE_NAME.to_owned(),
                [],
                extension_id,
                TypeBound::Copyable,
                extension_ref,
            ),
            Self::Context => CustomType::new(
                CONTEXT_TYPE_NAME.to_owned(),
                [],
                extension_id,
                TypeBound::Linear,
                extension_ref,
            ),
            Self::Func { inputs, outputs } => {
                Self::func_custom_type(inputs.clone(), outputs.clone(), extension_id, extension_ref)
            }
            Self::Result { outputs } => {
                Self::result_custom_type(outputs.clone(), extension_id, extension_ref)
            }
            Self::_Unreachable(x, _) => match *x {},
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Concrete instantiation(i.e. with type args applied) of op_defs defined in compute extensions
pub enum ComputeOp<T> {
    /// A `get_context` op.
    GetContext,
    /// A `dispose_context` op.
    DisposeContext,
    /// A `lookup_by_id` op.
    LookupById {
        /// The name of the function to be looked up.
        id: u64,
        /// The input signature of the function to be looked up.
        /// Note that row variables are allowed here.
        inputs: TypeRowRV,
        /// The output signature of the function to be looked up.
        /// Note that row variables are allowed here.
        outputs: TypeRowRV,
    },
    /// A `lookup_by_name` op.
    LookupByName {
        /// The name of the function to be looked up.
        name: String,
        /// The input signature of the function to be looked up.
        /// Note that row variables are allowed here.
        inputs: TypeRowRV,
        /// The output signature of the function to be looked up.
        /// Note that row variables are allowed here.
        outputs: TypeRowRV,
    },
    /// A `call` op.
    Call {
        /// The input signature of the function to be called
        /// Note that row variables are not allowed here.
        inputs: TypeRow,
        /// The output signature of the function to be called
        /// Note that row variables are not allowed here.
        outputs: TypeRow,
    },
    /// A `read_result` op.
    ReadResult {
        /// The output signature of the function to be called
        /// Note that row variables are not allowed here.
        outputs: TypeRow,
    },
    _Unreachable(std::convert::Infallible, PhantomData<T>),
}

macro_rules! compute_opdef {
    ($ext_id:expr, $ext:ty) => {
        use serde::{Deserialize, Serialize};
        use strum::{EnumIter, EnumString, IntoStaticStr};

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
        /// Simple enum of ops defined by the `tket.wasm` extension.
        pub enum ComputeOpDef {
            get_context,
            dispose_context,
            lookup_by_id,
            lookup_by_name,
            call,
            read_result,
        }

        impl From<ComputeType<$ext>> for CustomType {
            fn from(value: ComputeType<$ext>) -> Self {
                value.custom_type($ext_id, &EXTENSION_REF)
            }
        }

        impl From<ComputeType<$ext>> for Type {
            fn from(value: ComputeType<$ext>) -> Self {
                value.get_type($ext_id, &EXTENSION_REF)
            }
        }

        impl TryFrom<Type> for ComputeType<$ext> {
            type Error = ();

            fn try_from(value: Type) -> Result<Self, Self::Error> {
                let TypeEnum::Extension(custom_type) = value.as_type_enum() else {
                    Err(())?
                };

                custom_type.to_owned().try_into().map_err(|_| ())
            }
        }

        impl MakeOpDef for ComputeOpDef {
            fn opdef_id(&self) -> hugr::ops::OpName {
                <&'static str>::from(self).into()
            }

            fn init_signature(&self, extension_ref: &Weak<Extension>) -> SignatureFunc {
                let context_type = WasmType::Context.get_type(self.extension(), extension_ref);
                let module_type = WasmType::Module.get_type(self.extension(), extension_ref);
                match self {
                    // [usize] -> [Context]
                    Self::get_context => Signature::new(
                        usize_t(),
                        Type::from(WasmOp::get_context_return_type(
                            self.extension(),
                            extension_ref,
                        )),
                    )
                    .into(),
                    // [Context] -> []
                    Self::dispose_context => Signature::new(context_type, type_row![]).into(),
                    // <id: usize, inputs: TypeRow, outputs: TypeRow> [Module] -> [WasmType::Func { inputs, outputs }]
                    Self::lookup_by_id => {
                        let inputs = TypeRV::new_row_var_use(1, TypeBound::Copyable);
                        let outputs = TypeRV::new_row_var_use(2, TypeBound::Copyable);

                        let func_type = WasmType::func_custom_type(
                            inputs,
                            outputs,
                            self.extension(),
                            extension_ref,
                        )
                        .into();
                        PolyFuncTypeRV::new(
                            [
                                ID_PARAM.to_owned(),
                                INPUTS_PARAM.to_owned(),
                                OUTPUTS_PARAM.to_owned(),
                            ],
                            Signature::new(vec![module_type], vec![func_type]),
                        )
                        .into()
                    }
                    // <name: String, inputs: TypeRow, outputs: TypeRow> [Module] -> [WasmType::Func { inputs, outputs }]
                    Self::lookup_by_name => {
                        let inputs = TypeRV::new_row_var_use(1, TypeBound::Copyable);
                        let outputs = TypeRV::new_row_var_use(2, TypeBound::Copyable);

                        let func_type = WasmType::func_custom_type(
                            inputs,
                            outputs,
                            self.extension(),
                            extension_ref,
                        )
                        .into();
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
                            self.extension(),
                            extension_ref,
                        ));
                        let result_type = TypeRV::new_extension(WasmType::result_custom_type(
                            outputs,
                            self.extension(),
                            extension_ref,
                        ));

                        PolyFuncTypeRV::new(
                            [INPUTS_PARAM.to_owned(), OUTPUTS_PARAM.to_owned()],
                            FuncValueType::new(
                                vec![context_type.clone(), func_type.into(), inputs],
                                vec![result_type],
                            ),
                        )
                        .into()
                    }
                    Self::read_result => {
                        let context_type: TypeRV = context_type.into();
                        let outputs = TypeRV::new_row_var_use(0, TypeBound::Copyable);
                        let result_type = TypeRV::new_extension(WasmType::result_custom_type(
                            outputs.clone(),
                            self.extension(),
                            extension_ref,
                        ));
                        PolyFuncTypeRV::new(
                            [OUTPUTS_PARAM.to_owned()],
                            FuncValueType::new(vec![result_type], vec![context_type, outputs]),
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

        impl HasConcrete for ComputeOpDef {
            type Concrete = ComputeOp<$ext>;

            fn instantiate(&self, type_args: &[TypeArg]) -> Result<Self::Concrete, OpLoadError> {
                match self {
                    Self::get_context => {
                        let [] = type_args else {
                            Err(SignatureError::from(TermTypeError::WrongNumberArgs(
                                type_args.len(),
                                0,
                            )))?
                        };
                        Ok(WasmOp::GetContext)
                    }
                    Self::dispose_context => {
                        let [] = type_args else {
                            Err(SignatureError::from(TermTypeError::WrongNumberArgs(
                                type_args.len(),
                                0,
                            )))?
                        };
                        Ok(WasmOp::DisposeContext)
                    }
                    // <usize,in_row,out_row> [] -> []
                    Self::lookup_by_id => {
                        let Some([id_arg, inputs_arg, outputs_arg]): Option<[_; 3]> =
                            type_args.to_vec().try_into().ok()
                        else {
                            Err(SignatureError::from(TermTypeError::WrongNumberArgs(
                                type_args.len(),
                                3,
                            )))?
                        };

                        let Some(id) = id_arg.as_nat() else {
                            Err(SignatureError::from(TermTypeError::TypeMismatch {
                                term: Box::new(id_arg),
                                type_: Box::new(ID_PARAM.to_owned()),
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
                        Ok(WasmOp::LookupById {
                            id,
                            inputs,
                            outputs,
                        })
                    }
                    // <String,in_row,out_row> [] -> []
                    Self::lookup_by_name => {
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
                        Ok(WasmOp::LookupByName {
                            name,
                            inputs,
                            outputs,
                        })
                    }
                    Self::call => {
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
                    Self::read_result => {
                        let Some([outputs_arg]): Option<[_; 1]> =
                            type_args.to_vec().try_into().ok()
                        else {
                            Err(SignatureError::from(TermTypeError::WrongNumberArgs(
                                type_args.len(),
                                1,
                            )))?
                        };

                        let Ok(outputs) = TypeRowRV::try_from(outputs_arg.clone()) else {
                            Err(SignatureError::from(TermTypeError::TypeMismatch {
                                term: Box::new(outputs_arg),
                                type_: Box::new(OUTPUTS_PARAM.to_owned()),
                            }))?
                        };
                        Ok(WasmOp::ReadResult {
                            outputs: outputs.try_into()?,
                        })
                    }
                }
            }
        }
        impl TryFrom<&OpType> for ComputeOpDef {
            type Error = OpLoadError;

            fn try_from(value: &OpType) -> Result<Self, Self::Error> {
                Self::from_op(
                    value
                        .as_extension_op()
                        .ok_or(OpLoadError::NotMember(value.to_string()))?,
                )
            }
        }

        impl ComputeOp<$ext> {
            fn compute_op_def(&self) -> ComputeOpDef {
                match self {
                    Self::GetContext => ComputeOpDef::get_context,
                    Self::DisposeContext => ComputeOpDef::dispose_context,
                    Self::LookupById { .. } => ComputeOpDef::lookup_by_id,
                    Self::LookupByName { .. } => ComputeOpDef::lookup_by_name,
                    Self::Call { .. } => ComputeOpDef::call,
                    Self::ReadResult { .. } => ComputeOpDef::read_result,
                    Self::_Unreachable(x, _) => match *x {},
                }
            }

            fn get_context_return_type(
                extension_id: ExtensionId,
                extension_ref: &Weak<Extension>,
            ) -> SumType {
                option_type(WasmType::Context.get_type(extension_id, extension_ref))
            }
        }

        impl HasDef for ComputeOp<$ext> {
            type Def = ComputeOpDef;
        }

        impl MakeExtensionOp for ComputeOp<$ext> {
            fn op_id(&self) -> OpName {
                self.compute_op_def().opdef_id()
            }

            fn from_extension_op(ext_op: &ExtensionOp) -> Result<Self, OpLoadError>
            where
                Self: Sized,
            {
                ComputeOpDef::from_op(ext_op)?.instantiate(ext_op.args())
            }

            fn type_args(&self) -> Vec<TypeArg> {
                match self {
                    WasmOp::GetContext => vec![],
                    WasmOp::DisposeContext => vec![],
                    WasmOp::LookupById {
                        id,
                        inputs,
                        outputs,
                    } => {
                        let inputs = TypeArg::from(inputs.clone());
                        let outputs = TypeArg::from(outputs.clone());
                        vec![TypeArg::BoundedNat(*id), inputs, outputs]
                    }
                    WasmOp::LookupByName {
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
                    WasmOp::ReadResult { outputs } => vec![outputs.clone().into()],
                    WasmOp::_Unreachable(x, _) => match *x {},
                }
            }
        }
    };
}

pub mod wasm;
