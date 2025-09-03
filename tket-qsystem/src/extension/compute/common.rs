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
