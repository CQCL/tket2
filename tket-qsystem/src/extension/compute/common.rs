use hugr::types::{TypeRow, TypeRowRV};
use std::marker::PhantomData;

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
