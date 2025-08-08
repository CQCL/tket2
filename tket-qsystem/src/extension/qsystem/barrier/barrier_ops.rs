use std::ops::Deref;
use std::sync::Arc;

use hugr::{
    algorithms::mangle_name,
    builder::{BuildError, DFGBuilder, Dataflow, DataflowHugr, FunctionBuilder},
    extension::{
        prelude::{option_type, qb_t, UnpackTuple, UnwrapBuilder},
        Extension,
    },
    ops::{DataflowOpTrait, ExtensionOp, OpName},
    std_extensions::collections::{
        array::{array_type, op_builder::GenericArrayOpBuilder, Array, ArrayKind},
        borrow_array::BorrowArray,
        value_array::ValueArray,
    },
    types::{
        type_param::TypeParam, FuncValueType, PolyFuncTypeRV, Signature, SumType, Type, TypeArg,
        TypeBound, TypeRV,
    },
    Hugr, Wire,
};
use indexmap::IndexMap;

use crate::extension::qsystem::{barrier::qtype_analyzer::QTypeAnalyzer, QSystemOpBuilder};

use super::qtype_analyzer::{array_args, is_opt_qb};

/// Wrapper for ExtensionOp that implements Hash
#[derive(Clone, PartialEq, Eq)]
pub(super) struct OpHashWrapper(ExtensionOp);

impl From<ExtensionOp> for OpHashWrapper {
    fn from(op: ExtensionOp) -> Self {
        Self(op)
    }
}

impl OpHashWrapper {
    pub(super) fn extension_op(&self) -> &ExtensionOp {
        &self.0
    }
}

impl std::hash::Hash for OpHashWrapper {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.extension_id().hash(state);
        self.0.unqualified_id().hash(state);
        self.0.args().hash(state);
    }
}

/// Invert the signature of a function type.
fn invert_sig(sig: &PolyFuncTypeRV) -> PolyFuncTypeRV {
    let body = FuncValueType::new(sig.body().output().clone(), sig.body().input().clone());
    PolyFuncTypeRV::new(sig.params(), body)
}

/// Create and cache operations needed for barrier insertion
pub struct BarrierOperationFactory {
    /// Temporary extension used for placeholder operations.
    extension: Arc<Extension>,
    /// Function definitions for each instance of the operations.
    pub(super) funcs: IndexMap<OpHashWrapper, Hugr>,
    /// Type analyzer for determining qubit types
    type_analyzer: QTypeAnalyzer,
}

fn generic_array_unpack_sig<AK: ArrayKind>() -> PolyFuncTypeRV {
    PolyFuncTypeRV::new(
        vec![
            TypeParam::max_nat_type(),
            TypeParam::RuntimeType(TypeBound::Linear),
            TypeParam::new_list_type(TypeBound::Linear),
        ],
        FuncValueType::new(
            AK::ty_parametric(
                TypeArg::new_var_use(0, TypeParam::max_nat_type()),
                Type::new_var_use(1, TypeBound::Linear),
            )
            .unwrap(),
            TypeRV::new_row_var_use(2, TypeBound::Linear),
        ),
    )
}

/// Helper function to add array operations for any ArrayKind to the extension
fn add_array_ops<AK: ArrayKind>(
    ext: &mut Extension,
    ext_ref: &std::sync::Weak<Extension>,
    unpack_name: OpName,
    repack_name: OpName,
) -> Result<(), hugr::extension::ExtensionBuildError> {
    let array_unpack_sig = generic_array_unpack_sig::<AK>();
    // pack some wires into an array
    ext.add_op(
        repack_name,
        Default::default(),
        invert_sig(&array_unpack_sig),
        ext_ref,
    )?;
    // unpack an array into some wires
    ext.add_op(unpack_name, Default::default(), array_unpack_sig, ext_ref)?;
    Ok(())
}
impl BarrierOperationFactory {
    /// Temporary extension name.
    pub(super) const TEMP_EXT_NAME: hugr::hugr::IdentList =
        hugr::hugr::IdentList::new_static_unchecked("__tket.barrier.temp");
    // Temporary operation names.
    pub(super) const UNPACK_OPT: OpName = OpName::new_static("option_qb_unwrap");
    pub(super) const REPACK_OPT: OpName = OpName::new_static("option_qb_tag");
    pub(super) const WRAPPED_BARRIER: OpName = OpName::new_static("wrapped_barrier");
    pub(super) const ARRAY_UNPACK: OpName = OpName::new_static("array_unpack");
    pub(super) const ARRAY_REPACK: OpName = OpName::new_static("array_repack");
    pub(super) const VARRAY_UNPACK: OpName = OpName::new_static("varray_unpack");
    pub(super) const VARRAY_REPACK: OpName = OpName::new_static("varray_repack");
    pub(super) const BARRAY_UNPACK: OpName = OpName::new_static("barray_unpack");
    pub(super) const BARRAY_REPACK: OpName = OpName::new_static("barray_repack");
    pub(super) const TUPLE_UNPACK: OpName = OpName::new_static("tuple_unpack");
    pub(super) const TUPLE_REPACK: OpName = OpName::new_static("tuple_repack");

    /// Create a new instance of the BarrierOperationFactory.
    pub fn new() -> Self {
        Self {
            extension: Self::build_extension(),
            funcs: IndexMap::new(),
            type_analyzer: QTypeAnalyzer::new(),
        }
    }

    /// Gets a reference to the internal type analyzer
    pub fn type_analyzer(&mut self) -> &mut QTypeAnalyzer {
        &mut self.type_analyzer
    }

    fn build_extension() -> Arc<Extension> {
        Extension::new_arc(
            Self::TEMP_EXT_NAME,
            hugr::extension::Version::new(0, 0, 0),
            |ext, ext_ref| {
                let opt_unwrap_sig = Signature::new(Type::from(option_type(qb_t())), qb_t());
                // produce option of qubit
                ext.add_op(
                    Self::REPACK_OPT,
                    Default::default(),
                    invert_sig(&PolyFuncTypeRV::new([], opt_unwrap_sig.clone())),
                    ext_ref,
                )
                .unwrap();
                // unwrap option of qubit
                ext.add_op(
                    Self::UNPACK_OPT,
                    Default::default(),
                    opt_unwrap_sig,
                    ext_ref,
                )
                .unwrap();
                // version of runtime barrier that takes a variable number of qubits
                ext.add_op(
                    Self::WRAPPED_BARRIER,
                    Default::default(),
                    PolyFuncTypeRV::new(
                        vec![TypeParam::new_list_type(TypeBound::Linear)],
                        FuncValueType::new_endo(TypeRV::new_row_var_use(0, TypeBound::Linear)),
                    ),
                    ext_ref,
                )
                .unwrap();

                // Add array operations for all ArrayKind types
                add_array_ops::<Array>(ext, ext_ref, Self::ARRAY_UNPACK, Self::ARRAY_REPACK)
                    .unwrap();
                add_array_ops::<ValueArray>(ext, ext_ref, Self::VARRAY_UNPACK, Self::VARRAY_REPACK)
                    .unwrap();
                add_array_ops::<BorrowArray>(
                    ext,
                    ext_ref,
                    Self::BARRAY_UNPACK,
                    Self::BARRAY_REPACK,
                )
                .unwrap();

                let tuple_unpack_sig = PolyFuncTypeRV::new(
                    vec![
                        // incoming tuple row
                        TypeParam::new_list_type(TypeBound::Linear),
                        // unpacked row
                        TypeParam::new_list_type(TypeBound::Linear),
                    ],
                    FuncValueType::new(
                        Type::new_tuple(TypeRV::new_row_var_use(0, TypeBound::Linear)),
                        TypeRV::new_row_var_use(1, TypeBound::Linear),
                    ),
                );
                // pack some wires into a tuple
                ext.add_op(
                    Self::TUPLE_REPACK,
                    Default::default(),
                    invert_sig(&tuple_unpack_sig),
                    ext_ref,
                )
                .unwrap();
                // unpack a tuple into some wires
                ext.add_op(
                    Self::TUPLE_UNPACK,
                    Default::default(),
                    tuple_unpack_sig,
                    ext_ref,
                )
                .unwrap();
            },
        )
    }

    /// Get an operation from the extension.
    pub fn get_op(&self, name: &OpName, args: impl Into<Vec<TypeArg>>) -> Option<ExtensionOp> {
        ExtensionOp::new(self.extension.get_op(name)?.clone(), args).ok()
    }

    /// Cache a function definition for a given operation.
    pub fn cache_function<F>(
        &mut self,
        op: &ExtensionOp,
        mangle_args: &[TypeArg],
        func_builder: F,
    ) -> Result<(), BuildError>
    where
        F: FnOnce(&mut Self, &mut FunctionBuilder<Hugr>) -> Result<Vec<Wire>, BuildError>,
    {
        let key = op.clone().into();
        // clippy's suggested fix does not make the borrow checker happy
        #[allow(clippy::map_entry)]
        if !self.funcs.contains_key(&key) {
            let name = mangle_name(op.def().name(), mangle_args);
            let sig = op.signature().deref().clone();
            let mut func_b = FunctionBuilder::new(name, sig)?;
            let outs = func_builder(self, &mut func_b)?;
            let func_def = func_b.finish_hugr_with_outputs(outs)?;
            self.funcs.insert(key, func_def);
        }

        Ok(())
    }

    /// Apply a cached operation with the given name and arguments
    pub fn apply_cached_operation<I, F>(
        &mut self,
        builder: &mut impl Dataflow,
        op_name: &OpName,
        args: impl Into<Vec<TypeArg>>,
        mangle_args: &[TypeArg],
        inputs: I,
        func_builder: F,
    ) -> Result<hugr::builder::handle::Outputs, BuildError>
    where
        I: IntoIterator<Item = Wire>,
        F: FnOnce(&mut Self, &mut FunctionBuilder<Hugr>) -> Result<Vec<Wire>, BuildError>,
    {
        let op = self.get_op(op_name, args).unwrap();
        self.cache_function(&op, mangle_args, func_builder)?;
        Ok(builder.add_dataflow_op(op, inputs)?.outputs())
    }

    /// Insert an option unwrap operation.
    pub fn unpack_option(
        &mut self,
        builder: &mut impl Dataflow,
        opt_wire: Wire,
    ) -> Result<Wire, BuildError> {
        let mut outputs = self.apply_cached_operation(
            builder,
            &Self::UNPACK_OPT,
            [],
            &[],
            [opt_wire],
            |_, func_b| {
                let [in_wire] = func_b.input_wires_arr();
                let [out_wire] = func_b.build_expect_sum(1, option_type(qb_t()), in_wire, |_| {
                    "Value of type Option<qubit> is None so cannot apply runtime barrier to qubit."
                        .to_string()
                })?;
                Ok(vec![out_wire])
            },
        )?;
        Ok(outputs.next().unwrap())
    }

    /// Insert an option construction operation.
    pub fn repack_option(
        &mut self,
        builder: &mut impl Dataflow,
        wire: Wire,
    ) -> Result<Wire, BuildError> {
        let mut outputs = self.apply_cached_operation(
            builder,
            &Self::REPACK_OPT,
            [],
            &[],
            [wire],
            |_, func_b| {
                let [in_wire] = func_b.input_wires_arr();
                let out_wire =
                    func_b.make_sum(1, vec![hugr::type_row![], vec![qb_t()].into()], [in_wire])?;
                Ok(vec![out_wire])
            },
        )?;
        Ok(outputs.next().unwrap())
    }

    /// Build a runtime barrier across the given qubit wires
    pub fn build_runtime_barrier(
        &mut self,
        builder: &mut impl Dataflow,
        qubit_wires: Vec<Wire>,
    ) -> Result<hugr::builder::handle::Outputs, BuildError> {
        let size = qubit_wires.len();
        let qb_row = vec![qb_t(); size];
        let args = [TypeArg::List(
            qb_row.clone().into_iter().map(Into::into).collect(),
        )];

        self.apply_cached_operation(
            builder,
            &Self::WRAPPED_BARRIER,
            args,
            &[TypeArg::BoundedNat(size as u64)],
            qubit_wires,
            |_, func_b| func_b.build_wrapped_barrier(func_b.input_wires()),
        )
    }

    /// Generic array unpacking using the ArrayKind trait
    fn unpack_array<AK: ArrayKind>(
        &mut self,
        builder: &mut impl Dataflow,
        array_wire: Wire,
        size: u64,
        elem_ty: &Type,
        op_name: &OpName,
    ) -> Result<Vec<Wire>, BuildError> {
        let args = match self.array_args::<AK>(size, elem_ty) {
            Some(args) => args,
            None => return Ok(vec![array_wire]), // Not a qubit-containing array
        };

        let outputs = self.apply_cached_operation(
            builder,
            op_name,
            args.clone(),
            &args[..2],
            [array_wire],
            |slf, func_b| {
                let w = func_b.input().out_wire(0);
                let elems = func_b.add_generic_array_unpack::<AK>(elem_ty.clone(), size, w)?;

                let result: Vec<_> = elems
                    .into_iter()
                    .map(|wire| slf.unpack_container(func_b, elem_ty, wire))
                    .collect::<Result<Vec<_>, _>>()?
                    .concat();
                Ok(result)
            },
        )?;

        Ok(outputs.collect())
    }

    /// Helper function for array arguments
    fn array_args<AK: ArrayKind>(&mut self, size: u64, elem_ty: &Type) -> Option<[TypeArg; 3]> {
        let row = self
            .type_analyzer
            .unpack_type(&AK::ty(size, elem_ty.clone()))?;
        let args = [
            size.into(),
            elem_ty.clone().into(),
            TypeArg::List(row.into_iter().map(Into::into).collect()),
        ];
        Some(args)
    }

    /// Generic array repacking using the ArrayKind trait
    fn repack_array<AK: ArrayKind>(
        &mut self,
        builder: &mut impl Dataflow,
        elem_wires: impl IntoIterator<Item = Wire>,
        size: u64,
        elem_ty: &Type,
        op_name: &OpName,
    ) -> Result<Wire, BuildError> {
        let args = match self.array_args::<AK>(size, elem_ty) {
            Some(args) => args,
            None => {
                return Ok(elem_wires
                    .into_iter()
                    .next()
                    .expect("Non-qubit container should only have one wire."));
            }
        };

        let inner_row_len = self.type_analyzer.num_unpacked_wires(elem_ty);

        let mut outputs = self.apply_cached_operation(
            builder,
            op_name,
            args.clone(),
            &args[..2],
            elem_wires,
            |slf, func_b| {
                let input = func_b.input();
                let elems: Result<Vec<_>, _> = input
                    .outputs()
                    .collect::<Vec<_>>()
                    .chunks(inner_row_len)
                    .map(|chunk| slf.repack_container(func_b, elem_ty, chunk.to_vec()))
                    .collect();

                let array_wire = func_b.add_new_generic_array::<AK>(elem_ty.clone(), elems?)?;
                Ok(vec![array_wire])
            },
        )?;

        Ok(outputs.next().unwrap())
    }

    /// Generate tuple arguments
    fn tuple_args(&mut self, tuple_row: &[Type]) -> Option<[TypeArg; 2]> {
        let unpacked_row = self
            .type_analyzer
            .unpack_type(&Type::new_tuple(tuple_row.to_vec()))?;

        let args = [
            TypeArg::List(tuple_row.iter().cloned().map(Into::into).collect()),
            TypeArg::List(unpacked_row.into_iter().map(Into::into).collect()),
        ];

        Some(args)
    }

    /// Unpack a row of types into a flat list of wires containing all qubits and remaining types
    pub fn unpack_row(
        &mut self,
        builder: &mut impl Dataflow,
        types: &[Type],
        wires: impl IntoIterator<Item = Wire>,
    ) -> Result<Vec<Wire>, BuildError> {
        // Process each type in the row with its corresponding wire
        let unpacked: Result<Vec<_>, _> = types
            .iter()
            .zip(wires)
            .map(|(typ, wire)| self.unpack_container(builder, typ, wire))
            .collect();

        // Flatten the nested vector of wires
        Ok(unpacked?.concat())
    }

    /// Repack a flat list of wires into a row of structured types
    pub fn repack_row(
        &mut self,
        builder: &mut impl Dataflow,
        types: &[Type],
        wires: impl IntoIterator<Item = Wire>,
    ) -> Result<Vec<Wire>, BuildError> {
        let mut wires = wires.into_iter();
        types
            .iter()
            .map(|typ| {
                let wire_count = self.type_analyzer.num_unpacked_wires(typ);
                let type_wires = wires.by_ref().take(wire_count).collect();
                self.repack_container(builder, typ, type_wires)
            })
            .collect()
    }

    /// Unpack a tuple into individual wires
    pub fn unpack_tuple(
        &mut self,
        builder: &mut impl Dataflow,
        tuple_wire: Wire,
        tuple_row: &[Type],
    ) -> Result<Vec<Wire>, BuildError> {
        let tuple_row = tuple_row.to_vec();
        let args = match self.tuple_args(&tuple_row) {
            Some(args) => args,
            None => return Ok(vec![tuple_wire]), // Not a qubit-containing tuple
        };

        let outputs = self.apply_cached_operation(
            builder,
            &Self::TUPLE_UNPACK,
            args.clone(),
            &args[..1],
            [tuple_wire],
            |slf, func_b| {
                let w = func_b.input().out_wire(0);
                let unpacked_tuple_wires = func_b
                    .add_dataflow_op(UnpackTuple::new(tuple_row.clone().into()), [w])?
                    .outputs()
                    .collect::<Vec<_>>();

                let unpacked = slf.unpack_row(func_b, &tuple_row, unpacked_tuple_wires)?;
                Ok(unpacked)
            },
        )?;

        Ok(outputs.collect())
    }

    /// Repack wires into a tuple
    pub fn repack_tuple(
        &mut self,
        builder: &mut impl Dataflow,
        elem_wires: impl IntoIterator<Item = Wire>,
        tuple_row: &[Type],
    ) -> Result<Wire, BuildError> {
        let tuple_row = tuple_row.to_vec();
        let args = match self.tuple_args(&tuple_row) {
            Some(args) => args,
            None => {
                return Ok(elem_wires
                    .into_iter()
                    .next()
                    .expect("Non-qubit container should only have one wire."));
            }
        };

        let mut outputs = self.apply_cached_operation(
            builder,
            &Self::TUPLE_REPACK,
            args.clone(),
            &args[..1],
            elem_wires,
            |slf, func_b| {
                let in_wires = func_b.input().outputs().collect::<Vec<_>>();

                let repacked_elem_wires = slf.repack_row(func_b, &tuple_row, in_wires)?;
                let tuple_wire = func_b.make_tuple(repacked_elem_wires)?;

                Ok(vec![tuple_wire])
            },
        )?;

        Ok(outputs.next().unwrap())
    }

    /// Unpack a qubit containing type until all qubit wires are found.
    pub fn unpack_container(
        &mut self,
        builder: &mut impl Dataflow,
        typ: &Type,
        container_wire: Wire,
    ) -> Result<Vec<Wire>, BuildError> {
        if typ == &qb_t() {
            return Ok(vec![container_wire]);
        }
        if is_opt_qb(typ) {
            return Ok(vec![self.unpack_option(builder, container_wire)?]);
        }
        macro_rules! handle_array_type {
            ($array_kind:ty, $unpack_op:expr) => {
                if let Some((n, elem_ty)) = typ.as_extension().and_then(array_args::<$array_kind>) {
                    return self.unpack_array::<$array_kind>(
                        builder,
                        container_wire,
                        n,
                        elem_ty,
                        &$unpack_op,
                    );
                }
            };
        }

        handle_array_type!(Array, Self::ARRAY_UNPACK);
        handle_array_type!(ValueArray, Self::VARRAY_UNPACK);
        handle_array_type!(BorrowArray, Self::BARRAY_UNPACK);
        if let Some(row) = typ.as_sum().and_then(SumType::as_tuple) {
            let row: hugr::types::TypeRow =
                row.clone().try_into().expect("unexpected row variable.");
            return self.unpack_tuple(builder, container_wire, &row);
        }

        // No need to unpack if the type is not a qubit container.
        Ok(vec![container_wire])
    }

    /// Repack a qubit containing type from its unpacked wires.
    pub fn repack_container(
        &mut self,
        builder: &mut impl Dataflow,
        typ: &Type,
        unpacked_wires: Vec<Wire>,
    ) -> Result<Wire, BuildError> {
        if typ == &qb_t() {
            debug_assert!(unpacked_wires.len() == 1);
            return Ok(unpacked_wires[0]);
        }
        if is_opt_qb(typ) {
            debug_assert!(unpacked_wires.len() == 1);
            return self.repack_option(builder, unpacked_wires[0]);
        }
        macro_rules! handle_array_type {
            ($array_kind:ty, $repack_op:expr) => {
                if let Some((n, elem_ty)) = typ.as_extension().and_then(array_args::<$array_kind>) {
                    return self.repack_array::<$array_kind>(
                        builder,
                        unpacked_wires,
                        n,
                        elem_ty,
                        &$repack_op,
                    );
                }
            };
        }

        handle_array_type!(Array, Self::ARRAY_REPACK);
        handle_array_type!(ValueArray, Self::VARRAY_REPACK);
        handle_array_type!(BorrowArray, Self::BARRAY_REPACK);

        if let Some(row) = typ.as_sum().and_then(SumType::as_tuple) {
            let row: hugr::types::TypeRow =
                row.clone().try_into().expect("unexpected row variable.");
            return self.repack_tuple(builder, unpacked_wires, &row);
        }
        debug_assert!(unpacked_wires.len() == 1);
        Ok(unpacked_wires[0])
    }
}

impl Default for BarrierOperationFactory {
    fn default() -> Self {
        Self::new()
    }
}

/// Build a runtime barrier operation for an array of qubits
pub fn build_runtime_barrier_op(array_size: u64) -> Result<Hugr, BuildError> {
    let mut barr_builder = DFGBuilder::new(Signature::new_endo(array_type(array_size, qb_t())))?;
    let array_wire = barr_builder.input().out_wire(0);
    let out = barr_builder.add_runtime_barrier(array_wire, array_size)?;
    barr_builder.finish_hugr_with_outputs([out])
}

#[cfg(test)]
mod tests {
    use super::*;
    use hugr::{extension::prelude::bool_t, HugrView};
    use rstest::rstest;

    #[test]
    fn test_barrier_op_factory_creation() {
        let factory = BarrierOperationFactory::new();
        assert_eq!(factory.funcs.len(), 0);
    }

    #[test]
    fn test_option_unwrap_wrap() -> Result<(), BuildError> {
        let mut factory = BarrierOperationFactory::new();
        let mut builder = DFGBuilder::new(Signature::new_endo(Type::from(option_type(qb_t()))))?;

        let input = builder.input().out_wire(0);
        let unwrapped = factory.unpack_option(&mut builder, input)?;
        let wrapped = factory.repack_option(&mut builder, unwrapped)?;

        let hugr = builder.finish_hugr_with_outputs([wrapped])?;
        assert!(hugr.validate().is_ok());
        Ok(())
    }

    #[rstest]
    #[case::array(
        Array,
        BarrierOperationFactory::ARRAY_UNPACK,
        BarrierOperationFactory::ARRAY_REPACK
    )]
    #[case::value_array(
        ValueArray,
        BarrierOperationFactory::VARRAY_UNPACK,
        BarrierOperationFactory::VARRAY_REPACK
    )]
    #[case::borrow_array(
        BorrowArray,
        BarrierOperationFactory::BARRAY_UNPACK,
        BarrierOperationFactory::BARRAY_REPACK
    )]
    fn test_array_unpack_repack<AK: ArrayKind>(
        #[case] _kind: AK,
        #[case] unpack_op: OpName,
        #[case] repack_op: OpName,
    ) -> Result<(), BuildError> {
        let mut factory = BarrierOperationFactory::new();
        let array_size = 2;

        // Create the specific array type
        let array_type = AK::ty(array_size, qb_t());

        // Build a dataflow graph that unpacks and repacks the array
        let mut builder = DFGBuilder::new(Signature::new_endo(array_type))?;
        let input = builder.input().out_wire(0);

        // Unpack the array
        let unpacked =
            factory.unpack_array::<AK>(&mut builder, input, array_size, &qb_t(), &unpack_op)?;

        // Repack the array
        let repacked =
            factory.repack_array::<AK>(&mut builder, unpacked, array_size, &qb_t(), &repack_op)?;

        let hugr = builder.finish_hugr_with_outputs([repacked])?;
        assert!(hugr.validate().is_ok());

        Ok(())
    }

    #[test]
    fn test_tuple_unpack_repack() -> Result<(), BuildError> {
        let mut factory = BarrierOperationFactory::new();
        let tuple_row = vec![qb_t(), bool_t()];
        let tuple_type = Type::new_tuple(tuple_row.clone());

        let mut builder = DFGBuilder::new(Signature::new_endo(tuple_type))?;

        let input = builder.input().out_wire(0);
        let unpacked = factory.unpack_tuple(&mut builder, input, &tuple_row)?;
        assert_eq!(unpacked.len(), tuple_row.len());

        let repacked = factory.repack_tuple(&mut builder, unpacked, &tuple_row)?;
        let hugr = builder.finish_hugr_with_outputs([repacked])?;
        assert!(hugr.validate().is_ok());
        Ok(())
    }

    #[test]
    fn test_runtime_barrier() -> Result<(), BuildError> {
        let mut factory = BarrierOperationFactory::new();
        let mut builder = DFGBuilder::new(Signature::new_endo(vec![qb_t(), qb_t(), qb_t()]))?;

        let inputs = builder.input().outputs().collect::<Vec<_>>();
        let outputs = factory.build_runtime_barrier(&mut builder, inputs)?;

        let hugr = builder.finish_hugr_with_outputs(outputs)?;
        assert!(hugr.validate().is_ok());
        Ok(())
    }

    #[test]
    fn test_build_runtime_barrier_op() -> Result<(), BuildError> {
        let array_size = 4;
        let hugr = build_runtime_barrier_op(array_size)?;
        assert!(hugr.validate().is_ok());
        Ok(())
    }
}
