use hugr::algorithms::replace_types::NodeTemplate;
use hugr::algorithms::{mangle_name, ReplaceTypes};
use hugr::builder::handle::Outputs;
use hugr::builder::{BuildError, Container, FunctionBuilder};
use hugr::extension::prelude::{self, option_type, UnwrapBuilder};
use hugr::hugr::patch::insert_cut::InsertCut;
use hugr::hugr::patch::PatchHugrMut;
use hugr::hugr::IdentList;
use hugr::ops::{DataflowOpTrait, ExtensionOp, OpName, OpTrait};
use hugr::std_extensions::collections::array::{
    self, array_type, array_type_parametric, ArrayOpBuilder,
};
use hugr::types::type_param::TypeParam;
use hugr::types::{FuncValueType, PolyFuncTypeRV, SumType, TypeArg, TypeBound, TypeRV, TypeRow};
use hugr::{
    builder::{DFGBuilder, Dataflow, DataflowHugr},
    extension::prelude::{qb_t, Barrier},
    hugr::hugrmut::HugrMut,
    types::{Signature, Type},
    HugrView, IncomingPort, Node, OutgoingPort, Wire,
};
use hugr::{type_row, Extension, Hugr};
use itertools::Itertools;
use std::collections::HashMap;
use std::ops::Deref;
use std::sync::Arc;

use super::lower::insert_function;
use super::{LowerTk2Error, QSystemOpBuilder};

/// If a sum is an option of a single type, return the type.
fn as_unary_option(sum: &SumType) -> Option<&TypeRV> {
    // TODO upstream to impl SumType
    let vars: Vec<_> = sum.variants().collect();
    match &vars[..] {
        [x, y] if x.is_empty() && y.len() == 1 => Some(&y[0]),
        _ => None,
    }
}

/// If a type is an option of qubit.
fn is_opt_qb(ty: &Type) -> bool {
    if let Some(sum) = ty.as_sum() {
        if let Some(inner) = as_unary_option(sum) {
            return inner == &qb_t();
        }
    }
    false
}

/// If a custom type is an array, return size and element type.
fn array_args(ext: &hugr::types::CustomType) -> Option<(u64, &Type)> {
    array::array_type_def()
        .check_custom(ext)
        .ok()
        .and_then(|_| match ext.args() {
            [TypeArg::BoundedNat { n }, TypeArg::Type { ty: elem_ty }] => Some((*n, elem_ty)),
            _ => None,
        })
}

type Target = (Node, IncomingPort);

#[derive(Clone, PartialEq, Eq)]
struct OpHashWrapper(ExtensionOp);

impl From<ExtensionOp> for OpHashWrapper {
    fn from(op: ExtensionOp) -> Self {
        Self(op)
    }
}

impl std::hash::Hash for OpHashWrapper {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.extension_id().hash(state);
        self.0.def().name().hash(state);
        self.0.args().hash(state);
    }
}

/// Helper struct to record how a type is unpacked.
#[derive(Clone, Hash, PartialEq, Eq)]
enum UnpackedRow {
    /// No internal qubits, so not unpacked.
    Other(Type),
    /// Qubit container, unpacked to a row of types.
    QbContainer(Vec<Type>),
}
impl UnpackedRow {
    /// Number of wires in the unpacked row.
    fn num_wires(&self) -> usize {
        match self {
            UnpackedRow::Other(_) => 1,
            UnpackedRow::QbContainer(row) => row.len(),
        }
    }

    /// Row produced when unpacked.
    fn into_row(self) -> Vec<Type> {
        match self {
            UnpackedRow::Other(ty) => vec![ty],
            UnpackedRow::QbContainer(row) => row,
        }
    }

    /// Returns `true` if the unpacked row is [`QbContainer`].
    ///
    /// [`QbContainer`]: UnpackedRow::QbContainer
    #[must_use]
    fn is_qb_container(&self) -> bool {
        matches!(self, Self::QbContainer(..))
    }
}

// TODO pull out a trait for lowering-via-temp-extension
/// Insert runtime barriers by using temporary extension operations
/// that get lowered to function calls.
pub(super) struct BarrierFuncs {
    /// Temporary extension used for placeholder operations.
    extension: Arc<Extension>,
    /// Function definitions for each instance of the operations.
    funcs: HashMap<OpHashWrapper, Hugr>,
    /// Cache of unpacked types.
    qubit_ports: HashMap<Type, UnpackedRow>,
}

/// Invert the signature of a function type.
fn invert_sig(sig: &PolyFuncTypeRV) -> PolyFuncTypeRV {
    let body = FuncValueType::new(sig.body().output().clone(), sig.body().input().clone());
    PolyFuncTypeRV::new(sig.params(), body)
}

static TEMP_EXT_NAME: IdentList = IdentList::new_static_unchecked("__tket2.barrier.temp");

impl BarrierFuncs {
    // Temporary operation names.
    const UNWRAP_OPT: OpName = OpName::new_static("option_qb_unwrap");
    const TAG_OPT: OpName = OpName::new_static("option_qb_tag");
    const WRAPPED_BARRIER: OpName = OpName::new_static("wrapped_barrier");
    const ARRAY_UNPACK: OpName = OpName::new_static("array_unpack");
    const ARRAY_REPACK: OpName = OpName::new_static("array_repack");
    const TUPLE_UNPACK: OpName = OpName::new_static("tuple_unpack");
    const TUPLE_REPACK: OpName = OpName::new_static("tuple_repack");

    // SECTION: Construction and Initialization

    /// Signature for a function that unwraps an option type.
    fn unwrap_opt_sig(ty: Type) -> Signature {
        Signature::new(Type::from(option_type(ty.clone())), ty)
    }

    /// Signature for a function that wraps an option type in to Some.
    fn wrap_opt_sig(ty: Type) -> Signature {
        Signature::new(ty.clone(), Type::from(option_type(ty)))
    }

    /// Create a new instance of the [BarrierFuncs] struct.
    pub(super) fn new() -> Result<Self, LowerTk2Error> {
        let unwrap_h = {
            let mut b = FunctionBuilder::new(Self::UNWRAP_OPT, Self::unwrap_opt_sig(qb_t()))?;
            let [in_wire] = b.input_wires_arr();
            let [out_wire] = b.build_expect_sum(1, option_type(qb_t()), in_wire, |_| {
                "Value of type Option<qubit> is None so cannot apply runtime barrier to qubit."
                    .to_string()
            })?;
            b.finish_hugr_with_outputs([out_wire])?
        };

        let wrap_h = {
            let mut b = FunctionBuilder::new(Self::TAG_OPT, Self::wrap_opt_sig(qb_t()))?;
            let [in_wire] = b.input_wires_arr();
            let out_wire = b.make_sum(1, vec![type_row![], vec![qb_t()].into()], [in_wire])?;
            b.finish_hugr_with_outputs([out_wire])?
        };

        let extension = Extension::new_arc(
            TEMP_EXT_NAME.clone(),
            hugr::extension::Version::new(0, 0, 0),
            |ext, ext_ref| {
                // unwrap option of qubit
                ext.add_op(
                    Self::UNWRAP_OPT,
                    Default::default(),
                    Self::unwrap_opt_sig(qb_t()),
                    ext_ref,
                )
                .unwrap();
                // produce option of qubit
                ext.add_op(
                    Self::TAG_OPT,
                    Default::default(),
                    Self::wrap_opt_sig(qb_t()),
                    ext_ref,
                )
                .unwrap();
                // version of runtime barrier that takes a variable number of qubits
                ext.add_op(
                    Self::WRAPPED_BARRIER,
                    Default::default(),
                    PolyFuncTypeRV::new(
                        vec![TypeParam::new_list(TypeBound::Any)],
                        FuncValueType::new_endo(TypeRV::new_row_var_use(0, TypeBound::Any)),
                    ),
                    ext_ref,
                )
                .unwrap();
                let array_unpack_sig = PolyFuncTypeRV::new(
                    vec![
                        TypeParam::max_nat(),
                        TypeParam::Type { b: TypeBound::Any },
                        TypeParam::new_list(TypeBound::Any),
                    ],
                    FuncValueType::new(
                        array_type_parametric(
                            TypeArg::new_var_use(0, TypeParam::max_nat()),
                            Type::new_var_use(1, TypeBound::Any),
                        )
                        .unwrap(),
                        TypeRV::new_row_var_use(2, TypeBound::Any),
                    ),
                );
                // pack some wires in to an array
                ext.add_op(
                    Self::ARRAY_REPACK,
                    Default::default(),
                    invert_sig(&array_unpack_sig),
                    ext_ref,
                )
                .unwrap();
                // unpack an array in to some wires
                ext.add_op(
                    Self::ARRAY_UNPACK,
                    Default::default(),
                    array_unpack_sig,
                    ext_ref,
                )
                .unwrap();

                let tuple_unpack_sig = PolyFuncTypeRV::new(
                    vec![
                        // incoming tuple row
                        TypeParam::new_list(TypeBound::Any),
                        // unpacked row
                        TypeParam::new_list(TypeBound::Any),
                    ],
                    FuncValueType::new(
                        Type::new_tuple(TypeRV::new_row_var_use(0, TypeBound::Any)),
                        TypeRV::new_row_var_use(1, TypeBound::Any),
                    ),
                );
                // pack some wires in to a tuple
                ext.add_op(
                    Self::TUPLE_REPACK,
                    Default::default(),
                    invert_sig(&tuple_unpack_sig),
                    ext_ref,
                )
                .unwrap();
                // unpack a tuple in to some wires
                ext.add_op(
                    Self::TUPLE_UNPACK,
                    Default::default(),
                    tuple_unpack_sig,
                    ext_ref,
                )
                .unwrap();
            },
        );

        let unwrap_op: OpHashWrapper =
            (ExtensionOp::new(extension.get_op(&Self::UNWRAP_OPT).unwrap().clone(), []).unwrap())
                .into();
        let tag_op: OpHashWrapper =
            (ExtensionOp::new(extension.get_op(&Self::TAG_OPT).unwrap().clone(), []).unwrap())
                .into();

        Ok(Self {
            extension,
            funcs: HashMap::from_iter([(unwrap_op, unwrap_h), (tag_op, wrap_h)]),
            qubit_ports: HashMap::new(),
        })
    }

    // SECTION: Core Extension and Operation Management

    /// Get an operation from the extension.
    fn get_op(&self, name: &OpName, args: impl Into<Vec<TypeArg>>) -> Option<ExtensionOp> {
        ExtensionOp::new(self.extension.get_op(name)?.clone(), args).ok()
    }

    /// Cache a function definition for a given operation.
    fn cache_function(
        &mut self,
        op: &ExtensionOp,
        mangle_args: &[TypeArg],
        func_builder: impl FnOnce(
            &mut Self,
            &mut FunctionBuilder<Hugr>,
        ) -> Result<Vec<Wire>, BuildError>,
    ) -> Result<(), BuildError> {
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
    fn apply_cached_operation<I>(
        &mut self,
        builder: &mut impl Dataflow,
        op_name: &OpName,
        args: impl Into<Vec<TypeArg>>,
        mangle_args: &[TypeArg],
        inputs: I,
        func_builder: impl FnOnce(
            &mut Self,
            &mut FunctionBuilder<Hugr>,
        ) -> Result<Vec<Wire>, BuildError>,
    ) -> Result<Outputs, BuildError>
    where
        I: IntoIterator<Item = Wire>,
    {
        let op = self.get_op(op_name, args).unwrap();
        self.cache_function(&op, mangle_args, func_builder)?;
        Ok(builder.add_dataflow_op(op, inputs)?.outputs())
    }

    /// Record that temporary extension operations can be lowered to function calls.
    pub fn lower(
        self,
        hugr: &mut impl HugrMut<Node = Node>,
        lowerer: &mut ReplaceTypes,
    ) -> Result<(), LowerTk2Error> {
        for (op, func_def) in self.funcs {
            let func_node = insert_function(hugr, func_def);
            lowerer.replace_op(&op.0, NodeTemplate::Call(func_node, vec![]));
        }
        Ok(())
    }

    // SECTION: Type Analysis and Unpacking Logic

    /// Compute the row produced when a type is unpacked.
    /// Uses memoization to avoid recomputing the same type.
    fn unpack_type(&mut self, ty: &Type) -> UnpackedRow {
        match self.qubit_ports.get(ty) {
            Some(count) => count.clone(),
            None => {
                let qb = self._new_unpack_type(ty);
                self.qubit_ports.insert(ty.clone(), qb.clone());
                qb
            }
        }
    }

    /// Compute the row produced when a type is unpacked for the first time.
    fn _new_unpack_type(&mut self, ty: &Type) -> UnpackedRow {
        if ty == &qb_t() {
            return UnpackedRow::QbContainer(vec![qb_t()]);
        }

        if let Some(row) = ty.as_sum().and_then(SumType::as_tuple) {
            let inner_unpacked = row
                .iter()
                .map(|t| {
                    self.unpack_type(&t.clone().try_into_type().expect("unexpected row variable."))
                })
                .collect::<Vec<_>>();
            if inner_unpacked.iter().any(UnpackedRow::is_qb_container) {
                let unpacked_row: Vec<_> = inner_unpacked
                    .into_iter()
                    .map(UnpackedRow::into_row)
                    .concat();
                return UnpackedRow::QbContainer(unpacked_row);
            }

            // TODO should other sums containing qubits raise an error?
        }

        if let Some((size, elem_ty)) = ty.as_extension().and_then(array_args) {
            // Special case for Option[Qubit] since it is used in guppy qubit arrays.
            // Fragile - would be better with dedicated guppy array type.
            // Not sure how this can be improved without runtime barrier being able to
            // take a compile time unknown number of qubits.

            if is_opt_qb(elem_ty) {
                return UnpackedRow::QbContainer(vec![qb_t(); size as usize]);
            } else {
                let elem_wc = self.unpack_type(elem_ty);
                return match elem_wc {
                    UnpackedRow::Other(_) => UnpackedRow::Other(ty.clone()),
                    UnpackedRow::QbContainer(inner) => {
                        return UnpackedRow::QbContainer(vec![inner; size as usize].concat());
                    }
                };
            };
        }

        UnpackedRow::Other(ty.clone())
    }

    /// Filter out types in the generic barrier that contain qubits.
    fn filter_qubit_containers<H: HugrMut<Node = Node>>(
        &mut self,
        hugr: &H,
        barrier: &Barrier,
        node: Node,
    ) -> Vec<(Type, Target)> {
        barrier
            .type_row
            .iter()
            .enumerate()
            .filter_map(|(i, typ)| {
                let wc = self.unpack_type(typ);

                match wc {
                    UnpackedRow::Other(_) => None,
                    UnpackedRow::QbContainer(_) => {
                        let port = OutgoingPort::from(i);
                        let target = hugr
                            .single_linked_input(node, port)
                            .expect("linearity violation.");
                        Some((typ.clone(), target))
                    }
                }
            })
            .collect()
    }

    // SECTION: Basic Operation Calls

    /// Insert an option unwrap.
    fn call_unwrap(
        &mut self,
        builder: &mut impl Dataflow,
        opt_wire: Wire,
    ) -> Result<Wire, BuildError> {
        let call =
            builder.add_dataflow_op(self.get_op(&Self::UNWRAP_OPT, []).unwrap(), [opt_wire])?;
        Ok(call.out_wire(0))
    }

    /// Insert an option construction.
    fn call_wrap(&mut self, builder: &mut impl Dataflow, wire: Wire) -> Result<Wire, BuildError> {
        let call = builder.add_dataflow_op(self.get_op(&Self::TAG_OPT, []).unwrap(), [wire])?;
        Ok(call.out_wire(0))
    }

    /// Insert a runtime barrier operation across some qubit wires.
    fn call_wrapped_runtime_barrier(
        &mut self,
        builder: &mut impl Dataflow,
        qubit_wires: Vec<Wire>,
    ) -> Result<Outputs, BuildError> {
        let size = qubit_wires.len();
        let qb_row = vec![qb_t(); size];
        let args = [TypeArg::Sequence {
            elems: qb_row.clone().into_iter().map(Into::into).collect(),
        }];

        self.apply_cached_operation(
            builder,
            &Self::WRAPPED_BARRIER,
            args,
            &[TypeArg::BoundedNat { n: size as u64 }],
            qubit_wires,
            |_, func_b| func_b.build_wrapped_barrier(func_b.input_wires()),
        )
    }

    // SECTION: Container Handling

    /// Unpack an array in to wires.
    fn call_unpack_array(
        &mut self,
        builder: &mut impl Dataflow,
        array_wire: Wire,
        size: u64,
        elem_ty: &Type,
    ) -> Result<Vec<Wire>, BuildError> {
        let Ok(args) = self.array_args(size, elem_ty) else {
            return Ok(vec![array_wire]);
        };

        let outputs = self.apply_cached_operation(
            builder,
            &Self::ARRAY_UNPACK,
            args.clone(),
            &args[..2],
            [array_wire],
            |slf, func_b| {
                let w = func_b.input().out_wire(0);
                let elems = super::pop_all(func_b, w, size, elem_ty.clone())?;
                let unpacked: Vec<_> = elems
                    .into_iter()
                    .map(|wire| slf.unpack_container(func_b, elem_ty, wire))
                    .collect::<Result<Vec<_>, _>>()?
                    .concat();
                Ok(unpacked)
            },
        )?;

        Ok(outputs.collect_vec())
    }

    fn array_args(&mut self, size: u64, elem_ty: &Type) -> Result<[TypeArg; 3], ()> {
        let row = match self.unpack_type(&array_type(size, elem_ty.clone())) {
            UnpackedRow::QbContainer(row) => row,
            _ => return Err(()),
        };
        let args = [
            size.into(),
            elem_ty.clone().into(),
            TypeArg::Sequence {
                elems: row.into_iter().map_into().collect(),
            },
        ];
        Ok(args)
    }

    /// Repack an array from wires.
    fn call_repack_array(
        &mut self,
        builder: &mut impl Dataflow,
        elem_wires: impl IntoIterator<Item = Wire>,
        size: u64,
        elem_ty: &Type,
    ) -> Result<Wire, BuildError> {
        let Ok(args) = self.array_args(size, elem_ty) else {
            return Ok(elem_wires
                .into_iter()
                .next()
                .expect("Non-qubit container should only have one wire."));
        };

        let inner_row_len = self.unpack_type(elem_ty).num_wires();

        let mut outputs = self.apply_cached_operation(
            builder,
            &Self::ARRAY_REPACK,
            args.clone(),
            &args[..2],
            elem_wires,
            |slf, func_b| {
                let input = func_b.input();
                let elems: Result<Vec<_>, _> = input
                    .outputs()
                    .chunks(inner_row_len)
                    .into_iter()
                    .map(|chunk| {
                        let chunk = chunk.collect_vec();
                        slf.repack_container(func_b, elem_ty, chunk)
                    })
                    .collect();
                let array_wire = func_b.add_new_array(elem_ty.clone(), elems?)?;

                Ok(vec![array_wire])
            },
        )?;

        Ok(outputs.next().unwrap())
    }

    /// Unpack a tuple in to wires.
    fn call_unpack_tuple(
        &mut self,
        builder: &mut impl Dataflow,
        tuple_wire: Wire,
        tuple_row: &[Type],
    ) -> Result<Vec<Wire>, BuildError> {
        let tuple_row: Vec<_> = tuple_row.to_vec();
        let Ok(args) = self.tuple_args(tuple_row.as_slice()) else {
            return Ok(vec![tuple_wire]);
        };

        let outputs = self.apply_cached_operation(
            builder,
            &Self::TUPLE_UNPACK,
            args.clone(),
            &args[..1],
            [tuple_wire],
            |slf, func_b| {
                let w = func_b.input().out_wire(0);
                let unpacked_wires = func_b
                    .add_dataflow_op(prelude::UnpackTuple::new(tuple_row.clone().into()), [w])?
                    .outputs();
                let unpacked: Vec<_> = unpacked_wires
                    .into_iter()
                    .zip(tuple_row.clone())
                    .map(|(wire, elem_ty)| slf.unpack_container(func_b, &elem_ty, wire))
                    .collect::<Result<Vec<_>, _>>()?
                    .concat();
                Ok(unpacked)
            },
        )?;

        Ok(outputs.collect_vec())
    }

    fn tuple_args(&mut self, tuple_row: &[Type]) -> Result<[TypeArg; 2], ()> {
        let unpacked_row = match self.unpack_type(&Type::new_tuple(tuple_row.to_vec())) {
            UnpackedRow::QbContainer(row) => row,
            _ => return Err(()),
        };
        let args = [
            TypeArg::Sequence {
                elems: tuple_row.iter().cloned().map_into().collect(),
            },
            TypeArg::Sequence {
                elems: unpacked_row.into_iter().map_into().collect(),
            },
        ];
        Ok(args)
    }

    /// Repack a tuple from wires.
    fn call_repack_tuple(
        &mut self,
        builder: &mut impl Dataflow,
        elem_wires: impl IntoIterator<Item = Wire>,
        tuple_row: &[Type],
    ) -> Result<Wire, BuildError> {
        let tuple_row: Vec<_> = tuple_row.to_vec();
        let Ok(args) = self.tuple_args(tuple_row.as_slice()) else {
            return Ok(elem_wires
                .into_iter()
                .next()
                .expect("Non-qubit container should only have one wire."));
        };

        let elem_num_wires: Vec<_> = tuple_row
            .iter()
            .map(|t| self.unpack_type(t).num_wires())
            .collect();
        let mut outputs = self.apply_cached_operation(
            builder,
            &Self::TUPLE_REPACK,
            args.clone(),
            &args[..1],
            elem_wires,
            |slf, func_b| {
                let mut in_wires = func_b.input().outputs();
                let mut elem_out_wires = Vec::with_capacity(tuple_row.len());
                for (elem_ty, num_wires) in tuple_row.iter().zip(elem_num_wires) {
                    let elem_in_wires = in_wires.by_ref().take(num_wires).collect();
                    let repacked = slf.repack_container(func_b, elem_ty, elem_in_wires)?;
                    elem_out_wires.push(repacked);
                }

                let tuple_wire = func_b.make_tuple(elem_out_wires)?;
                Ok(vec![tuple_wire])
            },
        )?;
        Ok(outputs.next().unwrap())
    }

    /// Unpack a qubit containing type until all qubit wires are found.
    fn unpack_container(
        &mut self,
        builder: &mut impl Dataflow,
        typ: &Type,
        container_wire: Wire,
    ) -> Result<Vec<Wire>, BuildError> {
        if typ == &qb_t() {
            return Ok(vec![container_wire]);
        }
        if is_opt_qb(typ) {
            return Ok(vec![self.call_unwrap(builder, container_wire)?]);
        }
        if let Some((n, elem_ty)) = typ.as_extension().and_then(array_args) {
            return self.call_unpack_array(builder, container_wire, n, elem_ty);
        }
        if let Some(row) = typ.as_sum().and_then(SumType::as_tuple) {
            let row: TypeRow = row.clone().try_into().expect("unexpected row variable.");
            return self.call_unpack_tuple(builder, container_wire, &row);
        }

        // No need to unpack if the type is not a qubit container.
        Ok(vec![container_wire])
    }

    /// Unpack a qubit containing type until all qubit wires are found.
    fn repack_container(
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
            return self.call_wrap(builder, unpacked_wires[0]);
        }
        if let Some((n, elem_ty)) = typ.as_extension().and_then(array_args) {
            return self.call_repack_array(builder, unpacked_wires, n, elem_ty);
        }

        if let Some(row) = typ.as_sum().and_then(SumType::as_tuple) {
            let row: TypeRow = row.clone().try_into().expect("unexpected row variable.");
            return self.call_repack_tuple(builder, unpacked_wires, &row);
        }
        debug_assert!(unpacked_wires.len() == 1);
        Ok(unpacked_wires[0])
    }

    // SECTION: Barrier Operations

    /// Insert [RuntimeBarrier] after a [Barrier] in the Hugr.
    pub(super) fn insert_runtime_barrier(
        &mut self,
        hugr: &mut impl HugrMut<Node = Node>,
        b_node: Node,
        barrier: Barrier,
    ) -> Result<(), LowerTk2Error> {
        // 1. Find all qubit containing types in the barrier.
        let filtered_qbs = self.filter_qubit_containers(hugr, &barrier, b_node);

        if filtered_qbs.is_empty() {
            return Ok(());
        }
        let parent = hugr.get_parent(b_node).expect("Barrier can't be root.");

        if let [(typ, (targ_n, targ_p))] = filtered_qbs.as_slice() {
            // If the barrier is over a single array of qubits
            // we can insert a runtime barrier op directly.
            let shortcut = typ
                .as_extension()
                .and_then(|ext| array_args(ext))
                .and_then(|(size, elem_ty)| (elem_ty == &qb_t()).then_some(size))
                .map(|size| {
                    let barr_hugr = build_runtime_barrier_op(size)?;

                    let insert = InsertCut::new(parent, vec![(*targ_n, *targ_p)], barr_hugr);
                    insert.apply_hugr_mut(hugr)?;
                    Ok(())
                });
            if let Some(res) = shortcut {
                return res;
            }
        }
        let (row, targets) = filtered_qbs.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
        let insert_hugr: Hugr = self.packing_hugr(row)?;

        let inserter = InsertCut::new(parent, targets, insert_hugr);

        inserter.apply_hugr_mut(hugr)?;

        Ok(())
    }

    /// Construct the endofunction HUGR to unpack types, apply the runtime barrier across qubits, and repack.
    fn packing_hugr(&mut self, container_row: Vec<Type>) -> Result<Hugr, LowerTk2Error> {
        let mut dfg_b = DFGBuilder::new(Signature::new_endo(container_row.clone()))?;

        // pack the container row in to a tuple to use the tuple unpacking logic.
        let tuple_type = Type::new_tuple(container_row.clone());

        let input = dfg_b.input();
        let tuple = dfg_b.make_tuple(input.outputs())?;

        // unpack the tuple in to wires
        let unpacked_wires = self.unpack_container(&mut dfg_b, &tuple_type, tuple)?;

        // tag the qubit wires
        let tagged_wires: Vec<(bool, Wire)> = unpacked_wires
            .into_iter()
            .map(|wire| {
                let node_sig = dfg_b
                    .hugr()
                    .get_optype(wire.node())
                    .dataflow_signature()
                    .unwrap();
                (node_sig.port_type(wire.source()) == Some(&qb_t()), wire)
            })
            .collect();

        let qubit_wires = tagged_wires
            .iter()
            .filter(|(is_qb, _)| *is_qb)
            .map(|(_, w)| *w)
            .collect();

        // call the runtime barrier no all the wires
        let mut r_bar_outs = self.call_wrapped_runtime_barrier(&mut dfg_b, qubit_wires)?;

        // replace the qubit wires with the runtime barrier outputs
        let repack_wires = tagged_wires
            .into_iter()
            .map(|(is_qb, w)| {
                if is_qb {
                    r_bar_outs
                        .next()
                        .expect("Not enough runtime barrier outputs.")
                } else {
                    w
                }
            })
            .collect::<Vec<_>>();

        // repack the wires in to a tuple
        let repacked_tuple = self.repack_container(&mut dfg_b, &tuple_type, repack_wires)?;

        // separate back in to a row
        let new_container_wires = dfg_b
            .add_dataflow_op(
                prelude::UnpackTuple::new(container_row.clone().into()),
                [repacked_tuple],
            )?
            .outputs();
        Ok(dfg_b.finish_hugr_with_outputs(new_container_wires)?)
    }
}

fn build_runtime_barrier_op(array_size: u64) -> Result<Hugr, BuildError> {
    let mut barr_builder = DFGBuilder::new(Signature::new_endo(array_type(array_size, qb_t())))?;
    let array_wire = barr_builder.input().out_wire(0);
    let out = barr_builder.add_runtime_barrier(array_wire, array_size)?;
    barr_builder.finish_hugr_with_outputs([out])
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::extension::qsystem::{self, lower_tk2_op};
    use hugr::{
        builder::DFGBuilder,
        extension::prelude::{bool_t, option_type, qb_t},
        std_extensions::collections::array::array_type,
    };
    use hugr::{
        ops::{handle::NodeHandle, NamedOp},
        HugrView,
    };
    use itertools::Itertools;
    use rstest::rstest;
    fn opt_q_arr(size: u64) -> Type {
        array_type(size, option_type(qb_t()).into())
    }

    #[rstest]
    #[case(vec![qb_t(), qb_t()], 2, false)]
    #[case(vec![qb_t(), bool_t(), qb_t()], 2, false)]
    // special case, array of option qubit is unwrapped and unpacked
    #[case(vec![qb_t(), opt_q_arr(2)], 3, false)]
    // bare option of qubit is ignored
    #[case(vec![qb_t(), option_type(qb_t()).into()], 1, false)]
    #[case(vec![array_type(2, bool_t())], 0, false)]
    // special case, single array of qubits is passed directly to op without unpacking
    #[case(vec![array_type(3, qb_t())], 1, true)]
    #[case(vec![qb_t(), array_type(2, qb_t()), array_type(2, array_type(2, qb_t()))], 7, false)]
    #[case(vec![Type::new_tuple(vec![bool_t(), qb_t()]), qb_t()], 2, false)]
    #[case(vec![Type::new_tuple(vec![bool_t(), qb_t(), opt_q_arr(2)]), qb_t()], 4, false)]
    #[case(vec![Type::new_tuple(vec![bool_t(), qb_t(), array_type(2, Type::new_tuple(vec![bool_t(), qb_t()]))]), qb_t()], 4, false)]
    fn test_barrier(
        #[case] type_row: Vec<Type>,
        #[case] num_qb: usize,
        // whether it is the array[qubit] special case
        #[case] no_parent: bool,
    ) {
        // build a dfg with a generic barrier
        let (mut h, barr_n) = {
            let mut b = DFGBuilder::new(Signature::new_endo(type_row.clone())).unwrap();

            let barr_n = b.add_barrier(b.input_wires()).unwrap();
            (
                b.finish_hugr_with_outputs(barr_n.outputs()).unwrap(),
                barr_n.node(),
            )
        };

        // lower barrier to barrier + runtime barrier
        let lowered = lower_tk2_op(&mut h).unwrap_or_else(|e| panic!("{}", e));
        h.validate().unwrap_or_else(|e| panic!("{}", e));
        assert!(matches!(&lowered[..], [n] if barr_n == *n));

        let _barr_op: Barrier = h.get_optype(barr_n).cast().unwrap();

        let run_bar_n = if no_parent {
            h.nodes()
                .filter(|&r_barr_n| {
                    h.get_optype(r_barr_n).as_extension_op().is_some_and(|op| {
                        op.name().contains(qsystem::RUNTIME_BARRIER_NAME.as_str())
                    })
                })
                .exactly_one()
                .ok()
                .unwrap()
        } else {
            let run_barr_func_n = h
                .children(h.root())
                .filter(|&r_barr_n| {
                    h.get_optype(r_barr_n)
                        .as_func_defn()
                        .is_some_and(|op| op.name.contains(BarrierFuncs::WRAPPED_BARRIER.as_str()))
                })
                .exactly_one()
                .ok();
            if run_barr_func_n.is_none() {
                // if the runtime barrier function is never called
                // make sure it is because there are no qubits in the barrier
                let mut bf = BarrierFuncs::new().unwrap();
                let tuple_type = Type::new_tuple(type_row);
                assert!(matches!(bf.unpack_type(&tuple_type), UnpackedRow::Other(_)));
                return;
            }
            h.single_linked_input(run_barr_func_n.unwrap(), 0)
                .unwrap()
                .0
        };

        assert_eq!(h.all_linked_inputs(run_bar_n).count(), num_qb);

        // Check all temporary ops are removed
        assert!(h.nodes().all(|n| h
            .get_optype(n)
            .as_extension_op()
            .is_none_or(|op| op.extension_id() != &TEMP_EXT_NAME)));
    }
}
