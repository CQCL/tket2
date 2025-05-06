use hugr::algorithms::replace_types::NodeTemplate;
use hugr::algorithms::ReplaceTypes;
use hugr::builder::handle::Outputs;
use hugr::builder::{BuildError, Container, FunctionBuilder};
use hugr::extension::prelude::{self, option_type, UnwrapBuilder};
use hugr::hugr::rewrite::inline_dfg::InlineDFG;
use hugr::hugr::{IdentList, Rewrite};
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

// copied from
// https://github.com/CQCL/hugr/blob/c8090ca9089368de1a2de25e0071458ce5222d70/hugr-passes/src/monomorphize.rs#L353-L356
mod mangle {
    use std::fmt::Write;

    use itertools::Itertools;

    use super::*;
    struct TypeArgsList<'a>(&'a [TypeArg]);

    impl std::fmt::Display for TypeArgsList<'_> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            for arg in self.0 {
                f.write_char('$')?;
                write_type_arg_str(arg, f)?;
            }
            Ok(())
        }
    }

    fn escape_dollar(str: impl AsRef<str>) -> String {
        str.as_ref().replace("$", "\\$")
    }

    fn write_type_arg_str(arg: &TypeArg, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match arg {
            TypeArg::Type { ty } => {
                f.write_fmt(format_args!("t({})", escape_dollar(ty.to_string())))
            }
            TypeArg::BoundedNat { n } => f.write_fmt(format_args!("n({n})")),
            TypeArg::String { arg } => f.write_fmt(format_args!("s({})", escape_dollar(arg))),
            TypeArg::Sequence { elems } => {
                f.write_fmt(format_args!("seq({})", TypeArgsList(elems)))
            }
            TypeArg::Extensions { es } => f.write_fmt(format_args!(
                "es({})",
                es.iter().map(|x| x.deref()).join(",")
            )),
            // We are monomorphizing. We will never monomorphize to a signature
            // containing a variable.
            TypeArg::Variable { .. } => panic!("type_arg_str variable: {arg}"),
            _ => panic!("unknown type arg: {arg}"),
        }
    }

    /// We do our best to generate unique names. Our strategy is to pick out '$' as
    /// a special character.
    ///
    /// We:
    ///  - construct a new name of the form `{func_name}$$arg0$arg1$arg2` etc
    ///  - replace any existing `$` in the function name or type args string
    ///    representation with `r"\$"`
    ///  - We depend on the `Display` impl of `Type` to generate the string
    ///    representation of a `TypeArg::Type`. For other constructors we do the
    ///    simple obvious thing.
    ///  - For all TypeArg Constructors we choose a short prefix (e.g. `t` for type)
    ///    and use "t({arg})" as the string representation of that arg.
    pub fn mangle_name(name: &str, type_args: impl AsRef<[TypeArg]>) -> String {
        let name = escape_dollar(name);
        format!("${name}${}", TypeArgsList(type_args.as_ref()))
    }
}

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

#[derive(Clone, Hash, PartialEq, Eq)]
enum UnpackedRow {
    Other(Type),
    QbContainer(Vec<Type>),
}
impl UnpackedRow {
    fn num_wires(&self) -> usize {
        match self {
            UnpackedRow::Other(_) => 1,
            UnpackedRow::QbContainer(row) => row.len(),
        }
    }
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
pub(super) struct BarrierFuncs {
    extension: Arc<Extension>,
    funcs: HashMap<OpHashWrapper, Hugr>,
    qubit_ports: HashMap<Type, UnpackedRow>,
}

fn invert_sig(sig: &PolyFuncTypeRV) -> PolyFuncTypeRV {
    let body = FuncValueType::new(sig.body().output().clone(), sig.body().input().clone());
    PolyFuncTypeRV::new(sig.params(), body)
}

static TEMP_EXT_NAME: IdentList = IdentList::new_static_unchecked("__tket2.barrier.temp");

impl BarrierFuncs {
    const UNWRAP_OPT: OpName = OpName::new_static("option_qb_unwrap");
    const TAG_OPT: OpName = OpName::new_static("option_qb_tag");
    const WRAPPED_BARRIER: OpName = OpName::new_static("wrapped_barrier");
    const ARRAY_UNPACK: OpName = OpName::new_static("array_unpack");
    const ARRAY_REPACK: OpName = OpName::new_static("array_repack");
    const TUPLE_UNPACK: OpName = OpName::new_static("tuple_unpack");
    const TUPLE_REPACK: OpName = OpName::new_static("tuple_repack");
    /// Signature for a function that unwraps an option type.
    fn unwrap_opt_sig(ty: Type) -> Signature {
        Signature::new(Type::from(option_type(ty.clone())), ty)
    }

    /// Signature for a function that wraps an option type in to Some.
    fn wrap_opt_sig(ty: Type) -> Signature {
        Signature::new(ty.clone(), Type::from(option_type(ty)))
    }

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
                ext.add_op(
                    Self::UNWRAP_OPT,
                    Default::default(),
                    Self::unwrap_opt_sig(qb_t()),
                    ext_ref,
                )
                .unwrap();
                ext.add_op(
                    Self::TAG_OPT,
                    Default::default(),
                    Self::wrap_opt_sig(qb_t()),
                    ext_ref,
                )
                .unwrap();
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
                ext.add_op(
                    Self::ARRAY_REPACK,
                    Default::default(),
                    invert_sig(&array_unpack_sig),
                    ext_ref,
                )
                .unwrap();
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
                ext.add_op(
                    Self::TUPLE_REPACK,
                    Default::default(),
                    invert_sig(&tuple_unpack_sig),
                    ext_ref,
                )
                .unwrap();
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

    fn get_op(&self, name: &OpName, args: impl Into<Vec<TypeArg>>) -> Option<ExtensionOp> {
        ExtensionOp::new(self.extension.get_op(name)?.clone(), args).ok()
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
                let wc = self.num_unpacked_wires(typ);

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
    /// Count how many qubits are contained in a type.
    fn num_unpacked_wires(&mut self, ty: &Type) -> UnpackedRow {
        match self.qubit_ports.get(ty) {
            Some(count) => count.clone(),
            None => {
                let qb = self._num_unpacked_wires(ty);
                self.qubit_ports.insert(ty.clone(), qb.clone());
                qb
            }
        }
    }
    fn _num_unpacked_wires(&mut self, ty: &Type) -> UnpackedRow {
        if ty == &qb_t() {
            return UnpackedRow::QbContainer(vec![qb_t()]);
        }

        if let Some(row) = ty.as_sum().and_then(SumType::as_tuple) {
            let inner_unpacked = row
                .iter()
                .map(|t| {
                    self.num_unpacked_wires(
                        &t.clone().try_into_type().expect("unexpected row variable."),
                    )
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
                let elem_wc = self.num_unpacked_wires(elem_ty);
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

    fn call_unwrap(
        &mut self,
        builder: &mut impl Dataflow,
        opt_wire: Wire,
    ) -> Result<Wire, BuildError> {
        let call =
            builder.add_dataflow_op(self.get_op(&Self::UNWRAP_OPT, []).unwrap(), [opt_wire])?;
        Ok(call.out_wire(0))
    }

    fn call_wrap(&mut self, builder: &mut impl Dataflow, wire: Wire) -> Result<Wire, BuildError> {
        let call = builder.add_dataflow_op(self.get_op(&Self::TAG_OPT, []).unwrap(), [wire])?;
        Ok(call.out_wire(0))
    }

    fn cache_function(
        &mut self,
        op: &ExtensionOp,
        func_builder: impl FnOnce(&mut Self) -> Result<Hugr, BuildError>,
    ) -> Result<(), BuildError> {
        let key = op.clone().into();
        // clippy's suggested fix does not make the borrow checker happy
        #[allow(clippy::map_entry)]
        if !self.funcs.contains_key(&key) {
            let func_def: Hugr = func_builder(self)?;
            self.funcs.insert(key, func_def);
        }

        Ok(())
    }
    fn call_wrapped_runtime_barrier(
        &mut self,
        builder: &mut impl Dataflow,
        qubit_wires: Vec<Wire>,
    ) -> Result<Outputs, BuildError> {
        let size = qubit_wires.len();
        let qb_row = vec![qb_t(); size];

        let op = self
            .get_op(
                &Self::WRAPPED_BARRIER,
                [TypeArg::Sequence {
                    elems: qb_row.clone().into_iter().map(Into::into).collect(),
                }],
            )
            .unwrap();
        let sig = op.signature().deref().clone();
        self.cache_function(&op, |_| {
            let name = mangle::mangle_name(
                &Self::WRAPPED_BARRIER,
                [TypeArg::BoundedNat { n: size as u64 }],
            );
            let mut func_b = FunctionBuilder::new(name, sig)?;
            let outs = func_b.build_wrapped_barrier(func_b.input_wires())?;
            func_b.finish_hugr_with_outputs(outs)
        })?;

        Ok(builder.add_dataflow_op(op, qubit_wires)?.outputs())
    }

    fn call_unpack_array(
        &mut self,
        builder: &mut impl Dataflow,
        array_wire: Wire,
        size: u64,
        elem_ty: &Type,
    ) -> Result<Vec<Wire>, BuildError> {
        let row = if let UnpackedRow::QbContainer(row) =
            self.num_unpacked_wires(&array_type(size, elem_ty.clone()))
        {
            row
        } else {
            return Ok(vec![array_wire]);
        };

        let inner_row_len = self.num_unpacked_wires(elem_ty).num_wires();
        let args = [
            size.into(),
            elem_ty.clone().into(),
            TypeArg::Sequence {
                elems: row.into_iter().map_into().collect(),
            },
        ];
        let op = self.get_op(&Self::ARRAY_UNPACK, args.clone()).unwrap();
        self.cache_function(&op, |funcs| {
            let name = mangle::mangle_name(op.def().name(), &args[..2]);
            let sig = op.signature().deref().clone();
            let mut func_b = FunctionBuilder::new(name, sig)?;
            let w = func_b.input().out_wire(0);
            let elems = super::pop_all(&mut func_b, w, size, elem_ty.clone())?;
            let unpacked: Vec<_> = elems
                .into_iter()
                .map(|wire| funcs.unpack_container(&mut func_b, elem_ty, wire))
                .collect::<Result<Vec<_>, _>>()?
                .concat();
            func_b.finish_hugr_with_outputs(unpacked)
        })?;

        let outputs = builder
            .add_dataflow_op(op, [array_wire])?
            .outputs()
            .collect();

        let repack_op = self.get_op(&Self::ARRAY_REPACK, args.clone()).unwrap();

        self.cache_function(&repack_op, |funcs| {
            let name = mangle::mangle_name(repack_op.def().name(), &args[..2]);
            let sig = repack_op.signature().deref().clone();
            let mut func_b = FunctionBuilder::new(name, sig)?;
            let input = func_b.input();
            let elems: Result<Vec<_>, _> = input
                .outputs()
                .chunks(inner_row_len)
                .into_iter()
                .map(|chunk| {
                    let chunk = chunk.collect_vec();
                    funcs.repack_container(&mut func_b, elem_ty, chunk)
                })
                .collect();
            let array_wire = func_b.add_new_array(elem_ty.clone(), elems?)?;
            func_b.finish_hugr_with_outputs([array_wire])
        })?;

        Ok(outputs)
    }

    fn call_repack_array(
        &mut self,
        builder: &mut impl Dataflow,
        elem_wires: impl IntoIterator<Item = Wire>,
        size: u64,
        elem_ty: &Type,
    ) -> Result<Wire, BuildError> {
        let row = if let UnpackedRow::QbContainer(row) =
            self.num_unpacked_wires(&array_type(size, elem_ty.clone()))
        {
            row
        } else {
            return Ok(elem_wires
                .into_iter()
                .next()
                .expect("Non-qubit container should only have one wire."));
        };

        let args = [
            size.into(),
            elem_ty.clone().into(),
            TypeArg::Sequence {
                elems: row.into_iter().map_into().collect(),
            },
        ];

        // TODO common up with pack up to this point
        let repack_op = self.get_op(&Self::ARRAY_REPACK, args.clone()).unwrap();
        let repack_call = builder.add_dataflow_op(repack_op, elem_wires)?;
        Ok(repack_call.out_wire(0))
    }

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
    fn call_unpack_tuple(
        &mut self,
        builder: &mut impl Dataflow,
        tuple_wire: Wire,
        tuple_row: &[Type],
    ) -> Result<Vec<Wire>, BuildError> {
        let tuple_row: Vec<_> = tuple_row.to_vec();
        let unpacked_row = if let UnpackedRow::QbContainer(row) =
            self.num_unpacked_wires(&Type::new_tuple(tuple_row.clone()))
        {
            row
        } else {
            return Ok(vec![tuple_wire]);
        };
        let args = [
            TypeArg::Sequence {
                elems: tuple_row.iter().cloned().map_into().collect(),
            },
            TypeArg::Sequence {
                elems: unpacked_row.into_iter().map_into().collect(),
            },
        ];
        let op = self.get_op(&Self::TUPLE_UNPACK, args.clone()).unwrap();
        self.cache_function(&op, |funcs| {
            let name = mangle::mangle_name(op.def().name(), &args[..1]);
            let sig = op.signature().deref().clone();
            let mut func_b = FunctionBuilder::new(name, sig)?;
            let w = func_b.input().out_wire(0);
            let unpacked_wires = func_b
                .add_dataflow_op(prelude::UnpackTuple::new(tuple_row.clone().into()), [w])?
                .outputs();
            let unpacked: Vec<_> = unpacked_wires
                .into_iter()
                .zip(tuple_row.clone())
                .map(|(wire, elem_ty)| funcs.unpack_container(&mut func_b, &elem_ty, wire))
                .collect::<Result<Vec<_>, _>>()?
                .concat();
            func_b.finish_hugr_with_outputs(unpacked)
        })?;

        let outputs = builder
            .add_dataflow_op(op, [tuple_wire])?
            .outputs()
            .collect();

        let repack_op = self.get_op(&Self::TUPLE_REPACK, args.clone()).unwrap();
        let elem_num_wires: Vec<_> = tuple_row
            .iter()
            .map(|t| self.num_unpacked_wires(t).num_wires())
            .collect();
        self.cache_function(&repack_op, |funcs| {
            let name = mangle::mangle_name(repack_op.def().name(), &args[..1]);
            let sig = repack_op.signature().deref().clone();
            let mut func_b = FunctionBuilder::new(name, sig)?;
            let mut in_wires = func_b.input().outputs();
            let mut elem_out_wires = Vec::with_capacity(tuple_row.len());
            for (elem_ty, num_wires) in tuple_row.iter().zip(elem_num_wires) {
                let elem_in_wires = in_wires.by_ref().take(num_wires).collect();
                let repacked = funcs.repack_container(&mut func_b, elem_ty, elem_in_wires)?;
                elem_out_wires.push(repacked);
            }

            let tuple_wire = func_b.make_tuple(elem_out_wires)?;

            func_b.finish_hugr_with_outputs([tuple_wire])
        })?;
        Ok(outputs)
    }

    fn call_repack_tuple(
        &mut self,
        builder: &mut impl Dataflow,
        elem_wires: impl IntoIterator<Item = Wire>,
        tuple_row: &[Type],
    ) -> Result<Wire, BuildError> {
        let tuple_row: Vec<_> = tuple_row.to_vec();
        let unpacked_row = if let UnpackedRow::QbContainer(row) =
            self.num_unpacked_wires(&Type::new_tuple(tuple_row.clone()))
        {
            row
        } else {
            return Ok(elem_wires
                .into_iter()
                .next()
                .expect("Non-qubit container should only have one wire."));
        };
        let args = [
            TypeArg::Sequence {
                elems: tuple_row.iter().cloned().map_into().collect(),
            },
            TypeArg::Sequence {
                elems: unpacked_row.into_iter().map_into().collect(),
            },
        ];
        let repack_op = self.get_op(&Self::TUPLE_REPACK, args.clone()).unwrap();
        let repack_call = builder.add_dataflow_op(repack_op, elem_wires)?;
        Ok(repack_call.out_wire(0))
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

    /// Insert [RuntimeBarrier] after every [Barrier] in the Hugr.
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

                    let insert = InsertCut {
                        parent,
                        targets: vec![(*targ_n, *targ_p)],
                        insertion: barr_hugr,
                    };
                    insert
                        .apply(hugr)
                        // TODO handle error
                        .expect("failed to insert runtime barrier");
                    Ok(())
                });
            if let Some(res) = shortcut {
                return res;
            }
        }
        let (row, targets) = filtered_qbs.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
        let insert_hugr: Hugr = self.packing_hugr(row)?;

        let inserter = InsertCut {
            parent,
            targets,
            insertion: insert_hugr,
        };

        inserter
            .apply(hugr)
            // TODO handle error
            .expect("failed to insert runtime barrier");

        Ok(())
    }

    fn packing_hugr(&mut self, container_row: Vec<Type>) -> Result<Hugr, LowerTk2Error> {
        // TODO add comments for steps
        let mut dfg_b = DFGBuilder::new(Signature::new_endo(container_row.clone()))?;
        let tuple_type = Type::new_tuple(container_row.clone());

        let input = dfg_b.input();
        let tuple = dfg_b.make_tuple(input.outputs())?;

        let unpacked_wires = self.unpack_container(&mut dfg_b, &tuple_type, tuple)?;
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
        let mut r_bar_outs = self.call_wrapped_runtime_barrier(&mut dfg_b, qubit_wires)?;

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

        let repacked_tuple = self.repack_container(&mut dfg_b, &tuple_type, repack_wires)?;

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

// TODO upstream to hugr rewrite
struct InsertCut {
    parent: Node,
    targets: Vec<(Node, IncomingPort)>,
    insertion: Hugr,
}

impl InsertCut {
    fn apply(self, h: &mut impl HugrMut<Node = Node>) -> Result<HashMap<Node, Node>, ()> {
        assert!(self.insertion.root_type().is_dfg());
        let insert_res = h.insert_hugr(self.parent, self.insertion);
        let inserted_root = insert_res.new_root;
        for (i, (target, port)) in self.targets.into_iter().enumerate() {
            let (src_n, src_p) = h
                .single_linked_output(target, port)
                .expect("Incoming value edge has single connection.");
            h.disconnect(target, port);
            h.connect(src_n, src_p, inserted_root, i);
            h.connect(inserted_root, i, target, port);
        }
        let inline = InlineDFG(inserted_root.into());

        inline.apply(h).expect("inline failed");
        Ok(insert_res.node_map)
    }
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
                assert!(matches!(
                    bf.num_unpacked_wires(&tuple_type),
                    UnpackedRow::Other(_)
                ));
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
