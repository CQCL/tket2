use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::ops::Deref;
use std::sync::Arc;

use hugr::builder::handle::Outputs;
use hugr::builder::{BuildError, FunctionBuilder};
use hugr::extension::prelude::{self, option_type, UnwrapBuilder};
use hugr::hugr::rewrite::inline_dfg::InlineDFG;
use hugr::hugr::{IdentList, Rewrite};
use hugr::ops::handle::NodeHandle;
use hugr::ops::{DataflowOpTrait, ExtensionOp, OpName};
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

use super::{LowerTk2Error, QSystemOpBuilder};

/// Count how many qubits are contained in a type.
fn num_contained_qubits(ty: &Type) -> u64 {
    if ty == &qb_t() {
        return 1;
    }

    if let Some(row) = ty.as_sum().and_then(SumType::as_tuple) {
        return row
            .iter()
            .map(|t| {
                num_contained_qubits(&t.clone().try_into_type().expect("unexpected row variable."))
            })
            .sum();
        // TODO should other sums containing qubits raise an error?
    }

    if let Some((size, elem_ty)) = ty.as_extension().and_then(array_args) {
        // Special case for Option[Qubit] since it is used in guppy qubit arrays.
        // Fragile - would be better with dediccated guppy array type.
        // Not sure how this can be improved without runtime barrier being able to
        // take a compile time unknown number of qubits.
        return size
            * if is_opt_qb(elem_ty) {
                1
            } else {
                num_contained_qubits(elem_ty)
            };
    }

    0
}

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

struct QubitContainer {
    typ: Type,
    n_qb: u64,
}

type Target = (Node, IncomingPort);

/// Filter out types in the generic barrier that contain qubits.
fn filter_qubit_containers<H: HugrMut>(
    hugr: &H,
    barrier: &Barrier,
    node: Node,
) -> Vec<(QubitContainer, Target)> {
    barrier
        .type_row
        .iter()
        .enumerate()
        .filter_map(|(i, typ)| {
            let n_qb = num_contained_qubits(typ);
            (n_qb > 0).then(|| {
                let port = OutgoingPort::from(i);
                let target = hugr
                    .single_linked_input(node, port)
                    .expect("linearity violation.");
                (
                    QubitContainer {
                        typ: typ.clone(),
                        n_qb,
                    },
                    target,
                )
            })
        })
        .collect()
}

/// Leaf of a qubit [WireTree].
enum WireLeaf {
    Qubit(Wire),
    Other(Wire),
    OptQb(Wire),
}

/// Record the component wires once a qubit containing type is unpacked.
enum WireTree {
    Leaf(WireLeaf),
    Array(Type, Vec<WireTree>),
    Tuple(Vec<WireTree>),
}

impl WireTree {
    /// Recursively unpack the qubit wires from the tree.
    fn qubit_wires(&self) -> Vec<Wire> {
        match self {
            WireTree::Leaf(leaf) => match leaf {
                WireLeaf::Qubit(wire) | WireLeaf::OptQb(wire) => vec![*wire],
                WireLeaf::Other(_) => vec![],
            },
            WireTree::Tuple(children) | WireTree::Array(_, children) => {
                children.iter().flat_map(WireTree::qubit_wires).collect()
            }
        }
    }

    /// Given an iterator of qubit wires (in the same order as [`qubit_wires`]), update the tree
    fn update_wires(&mut self, qb_wires: &mut impl Iterator<Item = Wire>) {
        match self {
            WireTree::Leaf(leaf) => match leaf {
                WireLeaf::Qubit(p) | WireLeaf::OptQb(p) => {
                    *p = qb_wires.next().expect("Not enough ports.");
                }
                WireLeaf::Other(_) => {}
            },
            WireTree::Tuple(children) | WireTree::Array(_, children) => {
                for child in children {
                    child.update_wires(qb_wires);
                }
            }
        }
    }
}

// copied from
// https://github.com/CQCL/hugr/blob/c8090ca9089368de1a2de25e0071458ce5222d70/hugr-passes/src/monomorphize.rs#L353-L356
fn escape_dollar(str: impl AsRef<str>) -> String {
    str.as_ref().replace("$", "\\$")
}

pub(super) struct CallData {
    pub func_def: Hugr,
    pub op_nodes: Vec<Node>,
}

impl CallData {
    fn new(func_def: Hugr) -> Self {
        Self {
            func_def,
            op_nodes: vec![],
        }
    }
}
pub(super) struct BarrierFuncs {
    pub extension: Arc<Extension>,
    pub unwwrap: CallData,
    pub wrap: CallData,
    pub wrapped_barrier: HashMap<usize, CallData>,
    pub unpack_array: HashMap<(u64, Type), CallData>,
}

impl BarrierFuncs {
    const UNWRAP_OPT: OpName = OpName::new_static("__tk2_lower_option_qb_unwrap");
    const TAG_OPT: OpName = OpName::new_static("__tk2_lower_option_qb_tag");
    const WRAPPED_BARRIER: OpName = OpName::new_static("__tk2_lower_wrapped_barrier");
    const ARRAY_UNPACK: OpName = OpName::new_static("__tk2_lower_array_unpack");
    /// Signature for a function that unwraps an option type.
    pub(super) fn unwrap_opt_sig(ty: Type) -> Signature {
        Signature::new(Type::from(option_type(ty.clone())), ty)
    }

    /// Signature for a function that wraps an option type in to Some.
    pub(super) fn wrap_opt_sig(ty: Type) -> Signature {
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
            IdentList::new_static_unchecked("__tket2.barrier.temp"),
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
                ext.add_op(
                    Self::ARRAY_UNPACK,
                    Default::default(),
                    PolyFuncTypeRV::new(
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
                    ),
                    ext_ref,
                )
                .unwrap();
            },
        );

        Ok(Self {
            extension,
            unwwrap: CallData::new(unwrap_h),
            wrap: CallData::new(wrap_h),
            wrapped_barrier: HashMap::new(),
            unpack_array: HashMap::new(),
        })
    }

    fn get_op(&self, name: &OpName, args: impl Into<Vec<TypeArg>>) -> Option<ExtensionOp> {
        ExtensionOp::new(self.extension.get_op(name)?.clone(), args).ok()
    }

    fn call_unwrap(
        &mut self,
        builder: &mut impl Dataflow,
        opt_wire: Wire,
    ) -> Result<Wire, BuildError> {
        let call =
            builder.add_dataflow_op(self.get_op(&Self::UNWRAP_OPT, []).unwrap(), [opt_wire])?;
        self.unwwrap.op_nodes.push(call.node());
        Ok(call.out_wire(0))
    }

    fn call_wrap(&mut self, builder: &mut impl Dataflow, wire: Wire) -> Result<Wire, BuildError> {
        let call = builder.add_dataflow_op(self.get_op(&Self::TAG_OPT, []).unwrap(), [wire])?;
        self.wrap.op_nodes.push(call.node());
        Ok(call.out_wire(0))
    }

    fn call_hashed_func<K: std::hash::Hash + Eq + Clone>(
        builder: &mut impl Dataflow,
        in_wires: impl IntoIterator<Item = Wire>,
        op: ExtensionOp,
        func_builder: impl FnOnce(Signature) -> Result<Hugr, BuildError>,
        func_cache: &mut HashMap<K, CallData>,
        key: K,
    ) -> Result<Outputs, BuildError> {
        let sig = op.signature().deref().clone();
        let handle = builder.add_dataflow_op(op, in_wires)?;
        match func_cache.entry(key.clone()) {
            Entry::Occupied(mut ent) => ent.get_mut().op_nodes.push(handle.node()),
            Entry::Vacant(ent) => {
                let func_def: Hugr = func_builder(sig)?;
                ent.insert(CallData::new(func_def))
                    .op_nodes
                    .push(handle.node());
            }
        };

        Ok(handle.outputs())
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
        Self::call_hashed_func(
            builder,
            qubit_wires,
            op,
            |sig| {
                let mut name = Self::WRAPPED_BARRIER.to_string();
                name.push_str(&format!("${}", size));
                let mut func_b = FunctionBuilder::new(name, sig)?;
                let outs = func_b.build_wrapped_barrier(func_b.input_wires())?;
                func_b.finish_hugr_with_outputs(outs)
            },
            &mut self.wrapped_barrier,
            size,
        )
    }

    fn call_unpack_array(
        &mut self,
        builder: &mut impl Dataflow,
        array_wire: Wire,
        size: u64,
        elem_ty: Type,
    ) -> Result<Outputs, BuildError> {
        let op = self
            .get_op(
                &Self::ARRAY_UNPACK,
                [
                    size.into(),
                    elem_ty.clone().into(),
                    TypeArg::Sequence {
                        elems: vec![elem_ty.clone().into(); size as usize],
                    },
                ],
            )
            .unwrap();
        let elem_name = escape_dollar(elem_ty.to_string());
        let ty_clone = elem_ty.clone();
        Self::call_hashed_func(
            builder,
            [array_wire],
            op,
            |sig| {
                let mut name = Self::ARRAY_UNPACK.to_string();
                name.push_str(&format!("${}${}", size, elem_name));
                let mut func_b = FunctionBuilder::new(name, sig)?;
                let w = func_b.input().out_wire(0);
                let outs = super::pop_all(&mut func_b, w, size, elem_ty)?;
                func_b.finish_hugr_with_outputs(outs)
            },
            &mut self.unpack_array,
            (size, ty_clone),
        )
    }

    pub(super) fn into_calls(self) -> impl Iterator<Item = CallData> {
        [self.unwwrap, self.wrap]
            .into_iter()
            .chain(self.wrapped_barrier.into_values())
            .chain(self.unpack_array.into_values())
    }

    fn nodes_mut(&mut self) -> impl Iterator<Item = &mut Node> {
        self.unwwrap
            .op_nodes
            .iter_mut()
            .chain(self.wrap.op_nodes.iter_mut())
            .chain(
                self.wrapped_barrier
                    .values_mut()
                    .chain(self.unpack_array.values_mut())
                    .flat_map(|call| call.op_nodes.iter_mut()),
            )
    }

    /// Unpack a qubit containing type until all qubit wires are found.
    fn unpack_container(
        &mut self,
        builder: &mut impl Dataflow,
        typ: &Type,
        container_wire: Wire,
    ) -> Result<WireTree, LowerTk2Error> {
        if typ == &qb_t() {
            return Ok(WireTree::Leaf(WireLeaf::Qubit(container_wire)));
        }
        if is_opt_qb(typ) {
            let unwrapped = self.call_unwrap(builder, container_wire)?;
            return Ok(WireTree::Leaf(WireLeaf::OptQb(unwrapped)));
        }
        if let Some(ext) = typ.as_extension() {
            if let Some((n, elem_ty)) = array_args(ext) {
                // let unpacked = super::pop_all(builder, container_wire, n, elem_ty.clone())?;
                let unpacked =
                    self.call_unpack_array(builder, container_wire, n, elem_ty.clone())?;
                let children = unpacked
                    .into_iter()
                    .map(|elem_wire| self.unpack_container(builder, elem_ty, elem_wire))
                    .collect::<Result<_, _>>()?;
                let tree = WireTree::Array(elem_ty.clone(), children);
                return Ok(tree);
            }
        }
        if let Some(sum) = typ.as_sum() {
            if let Some(row) = sum.as_tuple() {
                let row: TypeRow = row.clone().try_into().expect("unexpected row variable.");
                let unpacked = builder
                    .add_dataflow_op(prelude::UnpackTuple::new(row.clone()), [container_wire])?;
                let children = row
                    .iter()
                    .enumerate()
                    .map(|(i, t)| self.unpack_container(builder, t, unpacked.out_wire(i)))
                    .collect::<Result<Vec<_>, _>>()?;
                return Ok(WireTree::Tuple(children));
            }
        }
        // No need to unpack if the type is not a qubit container.
        Ok(WireTree::Leaf(WireLeaf::Other(container_wire)))
    }
}
/// Repack a container given an updated [WireTree].
fn repack_container(
    builder: &mut impl Dataflow,
    wire_tree: &WireTree,
    barrier_funcs: &mut BarrierFuncs,
) -> Result<Wire, LowerTk2Error> {
    match wire_tree {
        WireTree::Leaf(leaf) => match leaf {
            WireLeaf::Qubit(wire) | WireLeaf::Other(wire) => Ok(*wire),
            WireLeaf::OptQb(wire) => Ok(barrier_funcs.call_wrap(builder, *wire)?),
        },
        WireTree::Array(elem_ty, children) => {
            let child_wires = children
                .iter()
                .map(|child| repack_container(builder, child, barrier_funcs))
                .collect::<Result<Vec<_>, _>>()?;
            let array_wire = builder.add_new_array(elem_ty.clone(), child_wires)?;
            Ok(array_wire)
        }
        WireTree::Tuple(children) => {
            let child_wires = children
                .iter()
                .map(|child| repack_container(builder, child, barrier_funcs))
                .collect::<Result<Vec<_>, _>>()?;
            let tuple_wire = builder.make_tuple(child_wires)?;
            Ok(tuple_wire)
        }
    }
}

/// Insert [RuntimeBarrier] after every [Barrier] in the Hugr.
pub(super) fn insert_runtime_barrier(
    hugr: &mut impl HugrMut,
    b_node: Node,
    barrier: Barrier,
    barrier_funcs: &mut BarrierFuncs,
) -> Result<(), LowerTk2Error> {
    if barrier.type_row.is_empty() {
        return Ok(());
    }
    // 1. Find all qubit containing types in the barrier.
    let filtered_qbs = filter_qubit_containers(hugr, &barrier, b_node);

    let parent = hugr.get_parent(b_node).expect("Barrier can't be root.");

    if let [(QubitContainer { typ, .. }, (targ_n, targ_p))] = filtered_qbs.as_slice() {
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
    let (qubit_containers, targets) = filtered_qbs.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
    let insert_hugr: Hugr = packing_hugr(barrier_funcs, &qubit_containers)?;

    let inserter = InsertCut {
        parent,
        targets,
        insertion: insert_hugr,
    };

    let node_map = inserter
        .apply(hugr)
        // TODO handle error
        .expect("failed to insert runtime barrier");

    // update all the extension op nodes that will be lowered later
    for node in barrier_funcs.nodes_mut() {
        let new_node = node_map.get(node).expect("node missing in map");
        *node = *new_node;
    }

    Ok(())
}

fn packing_hugr(
    barrier_funcs: &mut BarrierFuncs,
    containers: &[QubitContainer],
) -> Result<Hugr, LowerTk2Error> {
    let container_row: Vec<_> = containers
        .iter()
        .map(|container| container.typ.clone())
        .collect();
    // TODO add comments for steps
    let mut dfg_b = DFGBuilder::new(Signature::new_endo(container_row.clone()))?;
    let input = dfg_b.input();
    let mut wire_trees: Vec<WireTree> = container_row
        .iter()
        .enumerate()
        .map(|(port, ty)| barrier_funcs.unpack_container(&mut dfg_b, ty, input.out_wire(port)))
        .collect::<Result<_, _>>()?;
    let qubit_wires: Vec<_> = wire_trees
        .iter()
        .flat_map(|tree| tree.qubit_wires())
        .collect();
    // let r_bar_n = insert_wrapped_runtime_barrier(&mut dfg_b, qubit_wires)?;
    let mut r_bar_wires = barrier_funcs.call_wrapped_runtime_barrier(&mut dfg_b, qubit_wires)?;
    for tree in wire_trees.iter_mut() {
        tree.update_wires(&mut r_bar_wires);
    }
    let new_container_wires: Vec<Wire> = wire_trees
        .iter()
        .map(|tree| repack_container(&mut dfg_b, tree, barrier_funcs))
        .collect::<Result<_, _>>()?;
    Ok(dfg_b.finish_hugr_with_outputs(new_container_wires)?)
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
    fn apply(self, h: &mut impl HugrMut) -> Result<HashMap<Node, Node>, ()> {
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
    #[case(vec![qb_t(), qb_t(), bool_t()], 2, false)]
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
        if !no_parent {
            // check qubit number is expected value for row
            assert_eq!(
                num_contained_qubits(&Type::new_tuple(type_row.clone())),
                num_qb as u64
            );
        }
        // build a dfg with a generic barrier
        let (mut h, barr_n) = {
            let mut b = DFGBuilder::new(Signature::new_endo(type_row)).unwrap();

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
                .ok()
                .unwrap();
            h.single_linked_input(run_barr_func_n, 0).unwrap().0
        };

        assert_eq!(h.all_linked_inputs(run_bar_n).count(), num_qb);
    }
}
