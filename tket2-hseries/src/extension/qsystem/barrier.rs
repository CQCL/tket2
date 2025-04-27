use hugr::builder::FunctionBuilder;
use hugr::extension::prelude::{self, option_type, UnwrapBuilder};
use hugr::hugr::hugrmut::InsertionResult;
use hugr::ops::{OpTag, ValidateOp};
use hugr::std_extensions::collections::array::{self, array_type, ArrayOpBuilder};
use hugr::types::{SumType, TypeArg, TypeRV, TypeRow};
use hugr::{
    builder::{DFGBuilder, Dataflow, DataflowHugr},
    extension::prelude::{qb_t, Barrier},
    hugr::hugrmut::HugrMut,
    types::{Signature, Type},
    HugrView, IncomingPort, Node, OutgoingPort, Wire,
};
use hugr::{ops, type_row, Hugr};

use super::{LowerTk2Error, QSystemOpBuilder};

/// A [Barrier] output port for a qubit containing type.
struct QubitContainer<H: HugrView> {
    typ: Type,
    barrier_port: OutgoingPort,
    target: (H::Node, IncomingPort),
}

/// Check if the type tree contains any qubits.
fn is_qubit_container(ty: &Type) -> bool {
    if ty == &qb_t() {
        return true;
    }

    if let Some(sum) = ty.as_sum() {
        if is_opt_qb(sum) {
            // Special case for Option[Qubit] since it is used in guppy qubit arrays.
            // Will fail if a user passes None to a barrier. expecting Option[Qubit],
            // which can be surprising if you don't know how runtime barriers work.
            // Not sure how this can be handled without runtime barrier being able to
            // take a compile time unknown number of qubits.
            return true;
        }

        if let Some(row) = sum.as_tuple() {
            return row.iter().any(|t| {
                is_qubit_container(&t.clone().try_into_type().expect("unexpected row variable."))
            });
        }
        // TODO should other sums containing qubits raise an error?
    }

    if let Some(ext) = ty.as_extension() {
        if let Some((_, elem_ty)) = array_args(ext) {
            return is_qubit_container(elem_ty);
        }
    }

    false
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

/// If a sum is an option of qubit.
fn is_opt_qb(sum: &SumType) -> bool {
    if let Some(inner) = as_unary_option(sum) {
        return inner == &qb_t();
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

/// Filter out types in the generic barrier that contain qubits.
fn filter_qubit_containers<H: HugrView>(
    hugr: &H,
    barrier: &Barrier,
    node: H::Node,
) -> Vec<QubitContainer<H>> {
    barrier
        .type_row
        .iter()
        .enumerate()
        .filter(|&(_, ty)| is_qubit_container(ty))
        .map(|(i, ty)| {
            let barrier_port = OutgoingPort::from(i);
            let target = hugr
                .single_linked_input(node, barrier_port)
                .expect("linearity violation.");
            QubitContainer {
                typ: ty.clone(),
                barrier_port,
                target,
            }
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
    Tuple(TypeRow, Vec<WireTree>),
}

impl WireTree {
    /// Recursively unpack the qubit wires from the tree.
    fn qubit_wires(&self) -> Vec<Wire> {
        match self {
            WireTree::Leaf(leaf) => match leaf {
                WireLeaf::Qubit(wire) | WireLeaf::OptQb(wire) => vec![*wire],
                WireLeaf::Other(_) => vec![],
            },
            WireTree::Tuple(_, children) | WireTree::Array(_, children) => {
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
            WireTree::Tuple(_, children) | WireTree::Array(_, children) => {
                for child in children {
                    child.update_wires(qb_wires);
                }
            }
        }
    }
}

pub(super) struct BarrierFuncs {
    pub unwrap: Node,
    pub wrap: Node,
}

impl BarrierFuncs {
    /// Signature for a function that unwraps an option type.
    pub(super) fn unwrap_opt_sig(ty: Type) -> Signature {
        Signature::new(Type::from(option_type(ty.clone())), ty)
    }

    /// Signature for a function that wraps an option type in to Some.
    pub(super) fn wrap_opt_sig(ty: Type) -> Signature {
        Signature::new(ty.clone(), Type::from(option_type(ty)))
    }

    pub(super) fn new(hugr: &mut impl HugrMut) -> Result<Self, LowerTk2Error> {
        let unwrap_h = {
            let mut b =
                FunctionBuilder::new("__tk2_lower_option_qb_unwrap", Self::unwrap_opt_sig(qb_t()))?;
            let [in_wire] = b.input_wires_arr();
            let [out_wire] = b.build_expect_sum(1, option_type(qb_t()), in_wire, |_| {
                "Value of type Option<qubit> is None so cannot apply runtime barrier to qubit."
                    .to_string()
            })?;
            b.finish_hugr_with_outputs([out_wire])?
        };

        let wrap_h = {
            let mut b =
                FunctionBuilder::new("__tk2_lower_option_qb_tag", Self::wrap_opt_sig(qb_t()))?;
            let [in_wire] = b.input_wires_arr();
            let out_wire = b.make_sum(1, vec![type_row![], vec![qb_t()].into()], [in_wire])?;
            b.finish_hugr_with_outputs([out_wire])?
        };

        debug_assert!(hugr
            .root_type()
            .validity_flags()
            .allowed_children
            .is_superset(OpTag::FuncDefn));
        let unwrap = hugr.insert_hugr(hugr.root(), unwrap_h).new_root;
        let wrap = hugr.insert_hugr(hugr.root(), wrap_h).new_root;
        Ok(Self { unwrap, wrap })
    }

    fn call_unwrap(&self, hugr: &mut impl HugrMut, opt_wire: Wire) -> Wire {
        let call = ops::Call::try_new(BarrierFuncs::unwrap_opt_sig(qb_t()).into(), [])
            .expect("simple call");
        let call_port = call.called_function_port();
        let call_n = hugr.add_node_after(opt_wire.node(), call);
        hugr.connect(opt_wire.node(), opt_wire.source(), call_n, 0);
        hugr.connect(self.unwrap, 0, call_n, call_port);
        Wire::new(call_n, 0)
    }

    fn call_wrap(&self, hugr: &mut impl HugrMut, wire: &Wire) -> Wire {
        let call =
            ops::Call::try_new(BarrierFuncs::wrap_opt_sig(qb_t()).into(), []).expect("simple call");
        let call_port = call.called_function_port();
        let call_n = hugr.add_node_after(wire.node(), call);
        hugr.connect(wire.node(), wire.source(), call_n, 0);
        hugr.connect(self.wrap, 0, call_n, call_port);
        Wire::new(call_n, 0)
    }
}

/// Unpack a qubit containing type until all qubit wires are found.
fn unpack_container<H: HugrMut>(
    hugr: &mut H,
    typ: &Type,
    container_wire: Wire,
    barrier_funcs: &BarrierFuncs,
) -> Result<WireTree, LowerTk2Error> {
    if typ == &qb_t() {
        return Ok(WireTree::Leaf(WireLeaf::Qubit(container_wire)));
    }
    if let Some(ext) = typ.as_extension() {
        if let Some((n, elem_ty)) = array_args(ext) {
            let unpacked_dfg = build_unpack_array(hugr, container_wire, n, elem_ty)?;

            let children = (0..n as usize)
                .map(|i| {
                    let elem_wire = Wire::new(unpacked_dfg, i);
                    unpack_container(hugr, elem_ty, elem_wire, barrier_funcs)
                })
                .collect::<Result<_, _>>()?;
            let tree = WireTree::Array(elem_ty.clone(), children);
            return Ok(tree);
        }
    }
    if let Some(sum) = typ.as_sum() {
        if let Some(row) = sum.as_tuple() {
            let row: TypeRow = row.clone().try_into().expect("unexpected row variable.");

            let unpacked_dfg = build_unpack_tuple(hugr, container_wire, row.clone())?;
            let children = row
                .iter()
                .enumerate()
                .map(|(i, t)| unpack_container(hugr, t, Wire::new(unpacked_dfg, i), barrier_funcs))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(WireTree::Tuple(row, children));
        }

        if is_opt_qb(sum) {
            let unwrapped = barrier_funcs.call_unwrap(hugr, container_wire);
            return Ok(WireTree::Leaf(WireLeaf::OptQb(unwrapped)));
        }
    }
    // No need to unpack if the type is not a qubit container.
    Ok(WireTree::Leaf(WireLeaf::Other(container_wire)))
}

fn build_unpack_tuple(
    hugr: &mut impl HugrMut,
    tuple_wire: Wire,
    row: TypeRow,
) -> Result<Node, LowerTk2Error> {
    let unpack_hugr = {
        let mut dfg_b = DFGBuilder::new(Signature::new(Type::new_tuple(row.clone()), row.clone()))?;
        let inp = dfg_b.input().out_wire(0);
        let unpack = dfg_b.add_dataflow_op(prelude::UnpackTuple::new(row), [inp])?;
        dfg_b.finish_hugr_with_outputs(unpack.outputs())?
    };
    let parent = hugr.get_parent(tuple_wire.node()).expect("missing parent.");

    let res = insert_hugr_with_wires(hugr, unpack_hugr, parent, [tuple_wire]);
    Ok(res.new_root)
}

fn build_unpack_array(
    hugr: &mut impl HugrMut,
    array_wire: Wire,
    n: u64,
    elem_ty: &Type,
) -> Result<Node, LowerTk2Error> {
    let unpack_hugr = {
        let mut dfg_b = DFGBuilder::new(Signature::new(
            array_type(n, elem_ty.clone()),
            vec![elem_ty.clone(); n as usize],
        ))?;
        let inp = dfg_b.input().out_wire(0);
        let unpacked_wires = super::pop_all(&mut dfg_b, inp, n, elem_ty.clone())?;
        dfg_b.finish_hugr_with_outputs(unpacked_wires)?
    };
    let parent = hugr.get_parent(array_wire.node()).expect("missing parent.");
    let res = insert_hugr_with_wires(hugr, unpack_hugr, parent, [array_wire]);

    Ok(res.new_root)
}

/// Repack a container given an updated [WireTree].
fn repack_container<H: HugrMut>(
    hugr: &mut H,
    wire_tree: &WireTree,
    barrier_parent: Node,
    barrier_funcs: &BarrierFuncs,
) -> Result<Wire, LowerTk2Error> {
    match wire_tree {
        WireTree::Leaf(leaf) => match leaf {
            WireLeaf::Qubit(wire) | WireLeaf::Other(wire) => Ok(*wire),
            WireLeaf::OptQb(wire) => Ok(barrier_funcs.call_wrap(hugr, wire)),
        },
        WireTree::Array(elem_ty, children) => {
            let child_wires = children
                .iter()
                .map(|child| repack_container(hugr, child, barrier_parent, barrier_funcs))
                .collect::<Result<Vec<_>, _>>()?;
            let array_wire = build_new_array(hugr, barrier_parent, elem_ty.clone(), child_wires)?;
            Ok(array_wire)
        }
        WireTree::Tuple(row, children) => {
            let child_wires = children
                .iter()
                .map(|child| repack_container(hugr, child, barrier_parent, barrier_funcs))
                .collect::<Result<Vec<_>, _>>()?;
            let tuple_wire = build_new_tuple(hugr, barrier_parent, row.clone(), child_wires)?;
            Ok(tuple_wire)
        }
    }
}

fn build_new_tuple(
    hugr: &mut impl HugrMut,
    barrier_parent: Node,
    row: TypeRow,
    child_wires: Vec<Wire>,
) -> Result<Wire, LowerTk2Error> {
    let pack_hugr = {
        let mut dfg_b = DFGBuilder::new(Signature::new(row.clone(), Type::new_tuple(row)))?;

        let new_tuple = dfg_b.make_tuple(dfg_b.input_wires())?;
        dfg_b.finish_hugr_with_outputs([new_tuple])?
    };
    let new_tuple = insert_hugr_with_wires(hugr, pack_hugr, barrier_parent, child_wires).new_root;
    let tuple_wire = Wire::new(new_tuple, 0);
    Ok(tuple_wire)
}

fn build_new_array(
    hugr: &mut impl HugrMut,
    barrier_parent: Node,
    elem_ty: Type,
    elem_wires: Vec<Wire>,
) -> Result<Wire, LowerTk2Error> {
    let size = elem_wires.len();
    let pack_hugr = {
        let mut dfg_b = DFGBuilder::new(Signature::new(
            vec![elem_ty.clone(); size],
            array::array_type(size as u64, elem_ty.clone()),
        ))?;

        let new_arr = dfg_b.add_new_array(elem_ty, dfg_b.input_wires())?;
        dfg_b.finish_hugr_with_outputs([new_arr])?
    };
    let new_array = insert_hugr_with_wires(hugr, pack_hugr, barrier_parent, elem_wires).new_root;
    let array_wire = Wire::new(new_array, 0);
    Ok(array_wire)
}

fn insert_hugr_with_wires(
    base_hugr: &mut impl HugrMut,
    new_hugr: Hugr,
    parent: Node,
    in_wires: impl IntoIterator<Item = Wire>,
) -> InsertionResult {
    let res = base_hugr.insert_hugr(parent, new_hugr);

    for (in_port, wire) in in_wires.into_iter().enumerate() {
        base_hugr.connect(wire.node(), wire.source(), res.new_root, in_port);
    }

    res
}
/// Insert [RuntimeBarrier] after every [Barrier] in the Hugr.
pub(super) fn insert_runtime_barrier(
    hugr: &mut impl HugrMut,
    b_node: Node,
    barrier: Barrier,
    barrier_funcs: &BarrierFuncs,
) -> Result<(), LowerTk2Error> {
    if barrier.type_row.is_empty() {
        return Ok(());
    }
    // 1. Find all qubit containing types in the barrier.
    let qubit_containers = filter_qubit_containers(hugr, &barrier, b_node);

    let parent = hugr.get_parent(b_node).expect("Barrier can't be root.");

    if let [QubitContainer {
        typ,
        barrier_port,
        target: (targ_n, targ_p),
    }] = qubit_containers.as_slice()
    {
        // If the barrier is over a single array of qubits
        // we can insert a runtime barrier op directly.
        let shortcut = typ
            .as_extension()
            .and_then(|ext| array_args(ext))
            .and_then(|(size, elem_ty)| (elem_ty == &qb_t()).then_some(size))
            .map(|size| {
                let bar_out = Wire::new(b_node, *barrier_port);
                let r_bar_n = insert_runtime_barrier_op(hugr, parent, bar_out, size)?;
                hugr.disconnect(*targ_n, *targ_p);
                hugr.connect(r_bar_n, 0, *targ_n, *targ_p);
                Ok(())
            });
        if let Some(res) = shortcut {
            return res;
        }
    }

    // 2. Unpack qubits from wires leaving the [Barrier] and record in a tree.
    let mut wire_trees: Vec<WireTree> = qubit_containers
        .iter()
        .map(|qc| {
            unpack_container(
                hugr,
                &qc.typ,
                Wire::new(b_node, qc.barrier_port),
                barrier_funcs,
            )
        })
        .collect::<Result<_, _>>()?;

    // 3. Collect all qubit wires from the trees.
    let qubit_wires: Vec<_> = wire_trees
        .iter()
        .flat_map(|tree| tree.qubit_wires())
        .collect();

    let n_qbs = qubit_wires.len();

    // 4. Insert a new [RuntimeBarrier] with the qubit wires as inputs.

    let r_bar_n = insert_wrapped_runtime_barrier(hugr, parent, qubit_wires)?;

    // 5. Update the tree with wires leaving the [RuntimeBarrier].
    let mut r_bar_wires = (0..n_qbs).map(|p| Wire::new(r_bar_n, p));
    for tree in wire_trees.iter_mut() {
        tree.update_wires(&mut r_bar_wires);
    }

    // 6. Repack the containers with the new wires.
    let new_container_wires: Vec<Wire> = wire_trees
        .iter()
        .map(|tree| repack_container(hugr, tree, parent, barrier_funcs))
        .collect::<Result<_, _>>()?;

    // 7. Disconnect the targets of the original wire and connect the new container wires.
    for (old_container, new_wire) in qubit_containers.into_iter().zip(new_container_wires) {
        let (targ_n, targ_p) = old_container.target;
        hugr.disconnect(targ_n, targ_p);
        hugr.connect(new_wire.node(), new_wire.source(), targ_n, targ_p);
    }

    Ok(())
}

fn insert_wrapped_runtime_barrier(
    hugr: &mut impl HugrMut,
    parent: Node,
    qubit_wires: Vec<Wire>,
) -> Result<Node, LowerTk2Error> {
    let barr_hugr = {
        let mut barr_builder =
            DFGBuilder::new(Signature::new_endo(vec![qb_t(); qubit_wires.len()]))?;
        let outs = barr_builder.build_wrapped_barrier(barr_builder.input_wires())?;
        barr_builder.finish_hugr_with_outputs(outs)?
    };
    let res = insert_hugr_with_wires(hugr, barr_hugr, parent, qubit_wires);
    Ok(res.new_root)
}

fn insert_runtime_barrier_op(
    hugr: &mut impl HugrMut,
    parent: Node,
    in_wire: Wire,
    array_size: u64,
) -> Result<Node, LowerTk2Error> {
    let barr_hugr = {
        let mut barr_builder =
            DFGBuilder::new(Signature::new_endo(array_type(array_size, qb_t())))?;
        let array_wire = barr_builder.input().out_wire(0);
        let out = barr_builder.add_runtime_barrier(array_wire, array_size)?;
        barr_builder.finish_hugr_with_outputs([out])?
    };
    let res = insert_hugr_with_wires(hugr, barr_hugr, parent, [in_wire]);
    Ok(res.new_root)
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
    use itertools::Itertools;
    use rstest::rstest;

    fn opt_q_arr(size: u64) -> Type {
        array_type(size, option_type(qb_t()).into())
    }

    #[rstest]
    #[case(vec![qb_t(), qb_t()], 2)]
    #[case(vec![qb_t(), qb_t(), bool_t()], 2)]
    #[case(vec![qb_t(), opt_q_arr(2)], 3)]
    #[case(vec![array_type(2, bool_t())], 0)]
    // special case, single array of qubits is passed directly to op without unpacking
    #[case(vec![array_type(3, qb_t())], 1)]
    #[case(vec![qb_t(), array_type(2, qb_t()), array_type(2, array_type(2, qb_t()))], 7)]
    #[case(vec![Type::new_tuple(vec![bool_t(), qb_t()]), qb_t()], 2)]
    #[case(vec![Type::new_tuple(vec![bool_t(), qb_t(), opt_q_arr(2)]), qb_t()], 4)]
    #[case(vec![Type::new_tuple(vec![bool_t(), qb_t(), array_type(2, Type::new_tuple(vec![bool_t(), qb_t()]))]), qb_t()], 4)]
    fn test_barrier(#[case] type_row: Vec<hugr::types::Type>, #[case] num_qb: usize) {
        // build a dfg with a generic barrier
        use hugr::{
            ops::{handle::NodeHandle, NamedOp},
            HugrView,
        };

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

        let run_barr_n = h
            .nodes()
            .filter(|&r_barr_n| {
                h.get_optype(r_barr_n)
                    .as_extension_op()
                    .is_some_and(|op| op.name().contains(qsystem::RUNTIME_BARRIER_NAME.as_str()))
            })
            .exactly_one()
            .ok()
            .unwrap();

        let run_bar_dfg = h.get_parent(run_barr_n).unwrap();

        assert!(h.get_optype(run_bar_dfg).is_dfg());

        assert_eq!(h.all_linked_inputs(run_bar_dfg).count(), num_qb);
    }
}
