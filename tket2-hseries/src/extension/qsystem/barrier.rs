use hugr::hugr::hugrmut::InsertionResult;
use hugr::std_extensions::collections::array::{self, ArrayOpBuilder};
use hugr::types::{TypeArg, TypeRow};
use hugr::Hugr;
use hugr::{
    builder::{DFGBuilder, Dataflow, DataflowHugr},
    extension::prelude::{qb_t, Barrier},
    hugr::hugrmut::HugrMut,
    types::{Signature, Type},
    HugrView, IncomingPort, Node, OutgoingPort, Wire,
};

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
        if let Some(row) = sum.as_tuple() {
            return row.iter().any(|t| {
                is_qubit_container(&t.clone().try_into_type().expect("unexpected row variable."))
            });
        }
    }

    if let Some(ext) = ty.as_extension() {
        if let Some((_, elem_ty)) = array_args(ext) {
            return is_qubit_container(&elem_ty);
        }
    }

    false
}

/// If a custom type is an array, return size and element type.
fn array_args(ext: &hugr::types::CustomType) -> Option<(u64, Type)> {
    array::array_type_def()
        .check_custom(ext)
        .ok()
        .and_then(|_| match ext.args() {
            [TypeArg::BoundedNat { n }, TypeArg::Type { ty: elem_ty }] => {
                Some((*n, elem_ty.clone()))
            }
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
                WireLeaf::Qubit(port) => vec![*port],
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
                WireLeaf::Qubit(p) => {
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

/// Unpack a qubit containing type until all qubit wires are found.
fn unpack_container<H: HugrMut>(
    hugr: &mut H,
    typ: &Type,
    container_wire: Wire,
) -> Result<WireTree, LowerTk2Error> {
    if typ == &qb_t() {
        return Ok(WireTree::Leaf(WireLeaf::Qubit(container_wire)));
    }
    if let Some(ext) = typ.as_extension() {
        if let Some((n, elem_ty)) = array_args(ext) {
            let unpack_hugr = {
                let mut dfg_b = DFGBuilder::new(Signature::new(
                    typ.clone(),
                    vec![elem_ty.clone(); n as usize],
                ))?;
                let inp = dfg_b.input().out_wire(0);
                let unpacked_wires = super::pop_all(&mut dfg_b, inp, n, elem_ty.clone())?;
                dfg_b.finish_hugr_with_outputs(unpacked_wires)?
            };

            let parent = hugr
                .get_parent(container_wire.node())
                .expect("missing parent.");
            let unpacked_dfg =
                insert_hugr_with_wires(hugr, unpack_hugr, parent, &[container_wire]).new_root;

            return Ok(WireTree::Array(
                elem_ty.clone(),
                (0..n as usize)
                    .map(|i| unpack_container(hugr, &elem_ty, Wire::new(unpacked_dfg, i)))
                    .collect::<Result<Vec<_>, _>>()?,
            ));
        }
    }
    if let Some(sum) = typ.as_sum() {
        if let Some(_row) = sum.as_tuple() {
            todo!()
        }
    }
    Ok(WireTree::Leaf(WireLeaf::Other(container_wire)))
}

/// Repack a container given an updated [WireTree].
fn repack_container<H: HugrMut>(hugr: &mut H, wire_tree: &WireTree) -> Result<Wire, LowerTk2Error> {
    match wire_tree {
        WireTree::Leaf(leaf) => match leaf {
            WireLeaf::Qubit(wire) | WireLeaf::Other(wire) => Ok(*wire),
        },
        WireTree::Array(elem_ty, children) => {
            if children.is_empty() {
                panic!("Empty array.")
                // TODO return an empty unit array wire.
            }
            let child_wires = children
                .iter()
                .map(|child| repack_container(hugr, child))
                .collect::<Result<Vec<_>, _>>()?;
            let size = child_wires.len();
            let pack_hugr = {
                let mut dfg_b = DFGBuilder::new(Signature::new(
                    vec![elem_ty.clone(); size],
                    array::array_type(size as u64, elem_ty.clone()),
                ))?;

                let new_arr = dfg_b.add_new_array(elem_ty.clone(), dfg_b.input_wires())?;
                dfg_b.finish_hugr_with_outputs([new_arr])?
            };

            let parent = hugr
                .get_parent(child_wires[0].node())
                .expect("missing parent.");
            let new_array = insert_hugr_with_wires(hugr, pack_hugr, parent, &child_wires).new_root;
            Ok(Wire::new(new_array, 0))
        }
        WireTree::Tuple(_, _) => todo!(),
    }
}

fn insert_hugr_with_wires(
    base_hugr: &mut impl HugrMut,
    new_hugr: Hugr,
    parent: Node,
    in_wires: &[Wire],
) -> InsertionResult {
    let res = base_hugr.insert_hugr(parent, new_hugr);

    for (in_port, wire) in in_wires.iter().enumerate() {
        base_hugr.connect(wire.node(), wire.source(), res.new_root, in_port);
    }

    res
}
/// Insert [RuntimeBarrier] after every [Barrier] in the Hugr.
pub(super) fn insert_runtime_barrier(
    hugr: &mut impl HugrMut,
    node: Node,
    barrier: Barrier,
) -> Result<(), LowerTk2Error> {
    // 1. Find all qubit containing types in the barrier.
    let qubit_containers = filter_qubit_containers(hugr, &barrier, node);

    // 2. Unpack qubits from wires leaving the [Barrier] and record in a tree.
    let mut wire_trees: Vec<WireTree> = qubit_containers
        .iter()
        .map(|qc| unpack_container(hugr, &qc.typ, Wire::new(node, qc.barrier_port)))
        .collect::<Result<Vec<_>, _>>()?;

    // 3. Collect all qubit wires from the trees.
    let qubit_wires = wire_trees
        .iter()
        .flat_map(|tree| tree.qubit_wires())
        .collect::<Vec<_>>();

    // 4. Insert a new [RuntimeBarrier] with the qubit wires as inputs.
    let barr_hugr = {
        let mut barr_builder =
            DFGBuilder::new(Signature::new_endo(vec![qb_t(); qubit_wires.len()]))?;
        let outs = barr_builder.build_wrapped_barrier(barr_builder.input_wires())?;
        barr_builder.finish_hugr_with_outputs(outs)?
    };
    let parent = hugr.get_parent(node).expect("Barrier can't be root.");
    let r_bar_n = insert_hugr_with_wires(hugr, barr_hugr, parent, &qubit_wires).new_root;

    // 5. Update the tree with wires leaving the [RuntimeBarrier].
    let mut r_bar_wires = (0..qubit_wires.len()).map(|p| Wire::new(r_bar_n, p));
    for tree in wire_trees.iter_mut() {
        tree.update_wires(&mut r_bar_wires);
    }

    // 6. Repack the containers with the new wires.
    let new_container_wires: Vec<Wire> = wire_trees
        .iter()
        .map(|tree| repack_container(hugr, tree))
        .collect::<Result<Vec<_>, _>>()?;

    // 7. Disconnect the targets of the original wire and connect the new container wires.
    for (old_container, new_wire) in qubit_containers.into_iter().zip(new_container_wires) {
        let (targ_n, targ_p) = old_container.target;
        hugr.disconnect(targ_n, targ_p);
        hugr.connect(new_wire.node(), new_wire.source(), targ_n, targ_p);
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::extension::qsystem::{self, lower_tk2_op};
    use hugr::{
        builder::DFGBuilder,
        extension::prelude::{bool_t, qb_t},
        std_extensions::collections::array::array_type,
    };
    use itertools::Itertools;
    use rstest::rstest;

    #[rstest]
    #[case(vec![qb_t(), qb_t()], 2)]
    #[case(vec![qb_t(), qb_t(), bool_t()], 2)]
    #[case(vec![qb_t(), array_type(2, qb_t())], 3)]
    #[case(vec![array_type(2, bool_t())], 0)]
    #[case(vec![array_type(3, qb_t())], 3)]
    #[case(vec![qb_t(), array_type(2, qb_t()), array_type(2, array_type(2, qb_t()))], 7)]
    fn test_barrier(#[case] type_row: Vec<hugr::types::Type>, #[case] num_qb: usize) {
        // build a dfg with a barrier

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

        assert_eq!(h.all_linked_inputs(run_bar_dfg).count(), num_qb);
    }
}
