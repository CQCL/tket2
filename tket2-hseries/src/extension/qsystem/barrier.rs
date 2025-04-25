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

    false
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
    Tuple(Vec<WireTree>),
    Array(Vec<WireTree>),
}

impl WireTree {
    /// Recursively unpack the qubit wires from the tree.
    fn qubit_wires(&self) -> Vec<Wire> {
        match self {
            WireTree::Leaf(leaf) => match leaf {
                WireLeaf::Qubit(port) => vec![*port],
                WireLeaf::Other(_) => vec![],
            },
            WireTree::Tuple(children) | WireTree::Array(children) => {
                children.iter().flat_map(WireTree::qubit_wires).collect()
            }
        }
    }

    /// Given an iterator of qubit wires (in the same order as [`qubit_wires`]), update the tree
    fn update_ports(&mut self, qb_wires: &mut impl Iterator<Item = Wire>) {
        match self {
            WireTree::Leaf(leaf) => match leaf {
                WireLeaf::Qubit(p) => {
                    *p = qb_wires.next().expect("Not enough ports.");
                }
                WireLeaf::Other(_) => {}
            },
            WireTree::Tuple(children) | WireTree::Array(children) => {
                for child in children {
                    child.update_ports(qb_wires);
                }
            }
        }
    }
}

/// Unpack a qubit containing type until all qubit wires are found.
fn unpack_container<H: HugrMut>(
    _hugr: &mut H,
    b_node: Node,
    container: &QubitContainer<H>,
) -> Result<WireTree, LowerTk2Error> {
    if container.typ == qb_t() {
        return Ok(WireTree::Leaf(WireLeaf::Qubit(Wire::new(
            b_node,
            container.barrier_port,
        ))));
    }

    todo!()
}

/// Repack a container given an updated [WireTree].
fn repack_container<H: HugrMut>(
    _hugr: &mut H,
    wire_tree: &WireTree,
) -> Result<Wire, LowerTk2Error> {
    match wire_tree {
        WireTree::Leaf(leaf) => match leaf {
            WireLeaf::Qubit(wire) => Ok(*wire),
            WireLeaf::Other(_) => todo!(),
        },
        _ => todo!(),
    }
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
        .map(|qc| unpack_container(hugr, node, qc))
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
    // TODO use SimpleReplace once order bug fixed https://github.com/CQCL/hugr/issues/1974
    let parent = hugr.get_parent(node).expect("Barrier can't be root.");
    let insert_res = hugr.insert_hugr(parent, barr_hugr);
    let r_bar_n = insert_res.new_root;

    for (r_bar_port, wire) in qubit_wires.iter().enumerate() {
        hugr.connect(wire.node(), wire.source(), r_bar_n, r_bar_port);
    }

    // 5. Update the tree with wires leaving the [RuntimeBarrier].
    let mut r_bar_wires = (0..qubit_wires.len()).map(|p| Wire::new(r_bar_n, p));
    for tree in wire_trees.iter_mut() {
        tree.update_ports(&mut r_bar_wires);
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
    };
    use rstest::rstest;

    #[rstest]
    #[case(vec![qb_t(), qb_t()])]
    #[case(vec![qb_t(), qb_t(), bool_t()])]
    fn test_barrier(#[case] type_row: Vec<hugr::types::Type>) {
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

        // dfg containing runtime barrier wrapped by array construction/destruction
        let dfg = h.output_neighbours(barr_n).next().unwrap();
        assert!(h.get_optype(dfg).is_dfg());

        let r_barr = h.children(dfg).nth(3).unwrap(); // I, O, new_array, barrier
        assert!(h
            .get_optype(r_barr)
            .as_extension_op()
            .unwrap()
            .name()
            .contains(qsystem::RUNTIME_BARRIER_NAME.as_str()));
    }
}
