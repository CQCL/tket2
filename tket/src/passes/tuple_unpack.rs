//! Quick pass for removing redundant tuple pack/unpack operations.

use core::panic;

use hugr::builder::{DFGBuilder, Dataflow, DataflowHugr};
use hugr::extension::prelude::{MakeTuple, UnpackTuple};
use hugr::ops::{OpTrait, OpType};
use hugr::types::Type;
use hugr::{HugrView, Node};
use itertools::Itertools;

use crate::rewrite::{CircuitRewrite, Subcircuit};
use crate::Circuit;

/// Find tuple pack operations followed by tuple unpack operations
/// and generate rewrites to remove them.
pub fn find_tuple_unpack_rewrites(
    circ: &Circuit<impl HugrView<Node = Node>>,
) -> impl Iterator<Item = CircuitRewrite> + '_ {
    circ.hugr()
        .entry_descendants()
        .filter_map(|node| make_rewrite(circ, node))
}

/// Casts the optype to a MakeTuple operation, if possible.
fn as_make_tuple(optype: &OpType) -> Option<MakeTuple> {
    let ext_op = optype.as_extension_op()?;
    ext_op.cast::<MakeTuple>()
}

/// Casts the optype to an UnpackTuple operation, if possible.
fn as_unpack_tuple(optype: &OpType) -> Option<UnpackTuple> {
    let ext_op = optype.as_extension_op()?;
    ext_op.cast::<UnpackTuple>()
}

fn make_rewrite<T: HugrView<Node = Node>>(circ: &Circuit<T>, node: Node) -> Option<CircuitRewrite> {
    let optype = circ.hugr().get_optype(node);

    if let Some(unpack) = as_unpack_tuple(optype) {
        // Special case for unpack tuples with empty tuple types,
        // we can just remove the unpack operation even if the predecessor is not a make tuple.
        return remove_empty_unpack(circ, node, unpack);
    }
    let _ = as_make_tuple(optype)?;

    let tuple_types = optype.dataflow_signature().unwrap().input_types().to_vec();

    // See if it is followed by a tuple unpack
    let links = circ
        .hugr()
        .linked_inputs(node, 0)
        .map(|(neigh, _)| neigh)
        .collect_vec();

    if links.is_empty() {
        // Remove the make tuple operation
        return Some(remove_pack_unpack(circ, &tuple_types, node, vec![], 0));
    }

    let unpack_nodes = links
        .iter()
        .filter(|&&neigh| as_unpack_tuple(circ.hugr().get_optype(neigh)).is_some())
        .copied()
        .collect_vec();

    if unpack_nodes.is_empty() {
        return None;
    }

    // Remove all unpack operations, but only remove the pack operation if all neighbours are unpacks.
    let num_other_outputs = links.len() - unpack_nodes.len();
    Some(remove_pack_unpack(
        circ,
        &tuple_types,
        node,
        unpack_nodes,
        num_other_outputs,
    ))
}

/// Special case for unpack tuples with empty tuple types,
/// we can just remove the unpack operation even if the predecessor is not a make tuple.
fn remove_empty_unpack(
    circ: &Circuit<impl HugrView<Node = Node>>,
    node: Node,
    unpack: UnpackTuple,
) -> Option<CircuitRewrite> {
    // We only process empty tuple unpacks here.
    if !unpack.0.is_empty() {
        return None;
    }
    let (predecessor, _) = circ.hugr().single_linked_output(node, 0)?;
    let predecessor_optype = circ.hugr().get_optype(predecessor);
    if as_make_tuple(predecessor_optype).is_some() {
        // If the predecessor is a make tuple, we'd have had generated a removal rewrite from there already.
        return None;
    };

    // If the predecessor is a tag operation with only this node as successor,
    // we can remove both the tag and the unpack operation. Otherwise, remove
    // only the unpack operation.
    let nodes = if predecessor_optype.is_tag()
        && circ.hugr().single_linked_input(predecessor, 0).is_some()
    {
        vec![node, predecessor]
    } else {
        vec![node]
    };
    let subgraph = Subcircuit::try_from_nodes(nodes, circ).unwrap();
    let replacement = DFGBuilder::new(subgraph.signature(circ)).unwrap();
    let replacement = replacement.finish_hugr_with_outputs([]).unwrap();

    Some(
        subgraph
            .create_rewrite(circ, replacement.into())
            .unwrap_or_else(|e| {
                panic!("Failed to create rewrite for removing tuple pack/unpack operations. {e}")
            }),
    )
}

/// Returns a rewrite to remove a tuple pack operation that's followed by unpack operations,
/// and `other_tuple_links` other operations.
fn remove_pack_unpack<T: HugrView<Node = Node>>(
    circ: &Circuit<T>,
    tuple_types: &[Type],
    pack_node: Node,
    unpack_nodes: Vec<Node>,
    num_other_outputs: usize,
) -> CircuitRewrite {
    let num_unpack_outputs = tuple_types.len() * unpack_nodes.len();

    let mut nodes = unpack_nodes;
    nodes.push(pack_node);
    let subcirc = Subcircuit::try_from_nodes(nodes, circ).unwrap();
    let subcirc_signature = subcirc.signature(circ);

    // The output port order in `Subcircuit::try_from_nodes` is not too well defined.
    // Check that the outputs are in the expected order.
    debug_assert!(
        itertools::equal(
            subcirc_signature.output().iter(),
            tuple_types
                .iter()
                .cycle()
                .take(num_unpack_outputs)
                .chain(itertools::repeat_n(
                    &Type::new_tuple(tuple_types.to_vec()),
                    num_other_outputs
                ))
        ),
        "Unpacked tuple values must come before tupled values"
    );

    let mut replacement = DFGBuilder::new(subcirc_signature).unwrap();
    let mut outputs = Vec::with_capacity(num_unpack_outputs + num_other_outputs);

    // Wire the inputs directly to the unpack outputs
    outputs.extend(replacement.input_wires().cycle().take(num_unpack_outputs));

    // If needed, re-add the tuple pack node and connect its output to the tuple outputs.
    if num_other_outputs > 0 {
        let op = MakeTuple::new(tuple_types.to_vec().into());
        let [tuple] = replacement
            .add_dataflow_op(op, replacement.input_wires())
            .unwrap()
            .outputs_arr();
        outputs.extend(std::iter::repeat_n(tuple, num_other_outputs))
    }

    let replacement = replacement
        .finish_hugr_with_outputs(outputs)
        .unwrap_or_else(|e| {
            panic!("Failed to create replacement for removing tuple pack/unpack operations. {e}")
        })
        .into();

    subcirc
        .create_rewrite(circ, replacement)
        .unwrap_or_else(|e| {
            panic!("Failed to create rewrite for removing tuple pack/unpack operations. {e}")
        })
}

#[cfg(test)]
mod test {
    use super::*;
    use hugr::builder::FunctionBuilder;
    use hugr::extension::prelude::{bool_t, qb_t, UnpackTuple};

    use hugr::ops::Tag;
    use hugr::type_row;
    use hugr::types::Signature;
    use rstest::{fixture, rstest};

    /// A simple pack operation followed by an unpack operation.
    ///
    /// These can be removed entirely.
    #[fixture]
    fn simple_pack_unpack() -> Circuit {
        let mut h = FunctionBuilder::new(
            "simple_pack_unpack",
            Signature::new_endo(vec![qb_t(), bool_t()]),
        )
        .unwrap();
        let mut inps = h.input_wires();
        let qb1 = inps.next().unwrap();
        let b2 = inps.next().unwrap();

        let tuple = h.make_tuple([qb1, b2]).unwrap();

        let op = UnpackTuple::new(vec![qb_t(), bool_t()].into());
        let [qb1, b2] = h.add_dataflow_op(op, [tuple]).unwrap().outputs_arr();

        h.finish_hugr_with_outputs([qb1, b2]).unwrap().into()
    }

    /// A pack operation followed by two unpack operations from the same tuple.
    ///
    /// These can be removed entirely.
    #[fixture]
    fn multi_unpack() -> Circuit {
        let mut h = FunctionBuilder::new(
            "multi_unpack",
            Signature::new(
                vec![bool_t(), bool_t()],
                vec![bool_t(), bool_t(), bool_t(), bool_t()],
            ),
        )
        .unwrap();
        let mut inps = h.input_wires();
        let b1 = inps.next().unwrap();
        let b2 = inps.next().unwrap();

        let tuple = h.make_tuple([b1, b2]).unwrap();

        let op = UnpackTuple::new(vec![bool_t(), bool_t()].into());
        let [b1, b2] = h.add_dataflow_op(op, [tuple]).unwrap().outputs_arr();

        let op = UnpackTuple::new(vec![bool_t(), bool_t()].into());
        let [b3, b4] = h.add_dataflow_op(op, [tuple]).unwrap().outputs_arr();

        h.finish_hugr_with_outputs([b1, b2, b3, b4]).unwrap().into()
    }

    /// A pack operation followed by an unpack operation, where the tuple is also returned.
    ///
    /// The unpack operation can be removed, but the pack operation cannot.
    #[fixture]
    fn partial_unpack() -> Circuit {
        let mut h = FunctionBuilder::new(
            "partial_unpack",
            Signature::new(
                vec![bool_t(), bool_t()],
                vec![
                    bool_t(),
                    bool_t(),
                    Type::new_tuple(vec![bool_t(), bool_t()]),
                ],
            ),
        )
        .unwrap();
        let mut inps = h.input_wires();
        let b1 = inps.next().unwrap();
        let b2 = inps.next().unwrap();

        let tuple = h.make_tuple([b1, b2]).unwrap();

        let op = UnpackTuple::new(vec![bool_t(), bool_t()].into());
        let [b1, b2] = h.add_dataflow_op(op, [tuple]).unwrap().outputs_arr();

        h.finish_hugr_with_outputs([b1, b2, tuple]).unwrap().into()
    }

    /// A tag operation followed by an unpack of an empty tuple.
    ///
    /// Both should be removed.
    #[fixture]
    fn empty_tag_unpack() -> Circuit {
        let mut h =
            FunctionBuilder::new("empty_tag_unpack", Signature::new(vec![], vec![])).unwrap();

        let tuple = h
            .add_dataflow_op(Tag::new(0, vec![type_row![]]), [])
            .unwrap()
            .out_wire(0);
        let op = UnpackTuple::new(vec![].into());
        h.add_dataflow_op(op, [tuple]).unwrap();

        h.finish_hugr_with_outputs([]).unwrap().into()
    }

    /// A tag operation followed by an unpack of an empty tuple, but also connected to the output node.
    ///
    /// Only the unpack operation should be removed.
    #[fixture]
    fn empty_unpack() -> Circuit {
        let mut h = FunctionBuilder::new(
            "empty_unpack",
            Signature::new(vec![], vec![Type::new_tuple(type_row![])]),
        )
        .unwrap();

        let tuple = h
            .add_dataflow_op(Tag::new(0, vec![type_row![]]), [])
            .unwrap()
            .out_wire(0);
        let op = UnpackTuple::new(vec![].into());
        h.add_dataflow_op(op, [tuple]).unwrap();

        h.finish_hugr_with_outputs([tuple]).unwrap().into()
    }

    #[rstest]
    #[case::simple(simple_pack_unpack(), 1, 0)]
    #[case::multi(multi_unpack(), 1, 0)]
    #[case::partial(partial_unpack(), 1, 1)]
    #[case::empty_tag_unpack(empty_tag_unpack(), 1, 0)]
    #[case::empty_unpack(empty_unpack(), 1, 1)]
    fn test_pack_unpack(
        #[case] mut circ: Circuit,
        #[case] expected_rewrites: usize,
        #[case] remaining_commands: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut num_rewrites = 0;
        loop {
            let Some(rewrite) = find_tuple_unpack_rewrites(&circ).next() else {
                break;
            };
            num_rewrites += 1;
            rewrite.apply(&mut circ)?;
        }
        assert_eq!(num_rewrites, expected_rewrites);

        assert_eq!(circ.commands().count(), remaining_commands);

        Ok(())
    }
}
