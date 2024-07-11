//! Quick pass for removing redundant tuple pack/unpack operations.

use core::panic;

use hugr::builder::{DFGBuilder, Dataflow, DataflowHugr};
use hugr::ops::{MakeTuple, OpTrait, OpType};
use hugr::types::Type;
use hugr::{HugrView, Node};
use itertools::Itertools;

use crate::circuit::Command;
use crate::rewrite::{CircuitRewrite, Subcircuit};
use crate::Circuit;

/// Find tuple pack operations followed by tuple unpack operations
/// and generate rewrites to remove them.
pub fn find_tuple_unpack_rewrites(
    circ: &Circuit<impl HugrView>,
) -> impl Iterator<Item = CircuitRewrite> + '_ {
    circ.commands().filter_map(|cmd| make_rewrite(circ, cmd))
}

fn make_rewrite<T: HugrView>(circ: &Circuit<T>, cmd: Command<T>) -> Option<CircuitRewrite> {
    let cmd_optype = cmd.optype();
    let tuple_node = cmd.node();
    if !matches!(cmd_optype, OpType::MakeTuple(_)) {
        return None;
    }
    let tuple_types = cmd_optype
        .dataflow_signature()
        .unwrap()
        .input_types()
        .to_vec();

    // Make tuple should have a single output
    let Ok((_, wire)) = cmd.output_wires().exactly_one() else {
        panic!("MakeTuple at node {tuple_node} should have a single output wire.");
    };

    // See if it is followed by a tuple unpack
    let links = circ
        .hugr()
        .linked_inputs(tuple_node, wire.source())
        .map(|(neigh, _)| neigh)
        .collect_vec();

    if links.is_empty() {
        return None;
    }

    let unpack_nodes = links
        .iter()
        .filter(|&&neigh| match circ.hugr().get_optype(neigh) {
            OpType::UnpackTuple(_) => {
                debug_assert_eq!(
                    circ.hugr()
                        .get_optype(neigh)
                        .dataflow_signature()
                        .unwrap()
                        .output_types(),
                    tuple_types
                );
                true
            }
            _ => false,
        })
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
        tuple_node,
        unpack_nodes,
        num_other_outputs,
    ))
}

/// Returns a rewrite to remove a tuple pack operation that's followed by unpack operations,
/// and `other_tuple_links` other operations.
fn remove_pack_unpack<T: HugrView>(
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

    let mut replacement = DFGBuilder::new(subcirc.signature(circ)).unwrap();
    let mut outputs = Vec::with_capacity(num_unpack_outputs + num_other_outputs);

    // If needed, re-add the tuple pack node and connect its output to the tuple outputs.
    if num_other_outputs > 0 {
        let op = MakeTuple::new(tuple_types.to_vec().into());
        let [tuple] = replacement
            .add_dataflow_op(op, replacement.input_wires())
            .unwrap()
            .outputs_arr();
        outputs.extend(std::iter::repeat(tuple).take(num_other_outputs))
    }

    // Wire the inputs directly to the unpack outputs
    outputs.extend(replacement.input_wires().cycle().take(num_unpack_outputs));

    let replacement = replacement
        .finish_prelude_hugr_with_outputs(outputs)
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
    use hugr::extension::prelude::{BOOL_T, QB_T};
    use hugr::ops::{MakeTuple, UnpackTuple};
    use hugr::type_row;
    use hugr::types::Signature;
    use rstest::{fixture, rstest};

    /// A simple pack operation followed by an unpack operation.
    ///
    /// These can be removed entirely.
    #[fixture]
    fn simple_pack_unpack() -> Circuit {
        let mut h = DFGBuilder::new(Signature::new_endo(type_row![QB_T, BOOL_T])).unwrap();
        let mut inps = h.input_wires();
        let qb1 = inps.next().unwrap();
        let b2 = inps.next().unwrap();

        let op = MakeTuple::new(type_row![QB_T, BOOL_T]);
        let [tuple] = h.add_dataflow_op(op, [qb1, b2]).unwrap().outputs_arr();

        let op = UnpackTuple::new(type_row![QB_T, BOOL_T]);
        let [qb1, b2] = h.add_dataflow_op(op, [tuple]).unwrap().outputs_arr();

        h.finish_prelude_hugr_with_outputs([qb1, b2])
            .unwrap()
            .into()
    }

    /// A pack operation followed by two unpack operations from the same tuple.
    ///
    /// These can be removed entirely.
    #[fixture]
    fn multi_unpack() -> Circuit {
        let mut h = DFGBuilder::new(Signature::new(
            type_row![BOOL_T, BOOL_T],
            type_row![BOOL_T, BOOL_T, BOOL_T, BOOL_T],
        ))
        .unwrap();
        let mut inps = h.input_wires();
        let b1 = inps.next().unwrap();
        let b2 = inps.next().unwrap();

        let op = MakeTuple::new(type_row![BOOL_T, BOOL_T]);
        let [tuple] = h.add_dataflow_op(op, [b1, b2]).unwrap().outputs_arr();

        let op = UnpackTuple::new(type_row![BOOL_T, BOOL_T]);
        let [b1, b2] = h.add_dataflow_op(op, [tuple]).unwrap().outputs_arr();

        let op = UnpackTuple::new(type_row![BOOL_T, BOOL_T]);
        let [b3, b4] = h.add_dataflow_op(op, [tuple]).unwrap().outputs_arr();

        h.finish_prelude_hugr_with_outputs([b1, b2, b3, b4])
            .unwrap()
            .into()
    }

    /// A pack operation followed by an unpack operation, where the tuple is also returned.
    ///
    /// The unpack operation can be removed, but the pack operation cannot.
    #[fixture]
    fn partial_unpack() -> Circuit {
        let mut h = DFGBuilder::new(Signature::new(
            type_row![BOOL_T, BOOL_T],
            vec![BOOL_T, BOOL_T, Type::new_tuple(type_row![BOOL_T, BOOL_T])],
        ))
        .unwrap();
        let mut inps = h.input_wires();
        let b1 = inps.next().unwrap();
        let b2 = inps.next().unwrap();

        let op = MakeTuple::new(type_row![BOOL_T, BOOL_T]);
        let [tuple] = h.add_dataflow_op(op, [b1, b2]).unwrap().outputs_arr();

        let op = UnpackTuple::new(type_row![BOOL_T, BOOL_T]);
        let [b1, b2] = h.add_dataflow_op(op, [tuple]).unwrap().outputs_arr();

        h.finish_prelude_hugr_with_outputs([b1, b2, tuple])
            .unwrap()
            .into()
    }

    #[rstest]
    #[case::simple(simple_pack_unpack(), 1, 0)]
    #[case::multi(multi_unpack(), 1, 0)]
    #[case::partial(partial_unpack(), 1, 1)]
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
