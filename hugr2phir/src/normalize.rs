use hugr::builder::Dataflow;
use hugr::builder::DataflowHugr;
use hugr::HugrView;
use hugr::SimpleReplacement;

use hugr::hugr::hugrmut::HugrMut;
use hugr::hugr::views::SiblingSubgraph;
use hugr::ops::OpType;
use itertools::Itertools;
use tket2::extension::REGISTRY;

use hugr::ops::LeafOp;

use hugr::types::FunctionType;

use hugr::builder::DFGBuilder;

use hugr::types::TypeRow;

use hugr::Hugr;

fn identity_dfg(type_combination: TypeRow) -> Hugr {
    let identity_build = DFGBuilder::new(FunctionType::new(
        type_combination.clone(),
        type_combination,
    ))
    .unwrap();
    let inputs = identity_build.input_wires();
    identity_build
        .finish_hugr_with_outputs(inputs, &REGISTRY)
        .unwrap()
}

fn find_make_unmake(hugr: &impl HugrView) -> impl Iterator<Item = SimpleReplacement> + '_ {
    hugr.nodes().filter_map(|n| {
        let op = hugr.get_optype(n);

        let OpType::LeafOp(LeafOp::MakeTuple { tys }) = op else {
            return None;
        };

        let Ok(neighbour) = hugr.output_neighbours(n).exactly_one() else {
            return None;
        };

        let OpType::LeafOp(LeafOp::UnpackTuple { .. }) = hugr.get_optype(neighbour) else {
            return None;
        };

        let sibling_graph = SiblingSubgraph::try_from_nodes([n, neighbour], hugr)
            .expect("Make unmake should be valid subgraph.");

        let replacement = identity_dfg(tys.clone());
        sibling_graph
            .create_simple_replacement(hugr, replacement)
            .ok()
    })
}

/// Remove any pairs of MakeTuple immediately followed by UnpackTuple (an
/// identity operation)
pub(crate) fn remove_identity_tuples(circ: &mut impl HugrMut) {
    let rewrites: Vec<_> = find_make_unmake(circ).collect();
    // should be able to apply all in parallel unless there are copies...

    for rw in rewrites {
        circ.apply_rewrite(rw).unwrap();
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use hugr::extension::prelude::BOOL_T;
    use hugr::extension::prelude::QB_T;
    use hugr::type_row;
    use hugr::HugrView;

    fn make_unmake_tuple(type_combination: TypeRow) -> Hugr {
        let mut b = DFGBuilder::new(FunctionType::new(
            type_combination.clone(),
            type_combination.clone(),
        ))
        .unwrap();
        let input_wires = b.input_wires();

        let tuple = b
            .add_dataflow_op(
                LeafOp::MakeTuple {
                    tys: type_combination.clone(),
                },
                input_wires,
            )
            .unwrap();

        let unpacked = b
            .add_dataflow_op(
                LeafOp::UnpackTuple {
                    tys: type_combination,
                },
                tuple.outputs(),
            )
            .unwrap();

        b.finish_hugr_with_outputs(unpacked.outputs(), &REGISTRY)
            .unwrap()
    }
    #[test]
    fn test_remove_id_tuple() {
        let mut h = make_unmake_tuple(type_row![QB_T, BOOL_T]);

        assert_eq!(h.node_count(), 5);

        remove_identity_tuples(&mut h);

        assert_eq!(h.node_count(), 3);
    }
}
