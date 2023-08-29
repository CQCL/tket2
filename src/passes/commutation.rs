#![allow(unused)]
use hugr::{
    hugr::views::{HierarchyView, SiblingGraph},
    ops::handle::DfgID,
    Hugr, HugrView, Node, SimpleReplacement,
};

use crate::circuit::{command::Command, Circuit};

type Slice<'c> = Vec<Command<'c>>;
type SliceVec<'c> = Vec<Slice<'c>>;

fn load_slices<'c>(circ: &impl Circuit<'c>) -> SliceVec {
    todo!()
}

fn gen_rewrite<'c>(circ: &impl Circuit<'c>, commute_nodes: [Node; 2]) -> SimpleReplacement {
    todo!()
}

fn solve(mut h: Hugr) -> Result<Hugr, ()> {
    let circ: SiblingGraph<'_, DfgID> = SiblingGraph::new(&h, h.root());
    let mut slice_vec = load_slices(&circ);

    let mut slice_index: usize = 1;

    let mut done = false;
    loop {
        // keep going until reaching the end of the circuit
        let Some(current_slice) = slice_vec.get(slice_index + 1) else {
            break;
        };
        let commute_candidates: Vec<[Node; 2]> = find_candidates(current_slice, &circ);

        slice_index += 1;
    }
    Ok(h)
}

/// Return pairs of nodes, the first in the given slice, which commute.
fn find_candidates(
    current_slice: &Vec<Command<'_>>,
    circ: &SiblingGraph<'_, DfgID>,
) -> Vec<[Node; 2]> {
    current_slice
        .iter()
        .map(|command| {
            let node = command.node;

            circ.output_neighbours(node).filter_map(move |other| {
                // TODO filter to only neighbours which commutes on all
                // connecting ports
                Some([node, other])
            })
        })
        .flatten()
        .collect()
}

#[cfg(test)]
mod test {
    use crate::ops::test::t2_bell_circuit;
    use hugr::Hugr;
    use rstest::rstest;

    use super::*;

    #[rstest]
    fn commutation_simple_bell(t2_bell_circuit: Hugr) {
        solve(t2_bell_circuit).unwrap();
    }
}
