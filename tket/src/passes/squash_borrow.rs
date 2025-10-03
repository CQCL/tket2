//! Reorder and squash pairs of borrow and return nodes where possible.

pub mod analysis;

pub use analysis::BorrowAnalysis;
use analysis::BorrowOrReturn;
use hugr_core::{hugr::hugrmut::HugrMut, Node, Wire};

use std::collections::HashMap;

fn optimize_one_array<H: HugrMut<Node = Node>>(
    hugr: &mut H,
    source: Wire,
    nodes: impl Iterator<Item = (Node, BorrowOrReturn, u64)>,
) {
    let mut last_array_outport = source;
    let mut borrowed: HashMap<u64, (Node, Option<Node>)> = HashMap::new(); // index -> (borrow_node, optional return_node)
    let emit = |hugr: &mut H, node, borrow_from_in, borrow_from_out| {
        hugr.disconnect(last_array_outport.node(), last_array_outport.source());
        hugr.disconnect(node, borrow_from_in);
        hugr.connect(
            last_array_outport.node(),
            last_array_outport.source(),
            node,
            borrow_from_in,
        );
        last_array_outport = Wire::new(node, borrow_from_out);
    };
    for (node, borrow_or_return, index) in nodes {
        match (borrow_or_return, borrowed.get(&index)) {
            (BorrowOrReturn::Borrow, None) => {
                // initial borrow - record and emit
                borrowed.insert(index, (node, None));
                emit(hugr, node, borrow_from_in, borrow_from_out);
            }

            // "interesting" case - return after borrow - record but do not emit (yet)
            (BorrowOrReturn::Return, Some((prev_borrow, None))) => {
                borrowed.insert(index, (*prev_borrow, Some(node)));
            }

            (BorrowOrReturn::Borrow, Some((prev_borrow, Some(prev_return)))) => {
                // The interesting case...elide!
                let tgt = hugr
                    .single_linked_input(node, borrowed_in_port)
                    .expect("linear");
                hugr.connect(prev_borrow, prev_borrow_port, tgt.0, tgt.1);
                hugr.remove_node(prev_return);
                hugr.remove_node(node);
                borrowed.insert(index, (prev_borrow, None));
            }

            // The following should all have been ruled out by analysis earlier:
            (BorrowOrReturn::Borrow, Some((prev_borrow, None))) => panic!("Double borrow"),
            (BorrowOrReturn::Return, None) => panic!("Return without borrow"),
            (BorrowOrReturn::Return, Some((_, Some(_)))) => panic!("Double return"),
        }
    }
    // Finally....
    for (borrow, opt_return) in borrowed.values() {
        let return_node = opt_return.unwrap(); // analysis should have ensured this will work
        emit(hugr, return_node, borrow_from_in, borrow_from_out);
    }
}
