//! Reorder and squash pairs of borrow and return nodes where possible.

pub mod analysis;

pub use analysis::BorrowAnalysis;
use analysis::BorrowOrReturn;
use hugr::hugr::hugrmut::HugrMut;
use hugr::{IncomingPort, Node, OutgoingPort, Wire};

use std::collections::hash_map::Entry;
use std::collections::HashMap;

use crate::passes::squash_borrow::analysis::BorrowAction;

#[derive(Clone, Debug)]
struct BorrowFromPorts {
    inc: IncomingPort,
    out: OutgoingPort,
}

pub fn optimize_one_array<H: HugrMut<Node = Node>>(
    hugr: &mut H,
    source: Wire,
    nodes: impl Iterator<Item = BorrowAction>,
) {
    let mut last_array_outport = source;
    struct Borrow(Node, OutgoingPort);
    struct Return(Node, BorrowFromPorts);
    let mut borrowed: HashMap<u64, (Borrow, Option<Return>)> = HashMap::new();
    let mut emit = |hugr: &mut H, node, borrow_from: BorrowFromPorts| {
        hugr.disconnect(last_array_outport.node(), last_array_outport.source());
        hugr.disconnect(node, borrow_from.inc);
        hugr.connect(
            last_array_outport.node(),
            last_array_outport.source(),
            node,
            borrow_from.inc,
        );
        last_array_outport = Wire::new(node, borrow_from.out);
    };
    for BorrowAction {
        node,
        borrow_index_const: index,
        action,
        borrow_from,
    } in nodes
    {
        match (action, borrowed.entry(index)) {
            (BorrowOrReturn::Borrow(borrowed_out), Entry::Vacant(ve)) => {
                // initial borrow - record and emit
                ve.insert((Borrow(node, borrowed_out), None));
                emit(hugr, node, borrow_from);
            }

            // "interesting" case - return after borrow - record but do not emit (yet)
            (BorrowOrReturn::Return(_), Entry::Occupied(mut oe)) => {
                if oe.get_mut().1.replace(Return(node, borrow_from)).is_some() {
                    panic!("Double return");
                }
            }

            (BorrowOrReturn::Borrow(borrowed_out), Entry::Occupied(mut oe)) => {
                let (prev_borrow, prev_return) = oe.get_mut();
                let Some(prev_return) = prev_return.take() else {
                    panic!("Double borrow");
                };
                // The interesting case...elide!
                let tgt = hugr
                    .single_linked_input(node, borrowed_out)
                    .expect("linear");
                hugr.connect(prev_borrow.0, prev_borrow.1, tgt.0, tgt.1);
                hugr.remove_node(prev_return.0);
                hugr.remove_node(node);
            }

            (BorrowOrReturn::Return(_), Entry::Vacant(_)) => panic!("Return without borrow"),
        }
    }
    // Finally....
    for (_, opt_return) in borrowed.into_values() {
        let Return(return_node, borrow_from) = opt_return.unwrap(); // analysis should have ensured this will work
        emit(hugr, return_node, borrow_from);
    }
}
