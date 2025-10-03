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
    struct Return(Node, IncomingPort);
    // The map is from borrow index to (borrow node, optional return node)
    let mut borrowed: HashMap<u64, (Borrow, Option<(Return, BorrowFromPorts)>)> = HashMap::new();
    let mut elisions: Vec<(Return, Borrow)> = Vec::new();
    let mut emit = |node, borrow_from: BorrowFromPorts| {
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
                emit(node, borrow_from);
            }

            // "interesting" case - return after borrow - record but do not emit (yet)
            (BorrowOrReturn::Return(inc), Entry::Occupied(mut oe)) => {
                let (_, ret) = &mut oe.get_mut();
                if ret.replace((Return(node, inc), borrow_from)).is_some() {
                    panic!("Double return");
                }
            }

            (BorrowOrReturn::Borrow(borrowed_out), Entry::Occupied(mut oe)) => {
                let (_, prev_return) = oe.get_mut();
                let Some(prev_return) = prev_return.take() else {
                    panic!("Double borrow");
                };
                elisions.push((prev_return.0, Borrow(node, borrowed_out)));
            }

            (BorrowOrReturn::Return(_), Entry::Vacant(_)) => panic!("Return without borrow"),
        }
    }
    // Wire up final (non-elided) returns
    for (_, opt_return) in borrowed.into_values() {
        let (Return(return_node, _), borrow_from) = opt_return.unwrap(); // analysis should have ensured this will work
        emit(return_node, borrow_from);
    }
    for (ret, bor) in elisions {
        let src = hugr.single_linked_output(ret.0, ret.1).expect("linear");
        let tgt = hugr.single_linked_input(bor.0, bor.1).expect("linear");
        hugr.connect(src.0, src.1, tgt.0, tgt.1);
        hugr.remove_node(ret.0);
        hugr.remove_node(bor.0);
    }
}
