//! Reorder and squash pairs of borrow and return nodes where possible.

pub mod analysis;
pub use analysis::BorrowAnalysis;

use hugr::algorithms::ComposablePass;
use hugr::hugr::hugrmut::HugrMut;
use hugr::ops::{OpTag, OpTrait};
use hugr::{IncomingPort, Node, OutgoingPort, Port, Wire};

use std::collections::hash_map::Entry;
use std::collections::HashMap;

use crate::Circuit;
use analysis::{BorrowAnalysisError, DefaultBorrowAnalysis};

/// A pass for eliding `BorrowArray` reborrows of elements (with constant indices)
/// along with the preceding return.

#[derive(Clone, Debug, Default)]
pub struct BorrowSquashPass {
    allow_errors_same_dfg: bool,
    regions: Vec<Node>,
}

impl BorrowSquashPass {
    /// Controls whether analysis failures affect optimization.
    ///
    /// If `true`, then even when analysis fails (with a [BorrowAnalysisError])
    /// for some arrays in a Dataflow Sibling Graph, return-borrow pairs
    /// may still be elided for other arrays (as long as analysis succeeded on the latter).
    /// If `false`, then analysis failure for any array in a DSG prevents any elision
    /// in that DSG.
    ///
    /// Default is `false`.
    pub fn allow_errors_same_dfg(&mut self, allow: bool) {
        self.allow_errors_same_dfg = allow;
    }

    /// Add regions (subgraphs) in which to perform the pass.
    ///
    /// If no regions are specified, the pass is performed on all dataflow regions
    /// beneath the entrypoint.
    pub fn with_regions(mut self, regions: impl IntoIterator<Item = Node>) -> Self {
        self.regions.extend(regions);
        self
    }
}

impl<H: HugrMut<Node = Node>> ComposablePass<H> for BorrowSquashPass {
    type Error = BorrowAnalysisError<Node>;
    /// Pairs of (Return node, Borrow node) that were elided.
    type Result = Vec<(Node, Node)>;

    /// Perform the pass on the given hugr.
    fn run(&self, hugr: &mut H) -> Result<Vec<(Node, Node)>, BorrowAnalysisError<Node>> {
        let analysis = DefaultBorrowAnalysis::default();
        let temp: Vec<Node>; // to keep alive
        let regions = if self.regions.is_empty() {
            temp = hugr
                .entry_descendants()
                .filter(|n| OpTag::DataflowParent.is_superset(hugr.get_optype(*n).tag()))
                .collect();
            &temp
        } else {
            &self.regions
        };
        let mut results = Vec::new();
        for node in regions {
            let circ = Circuit::new(hugr.with_entrypoint(*node));
            for actions in analysis.run(&circ, self.allow_errors_same_dfg)? {
                results.extend(borrow_squash_array(hugr, actions));
            }
        }
        Ok(results)
    }
}

/// The ports by which the container array reaches and leaves a
/// particular borrow or return node.
#[derive(Clone, Debug)]
pub struct BorrowFromPorts {
    /// The port receiving the array before the borrow/return.
    inc: IncomingPort,
    /// The port returning the array after the borrow/return.
    out: OutgoingPort,
}

/// Whether a node is a borrow or return, along with the port for
/// the borrowed value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorrowOrReturn {
    /// A borrow action, containing the port on which the borrowed value is output.
    Borrow(OutgoingPort),
    /// A return action, containing the port on which the value to return is input.
    Return(IncomingPort),
}

impl BorrowOrReturn {
    /// Gets the port on which the borrowed value is output (for a [Self::Borrow])
    /// or input (for a [Self::Return]).
    pub fn borrowed_port(&self) -> Port {
        match self {
            BorrowOrReturn::Borrow(p) => (*p).into(),
            BorrowOrReturn::Return(p) => (*p).into(),
        }
    }
}

/// Information about a node that is either a borrow from or return to a resource.
#[derive(Clone, Debug)]
pub struct BorrowAction {
    /// The node in the Hugr that this instance describes
    pub node: Node,
    /// The constant index of the element being borrowed/returned
    pub borrow_index_const: u64,
    /// Whether this is a borrow or return, and the port for the borrowed value.
    pub action: BorrowOrReturn,
    /// The ports by which the container array reaches and leaves this node.
    pub borrow_from: BorrowFromPorts,
}

/// Elide return-borrow pairs for a single array, given `nodes` that are well-paired
/// i.e. from [BorrowAnalysis]
///
/// # Returns
///
/// Pairs of (Return node, Borrow node) that were elided.
///
/// # Panics
///
/// If `nodes` are not well-paired
fn borrow_squash_array<H: HugrMut<Node = Node>>(
    hugr: &mut H,
    nodes: Vec<BorrowAction>,
) -> Vec<(Node, Node)> {
    // Find the original source of the array and target. (These may have changed
    // since the analysis was run, e.g. if this is a nested array produced by an
    // elided borrow.)
    let mut current_array = {
        let Some(first) = nodes.first() else {
            return Vec::new();
        };
        to_wire(
            hugr.single_linked_output(first.node, first.borrow_from.inc)
                .expect("linear"),
        )
    };

    let final_array_target = {
        let last = nodes.last().unwrap();
        hugr.single_linked_input(last.node, last.borrow_from.out)
            .expect("linear")
    };

    struct Borrow(Node, OutgoingPort);
    struct Return(Node, IncomingPort);
    // The map is from borrow index to (borrow node, optional return node)
    let mut borrowed: HashMap<u64, (Borrow, Option<(Return, BorrowFromPorts)>)> = HashMap::new();
    let mut elisions: Vec<(Return, Borrow)> = Vec::new();
    let mut emit = |node, borrow_from: BorrowFromPorts| {
        hugr.disconnect(current_array.node(), current_array.source());
        hugr.disconnect(node, borrow_from.inc);
        hugr.connect(
            current_array.node(),
            current_array.source(),
            node,
            borrow_from.inc,
        );
        current_array = Wire::new(node, borrow_from.out);
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

            (BorrowOrReturn::Return(inc), Entry::Occupied(mut oe)) => {
                // return after borrow - record but do not emit (yet)
                let (_, ret) = &mut oe.get_mut();
                if ret.replace((Return(node, inc), borrow_from)).is_some() {
                    panic!("Double return");
                }
            }

            (BorrowOrReturn::Borrow(borrowed_out), Entry::Occupied(mut oe)) => {
                // Borrow after return (can't be borrow after borrow as per analysis)
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
        let (Return(return_node, _), borrow_from) = opt_return.expect("ensured by analysis");
        emit(return_node, borrow_from);
    }
    // Wire the last-emitted return to the same target as whichever return
    // was originally last (they can be reordered by the `borrowed` map).
    emit(
        final_array_target.0,
        BorrowFromPorts {
            inc: final_array_target.1,
            out: OutgoingPort::from(usize::MAX), // unused
        },
    );
    elisions
        .into_iter()
        .map(|(ret, bor)| {
            let src = hugr.single_linked_output(ret.0, ret.1).expect("linear");
            let tgt = hugr.single_linked_input(bor.0, bor.1).expect("linear");
            hugr.connect(src.0, src.1, tgt.0, tgt.1);
            hugr.remove_node(ret.0);
            hugr.remove_node(bor.0);
            (ret.0, bor.0)
        })
        .collect()
}

fn to_wire((n, p): (Node, OutgoingPort)) -> Wire {
    Wire::new(n, p)
}

#[cfg(test)]
mod test {
    use std::io::BufReader;

    use super::{analysis::find_const, to_wire, BorrowSquashPass};
    use crate::{extension::REGISTRY, Circuit};
    use hugr::{
        algorithms::ComposablePass, extension::simple_op::MakeExtensionOp, hugr::hugrmut::HugrMut,
        std_extensions::collections::borrow_array::BArrayUnsafeOpDef, Hugr, HugrView, Node,
    };
    use itertools::Itertools;
    use portgraph::NodeIndex;
    use rstest::{fixture, rstest};

    #[fixture]
    pub(super) fn borrow_circuit() -> Circuit {
        let reader =
            BufReader::new(include_bytes!("../../../test_files/squashing_inline.hugr").as_slice());
        let mut hugr = Hugr::load(reader, Some(&REGISTRY)).unwrap();
        hugr.set_entrypoint(Node::from(NodeIndex::new(1176)));
        Circuit::new(hugr)
    }

    #[rstest]
    fn test_borrow_squash(borrow_circuit: Circuit) {
        let mut h = borrow_circuit.into_hugr();
        let res = BorrowSquashPass::default()
            .with_regions([h.entrypoint()])
            .run(&mut h)
            .unwrap();
        h.validate().unwrap();
        assert_eq!(res.len(), 9); // Just what's been seen

        let get_index = |n| find_const(&h, to_wire(h.single_linked_output(n, 1).unwrap())).unwrap();
        let all_indices = |b| {
            h.entry_descendants()
                .filter(|n| {
                    h.get_optype(*n)
                        .as_extension_op()
                        .is_some_and(|eop| BArrayUnsafeOpDef::from_extension_op(eop) == Ok(b))
                })
                .map(get_index)
                .sorted()
                .collect_vec()
        };

        let indices = Vec::from_iter(0..=7);
        assert_eq!(all_indices(BArrayUnsafeOpDef::borrow), indices);
        assert_eq!(all_indices(BArrayUnsafeOpDef::r#return), indices);
    }
}
