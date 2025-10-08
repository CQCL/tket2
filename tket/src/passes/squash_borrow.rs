//! Reorder and squash pairs of borrow and return nodes where possible.

pub mod analysis;
pub use analysis::BorrowAnalysis;

use hugr::algorithms::ComposablePass;
use hugr::hugr::hugrmut::HugrMut;
use hugr::ops::{OpTag, OpTrait, OpType};
use hugr::{IncomingPort, Node, OutgoingPort, Port, Wire};
use itertools::{Either, Itertools};

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
        fn valid_circuit_parent(op: &OpType) -> bool {
            match op {
                OpType::FuncDefn(fd) => fd.signature().params().is_empty(),
                _ => OpTag::DataflowParent.is_superset(op.tag()),
            }
        }
        let temp: Vec<Node>; // to keep alive
        let regions = if self.regions.is_empty() {
            temp = hugr
                .entry_descendants()
                .filter(|n| valid_circuit_parent(hugr.get_optype(*n)))
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

/// A list or [BorrowOrReturn] nodes for a single resource, satisfying that
///   * each [Borrow] is followed by a [Return] (without any intervening [Borrow]), and
///   * each [Return] follows a [Borrow] (without any intervening [Return]),
///
/// ...both of the same element index.
///
/// [Borrow]: BRAction::Borrow
/// [Return]: BRAction::Return
#[derive(Clone, Debug)]
pub struct BorrowIntervals {
    actions: Vec<BorrowOrReturn>,
}

impl BorrowIntervals {
    /// Gets the list of borrow/return actions satisfying the
    /// pairing conditions.
    pub fn actions(&self) -> &[BorrowOrReturn] {
        &self.actions
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
pub enum BRAction {
    /// A borrow action, containing the port on which the borrowed value is output.
    Borrow(OutgoingPort),
    /// A return action, containing the port on which the value to return is input.
    Return(IncomingPort),
}

impl BRAction {
    /// Gets the port on which the borrowed value is output (for a [Self::Borrow])
    /// or input (for a [Self::Return]).
    pub fn borrowed_port(&self) -> Port {
        match self {
            BRAction::Borrow(p) => (*p).into(),
            BRAction::Return(p) => (*p).into(),
        }
    }
}

/// An element index to a borrow/return.
/// If right, i.e. non-constant, then no elision may be possible.
pub type BorrowIndex = Either<u64, Wire>;

/// Information about a node that is either a borrow from or return to a resource.
#[derive(Clone, Debug)]
pub struct BorrowOrReturn {
    /// The node in the Hugr that this instance describes
    pub node: Node,
    /// Whether this is a borrow or return, and the port for the borrowed value.
    pub action: BRAction,
    /// The (hopefully-)constant index of the element being borrowed/returned
    pub elem_index: BorrowIndex,
    /// The ports by which the container array reaches and leaves this node.
    pub borrow_from: BorrowFromPorts,
}

/// Elide return-borrow pairs for a single array.
///
/// # Returns
///
/// Pairs of (Return node, Borrow node) that were elided.
pub fn borrow_squash_array<H: HugrMut<Node = Node>>(
    hugr: &mut H,
    intervals: BorrowIntervals,
) -> Vec<(Node, Node)> {
    if intervals.actions().iter().any(|n| n.elem_index.is_right()) {
        // If any index is non-constant, for now don't elide anything.
        // (May be able to proceed very carefully...)
        return Vec::new();
    }

    struct Borrow(Node, OutgoingPort, BorrowFromPorts);
    struct Return(Node, IncomingPort, BorrowFromPorts);
    let mut borrowed: HashMap<u64, (Borrow, Option<Return>)> = HashMap::new(); // Key is elem index
    let mut elisions: Vec<(Return, Borrow)> = Vec::new();

    for BorrowOrReturn {
        node,
        elem_index: index,
        action,
        borrow_from,
    } in intervals.actions
    {
        // We bailed out if any indices were Right (i.e. non-Const) above.
        match (action, borrowed.entry(index.unwrap_left())) {
            (BRAction::Borrow(borrowed_out), Entry::Vacant(ve)) => {
                // initial borrow - record
                ve.insert((Borrow(node, borrowed_out, borrow_from), None));
            }

            (BRAction::Return(inc), Entry::Occupied(mut oe)) => {
                // return after borrow - record
                let (_, ret) = &mut oe.get_mut();
                if ret.replace(Return(node, inc, borrow_from)).is_some() {
                    panic!("Double return");
                }
            }

            (BRAction::Borrow(borrowed_out), Entry::Occupied(mut oe)) => {
                // Borrow after return - record both to be removed
                let (_, prev_return) = oe.get_mut();
                let Some(prev_return) = prev_return.take() else {
                    panic!("Double borrow"); // ensured by analysis
                };
                elisions.push((prev_return, Borrow(node, borrowed_out, borrow_from)));
            }

            (BRAction::Return(_), Entry::Vacant(_)) => panic!("Return without borrow"),
        }
    }
    fn elide_node<H: HugrMut<Node = Node>>(hugr: &mut H, n: Node, ports: &BorrowFromPorts) {
        let in_array = hugr
            .single_linked_output(n, ports.inc)
            .expect("array is linear");
        let out_array = hugr
            .single_linked_input(n, ports.out)
            .expect("array is linear");
        hugr.connect(in_array.0, in_array.1, out_array.0, out_array.1);
        hugr.remove_node(n);
    }
    let elided = elisions
        .into_iter()
        .map(|(ret, bor)| {
            // Pass Return-ed value directly through to users of Borrow-ed value
            let src = hugr.single_linked_output(ret.0, ret.1).expect("input");
            for tgt in hugr.linked_inputs(bor.0, bor.1).collect::<Vec<_>>() {
                hugr.connect(src.0, src.1, tgt.0, tgt.1);
            }
            elide_node(hugr, bor.0, &bor.2);
            elide_node(hugr, ret.0, &ret.2);
            (ret.0, bor.0)
        })
        .collect();
    // Now also elide Borrow-Returns if the returned value is exactly that from the borrow.
    // TODO ALAN noooo, this removes potential panics. Need to check against array size.
    for (bor, opt_return) in borrowed.into_values() {
        let ret = opt_return.unwrap(); // ensured by analysis
        if hugr.linked_inputs(bor.0, bor.1).exactly_one().ok() == Some((ret.0, ret.1)) {
            elide_node(hugr, bor.0, &bor.2);
            elide_node(hugr, ret.0, &ret.2);
        }
    }

    elided
}

#[cfg(test)]
mod test {
    use std::io::BufReader;

    use super::{analysis::find_const, BorrowSquashPass};
    use crate::{extension::REGISTRY, Circuit};
    use hugr::{
        algorithms::ComposablePass, extension::simple_op::MakeExtensionOp, hugr::hugrmut::HugrMut,
        std_extensions::collections::borrow_array::BArrayUnsafeOpDef, Hugr, HugrView, Node,
        OutgoingPort, Wire,
    };
    use itertools::Itertools;
    use portgraph::NodeIndex;
    use rstest::{fixture, rstest};

    fn to_wire((n, p): (Node, OutgoingPort)) -> Wire {
        Wire::new(n, p)
    }

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
            // NOTE ALAN: h.nodes() doesn't work, something is non-const...
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
