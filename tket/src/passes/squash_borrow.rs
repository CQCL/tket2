//! Reorder and squash pairs of borrow and return nodes where possible.

use derive_more::{Display, Error};
use hugr::algorithms::ComposablePass;
use hugr::extension::prelude::ConstUsize;
use hugr::extension::simple_op::MakeExtensionOp;
use hugr::hugr::hugrmut::HugrMut;
use hugr::ops::{OpTag, OpTrait, OpType, Value};
use hugr::std_extensions::arithmetic::conversions::ConvertOpDef;
use hugr::std_extensions::arithmetic::int_types::ConstInt;
use hugr::std_extensions::collections::borrow_array::BArrayUnsafeOpDef;
use hugr::types::{Term, Type};
use hugr::{HugrView, IncomingPort, Node, OutgoingPort, Port, PortIndex, Wire};
use itertools::{Either, Itertools};

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet, VecDeque};

/// Identifies array types and borrows/returns on them
pub trait IsBorrowReturn: Clone {
    /// Determine if the given node is a borrow or return node, and if so,
    /// return the ports identifying the operands
    ///
    /// # Errors
    ///
    /// [NodeInfoError] if the analysis cannot safely determine whether the
    /// node is a borrow/return with known operands. (Assuming a not-understood
    /// node is *not* a borrow/return might lead to unsafe transformation.)
    fn is_borrow_return<H: HugrView>(
        &self,
        node: H::Node,
        hugr: &H,
    ) -> Result<Option<BorrowReturnPorts>, NodeInfoError>;

    /// If the specified type is an array that can be borrowed from,
    /// return `Some` of:
    ///   * `Some(size)` if the array's size is statically known.
    ///   * `None` if the array's size is not statically known.
    /// Otherwise (if the type is not such an array), return `None`.
    fn get_array_size(&self, ty: &Type) -> Option<Option<u64>>;
}

/// A pass for eliding `BorrowArray` reborrows of elements (with constant indices)
/// along with the preceding return.
#[derive(Clone, Debug, Default)]
pub struct BorrowSquashPass<BR> {
    allow_errors_same_dfg: bool,
    regions: Vec<Node>,
    is_br: BR,
}

impl<BR> BorrowSquashPass<BR> {
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

impl<H: HugrMut<Node = Node>, BR: IsBorrowReturn> ComposablePass<H> for BorrowSquashPass<BR> {
    type Error = NodeInfoError;
    /// Pairs of (Return node, Borrow node) that were elided.
    type Result = Vec<(Node, Node)>;

    /// Perform the pass on the given hugr.
    fn run(&self, hugr: &mut H) -> Result<Vec<(Node, Node)>, NodeInfoError> {
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
        let mut seen = HashSet::new();
        for region in regions {
            // Start with all nodes not reachable along dataflow edges from other nodes. (INcludes Input.)
            let mut queue = VecDeque::from_iter(
                hugr.children(*region)
                    .filter(|n| hugr.in_value_types(*n).next().is_none())
                    .flat_map(|n| all_outs(hugr, n)),
            );

            while let Some(start) = queue.pop_front() {
                if !seen.insert(start) {
                    continue;
                }
                let (starts, elided) = borrow_squash_traversal(hugr, &self.is_br, start, true)?;
                queue.extend(starts);
                results.extend(elided);
            }
        }
        Ok(results)
    }
}

/// Reasons we may fail to determine whether a node is a borrow/return.
#[derive(Debug, Display, Error)]
pub enum NodeInfoError {
    /// Borrow op is not a dataflow op.
    #[display("expected dataflow op: {op}")]
    NodeNotDataflow {
        /// The operation that is not a dataflow op.
        op: OpType,
    },
    /// Borrow op has incorrect signature.
    #[display("borrow_node has incorrect signature")]
    BorrowNodeIncorrectSignature,
    /// An operation does not output the same array type as it inputs
    #[display("Array was input as {input} but returned as {output}")]
    InconsistentArrayType {
        /// The array type fed in on the [BorrowFromPorts::inc] port
        input: Type,
        /// The type returned on the [BorrowFromPorts::out] port
        output: Type,
    },
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BRAction {
    /// A borrow action, containing the port on which the borrowed value is output.
    Borrow(OutgoingPort),
    /// A return action, containing the port on which the value to return is input.
    Return(IncomingPort),
    /// Some other action that requires borrowedness of some indices to be as it
    /// would be in the unoptimized Hugr. Thus, we cannot optimize any boro-ret or
    /// ret-boro pair across this op.
    ///
    /// The indices are given by the combination of [BorrowReturnPorts::elem_index]
    /// and any indices in the `Vec` herein.
    ///
    /// If the op clobbers even a single index which is definitely unknown (not from
    /// an [IncomingPort]), then we must assume borrowedness of *all* indices must be
    /// preserved, so no optimization of any pairs around this op is possible; in such
    /// case, the outputs of the op can be considered to be fresh arrays so the
    /// appropriate return value from [IsBorrowReturn::is_borrow_return] is `None`.
    Clobber(Vec<IncomingPort>),
}

impl BRAction {
    /// Gets the port on which the borrowed value is output (for a [Self::Borrow])
    /// or input (for a [Self::Return]).
    pub fn borrowed_port(&self) -> Port {
        match self {
            BRAction::Borrow(p) => (*p).into(),
            BRAction::Return(p) => (*p).into(),
            BRAction::Clobber(_) => panic!("Clobber has no borrowed port"),
        }
    }
}

/// Ports common to a borrow or return op
#[derive(Debug, Clone)]
pub struct BorrowReturnPorts {
    /// Whether this is a borrow or return, and the port for the borrowed value.
    pub action: BRAction,
    /// Port on which the index to borrow, return, or clobber, is passed in
    pub elem_index: IncomingPort,
    /// The ports by which the container array reaches and leaves this node.
    pub borrow_from: BorrowFromPorts,
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

/// Implements [IsBorrowReturn] for `BorrowArray`s.
#[derive(Debug, Default, Display, Clone)]
#[allow(rustdoc::private_intra_doc_links)]
pub struct DefaultBorrowArray;

impl IsBorrowReturn for DefaultBorrowArray {
    fn is_borrow_return<H: HugrView>(
        &self,
        node: H::Node,
        hugr: &H,
    ) -> Result<Option<BorrowReturnPorts>, NodeInfoError> {
        let op = hugr.get_optype(node);

        let Some(ext_op) = op.as_extension_op() else {
            return Ok(None);
        };
        Ok(match BArrayUnsafeOpDef::from_extension_op(ext_op) {
            Ok(BArrayUnsafeOpDef::borrow) => {
                let op = hugr.get_optype(node);
                let sig = op
                    .dataflow_signature()
                    .ok_or_else(|| NodeInfoError::NodeNotDataflow { op: op.clone() })?;
                if sig.input_count() != 2 || sig.output_count() != 2 {
                    return Err(NodeInfoError::BorrowNodeIncorrectSignature);
                }
                Some(BorrowReturnPorts {
                    action: BRAction::Borrow(OutgoingPort::from(0)),
                    borrow_from: BorrowFromPorts {
                        inc: IncomingPort::from(0),
                        out: OutgoingPort::from(1),
                    },
                    elem_index: IncomingPort::from(1),
                })
            }
            Ok(BArrayUnsafeOpDef::r#return) => {
                let op = hugr.get_optype(node);
                let sig = op
                    .dataflow_signature()
                    .ok_or_else(|| NodeInfoError::NodeNotDataflow { op: op.clone() })?;

                if sig.input_count() != 3 || sig.output_count() != 1 {
                    return Err(NodeInfoError::BorrowNodeIncorrectSignature);
                }
                Some(BorrowReturnPorts {
                    action: BRAction::Return(IncomingPort::from(2)),
                    borrow_from: BorrowFromPorts {
                        inc: IncomingPort::from(0),
                        out: OutgoingPort::from(0),
                    },
                    elem_index: IncomingPort::from(1),
                })
            }
            _ => None,
        })
    }

    fn get_array_size(&self, ty: &Type) -> Option<Option<u64>> {
        let ext = ty.as_extension()?;
        (ext.name() == "borrow_array").then(|| {
            let [sz, _elem] = ext.args() else {
                panic!("BorrowArray must have two type arguments");
            };
            match sz {
                Term::BoundedNat(n) => Some(*n),
                _ => None,
            }
        })
    }
}

/// Elide return-borrow pairs for a single array.
///
/// If `recurse` is true, then also elide return-borrow pairs for sub-arrays
/// (i.e. if this is a nested array).
///
/// # Panics
///
/// if `start` does not exist in the Hugr
///
/// # Returns
///
/// * Pairs of (Return node, Borrow node) that were elided.
pub fn borrow_squash_array<H: HugrMut<Node = Node>>(
    hugr: &mut H,
    is_br: &impl IsBorrowReturn,
    start: Wire,
    recurse: bool,
) -> Result<Vec<(Node, Node)>, NodeInfoError> {
    Ok(borrow_squash_traversal(hugr, is_br, start, recurse)?.1)
}

/// Internal method to keep traversal private.
///
/// # Returns
///
/// A tuple of:
/// * New nodes that were discovered that may create arrays and thus should be analysed separately/later
/// * Elided (Return node, Borrow node) pairs
fn borrow_squash_traversal<H: HugrMut<Node = Node>>(
    hugr: &mut H,
    is_br: &impl IsBorrowReturn,
    start: Wire,
    recurse: bool,
) -> Result<(Vec<Wire>, Vec<(Node, Node)>), NodeInfoError> {
    let mut candidates = Vec::new();
    let array_ty = hugr
        .out_value_types(start.node())
        .find(|(p, _)| *p == start.source())
        .unwrap()
        .1;
    let Some(array_sz) = is_br.get_array_size(&array_ty) else {
        return Ok((
            hugr.linked_inputs(start.node(), start.source())
                .flat_map(|(n, _)| all_outs(hugr, n))
                .collect(),
            vec![],
        ));
    };

    struct Borrow(Node, OutgoingPort, BorrowFromPorts);
    struct Return(Node, IncomingPort, BorrowFromPorts);

    let mut borrowed: HashMap<u64, (Borrow, Option<Return>)> = HashMap::new(); // Key is elem index
    let mut rb_elisions: Vec<(Return, Borrow)> = Vec::new();
    let mut br_pairs: Vec<(u64, Borrow, Return)> = Vec::new();

    let mut array = start;
    while let Some((node, index, action, borrow_from)) =
        next_array_op(hugr, is_br, &mut candidates, array)?
    {
        array = Wire::new(node, borrow_from.out); // for next iteration

        match (action, borrowed.entry(index)) {
            (BRAction::Clobber(other_idxs), _) => {
                let Some(mut clobbered_indices) = other_idxs
                    .iter()
                    .map(|inp| find_const(hugr, node, *inp))
                    .collect::<Option<Vec<_>>>()
                else {
                    // At least one unknown index - must preserve all borrowedness
                    // so no elision possible across this op.
                    candidates.extend(all_outs(hugr, node));
                    break;
                };
                clobbered_indices.push(index);

                for idx in clobbered_indices {
                    if let Entry::Occupied(oe) = borrowed.entry(idx) {
                        let (boro, opt_ret) = oe.remove();
                        if let Some(ret) = opt_ret {
                            br_pairs.push((idx, boro, ret));
                        }
                    }
                }
            }
            (BRAction::Borrow(borrowed_out), Entry::Vacant(ve)) => {
                // initial borrow - record
                ve.insert((Borrow(node, borrowed_out, borrow_from), None));
            }

            (BRAction::Return(inc), Entry::Occupied(mut oe)) => {
                // return after borrow - record
                let (_, ret) = &mut oe.get_mut();
                if ret.replace(Return(node, inc, borrow_from)).is_some() {
                    oe.remove_entry(); // double return - this will panic, don't bother optimizing
                }
            }

            (BRAction::Borrow(borrowed_out), Entry::Occupied(mut oe)) => {
                // Borrow after return - record both to be removed
                let (_, prev_return) = oe.get_mut();
                match prev_return.take() {
                    Some(prev_return) => {
                        rb_elisions.push((prev_return, Borrow(node, borrowed_out, borrow_from)));
                    }
                    None => {
                        oe.remove_entry(); // double borrow - this will panic, don't bother optimizing
                    }
                };
            }

            (BRAction::Return(_), Entry::Vacant(_)) => (), // return without borrow - will panic, don't optimize
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
    let mut elided: Vec<_> = rb_elisions
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
    br_pairs.extend(
        borrowed
            .into_iter()
            .filter_map(|(idx, (bor, opt_return))| opt_return.map(|ret| (idx, bor, ret))),
    );
    for (idx, bor, ret) in br_pairs {
        if recurse {
            // Recurse to elide any intervening borrows/returns on the same array
            let (new_nodes, new_elided) =
                borrow_squash_traversal(hugr, is_br, Wire::new(bor.0, bor.1), true)?;
            // TODO can we avoid duplicates? (e.g. if we've seen them in this array - second traversal will do nothing)
            candidates.extend(new_nodes);
            elided.extend(new_elided);
        }
        // Don't elide unless we know the borrowed index is within bounds and would not panic.
        if array_sz.is_some_and(|sz| idx < sz)
            && hugr.linked_inputs(bor.0, bor.1).exactly_one().ok() == Some((ret.0, ret.1))
        {
            elide_node(hugr, bor.0, &bor.2);
            elide_node(hugr, ret.0, &ret.2);
            elided.push((bor.0, ret.0));
        }
    }
    Ok((candidates, elided))
}

fn next_array_op(
    hugr: &impl HugrView<Node = Node>,
    is_br: &impl IsBorrowReturn,
    candidates: &mut Vec<Wire>,
    array: Wire,
) -> Result<Option<(Node, u64, BRAction, BorrowFromPorts)>, NodeInfoError> {
    let next = hugr
        .single_linked_input(array.node(), array.source())
        .expect("array is linear");

    let Some(is_br) = is_br.is_borrow_return(next.0, hugr)? else {
        // Not a borrow/return - stop processing this array;
        // any outports of the node are considered fresh arrays.
        candidates.extend(all_outs(hugr, next.0));
        return Ok(None);
    };
    {
        let [input, output] = [
            (array.node(), array.source()),
            (next.0, is_br.borrow_from.out),
        ]
        .map(|(n, p)| hugr.out_value_types(n).find(|(p2, _)| p == *p2).unwrap().1);
        if input != output {
            return Err(NodeInfoError::InconsistentArrayType { input, output });
        }
    }
    if next.1 != is_br.borrow_from.inc {
        match is_br.action {
            BRAction::Clobber(_) => (), // Must be reachable along array inport
            BRAction::Return(rv) => {
                assert_eq!(rv, next.1);
                // this array ends here (by being returned to outer array).
                // Processing of outer array will continue after the Return.
            }
            BRAction::Borrow(_) => panic!("Array fed into unexpected port of borrow"),
        }
        return Ok(None);
    }
    let Some(idx) = find_const(hugr, next.0, is_br.elem_index) else {
        // Unknown index.
        // Hence, we must preserve borrowedness of all indices for this op;
        // so we can't elide anything before this op with anything after.
        // Hence, op can be considered as beginning of fresh array(s) on each outport.
        candidates.extend(all_outs(hugr, next.0));
        return Ok(None);
    };
    if matches!(is_br.action, BRAction::Clobber(_)) {
        // We'll process the borrow-from-out port here, but the others are potentially fresh arrays.
        candidates.extend(all_outs(hugr, next.0).filter(|w| w.source() != is_br.borrow_from.out));
    }
    Ok(Some((next.0, idx, is_br.action, is_br.borrow_from)))
}

fn all_outs(h: &impl HugrView<Node = Node>, n: Node) -> impl Iterator<Item = Wire> + '_ {
    h.out_value_types(n).map(move |(p, _)| Wire::new(n, p))
}

fn find_const<H: HugrView>(hugr: &H, n: H::Node, inp: IncomingPort) -> Option<u64> {
    let (mut curr_node, mut outp) = hugr.single_linked_output(n, inp).unwrap();

    fn is_const_conversion_op(op: &OpType) -> bool {
        matches!(op, OpType::LoadConstant(..))
            || op
                .as_extension_op()
                .is_some_and(|op| ConvertOpDef::from_extension_op(op) == Ok(ConvertOpDef::itousize))
    }

    let op = loop {
        if outp.index() > 0 {
            return None;
        }
        let op = hugr.get_optype(curr_node);
        if !is_const_conversion_op(op) {
            break op;
        }
        (curr_node, outp) = hugr
            .single_linked_output(curr_node, IncomingPort::from(0))
            .expect("invalid signature for conversion op");
    };

    let OpType::Const(const_op) = op else {
        return None;
    };
    if let Value::Extension { e } = &const_op.value {
        if let Some(c) = e.value().downcast_ref::<ConstUsize>() {
            return Some(c.value());
        }
        if let Some(c) = e.value().downcast_ref::<ConstInt>() {
            return Some(c.value_u());
        }
    }
    panic!("Unexpected index {:?}", const_op.value)
}

#[cfg(test)]
mod test {
    use std::io::BufReader;

    use super::{find_const, BorrowSquashPass};
    use crate::{extension::REGISTRY, passes::squash_borrow::DefaultBorrowArray, Circuit};
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
        let res = BorrowSquashPass::<DefaultBorrowArray>::default()
            .with_regions([h.entrypoint()])
            .run(&mut h)
            .unwrap();
        h.validate().unwrap();
        assert_eq!(res.len(), 9); // Just what's been seen

        let get_index = |n| find_const(&h, n, 1.into()).unwrap();
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
