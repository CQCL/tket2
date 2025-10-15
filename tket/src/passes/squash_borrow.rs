//! Elide pairs of return-borrow operations (e.g. [BorrowArray]) where possible.
//!
//! Specifically when the index is known to be the same, and when there is a preceding
//! borrow (not elided) that guarantees the elided ops would not have panicked.

use derive_more::{Display, Error};
use hugr::algorithms::ComposablePass;
use hugr::extension::prelude::ConstUsize;
use hugr::extension::simple_op::MakeExtensionOp;
use hugr::hugr::hugrmut::HugrMut;
use hugr::ops::{OpTag, OpTrait, OpType, Value};
use hugr::std_extensions::arithmetic::int_types::ConstInt;
use hugr::std_extensions::collections::borrow_array::{
    BArrayUnsafeOpDef, BorrowArray, BORROW_ARRAY_TYPENAME,
};
use hugr::types::Type;
use hugr::{HugrView, IncomingPort, Node, OutgoingPort, Wire};

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet, VecDeque};

/// Identifies array types and borrows/returns on them
pub trait IsBorrowReturn: Clone {
    /// Determine if the given node is a borrow or return node, and if so, return
    /// the ports identifying the operands. If not, return `Ok(None)` - this will
    /// prevent elision of any return-borrow-return spread over this node.
    ///
    /// # Errors
    ///
    /// [BorrowAnalysisError] if the analysis cannot safely determine whether the
    /// node is a borrow/return with known operands. (Generally, returning `Ok(None)`
    /// handles most situations by preventing optimization across a particular node;
    /// however, returning an error prevents optimization across the wider DSG.)
    fn is_borrow_return<H: HugrView>(
        &self,
        node: H::Node,
        hugr: &H,
    ) -> Result<Option<BorrowReturnPorts>, BorrowAnalysisError>;

    /// Tells whether the specified type is an array that can be borrowed from.
    fn is_array(&self, ty: &Type) -> bool;
}

/// A pass for eliding (e.g. `BorrowArray`) reborrows of elements (with constant indices)
/// along with the preceding return.
#[derive(Clone, Debug, Default)]
pub struct BorrowSquashPass<BR> {
    regions: Vec<Node>,
    is_br: BR,
}

impl<BR> BorrowSquashPass<BR> {
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
    type Error = BorrowAnalysisError;
    /// Pairs of (Return node, Borrow node) that were elided.
    type Result = Vec<(Node, Node)>;

    /// Perform the pass on the given hugr.
    ///
    /// Note it is recommended to run [ConstantFoldPass] first to make as many indices
    /// constant as possible.
    ///
    /// [ConstantFoldPass]: hugr_passes::constant_fold::ConstantFoldPass
    fn run(&self, hugr: &mut H) -> Result<Vec<(Node, Node)>, BorrowAnalysisError> {
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
                let elided = borrow_squash_traversal(hugr, &self.is_br, &mut queue, start, true)?;
                results.extend(elided);
            }
        }
        Ok(results)
    }
}

/// Reasons we may fail to determine whether a node is a borrow/return.
#[derive(Debug, Display, Error)]
pub enum BorrowAnalysisError {
    /// Borrow op is not a dataflow op.
    #[display("expected dataflow op: {op}")]
    NodeNotDataflow {
        /// The operation that is not a dataflow op.
        op: OpType,
    },
    /// Borrow op has incorrect signature.
    #[display("borrow_node has incorrect signature")]
    BorrowNodeIncorrectSignature,
    /// An operation does not return the array with the same type as the array had when created
    #[display("Array was created as {array_ty} but returned as {out_ty}")]
    InconsistentArrayType {
        /// The type as which the array was created
        array_ty: Type,
        /// The type which the op returns after borrowing from it
        out_ty: Type,
    },
}

/// The ports by which the container array reaches and leaves a
/// particular borrow or return node.
#[derive(Clone, Debug)]
pub struct BorrowFromPorts {
    /// The port receiving the array before the borrow/return.
    pub inc: IncomingPort,
    /// The port returning the array after the borrow/return.
    pub out: OutgoingPort,
}

/// Whether a node is a borrow or return, along with the port for
/// the borrowed value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BRAction {
    /// A borrow action, containing the port on which the borrowed value is output.
    Borrow(OutgoingPort),
    /// A return action, containing the port on which the value to return is input.
    Return(IncomingPort),
}

/// Ports common to a borrow or return op
#[derive(Debug, Clone)]
pub struct BorrowReturnPorts {
    /// Whether this is a borrow or return, and the port for the borrowed value.
    pub action: BRAction,
    /// Port on which the index (of the element to borrow or return) is passed in
    pub elem_index: IncomingPort,
    /// The ports by which the container array reaches and leaves this node.
    pub borrow_from: BorrowFromPorts,
}

impl IsBorrowReturn for BorrowArray {
    fn is_borrow_return<H: HugrView>(
        &self,
        node: H::Node,
        hugr: &H,
    ) -> Result<Option<BorrowReturnPorts>, BorrowAnalysisError> {
        let op = hugr.get_optype(node);

        let Some(ext_op) = op.as_extension_op() else {
            return Ok(None);
        };
        Ok(match BArrayUnsafeOpDef::from_extension_op(ext_op) {
            Ok(BArrayUnsafeOpDef::borrow) => {
                let op = hugr.get_optype(node);
                let sig = op
                    .dataflow_signature()
                    .ok_or_else(|| BorrowAnalysisError::NodeNotDataflow { op: op.clone() })?;
                if sig.input_count() != 2 || sig.output_count() != 2 {
                    return Err(BorrowAnalysisError::BorrowNodeIncorrectSignature);
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
                    .ok_or_else(|| BorrowAnalysisError::NodeNotDataflow { op: op.clone() })?;

                if sig.input_count() != 3 || sig.output_count() != 1 {
                    return Err(BorrowAnalysisError::BorrowNodeIncorrectSignature);
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

    fn is_array(&self, ty: &Type) -> bool {
        ty.as_extension()
            .is_some_and(|ext| ext.name() == &BORROW_ARRAY_TYPENAME)
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
) -> Result<Vec<(Node, Node)>, BorrowAnalysisError> {
    borrow_squash_traversal(hugr, is_br, &mut Vec::new(), start, recurse)
}

/// Internal method to keep traversal private.
/// Like [borrow_squash_array] but also pushes new Wires that may create arrays onto `candidates`.
fn borrow_squash_traversal<H: HugrMut<Node = Node>>(
    hugr: &mut H,
    is_br: &impl IsBorrowReturn,
    candidates: &mut impl Extend<Wire>,
    start: Wire,
    recurse: bool,
) -> Result<Vec<(Node, Node)>, BorrowAnalysisError> {
    let array_ty = wire_type(hugr, start);
    if !is_br.is_array(&array_ty) {
        for (n, _) in hugr.linked_inputs(start.node(), start.source()) {
            // Traverse successors until we find an array-creating op. Borrows/Returns
            // will be traversed when reached along their array input.
            if is_br.is_borrow_return(n, hugr)?.is_none() {
                candidates.extend(all_outs(hugr, n));
            }
        }
        return Ok(vec![]);
    };

    struct Borrow(Node, OutgoingPort, BorrowFromPorts);
    struct Return(Node, IncomingPort, BorrowFromPorts);

    let mut borrowed: HashMap<u64, (Borrow, Option<Return>)> = HashMap::new(); // Key is elem index
    let mut rb_elisions: Vec<(Return, Borrow)> = Vec::new();

    let mut array = start;
    while let Some((node, index, action, borrow_from)) =
        next_array_op(hugr, is_br, candidates, array, &array_ty)?
    {
        array = Wire::new(node, borrow_from.out); // for next iteration

        match (action, borrowed.entry(index)) {
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

    if recurse {
        // Elide any intervening borrows/returns on nested arrays
        for (bor, _) in borrowed.into_values() {
            let start = Wire::new(bor.0, bor.1);
            let new_elided = borrow_squash_traversal(hugr, is_br, candidates, start, true)?;
            elided.extend(new_elided);
            // TODO: it would be good to elide borrow-return (of the same value), but we would need to know
            // the index was not already borrowed, e.g. had just been returned before the borrow.
        }
    }

    Ok(elided)
}

fn next_array_op(
    hugr: &impl HugrView<Node = Node>,
    is_br: &impl IsBorrowReturn,
    candidates: &mut impl Extend<Wire>,
    array: Wire,
    array_ty: &Type,
) -> Result<Option<(Node, u64, BRAction, BorrowFromPorts)>, BorrowAnalysisError> {
    let (node, inport) = hugr
        .single_linked_input(array.node(), array.source())
        .expect("array is linear");
    let Some(is_br) = is_br.is_borrow_return(node, hugr)? else {
        // Not a borrow/return - stop processing this array;
        // any outports of the node are considered fresh arrays.
        candidates.extend(all_outs(hugr, node));
        return Ok(None);
    };

    if inport != is_br.borrow_from.inc {
        if let BRAction::Borrow(_) = is_br.action {
            panic!("Array fed into unexpected port of borrow")
        }
        // Op will be reachable along the incoming-array port, so some other call
        // (along that path) will process it.
        if let BRAction::Return(rv) = is_br.action {
            assert_eq!(rv, inport); // Nested array being returned into outer array.
        }
        return Ok(None);
    }

    {
        let out_ty = wire_type(hugr, Wire::new(node, is_br.borrow_from.out));
        if *array_ty != out_ty {
            let array_ty = array_ty.clone();
            return Err(BorrowAnalysisError::InconsistentArrayType { array_ty, out_ty });
        }
    }
    let Some(idx) = find_const(hugr, node, is_br.elem_index) else {
        // Unknown index.
        // Hence, we must preserve borrowedness of all indices for this op;
        // so we can't elide anything before this op with anything after.
        // Hence, op can be considered as beginning of fresh array(s) on each outport.
        candidates.extend(all_outs(hugr, node));
        return Ok(None);
    };
    Ok(Some((node, idx, is_br.action, is_br.borrow_from)))
}

fn wire_type(h: &impl HugrView<Node = Node>, w: Wire) -> Type {
    h.out_value_types(w.node())
        .find(|(p, _)| *p == w.source())
        .unwrap()
        .1
}

fn all_outs(h: &impl HugrView<Node = Node>, n: Node) -> impl Iterator<Item = Wire> + '_ {
    h.out_value_types(n).map(move |(p, _)| Wire::new(n, p))
}

fn find_const<H: HugrView>(hugr: &H, n: H::Node, inp: IncomingPort) -> Option<u64> {
    let (load_const, _) = hugr.single_linked_output(n, inp).expect("dataflow input");
    if !hugr.get_optype(load_const).is_load_constant() {
        return None;
    }

    let const_op = hugr
        .single_linked_output(load_const, 0)
        .and_then(|(n, _)| hugr.get_optype(n).as_const())
        .expect("LoadConstant input is constant");

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
    use std::{collections::BTreeSet, io::BufReader};

    use super::{find_const, BorrowSquashPass};
    use crate::extension::REGISTRY;
    use hugr::{
        algorithms::{const_fold::ConstantFoldPass, ComposablePass},
        extension::{prelude::qb_t, simple_op::MakeExtensionOp},
        hugr::hugrmut::HugrMut,
        ops::OpTrait,
        std_extensions::collections::{
            array::ArrayKind,
            borrow_array::{BArrayUnsafeOpDef, BorrowArray},
        },
        Hugr, HugrView, Node,
    };
    use itertools::Itertools;
    use portgraph::NodeIndex;
    use rstest::rstest;

    fn borrow_circuit() -> Hugr {
        let reader =
            BufReader::new(include_bytes!("../../../test_files/squashing_inline.hugr").as_slice());
        let mut hugr = Hugr::load(reader, Some(&REGISTRY)).unwrap();
        hugr.set_entrypoint(Node::from(NodeIndex::new(1176)));
        hugr
    }

    fn big_array() -> Hugr {
        let reader =
            BufReader::new(include_bytes!("../../../test_files/big_array.hugr").as_slice());
        Hugr::load(reader, Some(&REGISTRY)).unwrap()
    }

    #[rstest]
    #[case(borrow_circuit(), 9, Some(Vec::from_iter(0..=7)))]
    #[case(big_array(), 759, None)]
    fn test_borrow_squash(
        #[case] mut h: Hugr,
        #[case] expected_elisions: usize,
        #[case] expected_indices: Option<Vec<u64>>,
    ) {
        ConstantFoldPass::default().run(&mut h).unwrap();
        let orig_num_nodes = h.num_nodes();
        let res = BorrowSquashPass::<BorrowArray>::default()
            .run(&mut h)
            .unwrap();
        h.validate().unwrap();
        assert_eq!(res.len(), expected_elisions);
        assert_eq!(h.num_nodes(), orig_num_nodes - 2 * expected_elisions);

        if let Some(exp_indices) = expected_indices {
            assert_eq!(
                find_borrows(&h).map(|n| get_index(&h, n)).collect_vec(),
                exp_indices
            );
            assert_eq!(
                find_returns(&h).map(|n| get_index(&h, n)).collect_vec(),
                exp_indices
            );
        }
    }

    fn find_borrows<H: HugrView>(h: &H) -> impl Iterator<Item = H::Node> + '_ {
        h.entry_descendants().filter(|n| {
            h.get_optype(*n).as_extension_op().is_some_and(|eop| {
                BArrayUnsafeOpDef::from_extension_op(eop) == Ok(BArrayUnsafeOpDef::borrow)
            })
        })
    }

    fn find_returns<H: HugrView>(h: &H) -> impl Iterator<Item = H::Node> + '_ {
        h.entry_descendants().filter(|n| {
            h.get_optype(*n).as_extension_op().is_some_and(|eop| {
                BArrayUnsafeOpDef::from_extension_op(eop) == Ok(BArrayUnsafeOpDef::r#return)
            })
        })
    }

    fn get_index<H: HugrView>(h: &H, n: H::Node) -> u64 {
        find_const(h, n, 1.into()).unwrap()
    }

    #[rstest]
    fn test_nested_array() {
        let inner_array_type = BorrowArray::ty(5, qb_t());
        let outer_array_type = BorrowArray::ty(10, inner_array_type.clone());
        let reader =
            BufReader::new(include_bytes!("../../../test_files/nested_array.hugr").as_slice());
        let mut h = Hugr::load(reader, Some(&REGISTRY)).unwrap();
        let array_func = h
            .children(h.module_root())
            .find(|n| {
                h.get_optype(*n)
                    .as_func_defn()
                    .is_some_and(|fd| fd.func_name() == "nested_array")
            })
            .unwrap();
        h.set_entrypoint(array_func);
        ConstantFoldPass::default().run(&mut h).unwrap();
        // Sanity checks: all borrows are qs[0][1 or 2]
        for nodes in [find_borrows(&h).collect_vec(), find_returns(&h).collect()] {
            let mut outer_count = 0;
            for node in nodes {
                let expected_array_type = match get_index(&h, node) {
                    0 => {
                        outer_count += 1;
                        &outer_array_type
                    }
                    1 | 2 => &inner_array_type,
                    idx => panic!("Unexpected index {idx}"),
                };
                assert_eq!(
                    h.get_optype(node)
                        .dataflow_signature()
                        .unwrap()
                        .input_types()[0],
                    *expected_array_type
                );
            }
            // For each CX, two borrows or two returns of the outer array before the op, and two after
            assert_eq!(outer_count, 8);
        }
        // Two CXs, both borrowing their inputs and returning their outputs:
        let [cx1, cx2] = h
            .nodes()
            .filter(|n| {
                h.get_optype(*n)
                    .as_extension_op()
                    .is_some_and(|eop| eop.qualified_id() == "tket.quantum.CX")
            })
            .collect_array()
            .unwrap();
        for cx in [cx1, cx2] {
            assert!(BTreeSet::from_iter(find_returns(&h))
                .is_superset(&h.output_neighbours(cx).collect()));
            assert!(BTreeSet::from_iter(find_borrows(&h))
                .is_superset(&h.input_neighbours(cx).collect()));
        }

        let res = BorrowSquashPass::<BorrowArray>::default()
            .run(&mut h)
            .unwrap();
        h.validate().unwrap();
        assert_eq!(res.len(), 9);
        // Now, one borrow and one return from the outer array
        assert_eq!(
            find_borrows(&h)
                .map(|n| get_index(&h, n))
                .sorted()
                .collect_vec(),
            [0, 1, 2]
        );
        assert_eq!(
            find_returns(&h)
                .map(|n| get_index(&h, n))
                .sorted()
                .collect_vec(),
            [0, 1, 2]
        );
        // CX's should still be there (in same place), but now directly connected:
        for cx in [cx1, cx2] {
            assert!(h
                .get_optype(cx)
                .as_extension_op()
                .is_some_and(|eop| eop.qualified_id() == "tket.quantum.CX"));
        }
        assert!(h.output_neighbours(cx1).all(|n| n == cx2));
    }
}
