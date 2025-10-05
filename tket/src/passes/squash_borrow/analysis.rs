//! An analysis pass that identifies borrowed resources and their lifetimes.

use std::collections::btree_map::Entry;
use std::collections::BTreeMap;

use derive_more::derive::{Display, Error};
use hugr::core::HugrNode;
use hugr::extension::prelude::ConstUsize;
use hugr::extension::simple_op::MakeExtensionOp;
use hugr::hugr::views::sibling_subgraph::InvalidSubgraph;
use hugr::ops::{OpTrait, OpType, Value};
use hugr::std_extensions::arithmetic::conversions::ConvertOpDef;
use hugr::std_extensions::arithmetic::int_types::ConstInt;
use hugr::std_extensions::collections::borrow_array::BArrayUnsafeOpDef;
use hugr::types::Type;
use hugr::{Direction, HugrView, IncomingPort, Node, OutgoingPort, PortIndex, Wire};

use super::{BorrowOrReturn, BorrowFromPorts, BorrowIndex, BRAction};
use crate::resource::{
    CircuitUnit, ResourceFlow, ResourceId, ResourceScope, ResourceScopeConfig, UnsupportedOp,
};
use crate::Circuit;

/// A predicate that determines if a node is a borrow or return node.
///
/// Note: parametrized by [HugrNode] because it has to be to implement [ResourceFlow];
/// we can't actually operate over non-[Node] views because [Circuit::subgraph] is not parametrized.
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
    ) -> Result<Option<(BRAction, BorrowReturnPorts)>, NodeInfoError>;
}

/// An analysis pass that identifies borrowed resources and their lifetimes.
pub struct BorrowAnalysis<BR: IsBorrowReturn> {
    is_br: BR,
}

/// Reasons that a [IsBorrowReturn] may be unable to determine whether a node is a borrow/return.
#[derive(Debug, Display, Error)]
pub enum NodeInfoError {
    /// Borrow op is not a dataflow op.
    #[display("expected dataflow op: {op}")]
    NodeNotDataflow {
        /// The operation that is not a dataflow op.
        op: OpType,
    },
    /// Index was not a const....ALAN need to handle gracefully inside rather than fail analysis
    NonConstIndex,
    /// Borrow op has incorrect signature.
    #[display("borrow_node has incorrect signature")]
    BorrowNodeIncorrectSignature,
    /// Borrow index is not copyable.
    #[display("non-copyable borrow index")]
    NonCopyableBorrowIndex,
    /// Borrowed resource (array or element) is not linear.
    #[display("non-linear borrow of element {borrowed_ty} from array {borrow_from_ty}")]
    #[allow(missing_docs)]
    NonLinearBorrowedResource {
        borrowed_ty: Type,
        borrow_from_ty: Type,
    },
}

/// The errors that can occur when running the borrow analysis pass.
#[derive(Debug, Display, Error)]
pub enum BorrowAnalysisError<N: HugrNode> {
    /// The analysis could not run on an invalid subgraph
    InvalidSubgraph(#[error(source)] InvalidSubgraph),
    /// A node returns to an index that was not borrowed
    ReturnWithoutBorrow(#[error(not(source))] N),
    /// Two borrows of the same index without a return
    #[display("Nodes {_0} and {_1} both borrow the same element without an intervening return")]
    RepeatedBorrow(N, N),
    /// A borrow was not followed by a corresponding return
    BorrowNotReturned(#[error(not(source))] N),
    /// Could not analyse a node
    NodeInfoError(NodeInfoError),
}

/// Incomplete information about a borrow interval, used during the analysis.
#[derive(Clone, Debug, PartialEq)]
pub struct BorrowInfo {
    borrow_from_ty: Type,

    borrowed_ty: Type,

    borrow_index_ty: Type,
    borrow_index_const: u64,
}

impl BorrowInfo {
    /// Construct a new instance given the ports
    ///
    /// # Errors
    ///
    /// If the index is not a constant, or the node does not conform in some other way, return the Wire.
    ///
    /// # Panics
    ///
    /// If the ports are not valid
    fn try_from_ports<N: HugrNode>(
        hugr: &impl HugrView<Node = N>,
        node: N,
        action: BRAction,
        ports: &BorrowReturnPorts,
    ) -> Result<Self, NodeInfoError> {
        let sig = hugr
            .signature(node)
            .ok_or_else(|| NodeInfoError::NodeNotDataflow {
                op: hugr.get_optype(node).clone(),
            })?;

        let borrow_from_ty = sig.port_type(ports.borrow_from_in).expect("valid port");
        if borrow_from_ty != sig.port_type(ports.borrow_from_out).expect("valid port") {
            return Err(NodeInfoError::BorrowNodeIncorrectSignature);
        }

        let borrow_index_ty = sig.port_type(ports.elem_index).expect("valid port");
        let borrowed_ty = sig.port_type(action.borrowed_port()).expect("valid port");

        if !borrow_index_ty.copyable() {
            return Err(NodeInfoError::NonCopyableBorrowIndex);
        }

        if borrow_from_ty.copyable() || borrowed_ty.copyable() {
            let (borrow_from_ty, borrowed_ty) = (borrow_from_ty.clone(), borrowed_ty.clone());
            return Err(NodeInfoError::NonLinearBorrowedResource {
                borrow_from_ty,
                borrowed_ty,
            });
        }

        let borrow_index = Wire::from_connected_port(node, ports.elem_index, hugr);
        let borrow_index_const = find_const(hugr, borrow_index)
            .ok_or(NodeInfoError::NonConstIndex)? // flag the wire here, or return in Self
            .clone();

        Ok(Self {
            borrow_from_ty: borrow_from_ty.clone(),
            borrowed_ty: borrowed_ty.clone(),
            borrow_index_ty: borrow_index_ty.clone(),
            borrow_index_const,
        })
    }

    fn check_eq(&self, other: &Self) -> Result<(), String> {
        // Do not check borrow_from/borrow_from_outgoing ports
        if self == other {
            return Ok(());
        }

        let Self {
            borrow_from_ty,
            borrowed_ty,
            borrow_index_ty,
            borrow_index_const,
        } = self;
        fn compare<T: std::fmt::Debug + PartialEq>(a: T, b: T, name: &str) -> Option<String> {
            (a != b).then_some(format!("{}: {:?} != {:?}", name, a, b))
        }
        let msg = [
            compare(borrow_from_ty, &other.borrow_from_ty, "array_ty"),
            compare(borrowed_ty, &other.borrowed_ty, "elem_ty"),
            compare(borrow_index_ty, &other.borrow_index_ty, "b_index_ty"),
            compare(borrow_index_const, &other.borrow_index_const, "b_idx"),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join("\n");
        Err(msg)
    }
}

impl<BR: IsBorrowReturn> BorrowAnalysis<BR> {
    /// Run the borrow analysis on the given circuit, i.e. on the DFG sibling
    /// graph of the circuit entrypoint.
    ///
    /// Find all pairs of borrow and return nodes that match within the DFG and
    /// return them as borrow intervals. Nodes may not be matched successfully
    /// for a variety of reasons:
    ///  - the borrow index is not a statically known constant. At the present time
    ///    the analysis will fail with [NodeInfoError::NonConstIndex] rather
    ///    than returning [BorrowIndex::Right].
    ///  - it cannot be established that the borrow and return nodes match up
    ///  - the nodes are another unknown op
    ///
    /// In such cases, an error will be returned without any results.
    pub fn run<H: HugrView<Node = Node> + Clone>(
        &self,
        circuit: &Circuit<H>,
        localize_errors: bool,
    ) -> Result<Vec<Vec<BorrowOrReturn>>, BorrowAnalysisError<H::Node>> {
        let subgraph = match circuit.subgraph() {
            Ok(sg) => sg,
            Err(InvalidSubgraph::EmptySubgraph) => return Ok(vec![]),
            Err(e) => return Err(BorrowAnalysisError::InvalidSubgraph(e)),
        };
        // TODO Possibly add ResourceScope::from_circuit_with_config?
        let res_tracker =
            ResourceScope::with_config(circuit.hugr(), subgraph, &self.resource_scope_config());
        let intervals = self.gather_intervals(&res_tracker);
        if localize_errors {
            Ok(intervals.filter_map(Result::ok).collect())
        } else {
            intervals.collect::<Result<Vec<_>, _>>()
        }
    }

    fn gather_intervals<'a, H: HugrView<Node = Node> + Clone + 'a>(
        &'a self,
        circuit: &'a ResourceScope<&'a H>,
    ) -> impl Iterator<Item = Result<Vec<BorrowOrReturn>, BorrowAnalysisError<H::Node>>> + 'a {
        circuit
            .get_resource_starts()
            .map(|(start_node, resource_id)| {
                self.check_actions_paired(circuit, resource_id, start_node)
            })
    }

    /// Traverse the resource path of the given resource and find all pairs
    /// of borrow and return nodes that match up.
    fn check_actions_paired<H: HugrView<Node = Node>>(
        &self,
        circuit: &ResourceScope<&H>,
        resource_id: ResourceId,
        inp_node: H::Node,
    ) -> Result<Vec<BorrowOrReturn>, BorrowAnalysisError<H::Node>> {
        // Analysis only produces constant indices for now
        let mut interval_starts: BTreeMap<u64, (BorrowInfo, Node)> = BTreeMap::new();
        let mut actions = Vec::new();
        let mut must_be_last = None;
        let mut first = true;

        for node in circuit.resource_path_iter(resource_id, inp_node, Direction::Outgoing) {
            if let Some(isnt_last) = must_be_last {
                panic!(
                    "Resource path continued through node {isnt_last} with type {:?}",
                    circuit.hugr().get_optype(isnt_last)
                );
            }

            let is_br = self
                .is_br
                .is_borrow_return(node, circuit.hugr())
                .map_err(BorrowAnalysisError::NodeInfoError)?
                .filter(|(act, ports)| {
                    let bf = circuit.get_circuit_unit(node, ports.borrow_from_in);
                    assert_eq!(bf, circuit.get_circuit_unit(node, ports.borrow_from_out));
                    bf.unwrap() == CircuitUnit::Resource(resource_id) || {
                        // Ignore nested array creation/ending (borrowing from/returning back to parent)
                        assert_eq!(
                            Some(CircuitUnit::Resource(resource_id)),
                            circuit.get_circuit_unit(node, act.borrowed_port())
                        );
                        false
                    }
                });

            if first {
                // First node on path creates the resource, does not borrow from it
                first = false;
                debug_assert!(circuit.is_resource_start(node, resource_id));
                assert!(is_br.is_none());
                continue;
            };

            let Some((action, ports)) = is_br else {
                // Some other op that uses the resource, so we are done tracking borrows
                must_be_last = Some(node);
                continue;
            };
            let info = BorrowInfo::try_from_ports(circuit.hugr(), node, action, &ports)
                .map_err(BorrowAnalysisError::NodeInfoError)?;

            match action {
                BRAction::Borrow(_) => {
                    let ve = match interval_starts.entry(info.borrow_index_const) {
                        Entry::Occupied(oe) => {
                            return Err(BorrowAnalysisError::RepeatedBorrow(oe.get().1, node))
                        }
                        Entry::Vacant(ve) => ve,
                    };
                    actions.push(BorrowOrReturn {
                        node,
                        borrow_index_const: BorrowIndex::Left(info.borrow_index_const),
                        action,
                        borrow_from: BorrowFromPorts {
                            inc: ports.borrow_from_in,
                            out: ports.borrow_from_out,
                        },
                    });
                    ve.insert((info, node));
                }
                BRAction::Return(_) => {
                    let Some(interval_start) = interval_starts.remove(&info.borrow_index_const)
                    else {
                        return Err(BorrowAnalysisError::ReturnWithoutBorrow(node));
                    };
                    // Perhaps we should return the error here, but let's see how it's triggered
                    interval_start.0.check_eq(&info).unwrap();

                    actions.push(BorrowOrReturn {
                        node,
                        borrow_index_const: BorrowIndex::Left(info.borrow_index_const),
                        action,
                        borrow_from: BorrowFromPorts {
                            inc: ports.borrow_from_in,
                            out: ports.borrow_from_out,
                        },
                    })
                }
            }
        }
        // Would like to:
        // assert!(must_be_last.is_some());
        // here, but the ResourceScope does not include e.g. final Output node.

        if let Some((_, n)) = interval_starts.into_values().next() {
            return Err(BorrowAnalysisError::BorrowNotReturned(n));
        }

        Ok(actions)
    }

    fn resource_scope_config<H: HugrView>(&self) -> ResourceScopeConfig<'_, &H> {
        std::iter::once(self.is_br.clone().into_boxed()).collect()
    }
}

pub(super) fn find_const<H: HugrView>(hugr: &H, wire: Wire<H::Node>) -> Option<u64> {
    if wire.source().index() > 0 {
        return None;
    }

    fn is_const_conversion_op(op: &OpType) -> bool {
        if matches!(op, OpType::LoadConstant(..)) {
            true
        } else if let Some(op) = op.as_extension_op() {
            ConvertOpDef::from_extension_op(op) == Ok(ConvertOpDef::itousize)
        } else {
            false
        }
    }

    let mut curr_node = wire.node();
    loop {
        let op = hugr.get_optype(curr_node);
        if is_const_conversion_op(op) {
            (curr_node, _) = hugr
                .single_linked_output(curr_node, IncomingPort::from(0))
                .expect("invalid signature for conversion op");
            continue;
        }

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
}

impl<'h, H: HugrView, BR: IsBorrowReturn> ResourceFlow<&'h H> for BR {
    fn map_resources(
        &self,
        node: H::Node,
        hugr: &&'h H,
        inputs: &[Option<ResourceId>],
    ) -> Result<Vec<Option<ResourceId>>, UnsupportedOp> {
        Ok(
            match self
                .is_borrow_return(node, hugr)
                .map_err(|_| UnsupportedOp(hugr.get_optype(node).clone()))?
            {
                Some((BRAction::Borrow(_), _)) => {
                    let borrowed_resource = inputs[0].expect("linear input");
                    vec![None, Some(borrowed_resource)]
                }
                Some((BRAction::Return(_), _)) => {
                    let borrowed_resource = inputs[0].expect("linear input");
                    vec![Some(borrowed_resource)]
                }
                None => {
                    // All borrows should have been returned before any other op.
                    // Thus, any op touching an array, effectively creates a new array
                    // (begins a fresh sequence of ops)
                    vec![None; hugr.value_types(node, Direction::Outgoing).count()]
                }
            },
        )
    }
}

/// Ports common to a borrow or return op
#[derive(Debug, Clone)]
pub struct BorrowReturnPorts {
    borrow_from_in: IncomingPort,
    elem_index: IncomingPort,
    borrow_from_out: OutgoingPort,
}

/// Implements [IsBorrowReturn] for `BorrowArray`s.
#[derive(Debug, Display, Clone)]
pub struct DefaultBorrowArray;

/// Default [BorrowAnalysis] for `BorrowArray`s
pub type DefaultBorrowAnalysis = BorrowAnalysis<DefaultBorrowArray>;

impl Default for DefaultBorrowAnalysis {
    fn default() -> Self {
        Self {
            is_br: DefaultBorrowArray,
        }
    }
}

impl IsBorrowReturn for DefaultBorrowArray {
    fn is_borrow_return<H: HugrView>(
        &self,
        node: H::Node,
        hugr: &H,
    ) -> Result<Option<(BRAction, BorrowReturnPorts)>, NodeInfoError> {
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
                let ports = BorrowReturnPorts {
                    borrow_from_in: IncomingPort::from(0),
                    elem_index: IncomingPort::from(1),
                    borrow_from_out: OutgoingPort::from(1),
                };

                Some((BRAction::Borrow(OutgoingPort::from(0)), ports))
            }
            Ok(BArrayUnsafeOpDef::r#return) => {
                let op = hugr.get_optype(node);
                let sig = op
                    .dataflow_signature()
                    .ok_or_else(|| NodeInfoError::NodeNotDataflow { op: op.clone() })?;

                if sig.input_count() != 3 || sig.output_count() != 1 {
                    return Err(NodeInfoError::BorrowNodeIncorrectSignature);
                }
                let ports = BorrowReturnPorts {
                    borrow_from_in: IncomingPort::from(0),
                    elem_index: IncomingPort::from(1),
                    borrow_from_out: OutgoingPort::from(0),
                };
                Some((BRAction::Return(IncomingPort::from(2)), ports))
            }
            _ => None,
        })
    }
}

#[cfg(test)]
mod tests {

    use hugr::ops::ExtensionOp;
    use rstest::rstest;

    use crate::Circuit;

    use super::super::test::borrow_circuit;
    use super::*;

    #[rstest]
    fn test_borrow_analysis(borrow_circuit: Circuit) {
        let res = DefaultBorrowAnalysis::default()
            .run(&borrow_circuit, false)
            .unwrap();

        assert_eq!(res.len(), 38); // Arbitrary!

        fn is_borrow_ret(eop: &ExtensionOp) -> bool {
            matches!(
                BArrayUnsafeOpDef::from_extension_op(eop),
                Ok(BArrayUnsafeOpDef::borrow) | Ok(BArrayUnsafeOpDef::r#return)
            )
        }
        let h = borrow_circuit.hugr();
        let num_boro_rets = h
            .children(h.entrypoint())
            .filter(|n| {
                h.get_optype(*n)
                    .as_extension_op()
                    .is_some_and(is_borrow_ret)
            })
            .count();
        assert_eq!(res.iter().map(|v| v.len()).sum::<usize>(), num_boro_rets);
    }
}
