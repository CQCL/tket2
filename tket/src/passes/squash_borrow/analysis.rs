//! An analysis pass that identifies borrowed resources and their lifetimes.

use std::collections::BTreeMap;

use derive_more::derive::{Display, Error};
use hugr::core::HugrNode;
use hugr::extension::simple_op::MakeExtensionOp;
use hugr::hugr::views::sibling_subgraph::InvalidSubgraph;
use hugr::ops::{constant, OpTrait, OpType};
use hugr::std_extensions::arithmetic::conversions::ConvertOpDef;
use hugr::types::Signature;
use hugr::{types::Type, HugrView};
use hugr::{Direction, IncomingPort, OutgoingPort, Port, PortIndex, Wire};

use crate::resource::{
    CircuitUnit, DefaultResourceFlow, ResourceFlow, ResourceId, ResourceScope, ResourceScopeConfig,
    UnsupportedOp,
};
use crate::Circuit;

/// An analysis pass that identifies borrowed resources and their lifetimes.
pub struct BorrowAnalysis<H: HugrView = hugr::Hugr> {
    /// A predicate that determines if a node is a borrow node.
    pub is_borrow_node: Box<dyn Fn(H::Node, &H) -> bool>,
    /// A predicate that determines if a node is a return node.
    pub is_return_node: Box<dyn Fn(H::Node, &H) -> bool>,
}

/// The errors that can occur when running the borrow analysis pass.
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
    /// Index was not a const....TODO need to handle gracefully
    NonConstIndex,
}

#[derive(Debug, Display, Error)]
enum TrackingError {
    MissingBorrow,
    MissingReturn,
    MismatchedInfo(#[error(not(source))] String),
}

/// Lifespan of a borrowed resource, represented as an interval on the resource
/// that is borrowed from.
#[derive(Clone, Debug)]
pub struct BorrowInterval<N> {
    /// The resource ID that this interval borrows.
    pub borrowed_resource: ResourceId,
    /// The resource ID that this interval borrows from.
    pub borrow_from_resource: ResourceId,
    /// The statically known index used to borrow the resource.
    pub borrow_index: constant::Value,
    /// The node that borrowed the resource.
    ///
    /// This will always be the first node on the resource path of the borrowed
    /// resource.
    ///
    /// This node satisfies that [BorrowAnalysis::is_borrow_node] returns true.
    /// Typically chosen to be a `BArrayUnsafeOpDef::borrow` op or similar.
    pub borrow_node: N,
    /// The node that returns the resource, i.e. the node that "consumes" the
    /// borrowed resource.
    ///
    /// This node satisfies that [BorrowAnalysis::is_return_node] returns true.
    /// Typically chosen to be a `BArrayUnsafeOpDef::return` op or similar.
    pub return_node: N,
}

/// Incomplete information about a borrow interval, used during the analysis.
#[derive(Clone, Debug, PartialEq)]
struct BorrowInfo<N> {
    borrow_from_ty: Type,
    borrow_from_resource: ResourceId,

    borrowed_ty: Type,
    borrowed_resource: ResourceId,

    borrow_index_ty: Type,
    borrow_index_const: constant::Value,

    borrow_node: Option<N>,
    return_node: Option<N>,
}

impl<N: HugrNode> BorrowInfo<N> {
    fn try_from_borrow_node(
        borrow_node: N,
        circuit: &ResourceScope<impl HugrView<Node = N>>,
    ) -> Result<Self, BorrowAnalysisError> {
        let op = circuit.hugr().get_optype(borrow_node);
        let sig = op
            .dataflow_signature()
            .ok_or_else(|| BorrowAnalysisError::NodeNotDataflow { op: op.clone() })?;
        let (borrow_from, borrow_index, borrowed) = parse_borrow_signature(&sig)?;

        Ok(Self {
            borrow_node: Some(borrow_node),
            ..Self::try_from_ports(circuit, borrow_node, borrow_from, borrow_index, borrowed)
                .map_err(|_| BorrowAnalysisError::NonConstIndex)?
        })
    }

    fn try_from_return_node(
        return_node: N,
        circuit: &ResourceScope<impl HugrView<Node = N>>,
    ) -> Result<Self, BorrowAnalysisError> {
        let op = circuit.hugr().get_optype(return_node);
        let sig = op
            .dataflow_signature()
            .ok_or_else(|| BorrowAnalysisError::NodeNotDataflow { op: op.clone() })?;
        let (borrow_from, borrow_index, borrowed) = parse_return_signature(&sig)?;

        Ok(Self {
            return_node: Some(return_node),
            ..Self::try_from_ports(circuit, return_node, borrow_from, borrow_index, borrowed)
                .map_err(|_| BorrowAnalysisError::NonConstIndex)?
        })
    }

    /// Prefer using [Self::try_from_borrow_node] or
    /// [Self::try_from_return_node].
    ///
    /// This will panic if the ports are not valid.
    fn try_from_ports(
        circuit: &ResourceScope<impl HugrView<Node = N>>,
        node: N,
        borrow_from: Port,
        borrow_index: Port,
        borrowed: Port,
    ) -> Result<Self, Wire<N>> {
        let borrow_from_resource = circuit
            .get_circuit_unit(node, borrow_from)
            .and_then(|v| v.as_resource())
            .expect("valid port");
        let borrow_from_ty = circuit
            .hugr()
            .signature(node)
            .and_then(|sig| sig.port_type(borrow_from).cloned())
            .expect("valid port");

        let borrow_index_id = circuit
            .get_circuit_unit(node, borrow_index)
            .and_then(|v| v.as_copyable_wire())
            .expect("valid port");
        let borrow_index_const = circuit
            .as_const(borrow_index_id)
            .ok_or(borrow_index_id)?
            .clone();
        let borrow_index_ty = circuit
            .hugr()
            .signature(node)
            .and_then(|sig| sig.port_type(borrow_index).cloned())
            .expect("valid port");

        let borrowed_resource = circuit
            .get_circuit_unit(node, borrowed)
            .and_then(|v| v.as_resource())
            .expect("valid port");
        let borrowed_ty = circuit
            .hugr()
            .get_optype(node)
            .dataflow_signature()
            .and_then(|sig| sig.port_type(borrowed).cloned())
            .expect("valid port");

        Ok(Self {
            borrow_from_ty,
            borrow_from_resource,
            borrowed_ty,
            borrowed_resource,
            borrow_index_ty,
            borrow_index_const,
            borrow_node: None,
            return_node: None,
        })
    }

    fn try_merge(
        mut interval_start: Self,
        mut interval_end: Self,
    ) -> Result<BorrowInterval<N>, TrackingError> {
        let borrow_node = interval_start
            .borrow_node
            .ok_or(TrackingError::MissingBorrow)?;
        let return_node = interval_end
            .return_node
            .ok_or(TrackingError::MissingReturn)?;
        interval_start.return_node = interval_end.return_node;
        interval_end.borrow_node = interval_start.borrow_node;

        if interval_start != interval_end {
            let Self {
                borrow_from_ty,
                borrow_from_resource,
                borrowed_ty,
                borrowed_resource,
                borrow_index_ty,
                borrow_index_const,
                borrow_node: _,
                return_node: _,
            } = interval_start;
            fn compare<T: std::fmt::Debug + PartialEq>(a: T, b: T, name: &str) -> Option<String> {
                (a != b).then_some(format!("{}: {:?} != {:?}", name, a, b))
            }
            let msg = [
                compare(borrow_from_ty, interval_end.borrow_from_ty, "array_ty"),
                compare(borrow_from_resource, interval_end.borrow_from_resource, "b"),
                compare(borrowed_ty, interval_end.borrowed_ty, "elem_ty"),
                compare(borrowed_resource, interval_end.borrowed_resource, "b_res"),
                compare(borrow_index_ty, interval_end.borrow_index_ty, "b_index_ty"),
                compare(borrow_index_const, interval_end.borrow_index_const, "b_idx"),
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .join("\n");
            return Err(TrackingError::MismatchedInfo(msg));
        }
        Ok(BorrowInterval {
            borrowed_resource: interval_start.borrowed_resource,
            borrow_from_resource: interval_start.borrow_from_resource,
            borrow_index: interval_start.borrow_index_const,
            borrow_node,
            return_node,
        })
    }
}

impl<H: Clone + HugrView<Node = hugr::Node>> BorrowAnalysis<H> {
    /// Run the borrow analysis on the given circuit, i.e. on the DFG sibling
    /// graph of the circuit entrypoint.
    ///
    /// Find all pairs of borrow and return nodes that match within the DFG and
    /// return them as borrow intervals. Nodes may not be matched successfully
    /// for a variety of reasons:
    ///  - the borrow index is not a statically known constant
    ///  - it cannot be established that the borrow and return nodes match up
    ///  - the nodes are another unknown op
    ///
    /// In those cases, the analysis pass proceeds ignoring the node, which may
    /// result in missing borrow intervals.
    pub fn run(
        &self,
        circuit: &Circuit<H>,
    ) -> Result<Vec<BorrowInterval<H::Node>>, InvalidSubgraph> {
        let circuit = ResourceScope::with_config(
            circuit.hugr(),
            circuit.subgraph()?,
            &self.resource_scope_config(),
        );

        Ok(circuit
            .get_resource_starts()
            .filter(|(n, _)| !self.is_borrow_node(*n, circuit.hugr()))
            .flat_map(|(node, resource_id)| self.get_borrow_intervals(&circuit, resource_id, node))
            .collect())
    }

    /// Check if a node is a borrow node.
    ///
    /// A borrow node must have the signature
    /// ```ignore
    /// borrow_array<size, T>, index -> T, borrow_array<size, T>
    /// ```
    /// where `T` is a linear type and `index` is a copyable type.
    pub fn is_borrow_node(&self, node: H::Node, hugr: &H) -> bool {
        (self.is_borrow_node)(node, hugr)
    }

    /// Check if a node is a return node.
    ///
    /// A return node must have the signature
    /// ```ignore
    /// borrow_array<size, T>, index, T -> borrow_array<size, T>
    /// ```
    /// where `T` is a linear type and `index` is a copyable type.
    pub fn is_return_node(&self, node: H::Node, hugr: &H) -> bool {
        (self.is_return_node)(node, hugr)
    }

    /// Traverse the resource path of the given resource and find all pairs
    /// of borrow and return nodes that match up.
    fn get_borrow_intervals(
        &self,
        circuit: &ResourceScope<&H>,
        resource_id: ResourceId,
        inp_node: H::Node,
    ) -> Vec<BorrowInterval<H::Node>> {
        let mut interval_starts = BTreeMap::new();
        let mut complete_intervals = Vec::new();

        for node in circuit.resource_path_iter(resource_id, inp_node, Direction::Outgoing) {
            match node {
                borrow_node if self.is_borrow_node(node, circuit.hugr()) => {
                    let Ok(interval_start) = BorrowInfo::try_from_borrow_node(borrow_node, circuit)
                    else {
                        // Could not parse this borrow, just ignore it.
                        continue;
                    };

                    let already_inserted =
                        interval_starts.insert(interval_start.borrowed_resource, interval_start);

                    debug_assert!(already_inserted.is_none());
                }
                return_node if self.is_return_node(node, circuit.hugr()) => {
                    let Ok(interval_end) = BorrowInfo::try_from_return_node(return_node, circuit)
                    else {
                        // Could not parse this return, just ignore it.
                        continue;
                    };

                    let Some(interval_start) =
                        interval_starts.remove(&interval_end.borrowed_resource)
                    else {
                        // could not find the interval start, ignore
                        continue;
                    };

                    let Ok(complete_interval) = BorrowInfo::try_merge(interval_start, interval_end)
                    else {
                        // the pair of borrow and return does not match up, ignore
                        continue;
                    };

                    complete_intervals.push(complete_interval);
                }
                _ => {}
            }
        }

        complete_intervals
    }

    fn resource_scope_config(&self) -> ResourceScopeConfig<'_, &H> {
        [
            HandleBorrowReturn::new(&self.is_borrow_node, &self.is_return_node).into_boxed(),
            DefaultResourceFlow.into_boxed(),
        ]
        .into_iter()
        .collect()
    }
}

impl<H: HugrView> ResourceScope<H> {
    fn get_value_definition(&self, wire: Wire<H::Node>) -> Option<(H::Node, OutgoingPort)> {
        let mut all_out_value_ports = self
            .nodes()
            .iter()
            .flat_map(|&n| self.hugr().out_value_types(n).map(move |(p, _)| (n, p)));
        all_out_value_ports.find(|&(n, p)| {
            self.get_circuit_unit(n, p).expect("valid port") == CircuitUnit::Copyable(wire)
        })
    }

    fn as_const(&self, wire: Wire<H::Node>) -> Option<&constant::Value> {
        let (def_node, def_port) = self.get_value_definition(wire)?;

        if def_port.index() > 0 {
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

        let mut curr_node = def_node;
        let mut op;
        while {
            op = self.hugr().get_optype(curr_node);
            is_const_conversion_op(op)
        } {
            (curr_node, _) = self
                .hugr()
                .single_linked_output(curr_node, IncomingPort::from(0))
                .expect("invalid signature for conversion op");
        }

        if let OpType::Const(const_op) = op {
            Some(&const_op.value)
        } else {
            None
        }
    }
}

struct HandleBorrowReturn<'a, H: HugrView> {
    /// A predicate that determines if a node is a borrow node.
    pub is_borrow_node: &'a Box<dyn Fn(H::Node, &H) -> bool>,
    /// A predicate that determines if a node is a return node.
    pub is_return_node: &'a Box<dyn Fn(H::Node, &H) -> bool>,
}

impl<'a, H: HugrView> HandleBorrowReturn<'a, H> {
    pub fn new(
        is_borrow_node: &'a Box<dyn Fn(H::Node, &H) -> bool>,
        is_return_node: &'a Box<dyn Fn(H::Node, &H) -> bool>,
    ) -> Self {
        Self {
            is_borrow_node,
            is_return_node,
        }
    }
}

impl<'a, 'h, H: HugrView> ResourceFlow<&'h H> for HandleBorrowReturn<'a, H> {
    fn map_resources(
        &self,
        node: <H>::Node,
        hugr: &&'h H,
        inputs: &[Option<ResourceId>],
    ) -> Result<Vec<Option<ResourceId>>, UnsupportedOp> {
        if (self.is_borrow_node)(node, hugr) {
            let borrowed_resource = inputs[0].expect("linear input");
            Ok(vec![None, Some(borrowed_resource)])
        } else if (self.is_return_node)(node, hugr) {
            let borrowed_resource = inputs[0].expect("linear input");
            Ok(vec![Some(borrowed_resource)])
        } else {
            Err(UnsupportedOp(hugr.get_optype(node).clone()))
        }
    }
}

fn parse_borrow_signature(sig: &Signature) -> Result<(Port, Port, Port), BorrowAnalysisError> {
    let borrow_from_port = IncomingPort::from(0);
    let borrow_index_port = IncomingPort::from(1);
    let borrowed_port = OutgoingPort::from(0);

    let borrow_from_port_outgoing = OutgoingPort::from(1);

    let borrow_from_ty = sig.port_type(borrow_from_port).unwrap();
    let borrow_index_ty = sig.port_type(borrow_index_port).unwrap();
    let borrowed_ty = sig.port_type(borrowed_port).unwrap();

    if !borrow_index_ty.copyable() {
        return Err(BorrowAnalysisError::NonCopyableBorrowIndex);
    }

    if borrow_from_ty.copyable() || borrowed_ty.copyable() {
        let (borrow_from_ty, borrowed_ty) = (borrow_from_ty.clone(), borrowed_ty.clone());
        return Err(BorrowAnalysisError::NonLinearBorrowedResource {
            borrow_from_ty,
            borrowed_ty,
        });
    }

    if sig.port_type(borrow_from_port_outgoing) != Some(borrow_from_ty)
        || sig.input_count() != 2
        || sig.output_count() != 2
    {
        return Err(BorrowAnalysisError::BorrowNodeIncorrectSignature);
    }

    Ok((
        borrow_from_port.into(),
        borrow_index_port.into(),
        borrowed_port.into(),
    ))
}

fn parse_return_signature(sig: &Signature) -> Result<(Port, Port, Port), BorrowAnalysisError> {
    let borrow_from_port = IncomingPort::from(0);
    let borrow_index_port = IncomingPort::from(1);
    let borrowed_port = IncomingPort::from(2);

    let borrow_from_ty = sig.port_type(borrow_from_port).unwrap();
    let borrow_index_ty = sig.port_type(borrow_index_port).unwrap();
    let borrowed_ty = sig.port_type(borrowed_port).unwrap();

    if !borrow_index_ty.copyable() {
        return Err(BorrowAnalysisError::NonCopyableBorrowIndex);
    }

    if borrow_from_ty.copyable() || borrowed_ty.copyable() {
        let (borrow_from_ty, borrowed_ty) = (borrow_from_ty.clone(), borrowed_ty.clone());
        return Err(BorrowAnalysisError::NonLinearBorrowedResource {
            borrow_from_ty,
            borrowed_ty,
        });
    }

    if sig.port_type(OutgoingPort::from(0)) != Some(&borrow_from_ty)
        || sig.input_count() != 3
        || sig.output_count() != 1
    {
        return Err(BorrowAnalysisError::BorrowNodeIncorrectSignature);
    }

    Ok((
        borrow_from_port.into(),
        borrow_index_port.into(),
        borrowed_port.into(),
    ))
}

#[cfg(test)]
mod tests {
    use std::io::BufReader;

    use hugr::{
        hugr::hugrmut::HugrMut, std_extensions::collections::borrow_array::BArrayUnsafeOpDef, Hugr,
        Node,
    };
    use portgraph::NodeIndex;
    use rstest::{fixture, rstest};

    use crate::{extension::REGISTRY, Circuit};

    use super::*;

    #[fixture]
    fn borrow_circuit() -> Circuit {
        let reader = BufReader::new(
            include_bytes!("../../../../test_files/squashing_inline.hugr").as_slice(),
        );
        let mut hugr = Hugr::load(reader, Some(&REGISTRY)).unwrap();
        hugr.set_entrypoint(Node::from(NodeIndex::new(1176)));
        Circuit::new(hugr)
    }

    #[fixture]
    fn inline_borrow_analysis() -> BorrowAnalysis<Hugr> {
        BorrowAnalysis {
            is_borrow_node: Box::new(|node, hugr: &Hugr| {
                let op = hugr.get_optype(node);
                let Some(ext_op) = op.as_extension_op() else {
                    return false;
                };
                BArrayUnsafeOpDef::from_extension_op(ext_op) == Ok(BArrayUnsafeOpDef::borrow)
            }),
            is_return_node: Box::new(|node, hugr: &Hugr| {
                let op = hugr.get_optype(node);
                let Some(ext_op) = op.as_extension_op() else {
                    return false;
                };
                BArrayUnsafeOpDef::from_extension_op(ext_op) == Ok(BArrayUnsafeOpDef::r#return)
            }),
        }
    }

    /// Make sure that the resources flow correctly through borrow and return
    /// nodes.
    #[rstest]
    fn test_borrow_flow(inline_borrow_analysis: BorrowAnalysis, borrow_circuit: Circuit) {
        let scope = ResourceScope::with_config(
            borrow_circuit.hugr(),
            borrow_circuit.subgraph().unwrap(),
            &inline_borrow_analysis.resource_scope_config(),
        );

        for resource_start in borrow_circuit
            .hugr()
            .node_outputs(borrow_circuit.input_node())
        {
            let resource_id = scope
                .get_circuit_unit(borrow_circuit.input_node(), resource_start)
                .and_then(|v| v.as_resource());
            println!("resource_id: {:?}", resource_id);
            println!(
                "resource_path: {:?}",
                scope
                    .resource_path_iter(
                        resource_id.unwrap(),
                        borrow_circuit.input_node(),
                        Direction::Outgoing
                    )
                    .collect::<Vec<_>>()
            );
        }
    }

    #[rstest]
    fn test_borrow_analysis(inline_borrow_analysis: BorrowAnalysis, borrow_circuit: Circuit) {
        let res = inline_borrow_analysis.run(&borrow_circuit).unwrap();

        assert_eq!(res.len(), 17);
    }
}
