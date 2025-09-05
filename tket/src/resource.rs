//! Resource tracking within HUGR DFG subgraphs.
//!
//! This module implements the resource tracking system. It provides
//! facilities for tracking linear resources (such as qubits) through quantum
//! circuits represented as HUGR subgraphs.
//!
//! # Overview
//!
//! HUGR has a notion of "Value": the data that corresponds to a wire within a
//! dataflow graph. It further has a notion of "linear value" a.k.a non-copyable
//! value: a value that cannot be copied or discarded (implicitly).
//!
//! As far as HUGR is concerned, a linear value (or any value, for that matter)
//! is born at an op's output and dies at the next op's input. TKET introduces
//! the notion of "Resource" to extend the lifetime of a linear value over
//! multiple ops.
//!
//! If a linear value appears both in an op's input and output, we say that it
//! is "resource-preserving". Using [`ResourceFlow`], we can track resources
//! as they "flow" through multiple operations. The chains of
//! resource-preserving ops acting on a same resource form a so-called resource
//! path.
//!
//! # Resources and Copyable Values
//!
//! Resource tracking distinguishes between two types of values:
//!
//! - **Linear resources**: Non-copyable values that form resource paths through
//!   the circuit. Each resource has a unique [`ResourceId`] and operations on
//!   the same resource are ordered by a [`Position`]. The [`ResourceFlow`]
//!   trait determines how resources are passed through, discarded or created by
//!   an op at its linear ports.
//! - **Copyable values**: Regular values that can be copied and discarded
//!   freely. Each is identified uniquely by a [`hugr::Wire`], i.e. the outgoing
//!   port defining the value. These are not tracked across ops and do not form
//!   resources.
//!
//! # Resource Scope
//!
//! Tracking resources is not free: there is a one-off linear cost to compute
//! the resource paths, plus a linear memory cost to store them.
//!
//! Use a [`SiblingSubgraph`] to define a region of a `HUGR`, within which
//! resources should be tracked. You can then construct a resource-tracked scope
//! using [`ResourceScope::new`].
//!
//! [`SiblingSubgraph`]: hugr::hugr::views::SiblingSubgraph

// Public API exports
pub use flow::{DefaultResourceFlow, ResourceFlow, UnsupportedOp};
use hugr::{
    extension::simple_op::MakeExtensionOp,
    hugr::hugrmut::HugrMut,
    ops::{constant, OpType},
    std_extensions::arithmetic::{conversions::ConvertOpDef, float_types::ConstF64},
    HugrView, IncomingPort, PortIndex,
};
use hugr_core::hugr::internal::NodeType;
pub use interval::{Interval, InvalidInterval};
use itertools::Itertools;
pub use scope::{CircuitRewriteError, ResourceScope, ResourceScopeConfig};
pub use types::{CircuitUnit, Position, ResourceAllocator, ResourceId};

use crate::{
    circuit::{CircuitHash, HashError},
    extension::rotation::{ConstRotation, RotationOp},
    rewrite::trace::RewriteTrace,
};

// Internal modules
mod convex_checker;
mod flow;
mod interval;
mod scope;
mod types;

// Below a bunch of methods that delegate to circuit.
// TODO: clean up once we decide what to do with the `Circuit` type.

impl<H: HugrMut> ResourceScope<H> {
    /// Enable rewrite tracing for the circuit.
    #[inline]
    pub fn enable_rewrite_tracing(&mut self) {
        self.as_circuit_mut().enable_rewrite_tracing();
    }

    /// Register a rewrite applied to the circuit.
    ///
    /// Returns `true` if the rewrite was successfully registered, or `false` if
    /// it was ignored.
    #[inline]
    pub fn add_rewrite_trace(&mut self, rewrite: impl Into<RewriteTrace>) -> bool {
        self.as_circuit_mut().add_rewrite_trace(rewrite)
    }
}

impl<H: HugrView> ResourceScope<H> {
    /// Returns the traces of rewrites applied to the circuit.
    ///
    /// Returns `None` if rewrite tracing is not enabled for this circuit.
    #[inline]
    pub fn rewrite_trace(&self) -> Option<impl Iterator<Item = RewriteTrace> + '_> {
        self.as_circuit()
            .rewrite_trace()
            .map(|rs| rs.collect_vec().into_iter())
    }

    /// The number of operations in the circuit.
    ///
    /// This includes [`TketOp`]s, pytket ops, and any other custom operations.
    ///
    /// Nested circuits are traversed to count their operations.
    ///
    ///   [`TketOp`]: crate::TketOp
    pub fn num_operations(&self) -> usize {
        self.as_circuit().num_operations()
    }

    /// Returns the node containing the circuit definition.
    pub fn parent(&self) -> H::Node {
        self.as_circuit().parent()
    }

    /// The constant value of a circuit unit.
    pub fn as_const_value(&self, unit: CircuitUnit<H::Node>) -> Option<&constant::Value> {
        let (mut curr_node, outport) = match unit {
            CircuitUnit::Resource(..) => None,
            CircuitUnit::Copyable(wire) => Some((wire.node(), wire.source())),
        }?;

        if outport.index() > 0 {
            return None;
        }

        fn is_const_conversion_op(op: &OpType) -> bool {
            if matches!(op, OpType::LoadConstant(..)) {
                true
            } else if let Some(op) = op.as_extension_op() {
                if let Ok(op) = ConvertOpDef::from_extension_op(op) {
                    op == ConvertOpDef::itousize
                } else if let Ok(op) = RotationOp::from_extension_op(op) {
                    matches!(
                        op,
                        RotationOp::from_halfturns_unchecked | RotationOp::from_halfturns
                    )
                } else {
                    false
                }
            } else {
                false
            }
        }

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

    /// The constant f64 value of a circuit unit (if it is a constant f64).
    pub fn as_const_f64(&self, unit: CircuitUnit<H::Node>) -> Option<f64> {
        let const_val = self.as_const_value(unit)?;
        if let Some(const_rot) = const_val.get_custom_value::<ConstRotation>() {
            return Some(const_rot.half_turns());
        } else if let Some(const_f64) = const_val.get_custom_value::<ConstF64>() {
            return Some(const_f64.value());
        } else {
            panic!("unknown constant type: {:?}", const_val);
        }
    }
}

impl<H: HugrView<Node = hugr::Node>> CircuitHash for ResourceScope<H> {
    fn circuit_hash(&self, parent: hugr::Node) -> Result<u64, HashError> {
        self.as_circuit().circuit_hash(parent)
    }
}

impl<H: HugrView> NodeType for ResourceScope<H> {
    type Node = H::Node;
}

#[cfg(test)]
pub(crate) mod tests {
    use hugr::{
        builder::{DFGBuilder, Dataflow, DataflowHugr},
        extension::prelude::qb_t,
        types::Signature,
        CircuitUnit, Hugr,
    };

    use itertools::Itertools;
    use rstest::rstest;

    use crate::{
        extension::rotation::{rotation_type, ConstRotation},
        resource::scope::tests::ResourceScopeReport,
        utils::build_simple_circuit,
        Circuit, TketOp,
    };

    use super::ResourceScope;

    /// A two-qubit circuit with `n_cx` CNOTs.
    pub fn cx_circuit(n_cx: usize) -> Hugr {
        build_simple_circuit(2, |circ| {
            for _ in 0..n_cx {
                circ.append(TketOp::CX, [0, 1])?;
            }
            Ok(())
        })
        .unwrap()
        .into_hugr()
    }

    // Gate being commuted has a non-linear input
    pub fn cx_rz_circuit(n_qubits: usize, add_rz: bool, add_const_rz: bool) -> Hugr {
        let build = || {
            let out_qb_row = vec![qb_t(); n_qubits];
            let mut inp_qb_row = out_qb_row.clone();
            if add_rz {
                inp_qb_row.push(rotation_type());
            };
            let mut dfg = DFGBuilder::new(Signature::new(inp_qb_row, out_qb_row))?;

            let (qubits, f) = if add_rz {
                let mut inputs = dfg.input_wires().collect_vec();
                let f = inputs.pop().unwrap();
                (inputs, Some(f))
            } else {
                (dfg.input_wires().collect_vec(), None)
            };

            let mut circ = dfg.as_circuit(qubits);

            for i in 0..n_qubits {
                circ.append(TketOp::H, [i])?;
            }
            for i in (0..n_qubits).step_by(2) {
                if i + 1 < n_qubits {
                    circ.append(TketOp::CX, [i, i + 1])?;
                }
            }
            if let Some(f) = f {
                for i in 0..n_qubits {
                    circ.append_and_consume(
                        TketOp::Rz,
                        [CircuitUnit::Linear(i), CircuitUnit::Wire(f)],
                    )?;
                }
            }
            if add_const_rz {
                let const_angle = circ.add_constant(ConstRotation::PI_2);
                for i in 0..n_qubits {
                    circ.append_and_consume(
                        TketOp::Rz,
                        [CircuitUnit::Linear(i), CircuitUnit::Wire(const_angle)],
                    )?;
                }
            }
            let qbs = circ.finish();
            dfg.finish_hugr_with_outputs(qbs)
        };
        build().unwrap()
    }

    #[rstest]
    #[case(2, false, false)]
    #[case(2, true, false)]
    #[case(2, false, true)]
    #[case(2, true, true)]
    #[case(4, false, false)]
    #[case(4, true, false)]
    #[case(4, false, true)]
    #[case(4, true, true)]
    fn test_resource_scope_creation(
        #[case] n_qubits: usize,
        #[case] add_rz: bool,
        #[case] add_const_rz: bool,
    ) {
        let circ = cx_rz_circuit(n_qubits, add_rz, add_const_rz);
        let subgraph = Circuit::from(&circ).try_to_subgraph().unwrap();
        let scope = ResourceScope::new(&circ, subgraph);
        let info = ResourceScopeReport::from(&scope);

        let mut name = format!("{n_qubits}_qubits");
        if add_rz {
            name.push('_');
            name.push_str("add_rz");
        }
        if add_const_rz {
            name.push('_');
            name.push_str("add_const_rz");
        }

        assert_eq!(info.resource_paths.len(), n_qubits);
        assert_eq!(info.n_copyable, add_const_rz as usize + add_rz as usize);
        insta::assert_snapshot!(name, info);
    }
}
