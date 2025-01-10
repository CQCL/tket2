//! Provides `LazifyMeasurePass` which replaces [Tket2Op::Measure] nodes with
//! [QSystemOp::Measure] nodes.
//!
//! [Tket2Op::Measure]: tket2::Tk2Op::Measure
//! [QSystemOp::Measure]: crate::extension::qsystem::QSystemOp::Measure
use std::{iter, mem};

use derive_more::{Display, Error, From};
use hugr::{
    algorithms::{
        ensure_no_nonlocal_edges,
        non_local::NonLocalEdgesError,
        validation::{ValidatePassError, ValidationLevel},
    },
    extension::{
        prelude::bool_t,
        simple_op::{HasConcrete as _, MakeRegisteredOp as _},
    },
    hugr::{hugrmut::HugrMut, Rewrite},
    ops::{DataflowOpTrait as _, OpTrait as _, OpType},
    types::TypeArg,
    HugrView, Node, OutgoingPort,
};
use itertools::Itertools as _;
use tket2::Tk2Op;

use crate::extension::{
    futures::{future_type, FutureOpDef},
    qsystem::QSystemOp,
};

/// A HUGR -> HUGR pass that replaces measurement ops with lazy `tket2.qsystem`
/// measurement ops.
///
/// [Tk2Op::Measure], [QSystemOp::Measure], and [QSystemOp::MeasureReset] nodes
/// are replaced by [QSystemOp::LazyMeasure] and [QSystemOp::LazyMeasureReset]
/// nodes.
///
/// To construct a `LazifyMeasurePass` use [Default::default].
///
/// The HUGR must not contain any non-local edges. If validation is enabled,
/// this precondition will be verified.
#[derive(Default)]
pub struct LazifyMeasurePass(ValidationLevel);

#[derive(Error, Debug, Display, From)]
#[non_exhaustive]
/// An error reported from [LazifyMeasurePass].
pub enum LazifyMeasurePassError {
    /// The HUGR was invalid either before or after a pass ran.
    ValidationError(ValidatePassError),
    /// The HUGR was found to contain non-local edges.
    NonLocalEdgesError(NonLocalEdgesError),
    #[display("LazifyMeasureRewrite applied to node {} with invalid optype {}", 0, 1)]
    /// A [LazifyMeasureRewrite] failed during the running of the pass.
    InvalidOpType(Node, OpType),
}

impl LazifyMeasurePass {
    /// Run `LazifyMeasurePass` on the given [HugrMut]. `registry` is used for
    /// validation, if enabled.
    pub fn run(&self, hugr: &mut impl HugrMut) -> Result<(), LazifyMeasurePassError> {
        self.0.run_validated_pass(hugr, |hugr, level| {
            if *level != ValidationLevel::None {
                ensure_no_nonlocal_edges(hugr)?;
            }
            replace_measure_ops(hugr)?;
            Ok(())
        })
    }

    /// Returns a new `LazifyMeasurePass` with the given [ValidationLevel].
    pub fn with_validation_level(mut self, level: ValidationLevel) -> Self {
        self.0 = level;
        self
    }
}

/// Implementation of [LazifyMeasurePass].
///
/// No validation is done here.
pub fn replace_measure_ops(hugr: &mut impl HugrMut) -> Result<Vec<Node>, LazifyMeasurePassError> {
    let mut nodes_and_rewrites = hugr
        .nodes()
        .filter_map(|n| {
            LazifyMeasureRewrite::replacement_for_op(hugr.get_optype(n)).map(|r| {
                (
                    n,
                    LazifyMeasureRewrite {
                        node: n,
                        replacement: r,
                    },
                )
            })
        }).collect_vec();
    nodes_and_rewrites.sort_by_key(|x| x.0);


    nodes_and_rewrites.into_iter().map(|(n,rw)| {
        hugr.apply_rewrite(rw)?;
        Ok(n)
    }).try_collect()
}

/// A rewrite used in [LazifyMeasurePass]
///
/// When applied to a HUGR `node` will be replaced with a new node of optype
/// `replacement`.
///
/// The signatures must match, excepting that `replacement` may
/// have outgoing ports with type `tket2.futures.future<bool>` where `node`.
/// has `bool`.
///
/// Applying this rewrite will panic if replacement is not one of
/// [QSystemOp::LazyMeasure] or [QSystemOp::LazyMeasureReset].
pub struct LazifyMeasureRewrite {
    /// The node that will be replaced by the rewrite.
    pub node: Node,
    /// The [OpType] with which `node` will be replaced.
    pub replacement: QSystemOp,
}

impl LazifyMeasureRewrite {
    fn rewire(
        &self,
        hugr: &mut impl HugrMut,
        new_node: Node,
    ) -> Result<(), LazifyMeasurePassError> {
        // We will insert and wire up self.replacement, then future.dup or free any
        // future<bool> outputs as necessary. We will then insert one future.read op
        // per future<bool> wire and connect those read ops to the outputs of self.node.
        //
        // We do not use a SimpleReplacement here because the replacement of
        // self.node has a different signature. We instead use HugrMut directly.
        for inport in hugr.node_inputs(self.node).collect_vec() {
            if let Some((linked_node, outport)) = hugr.single_linked_output(self.node, inport) {
                hugr.disconnect(self.node, inport);
                hugr.connect(linked_node, outport, new_node, inport);
            }
        }

        // the outports of the new node that have had their type changed
        // from <bool> -> future<bool>
        let future_ports = self.future_ports(hugr.get_optype(self.node))?;

        let outports = {
            let mut v = hugr.node_outputs(self.node).collect_vec();
            v.sort();
            v
        };
        for outport in outports {
            let uses = {
                let mut v = hugr.linked_inputs(self.node, outport).collect_vec();
                v.sort();
                v
            };
            hugr.disconnect(self.node, outport);
            if !future_ports.contains(&outport) {
                // This case is for outports whose type has not changed
                for (linked_node, inport) in uses {
                    hugr.connect(new_node, outport, linked_node, inport);
                }
            } else if uses.is_empty() {
                // This case is for outports whose type has changed and which
                // have no linked inports.
                let free_node = hugr.add_node_after(
                    new_node,
                    FutureOpDef::Free
                        .instantiate(&[TypeArg::from(bool_t())])
                        .unwrap(),
                );
                hugr.connect(new_node, outport, free_node, 0);
            } else {
                // This case is for outports whose type has change and which
                // have one or more uses.

                // First we create enough future outports for our uses.
                let mut future_wires = vec![(new_node, outport)];

                for i in 1..uses.len() {
                    let prev_wire = &mut future_wires[i - 1];
                    let dup_node = hugr.add_node_after(
                        prev_wire.0,
                        FutureOpDef::Dup
                            .instantiate(&[TypeArg::from(bool_t())])
                            .unwrap(),
                    );
                    let (prev_node, prev_port) = mem::replace(prev_wire, (dup_node, 0.into()));
                    hugr.disconnect(prev_node, prev_port);
                    hugr.connect(prev_node, prev_port, dup_node, 0);

                    future_wires.push((dup_node, 1.into()));
                    debug_assert_eq!(future_wires.len(), uses.len());
                    debug_assert_eq!(
                        {
                            let mut x = future_wires.clone();
                            x.dedup();
                            x
                        },
                        future_wires
                    );
                }

                // we consume each future wire by reading and connecting it to a
                // use
                for (i, (linked_node, inport)) in uses.into_iter().enumerate() {
                    let read_node = hugr.add_node_after(
                        linked_node,
                        FutureOpDef::Read
                            .instantiate(&[TypeArg::from(bool_t())])
                            .unwrap(),
                    );
                    hugr.connect(future_wires[i].0, future_wires[i].1, read_node, 0);
                    hugr.connect(read_node, 0, linked_node, inport);
                }
            }
        }
        Ok(())
    }

    fn replacement_for_op(optype: &OpType) -> Option<QSystemOp> {
        // We don't lazify Tk2Op::Measure because there is no benefit.
        // Implementing the semantics of that op with QSystemOps requires
        // immediately branching on the value of the measurement, defeating any
        // lazyness. See QSystemOpBuilder::build_measure_flip.
        if let Some(Tk2Op::MeasureFree) = optype.cast() {
            Some(QSystemOp::LazyMeasure)
        } else if let Some(QSystemOp::Measure) = optype.cast() {
            Some(QSystemOp::LazyMeasure)
        } else if let Some(QSystemOp::MeasureReset) = optype.cast() {
            Some(QSystemOp::LazyMeasureReset)
        } else {
            None
        }
    }

    fn future_ports(&self, optype: &OpType) -> Result<Vec<OutgoingPort>, LazifyMeasurePassError> {
        let replacement = Self::replacement_for_op(optype).ok_or(
            LazifyMeasurePassError::InvalidOpType(self.node, optype.clone()),
        )?;

        let result = match replacement {
            QSystemOp::LazyMeasure => Ok(vec![0.into()]),
            QSystemOp::LazyMeasureReset => Ok(vec![1.into()]),
            op => panic!("bug: invalid replacement for {optype}: {op}"),
        };
        #[cfg(debug_assertions)]
        {
            let (before_sig, after_sig) = (
                optype.dataflow_signature().unwrap(),
                self.replacement
                    .to_extension_op()
                    .unwrap()
                    .signature()
                    .into_owned(),
            );

            assert_eq!(before_sig.input(), after_sig.input());

            assert!(
                itertools::zip_eq(before_sig.output().iter(), after_sig.output().iter())
                    .all(|(before, after)| before == after
                        || (before == &bool_t() && after == &future_type(bool_t())))
            )
        }
        result
    }
}

impl Rewrite for LazifyMeasureRewrite {
    type ApplyResult = ();
    type Error = LazifyMeasurePassError;
    const UNCHANGED_ON_FAILURE: bool = true;

    fn apply(self, hugr: &mut impl HugrMut) -> Result<Self::ApplyResult, Self::Error> {
        let new_node = hugr.add_node_before(self.node, self.replacement);
        self.rewire(hugr, new_node)?;
        hugr.remove_node(self.node);
        Ok(())
    }

    fn verify(&self, h: &impl HugrView) -> Result<(), Self::Error> {
        let _ = self.future_ports(h.get_optype(self.node))?;
        Ok(())
    }

    fn invalidation_set(&self) -> impl Iterator<Item = Node> {
        iter::once(self.node)
    }
}

#[cfg(test)]
mod test {

    use hugr::{
        builder::{DFGBuilder, Dataflow as _, DataflowHugr as _},
        extension::prelude::qb_t,
        types::Signature,
    };

    use crate::extension::{
        futures::FutureOpDef,
        qsystem::{QSystemOp, QSystemOpBuilder as _},
    };

    use super::*;

    #[test]
    fn simple() {
        let mut hugr = {
            let mut builder =
                DFGBuilder::new(Signature::new(vec![qb_t(), qb_t()], bool_t())).unwrap();
            let [qb1, qb2] = builder.input_wires_arr();
            let [r1] = builder
                .add_dataflow_op(Tk2Op::MeasureFree, [qb1])
                .unwrap()
                .outputs_arr();
            let [qb2, _r2] = builder.add_measure_reset(qb2).unwrap();
            let _r3 = builder.add_measure(qb2).unwrap();
            builder.finish_hugr_with_outputs([r1]).unwrap()
        };
        LazifyMeasurePass::default().run(&mut hugr).unwrap();
        hugr.validate().unwrap();

        let mut num_read = 0;
        let mut num_lazy_measure = 0;
        let mut num_lazy_measure_reset = 0;
        for n in hugr.nodes() {
            let ot = hugr.get_optype(n);
            if let Some(FutureOpDef::Read) = ot.cast() {
                num_read += 1;
            } else if let Some(QSystemOp::LazyMeasure) = ot.cast() {
                num_lazy_measure += 1;
            } else if let Some(QSystemOp::LazyMeasureReset) = ot.cast() {
                num_lazy_measure_reset += 1;
            } else {
                assert_eq!(ot.cast::<Tk2Op>(), None)
            }
        }

        assert_eq!(1, num_read);
        assert_eq!(2, num_lazy_measure);
        assert_eq!(1, num_lazy_measure_reset);
    }

    #[test]
    fn multiple_uses() {
        let mut builder =
            DFGBuilder::new(Signature::new(qb_t(), vec![bool_t(), bool_t()])).unwrap();
        let [qb] = builder.input_wires_arr();
        let [bool] = builder
            .add_dataflow_op(Tk2Op::MeasureFree, [qb])
            .unwrap()
            .outputs_arr();
        let mut hugr = builder.finish_hugr_with_outputs([bool, bool]).unwrap();
        LazifyMeasurePass::default().run(&mut hugr).unwrap();
        hugr.validate().unwrap();
    }

    #[test]
    fn no_uses() {
        let mut builder = DFGBuilder::new(Signature::new_endo(qb_t())).unwrap();
        let [qb] = builder.input_wires_arr();
        let [qb, _] = builder
            .add_dataflow_op(Tk2Op::Measure, [qb])
            .unwrap()
            .outputs_arr();
        let mut hugr = builder.finish_hugr_with_outputs([qb]).unwrap();
        LazifyMeasurePass::default().run(&mut hugr).unwrap();
        assert!(hugr.validate_no_extensions().is_ok());
    }
}
