//! Provides `LazifyMeasurePass` which replaces [Tket2Op::Measure] nodes with
//! [QSystemOp::Measure] nodes.
//!
//! [Tket2Op::Measure]: tket2::Tk2Op::Measure
//! [QSystemOp::Measure]: crate::extension::qsystem::QSystemOp::Measure
use std::{collections::HashMap, iter};

use delegate::delegate;
use derive_more::{Display, Error, From};
use hugr::{
    algorithms::{
        ensure_no_nonlocal_edges,
        non_local::NonLocalEdgesError,
        validation::{ValidatePassError, ValidationLevel},
    },
    builder::{DFGBuilder, Dataflow, DataflowHugr as _},
    extension::{
        prelude::{bool_t, qb_t},
        simple_op::MakeRegisteredOp as _,
    },
    hugr::{hugrmut::HugrMut, views::SiblingSubgraph, Rewrite, SimpleReplacementError},
    ops::{handle::NodeHandle as _, DataflowOpTrait as _, OpTrait as _},
    types::Signature,
    HugrView, Node, SimpleReplacement, Wire,
};
use itertools::Itertools as _;
use tket2::Tk2Op;

use crate::extension::{futures::FutureOpBuilder as _, qsystem::QSystemOp};

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
    /// A [LazyMeasureRewrite] was constructed targetting an invalid op.
    #[display("A LazyMeasureRewrite was constructed for node {node} with an invalid signature.\nExpected: {expected_signature}\nActual: {}", actual_signature.as_ref().map_or("None".to_string(), |x| format!("{x}")))]
    #[allow(missing_docs)]
    InvalidOp {
        node: Node,
        expected_signature: Signature,
        actual_signature: Option<Signature>,
    },
    /// A [SimpleReplacement] failed during the running of the pass.
    SimpleReplacementError(SimpleReplacementError),
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
            let optype = hugr.get_optype(n);
            if let Some(Tk2Op::MeasureFree) = optype.cast() {
                Some(LazifyMeasureRewrite::try_new_measure(n, &hugr))
            } else if let Some(QSystemOp::Measure) = optype.cast() {
                Some(LazifyMeasureRewrite::try_new_measure(n, &hugr))
            } else if let Some(QSystemOp::MeasureReset) = optype.cast() {
                Some(LazifyMeasureRewrite::try_new_measure_reset(n, &hugr))
            } else {
                None
            }
            .map(|x| x.map(|y| (n, y)))
        })
        .collect::<Result<Vec<_>, _>>()?;

    nodes_and_rewrites.sort_by_key(|x| x.0);

    nodes_and_rewrites
        .into_iter()
        .map(|(n, rw)| {
            hugr.apply_rewrite(rw)?;
            Ok(n)
        })
        .try_collect()
}

/// A rewrite used in [LazifyMeasurePass] to replace strict measure ops with
/// either [QSystem::LazyMeasure] or [QSystem::LazyMeasureReset].
pub struct LazifyMeasureRewrite(SimpleReplacement);

impl LazifyMeasureRewrite {
    /// Construct a new `LazifyMeasureRewrite` replacing `node` with a
    /// [QSystemOp::LazyMeasure].
    ///
    /// Fails if node does not have signature `[QB] -> [BOOL]`
    pub fn try_new_measure(
        node: Node,
        hugr: impl HugrView,
    ) -> Result<Self, LazifyMeasurePassError> {
        Self::check_signature(node, QSystemOp::LazyMeasure, hugr.get_optype(node))?;

        let subgraph = SiblingSubgraph::from_node(node, &hugr);
        let uses = {
            let mut v = hugr.linked_inputs(node, 0).collect_vec();
            v.sort();
            v
        };
        let (lazy_measure_node, replacement) = {
            let bool_uses = uses.len();
            let mut builder =
                DFGBuilder::new(Signature::new(qb_t(), vec![bool_t(); bool_uses])).unwrap();
            let [qb] = builder.input_wires_arr();
            let (lazy_measure_node, future_wire) = {
                let handle = builder
                    .add_dataflow_op(QSystemOp::LazyMeasure, [qb])
                    .unwrap();
                (handle.node(), handle.out_wire(0))
            };
            let out_wires = Self::build_futures_gadget(&mut builder, future_wire, bool_uses);
            (
                lazy_measure_node,
                builder.finish_hugr_with_outputs(out_wires).unwrap(),
            )
        };
        let nu_inp = HashMap::from_iter([((lazy_measure_node, 0.into()), (node, 0.into()))]);
        let nu_out = iter::zip(uses, (0..).map_into()).collect();

        Ok(Self(SimpleReplacement::new(
            subgraph,
            replacement,
            nu_inp,
            nu_out,
        )))
    }

    /// Construct a new `LazifyMeasureRewrite` replacing `node` with a
    /// [QSystemOp::LazyMeasureReset].
    ///
    /// Fails if node does not have signature `[QB] -> [QB,BOOL]`
    pub fn try_new_measure_reset(
        node: Node,
        hugr: impl HugrView,
    ) -> Result<Self, LazifyMeasurePassError> {
        Self::check_signature(node, QSystemOp::LazyMeasureReset, hugr.get_optype(node))?;

        let subgraph = SiblingSubgraph::from_node(node, &hugr);
        let uses = {
            let mut v = hugr.linked_inputs(node, 1).collect_vec();
            v.sort();
            v
        };
        let (lazy_measure_reset_node, replacement) = {
            let bool_uses = uses.len();
            let mut builder = {
                let outputs = iter::once(qb_t())
                    .chain(itertools::repeat_n(bool_t(), bool_uses))
                    .collect_vec();
                DFGBuilder::new(Signature::new(qb_t(), outputs)).unwrap()
            };
            let [qb] = builder.input_wires_arr();
            let (lazy_measure_reset_node, [qb_wire, future_wire]) = {
                let handle = builder
                    .add_dataflow_op(QSystemOp::LazyMeasureReset, [qb])
                    .unwrap();
                (handle.node(), handle.outputs_arr())
            };
            let out_wires = Self::build_futures_gadget(&mut builder, future_wire, bool_uses);
            (
                lazy_measure_reset_node,
                builder
                    .finish_hugr_with_outputs(iter::once(qb_wire).chain(out_wires))
                    .unwrap(),
            )
        };
        let nu_inp = HashMap::from_iter([((lazy_measure_reset_node, 0.into()), (node, 0.into()))]);
        let qb_use = hugr.single_linked_input(node, 0).unwrap(); // qubit is linear so this can't fail
        let nu_out = iter::zip(iter::once(qb_use).chain(uses), (0..).map_into()).collect();

        Ok(Self(SimpleReplacement::new(
            subgraph,
            replacement,
            nu_inp,
            nu_out,
        )))
    }

    fn build_futures_gadget(builder: &mut impl Dataflow, wire: Wire, num_uses: usize) -> Vec<Wire> {
        let future_wires = if num_uses == 0 {
            builder.add_free(wire, bool_t()).unwrap();
            vec![]
        } else {
            let mut future_wires = vec![wire];
            for _ in 1..num_uses {
                let prev_wire = future_wires.last_mut().unwrap();
                let [wire1, wire2] = builder.add_dup(*prev_wire, bool_t()).unwrap();
                *prev_wire = wire1;
                future_wires.push(wire2);
            }
            future_wires
        };
        debug_assert_eq!(future_wires.len(), num_uses);

        future_wires
            .into_iter()
            .map(|w| builder.add_read(w, bool_t()).unwrap()[0])
            .collect_vec()
    }

    fn check_signature(
        node: Node,
        qsystem_op: QSystemOp,
        op_to_replace: &hugr::ops::OpType,
    ) -> Result<(), LazifyMeasurePassError> {
        let expected_signature = qsystem_op
            .to_extension_op()
            .unwrap()
            .signature()
            .into_owned();
        let actual_signature = op_to_replace
            .dataflow_signature()
            .map(std::borrow::Cow::into_owned);
        if let Some(sig) = &actual_signature {
            if (expected_signature.input(), expected_signature.output())
                == (sig.input(), sig.output())
            {
                return Ok(());
            }
        }
        Err(LazifyMeasurePassError::InvalidOp {
            node,
            expected_signature,
            actual_signature,
        })
    }
}

impl Rewrite for LazifyMeasureRewrite {
    type ApplyResult = <SimpleReplacement as Rewrite>::ApplyResult;
    type Error = <SimpleReplacement as Rewrite>::Error;
    const UNCHANGED_ON_FAILURE: bool = <SimpleReplacement as Rewrite>::UNCHANGED_ON_FAILURE;

    delegate! {
        to self.0 {
            fn apply(self, hugr: &mut impl HugrMut) -> Result<Self::ApplyResult, Self::Error>;
            fn verify(&self, h: &impl HugrView) -> Result<(), Self::Error>;
            fn invalidation_set(&self) -> impl Iterator<Item = Node>;
        }
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
