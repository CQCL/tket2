//! Provides `LazifyMeasurePass` which replaces [Tket2Op::Measure] nodes with
//! [QSystemOp::Measure] nodes.
//!
//! [Tket2Op::Measure]: tket2::Tk2Op::Measure
//! [QSystemOp::Measure]: crate::extension::qsystem::QSystemOp::Measure

use derive_more::{Display, Error, From};
use hugr::{
    algorithms::{
        // ensure_no_nonlocal_edges,
        non_local::NonLocalEdgesError,
        ComposablePass,
    },
    builder::{DFGBuilder, Dataflow, DataflowHugr as _},
    core::HugrNode,
    extension::prelude::{bool_t, qb_t},
    hugr::{
        hugrmut::HugrMut,
        patch::{PatchHugrMut, PatchVerification},
        views::{RootChecked, SiblingSubgraph},
    },
    ops::{
        handle::{DfgID, NodeHandle as _},
        OpTrait as _,
    },
    types::Signature,
    HugrView, Node, Wire,
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
#[derive(Default, Debug, Clone)]
pub struct LazifyMeasurePass;

impl<H: HugrMut<Node = Node>> ComposablePass<H> for LazifyMeasurePass {
    type Error = LazifyMeasurePassError<Node>;
    type Result = ();

    fn run(&self, hugr: &mut H) -> Result<(), LazifyMeasurePassError<Node>> {
        // TODO uncomment once https://github.com/CQCL/hugr/issues/1234 is complete
        // ensure_no_nonlocal_edges(hugr)?;
        replace_measure_ops(hugr)?;
        Ok(())
    }
}

#[derive(Error, Debug, Display, From)]
#[non_exhaustive]
/// An error reported from [LazifyMeasurePass].
pub enum LazifyMeasurePassError<N> {
    /// The HUGR was found to contain non-local edges.
    NonLocalEdgesError(NonLocalEdgesError<N>),
    /// A [LazifyMeasureRewrite] was constructed targetting an invalid op.
    #[display("A LazifyMeasureRewrite was constructed for node {node} with an invalid signature.\nExpected: {expected_signature}\nActual: {}", actual_signature.as_ref().map_or("None".to_string(), |x| format!("{x}")))]
    #[allow(missing_docs)]
    InvalidOp {
        node: N,
        expected_signature: Signature,
        actual_signature: Option<Signature>,
    },
}

/// Implementation of [LazifyMeasurePass].
///
/// No validation is done here.
pub fn replace_measure_ops(
    hugr: &mut impl HugrMut<Node = Node>,
) -> Result<Vec<Node>, LazifyMeasurePassError<Node>> {
    let nodes_and_rewrites = hugr
        .nodes()
        .filter_map(|n| {
            let optype = hugr.get_optype(n);
            if let Some(Tk2Op::MeasureFree) = optype.cast() {
                Some(LazifyMeasureRewrite::Measure(n))
            } else if let Some(QSystemOp::Measure) = optype.cast() {
                Some(LazifyMeasureRewrite::Measure(n))
            } else if let Some(QSystemOp::MeasureReset) = optype.cast() {
                Some(LazifyMeasureRewrite::MeasureReset(n))
            } else {
                None
            }
            .map(|x| (n, x))
        })
        .collect_vec();

    nodes_and_rewrites
        .into_iter()
        .map(|(n, rw)| {
            hugr.apply_patch(rw)?;
            Ok(n)
        })
        .try_collect()
}

/// A rewrite used in [LazifyMeasurePass] to replace strict measure ops with
/// either [QSystemOp::LazyMeasure] or [QSystemOp::LazyMeasureReset].
pub enum LazifyMeasureRewrite<N = Node> {
    /// Rewrite an op with signature `[QB] -> [BOOL]` to a `LazyMeasure`
    Measure(N),
    /// Rewrite an op with signature `[QB] -> [QB,BOOL]` to a `LazyMeasureReset`
    MeasureReset(N),
}

impl<N: HugrNode> LazifyMeasureRewrite<N> {
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

    // We check that the signature of `op_to_replace` is correct, given the
    // `qsystem_op` we intend to replace it with.
    //
    // Note that calling this private function with a non-sensical `qsystem_op`
    // (i.e.  not LazyMeasure or LazyMeasureReset) will panic.
    fn check_signature(
        node: N,
        qsystem_op: QSystemOp,
        op_to_replace: &hugr::ops::OpType,
    ) -> Result<(), LazifyMeasurePassError<N>> {
        let actual_signature = op_to_replace.dataflow_signature().map(|x| x.into_owned());
        match qsystem_op {
            QSystemOp::LazyMeasure => {
                let expected_signature = Signature::new(qb_t(), bool_t());
                if !actual_signature
                    .as_ref()
                    .is_some_and(|x| x.io() == expected_signature.io())
                {
                    Err(LazifyMeasurePassError::InvalidOp {
                        node,
                        expected_signature,
                        actual_signature,
                    })?
                }
            }
            QSystemOp::LazyMeasureReset => {
                let expected_signature = Signature::new(qb_t(), vec![qb_t(), bool_t()]);
                if !actual_signature
                    .as_ref()
                    .is_some_and(|x| x.io() == expected_signature.io())
                {
                    Err(LazifyMeasurePassError::InvalidOp {
                        node,
                        expected_signature,
                        actual_signature,
                    })?
                }
            }
            op => panic!("bug: {op} is unsupported"),
        }
        Ok(())
    }

    fn get_measure_node(&self) -> N {
        match *self {
            Self::Measure(n) => n,
            Self::MeasureReset(n) => n,
        }
    }

    fn get_lazy_measure_op(&self) -> QSystemOp {
        match self {
            Self::Measure(_) => QSystemOp::LazyMeasure,
            Self::MeasureReset(_) => QSystemOp::LazyMeasureReset,
        }
    }

    /// The signature of the replacement graph.
    fn repl_signature(&self, h: &impl HugrView<Node = N>) -> Signature {
        let inp_sig = vec![qb_t()];
        let out_sig = match self {
            Self::Measure(_) => {
                let n_bools = h.linked_inputs(self.get_measure_node(), 0).count();
                vec![bool_t(); n_bools]
            }
            Self::MeasureReset(_) => {
                let n_bools = h.linked_inputs(self.get_measure_node(), 1).count();
                let mut out_sig = Vec::with_capacity(n_bools + 1);
                out_sig.push(qb_t());
                out_sig.extend(itertools::repeat_n(bool_t(), n_bools));
                out_sig
            }
        };
        Signature::new(inp_sig, out_sig)
    }
}

impl<N: HugrNode + 'static> PatchVerification for LazifyMeasureRewrite<N> {
    type Node = N;
    type Error = LazifyMeasurePassError<N>;

    fn verify(&self, h: &impl HugrView<Node = Self::Node>) -> Result<(), Self::Error> {
        let node = self.get_measure_node();
        Self::check_signature(node, self.get_lazy_measure_op(), h.get_optype(node))
    }

    fn invalidation_set(&self) -> impl Iterator<Item = Self::Node> {
        std::iter::once(self.get_measure_node())
    }
}

impl<N: HugrNode + 'static> PatchHugrMut for LazifyMeasureRewrite<N> {
    type Outcome = ();

    const UNCHANGED_ON_FAILURE: bool = true;

    fn apply_hugr_mut(
        self,
        hugr: &mut impl HugrMut<Node = Self::Node>,
    ) -> Result<Self::Outcome, Self::Error> {
        let meas_node = self.get_measure_node();

        // 1. Check valid signature
        Self::check_signature(
            meas_node,
            self.get_lazy_measure_op(),
            hugr.get_optype(meas_node),
        )?;

        // 2. Store the ports `meas_node` is connected to
        let inputs = hugr
            .node_inputs(meas_node)
            .map(|p| hugr.linked_outputs(meas_node, p).collect_vec())
            .collect_vec();
        let outputs = hugr
            .node_outputs(meas_node)
            .map(|p| hugr.linked_inputs(meas_node, p).collect_vec())
            .collect_vec();
        let sig = self.repl_signature(hugr);

        // 3. Remove the `meas_node` from the HUGR (and store its parent)
        let meas_parent = hugr.get_parent(meas_node).expect("measure has parent");
        hugr.remove_node(meas_node);

        // 4. Create the replacement graph
        let (repl_meas, replacement) = {
            let n_bools = sig.output_count() - matches!(self, Self::MeasureReset(_)) as usize;
            let mut builder = DFGBuilder::new(sig).unwrap();
            let [qb] = builder.input_wires_arr();
            let (repl_meas, mut meas_out_wires) = {
                let handle = builder
                    .add_dataflow_op(self.get_lazy_measure_op(), [qb])
                    .unwrap();
                (handle.node(), handle.outputs().collect_vec())
            };
            let future_wire = meas_out_wires.pop().unwrap();
            let out_wires = Self::build_futures_gadget(&mut builder, future_wire, n_bools);
            (
                repl_meas,
                builder
                    .finish_hugr_with_outputs(meas_out_wires.pop().into_iter().chain(out_wires))
                    .unwrap(),
            )
        };

        // 5. Insert the replacement graph into the HUGR (minus IO and root)
        let node_map = hugr.insert_subgraph(
            meas_parent,
            &replacement,
            &SiblingSubgraph::try_new_dataflow_subgraph(
                RootChecked::<_, DfgID>::try_new(&replacement).unwrap(),
            )
            .unwrap(),
        );

        // 6. Connect old inputs to new `meas_node`
        for (pos, host_nps) in inputs.into_iter().enumerate() {
            for (src_node, src_port) in host_nps {
                hugr.connect(src_node, src_port, node_map[&repl_meas], pos);
            }
        }

        // 7. Connect old outputs to replacement graph outputs
        let [_, out_node] = replacement
            .get_io(replacement.entrypoint())
            .expect("valid dfg");
        // flatten boolean outputs
        let outputs = {
            let bool_ind = matches!(self, Self::MeasureReset(_)) as usize;
            let mut outputs = outputs;
            let bools = outputs.remove(bool_ind);
            outputs.splice(bool_ind..bool_ind, bools.into_iter().map(|x| vec![x]));
            outputs
        };
        for (pos, dst_node_ports) in outputs.into_iter().enumerate() {
            for (dst_node, dst_port) in dst_node_ports {
                let (src_node, src_port) = if let Some((repl_node, repl_port)) =
                    replacement.single_linked_output(out_node, pos)
                {
                    (repl_node, repl_port)
                } else {
                    // order edge
                    debug_assert_eq!(
                        hugr.get_optype(dst_node).other_input_port(),
                        Some(dst_port),
                        "missing dataflow port in replacement"
                    );
                    (
                        repl_meas,
                        replacement
                            .get_optype(repl_meas)
                            .other_output_port()
                            .expect("measure has other port"),
                    )
                };
                hugr.connect(node_map[&src_node], src_port, dst_node, dst_port);
            }
        }

        Ok(())
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
        LazifyMeasurePass.run(&mut hugr).unwrap();
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
        LazifyMeasurePass.run(&mut hugr).unwrap();
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
        LazifyMeasurePass.run(&mut hugr).unwrap();
        assert!(hugr.validate().is_ok());
    }

    #[test]
    fn measure_with_order_edges() {
        let mut hugr = {
            let mut builder = DFGBuilder::new(Signature::new(vec![qb_t(); 4], vec![])).unwrap();
            let [meas1, meas2, meas3, meas4] = builder
                .input_wires_arr()
                .map(|qb| builder.add_measure(qb).unwrap());
            builder.set_order(&meas1.node(), &meas2.node());
            builder.set_order(&meas2.node(), &meas3.node());
            builder.set_order(&meas2.node(), &meas4.node());
            builder.finish_hugr_with_outputs([]).unwrap()
        };
        hugr.validate().unwrap();
        LazifyMeasurePass.run(&mut hugr).unwrap();
        hugr.validate().unwrap();
    }
}
