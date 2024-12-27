//! Provides `LazifyMeasurePass` which replaces [Tket2Op::Measure] nodes with
//! [QSystemOp::Measure] nodes.
//!
//! [Tket2Op::Measure]: tket2::Tk2Op::Measure
//! [QSystemOp::Measure]: crate::extension::qsystem::QSystemOp::Measure
use std::collections::{HashMap, HashSet};

use derive_more::{Display, Error, From};
use hugr::{
    algorithms::{
        ensure_no_nonlocal_edges,
        non_local::NonLocalEdgesError,
        validation::{ValidatePassError, ValidationLevel},
    },
    builder::{DFGBuilder, Dataflow, DataflowHugr},
    extension::prelude::{bool_t, qb_t},
    hugr::{hugrmut::HugrMut, views::SiblingSubgraph, Rewrite, SimpleReplacementError},
    types::Signature,
    Hugr, HugrView, IncomingPort, Node, OutgoingPort, SimpleReplacement,
};
use tket2::Tk2Op;

use lazy_static::lazy_static;

use crate::extension::{futures::FutureOpBuilder, qsystem::QSystemOpBuilder};

/// A `Hugr -> Hugr` pass that replaces [Tk2Op::Measure] nodes with
/// [QSystemOp::Measure] nodes. To construct a `LazifyMeasurePass` use
/// [Default::default].
///
/// The `Hugr` must not contain any non-local edges. If validation is enabled,
/// this precondition will be verified.
///
/// [Tket2Op::Measure]: tket2::Tk2Op::Measure
/// [QSystemOp::Measure]: crate::extension::qsystem::QSystemOp::Measure
#[derive(Default)]
pub struct LazifyMeasurePass(ValidationLevel);

#[derive(Error, Debug, Display, From)]
#[non_exhaustive]
/// An error reported from [LazifyMeasurePass].
pub enum LazifyMeasurePassError {
    /// The [Hugr] was invalid either before or after a pass ran.
    ValidationError(ValidatePassError),
    /// The [Hugr] was found to contain non-local edges.
    NonLocalEdgesError(NonLocalEdgesError),
    /// A [SimpleReplacement] failed during the running of the pass.
    SimpleReplacementError(SimpleReplacementError),
}

impl LazifyMeasurePass {
    /// Run `LazifyMeasurePass` on the given [HugrMut]. `registry` is used for
    /// validation, if enabled.
    pub fn run(&self, hugr: &mut impl HugrMut) -> Result<(), LazifyMeasurePassError> {
        self.0.run_validated_pass(hugr, |hugr, validation_level| {
            if validation_level != &ValidationLevel::None {
                ensure_no_nonlocal_edges(hugr)?;
            }
            let mut state = State::new(hugr.nodes().filter_map(
                |n| match hugr.get_optype(n).cast() {
                    Some(Tk2Op::Measure) => Some(WorkItem::ReplaceMeasure(n)),
                    _ => None,
                },
            ));
            while state.work_one(hugr)? {}
            Ok(())
        })
    }

    /// Returns a new `LazifyMeasurePass` with the given [ValidationLevel].
    pub fn with_validation_level(mut self, level: ValidationLevel) -> Self {
        self.0 = level;
        self
    }
}

enum WorkItem {
    ReplaceMeasure(Node),
}

struct State {
    worklist: Vec<WorkItem>,
}

impl State {
    fn new(items: impl IntoIterator<Item = WorkItem>) -> Self {
        let worklist = items.into_iter().collect();
        Self { worklist }
    }

    fn work_one(&mut self, hugr: &mut impl HugrMut) -> Result<bool, LazifyMeasurePassError> {
        let Some(item) = self.worklist.pop() else {
            return Ok(false);
        };
        self.worklist.extend(item.work(hugr)?);
        Ok(true)
    }
}

lazy_static! {
    static ref MEASURE_READ_HUGR: Hugr = {
        let mut builder = DFGBuilder::new(Signature::new(qb_t(), vec![qb_t(), bool_t()])).unwrap();
        let [qb] = builder.input_wires_arr();
        let [qb, lazy_r] = builder.add_lazy_measure(qb).unwrap();
        let r = builder.add_read(lazy_r, bool_t()).unwrap();
        builder.finish_hugr_with_outputs([qb, r]).unwrap()
    };
}

fn measure_replacement(num_dups: usize) -> Hugr {
    let mut out_types = vec![qb_t()];
    out_types.extend((0..num_dups).map(|_| bool_t()));
    let num_out_types = out_types.len();
    let mut builder = DFGBuilder::new(Signature::new(qb_t(), out_types)).unwrap();
    let [qb] = builder.input_wires_arr();
    let [qb, mut future_r] = builder.add_lazy_measure(qb).unwrap();
    let mut future_rs = vec![];
    if num_dups > 0 {
        for _ in 0..num_dups - 1 {
            let [r1, r2] = builder.add_dup(future_r, bool_t()).unwrap();
            future_rs.push(r1);
            future_r = r2;
        }
        future_rs.push(future_r)
    } else {
        builder.add_free(future_r, bool_t()).unwrap();
    }
    let mut rs = vec![qb];
    rs.extend(
        future_rs
            .into_iter()
            .map(|r| builder.add_read(r, bool_t()).unwrap()),
    );
    assert_eq!(num_out_types, rs.len());
    assert_eq!(num_out_types, num_dups + 1);
    builder.finish_hugr_with_outputs(rs).unwrap()
}

fn simple_replace_measure(
    hugr: &impl HugrView,
    node: Node,
) -> (HashSet<(Node, IncomingPort)>, SimpleReplacement) {
    assert!(
        hugr.get_optype(node).cast() == Some(Tk2Op::Measure),
        "{:?}",
        hugr.get_optype(node)
    );
    let g = SiblingSubgraph::try_from_nodes([node], hugr).unwrap();
    let num_uses_of_bool = hugr.linked_inputs(node, OutgoingPort::from(1)).count();
    let replacement_hugr = measure_replacement(num_uses_of_bool);
    let [i, o] = replacement_hugr.get_io(replacement_hugr.root()).unwrap();

    // A map from (target ports of edges from the Input node of `replacement`) to (target ports of
    // edges from nodes not in `removal` to nodes in `removal`).
    let nu_inp = replacement_hugr
        .all_linked_inputs(i)
        .map(|(n, p)| ((n, p), (node, p)))
        .collect();

    // qubit is linear, there must be exactly one
    let (target_node, target_port) = hugr
        .single_linked_input(node, OutgoingPort::from(0))
        .unwrap();
    // A map from (target ports of edges from nodes in `removal` to nodes not in `removal`) to
    // (input ports of the Output node of `replacement`).
    let mut nu_out: HashMap<_, _> = [((target_node, target_port), IncomingPort::from(0))]
        .into_iter()
        .collect();
    nu_out.extend(
        hugr.linked_inputs(node, OutgoingPort::from(1))
            .enumerate()
            .map(|(i, target)| (target, IncomingPort::from(i + 1))),
    );
    assert_eq!(nu_out.len(), 1 + num_uses_of_bool);
    assert_eq!(nu_out.len(), replacement_hugr.in_value_types(o).count());

    let nu_out_set = nu_out.keys().copied().collect();
    (
        nu_out_set,
        SimpleReplacement::new(g, replacement_hugr, nu_inp, nu_out),
    )
}

impl WorkItem {
    fn work(
        self,
        hugr: &mut impl HugrMut,
    ) -> Result<impl IntoIterator<Item = Self>, LazifyMeasurePassError> {
        match self {
            Self::ReplaceMeasure(node) => {
                // for now we read immediately, but when we don't the first
                // results are the linked inputs we must return
                let (_, replace) = simple_replace_measure(hugr, node);
                replace.apply(hugr)?;
                Ok(std::iter::empty())
            }
        }
    }
}

#[cfg(test)]
mod test {

    use crate::extension::{futures::FutureOpDef, qsystem::QSystemOp};

    use super::*;

    #[test]
    fn simple() {
        let mut hugr = {
            let mut builder =
                DFGBuilder::new(Signature::new(qb_t(), vec![qb_t(), bool_t()])).unwrap();
            let [qb] = builder.input_wires_arr();
            let outs = builder
                .add_dataflow_op(Tk2Op::Measure, [qb])
                .unwrap()
                .outputs();
            builder.finish_hugr_with_outputs(outs).unwrap()
        };
        assert!(hugr.validate_no_extensions().is_ok());
        LazifyMeasurePass::default().run(&mut hugr).unwrap();
        assert!(hugr.validate_no_extensions().is_ok());
        let mut num_read = 0;
        let mut num_lazy_measure = 0;
        for n in hugr.nodes() {
            let ot = hugr.get_optype(n);
            if let Some(FutureOpDef::Read) = ot.cast() {
                num_read += 1;
            } else if let Some(QSystemOp::LazyMeasure) = ot.cast() {
                num_lazy_measure += 1;
            } else {
                assert_eq!(ot.cast::<Tk2Op>(), None)
            }
        }

        assert_eq!(1, num_read);
        assert_eq!(1, num_lazy_measure);
    }

    #[test]
    fn multiple_uses() {
        let mut builder =
            DFGBuilder::new(Signature::new(qb_t(), vec![qb_t(), bool_t(), bool_t()])).unwrap();
        let [qb] = builder.input_wires_arr();
        let [qb, bool] = builder
            .add_dataflow_op(Tk2Op::Measure, [qb])
            .unwrap()
            .outputs_arr();
        let mut hugr = builder.finish_hugr_with_outputs([qb, bool, bool]).unwrap();

        assert!(hugr.validate_no_extensions().is_ok());
        LazifyMeasurePass::default().run(&mut hugr).unwrap();
        assert!(hugr.validate_no_extensions().is_ok());
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

        assert!(hugr.validate_no_extensions().is_ok());
        LazifyMeasurePass::default().run(&mut hugr).unwrap();
        assert!(hugr.validate_no_extensions().is_ok());
    }
}
