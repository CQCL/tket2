use std::collections::{HashMap, HashSet};

use hugr::{
    builder::{DFGBuilder, Dataflow, DataflowHugr},
    extension::prelude::{BOOL_T, QB_T},
    hugr::{hugrmut::HugrMut, views::SiblingSubgraph, Rewrite},
    types::FunctionType,
    Hugr, HugrView, IncomingPort, Node, SimpleReplacement,
};
use tket2::Tk2Op;

use lazy_static::lazy_static;

use crate::extension::{
    futures::FutureOpBuilder,
    quantum_lazy::{self, LazyQuantumOpBuilder},
};
/// TODO docs
pub struct LazifyMeaurePass;

type Error = Box<dyn std::error::Error>;
impl LazifyMeaurePass {
    /// TODO docs
    pub fn run(&self, hugr: &mut impl HugrMut) -> Result<(), Error> {
        let mut state = State::new(hugr.nodes().filter_map(
            |n| match hugr.get_optype(n).try_into() {
                Ok(Tk2Op::Measure) => Some(WorkItem::ReplaceMeasure(n)),
                _ => None,
            },
        ));
        while state.work_one(hugr)? {}
        Ok(())
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

    fn work_one(&mut self, hugr: &mut impl HugrMut) -> Result<bool, Error> {
        let Some(item) = self.worklist.pop() else {
            return Ok(false);
        };
        self.worklist.extend(item.work(hugr)?);
        Ok(true)
    }
}

lazy_static! {
    static ref MEASURE_READ_HUGR: Hugr = {
        let mut builder = DFGBuilder::new(FunctionType::new(QB_T, vec![QB_T, BOOL_T])).unwrap();
        let [qb] = builder.input_wires_arr();
        let [qb, lazy_r] = builder.add_lazy_measure(qb).unwrap();
        let [r] = builder.add_read(lazy_r, BOOL_T).unwrap();
        builder
            .finish_hugr_with_outputs([qb, r], &quantum_lazy::REGISTRY)
            .unwrap()
    };
}

fn simple_replace_measure(
    hugr: &impl HugrView,
    node: Node,
) -> (HashSet<(Node, IncomingPort)>, SimpleReplacement) {
    assert!(
        hugr.get_optype(node).try_into() == Ok(Tk2Op::Measure),
        "{:?}",
        hugr.get_optype(node)
    );
    let g = SiblingSubgraph::try_from_nodes([node], hugr).unwrap();
    let replacement_hugr = MEASURE_READ_HUGR.clone();
    let [i, _] = replacement_hugr.get_io(replacement_hugr.root()).unwrap();

    // A map from (target ports of edges from the Input node of `replacement`) to (target ports of
    // edges from nodes not in `removal` to nodes in `removal`).
    let nu_inp = replacement_hugr
        .all_linked_inputs(i)
        .map(|(n, p)| ((n, p), (node, p)))
        .collect();

    // A map from (target ports of edges from nodes in `removal` to nodes not in `removal`) to
    // (input ports of the Output node of `replacement`).
    let nu_out: HashMap<_, _> = hugr
        .all_linked_inputs(node)
        .map(|(n, p)| ((n, p), p))
        .collect();

    let nu_out_set = nu_out.keys().copied().collect();
    (
        nu_out_set,
        SimpleReplacement::new(g, replacement_hugr, nu_inp, nu_out),
    )
}

impl WorkItem {
    fn work(self, hugr: &mut impl HugrMut) -> Result<impl IntoIterator<Item = Self>, Error> {
        match self {
            Self::ReplaceMeasure(node) => {
                // for now we read immeidately, but when we don't the first
                // result are the linked inputs we must return
                let (_, replace) = simple_replace_measure(hugr, node);
                replace.apply(hugr)?;
                Ok(std::iter::empty())
            }
        }
    }
}

#[cfg(test)]
mod test {
    use cool_asserts::assert_matches;

    use hugr::{
        extension::{ExtensionRegistry, PRELUDE},
        std_extensions::arithmetic::float_types,
    };
    use tket2::extension::TKET2_EXTENSION;

    use crate::extension::{
        futures::{self, FutureOp},
        quantum_lazy::LazyQuantumOp,
    };

    use super::*;

    lazy_static! {
        pub static ref REGISTRY: ExtensionRegistry = ExtensionRegistry::try_new([
            quantum_lazy::EXTENSION.to_owned(),
            futures::EXTENSION.to_owned(),
            TKET2_EXTENSION.to_owned(),
            PRELUDE.to_owned(),
            float_types::EXTENSION.clone(),
        ])
        .unwrap();
    }
    #[test]
    fn simple() {
        let mut hugr = {
            let mut builder = DFGBuilder::new(FunctionType::new(QB_T, vec![QB_T, BOOL_T])).unwrap();
            let [qb] = builder.input_wires_arr();
            let outs = builder
                .add_dataflow_op(Tk2Op::Measure, [qb])
                .unwrap()
                .outputs();
            builder.finish_hugr_with_outputs(outs, &REGISTRY).unwrap()
        };
        assert!(hugr.validate_no_extensions(&REGISTRY).is_ok());
        println!("{}", hugr.mermaid_string());
        LazifyMeaurePass.run(&mut hugr).unwrap();
        println!("{}", hugr.mermaid_string());
        assert!(hugr.validate_no_extensions(&REGISTRY).is_ok());
        let mut num_read = 0;
        let mut num_lazy_measure = 0;
        for n in hugr.nodes() {
            let ot = hugr.get_optype(n);
            if let Ok(FutureOp::Read) = ot.try_into() {
                num_read += 1;
            } else if let Ok(LazyQuantumOp::LazyMeasure) = ot.try_into() {
                num_lazy_measure += 1;
            } else {
                assert_matches!(Tk2Op::try_from(ot), Err(_))
            }
        }

        assert_eq!(1, num_read);
        assert_eq!(1, num_lazy_measure);
    }
}
