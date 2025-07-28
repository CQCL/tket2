use hugr::algorithms::replace_types::{NodeTemplate, ReplaceTypesError, ReplacementOptions};
use hugr::algorithms::{ComposablePass, ReplaceTypes};
use hugr::builder::{Container, DFGBuilder};
use hugr::types::{Signature, Term};
use hugr::{hugr::hugrmut::HugrMut, std_extensions::collections::array::array_type_def, Node};
use tket::extension::guppy::{DROP_OP_NAME, GUPPY_EXTENSION};

#[derive(Default, Debug, Clone)]
pub struct LowerDropsPass;

impl<H: HugrMut<Node = Node>> ComposablePass<H> for LowerDropsPass {
    type Error = ReplaceTypesError;

    /// Returns whether any drops were lowered
    type Result = bool;

    fn run(&self, hugr: &mut H) -> Result<Self::Result, Self::Error> {
        let mut rt = ReplaceTypes::default();
        rt.replace_parametrized_op_with(
            GUPPY_EXTENSION.get_op(DROP_OP_NAME.as_str()).unwrap(),
            |targs| {
                let [Term::Runtime(ty)] = targs else {
                    panic!("Expected just one type")
                };
                // The Hugr here is invalid, so we have to pull it out manually
                let mut dfb = DFGBuilder::new(Signature::new(ty.clone(), vec![])).unwrap();
                let h = std::mem::take(dfb.hugr_mut());
                Some(NodeTemplate::CompoundOp(Box::new(h)))
            },
            ReplacementOptions::default().with_linearization(true),
        );
        rt.run(hugr)
    }
}
