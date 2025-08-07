//! Providing replacements for pattern matches.

use hugr::{
    builder::{DFGBuilder, Dataflow, DataflowHugr},
    extension::prelude::qb_t,
    hugr::views::SiblingSubgraph,
    types::Signature,
    HugrView,
};
use itertools::Itertools;

use crate::Circuit;

/// Provide possible replacements for a pattern match.
pub trait CircuitReplacer<MatchInfo> {
    /// Get the possible replacements for a pattern match.
    ///
    /// The order (and signature) of the inputs and outputs on the returned circuits must match
    /// the order of the boundary ports in `subgraph`.
    fn replace_match<H: HugrView>(
        &self,
        subgraph: &SiblingSubgraph<H::Node>,
        hugr: H,
        match_info: MatchInfo,
    ) -> Vec<Circuit>;
}

/// Replace a match with the identity circuit.
#[derive(Debug, Copy, Clone, Default)]
pub struct ReplaceWithIdentity;

impl ReplaceWithIdentity {
    /// Get the identity circuit for a given signature.
    pub fn get_replacement_identity(sig: Signature) -> Result<Circuit, &'static str> {
        let qb_inputs = sig.input_types().iter().filter(|&t| t == &qb_t());
        if !qb_inputs.zip_eq(sig.output_types()).all_equal() {
            return Err("Unsupported signature for ReplaceWithIdentity: output types must be qubit only and must map input qubits one-to-one");
        }
        let builder = DFGBuilder::new(sig).unwrap();
        let outputs = builder.input_wires();
        let circ = builder.finish_hugr_with_outputs(outputs).unwrap();
        Ok(Circuit::new(circ))
    }
}

impl<M> CircuitReplacer<M> for ReplaceWithIdentity {
    fn replace_match<H: hugr::HugrView>(
        &self,
        subgraph: &hugr::hugr::views::SiblingSubgraph<H::Node>,
        hugr: H,
        _match_info: M,
    ) -> Vec<Circuit> {
        let sig = subgraph.signature(&hugr);
        vec![Self::get_replacement_identity(sig).unwrap()]
    }
}
