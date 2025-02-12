//! Pattern matching for circuits.
//!
//! This module provides a way to define circuit patterns and match
//! them against circuits.
//!
//! # Examples
//! ```
//! use tket2::portmatching::{CircuitPattern, PatternMatcher};
//! use tket2::Tk2Op;
//! use hugr::builder::{DFGBuilder, Dataflow, DataflowHugr};
//! use hugr::extension::prelude::qb_t;
//! use hugr::ops::handle::NodeHandle;
//! use hugr::types::Signature;
//!
//! # fn doctest() -> Result<(), Box<dyn std::error::Error>> {
//! // Define a simple pattern that matches a single qubit allocation.
//! let circuit_pattern = {
//!     let mut dfg = DFGBuilder::new(Signature::new(vec![], vec![qb_t()]))?;
//!     let alloc = dfg.add_dataflow_op(Tk2Op::QAlloc, [])?;
//!     dfg.finish_hugr_with_outputs(alloc.outputs())
//! }?.into();
//! let pattern = CircuitPattern::try_from_circuit(&circuit_pattern)?;
//!
//! // Define a circuit that contains a qubit allocation.
//! //
//! // -----[Z]--x---
//! //           |
//! //  0|--[Z]--o---
//! let (circuit, alloc_node) = {
//!     let mut dfg = DFGBuilder::new(Signature::new(vec![qb_t()], vec![qb_t(), qb_t()]))?;
//!     let [input_wire] = dfg.input_wires_arr();
//!     let alloc = dfg.add_dataflow_op(Tk2Op::QAlloc, [])?;
//!     let [alloc_wire] = alloc.outputs_arr();
//!
//!     let mut circuit = dfg.as_circuit(vec![input_wire, alloc_wire]);
//!     circuit
//!         .append(Tk2Op::Z, [1])?
//!         .append(Tk2Op::Z, [0])?
//!         .append(Tk2Op::CX, [1, 0])?;
//!     let outputs = circuit.finish();
//!
//!     let circuit = dfg.finish_hugr_with_outputs(outputs)?.into();
//!     (circuit, alloc.node())
//! };
//!
//! // Create a pattern matcher and find matches.
//! let matcher = PatternMatcher::from_patterns(vec![pattern]);
//! let matches = matcher.find_matches(&circuit);
//!
//! assert_eq!(matches.len(), 1);
//! assert_eq!(matches[0].nodes(), [alloc_node]);
//! # Ok(())
//! # }
//! ```

pub mod branch;
pub mod indexing;
pub mod matcher;
pub mod pattern;
pub mod predicate;

pub use indexing::{HugrVariableID, HugrVariableValue};
pub use matcher::{MatchOp, PatternMatch, PatternMatcher};
pub use pattern::CircuitPattern;
pub use predicate::{Constraint, Predicate};

#[cfg(test)]
mod tests {
    use crate::extension::rotation::rotation_type;
    use crate::{Circuit, Tk2Op};
    use hugr::{
        builder::{DFGBuilder, Dataflow, DataflowHugr},
        extension::prelude::qb_t,
        types::Signature,
    };
    use rstest::fixture;

    /// A circuit with two rotation gates in sequence, sharing a param
    #[fixture]
    pub(super) fn circ_with_copy() -> Circuit {
        let input_t = vec![qb_t(), rotation_type()];
        let output_t = vec![qb_t()];
        let mut h = DFGBuilder::new(Signature::new(input_t, output_t)).unwrap();

        let mut inps = h.input_wires();
        let qb = inps.next().unwrap();
        let f = inps.next().unwrap();

        let res = h.add_dataflow_op(Tk2Op::Rx, [qb, f]).unwrap();
        let qb = res.outputs().next().unwrap();
        let res = h.add_dataflow_op(Tk2Op::Rx, [qb, f]).unwrap();
        let qb = res.outputs().next().unwrap();

        h.finish_hugr_with_outputs([qb]).unwrap().into()
    }

    /*
    TODO: REACTIVATE THESE TESTS ONCE MATCHERS ARE UPDATED

    #[fixture]
    fn lhs() -> Circuit {
        let mut h = DFGBuilder::new(Signature::new(vec![], vec![qb_t()])).unwrap();

        let res = h.add_dataflow_op(Tk2Op::QAlloc, []).unwrap();
        let q = res.out_wire(0);

        h.finish_hugr_with_outputs([q]).unwrap().into()
    }

    #[fixture]
    pub fn circ() -> Circuit {
        let mut h = DFGBuilder::new(Signature::new(vec![qb_t()], vec![qb_t()])).unwrap();
        let mut inps = h.input_wires();
        let q_in = inps.next().unwrap();

        let res = h.add_dataflow_op(Tk2Op::QAlloc, []).unwrap();
        let q_out = res.out_wire(0);
        let res = h.add_dataflow_op(Tk2Op::CZ, [q_in, q_out]).unwrap();
        let q_in = res.out_wire(0);
        let q_out = res.out_wire(1);
        h.add_dataflow_op(Tk2Op::QFree, [q_in]).unwrap();

        h.finish_hugr_with_outputs([q_out]).unwrap().into()
    }

    #[ignore = "wip"]
    #[rstest]
    fn simple_match(circ: Circuit, lhs: Circuit) {
        let p = CircuitPattern::try_from_circuit(&lhs).unwrap();
        let m = PatternMatcher::from_patterns(vec![p]);

        let matches = m.find_matches(&circ).collect_vec();
        assert_eq!(matches.len(), 1);
    }*/
}
