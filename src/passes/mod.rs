pub mod classical;
// pub mod redundancy;
pub mod squash;
pub mod patterns;

use crate::{
    circuit::{
        circuit::{Circuit, CircuitError, CircuitRewrite},
        operation::Op,
    },
    graph::{
        substitute::BoundedSubgraph,
    },
};

/// Repeatedly apply all available rewrites reported by finder closure until no more are found.
///
/// # Errors
///
/// This function will return an error if rewrite application fails.
pub fn apply_exhaustive<F>(mut circ: Circuit, finder: F) -> Result<(Circuit, bool), CircuitError>
where
    F: Fn(&Circuit) -> Vec<CircuitRewrite>,
{
    let mut success = false;
    loop {
        // assuming all the returned rewrites are non-overlapping
        // or filter to make them non-overlapping
        // then in theory, they can all be applied in parallel
        let rewrites = finder(&circ);
        if rewrites.is_empty() {
            break;
        }
        success = true;
        for rewrite in rewrites {
            circ.apply_rewrite(rewrite)?;
        }
    }

    Ok((circ, success))
}

/// Repeatedly apply first reported rewrite
///
/// # Errors
///
/// This function will return an error if rewrite application fails.
pub fn apply_greedy<F>(mut circ: Circuit, finder: F) -> Result<(Circuit, bool), CircuitError>
where
    F: Fn(&Circuit) -> Option<CircuitRewrite>,
{
    let mut success = false;
    while let Some(rewrite) = finder(&circ) {
        success |= true;
        circ.apply_rewrite(rewrite)?;
    }

    Ok((circ, success))
}

// pub struct PatternRewrites<'p, I, F> {
//     match_iter: I,
//     pattern: CircFixedStructPattern<'p, F>,
//     boundary_ports: [Vec<NodePort>; 2],
// }

// impl<'p, I, F> PatternRewrites<'p, I, F> {
//     pub fn new(
//         match_iter: I,
//         pattern: CircFixedStructPattern<'p, F>,
//         boundary_ports: [Vec<NodePort>; 2],
//     ) -> Self {
//         Self {
//             match_iter,
//             pattern,
//             boundary_ports,
//         }
//     }
// }

// impl<I: Iterator<Item = Match<DefaultIx>>, F: Fn(NodeIndex, &VertexProperties) -> bool> Iterator
//     for PatternRewrites<'_, I, F>
// {
//     type Item = CircuitRewrite;

//     fn next(&mut self) -> Option<Self::Item> {
//         self.match_iter.next().map(|pmatch| {
//             let in_edges: Vec<_> = self.boundary_ports[0]
//                 .iter()
//                 .map(|np| {
//                     self.pattern
//                         .graph
//                         .edge_at_port(
//                             NodePort::new(*pmatch.get(&np.node).unwrap(), np.port),
//                             Direction::Incoming,
//                         )
//                         .unwrap()
//                 })
//                 .collect();
//             let out_edges: Vec<_> = self.boundary_ports[1]
//                 .iter()
//                 .map(|np| {
//                     self.pattern
//                         .graph
//                         .edge_at_port(
//                             NodePort::new(*pmatch.get(&np.node).unwrap(), np.port),
//                             Direction::Outgoing,
//                         )
//                         .unwrap()
//                 })
//                 .collect();
//             let subg = BoundedSubgraph::new(pmatch.values().cloned().into(), [in_edges, out_edges]);

//             let (newcirc, phase) = (rewrite_closure)(circ, pmatch);

//             CircuitRewrite::new(subg, newcirc.into(), phase)
//         })
//     }
// }
// #[cfg(test)]
// mod tests {
//     use symengine::Expression;

//     use crate::{
//         circuit::{
//             circuit::{Circuit, UnitID},
//             operation::{Op, Param},
//         },
//         graph::graph::PortIndex,
//     };
//     use tket_json_rs::circuit_json::SerialCircuit;

//     use super::redundancy::remove_redundancies;

//     #[test]
//     fn test_remove_redundancies() {
//         // circuit with only redundant gates; identity unitary
//         //[Rz(a) q[0];, Rz(-a) q[0];, CX q[0], q[1];, CX q[0], q[1];, Rx(2) q[1];]
//         let qubits = vec![
//             UnitID::Qubit {
//                 name: "q".into(),
//                 index: vec![0],
//             },
//             UnitID::Qubit {
//                 name: "q".into(),
//                 index: vec![0],
//             },
//         ];
//         let mut circ = Circuit::with_uids(qubits);

//         circ.append_op(Op::Rz(Param::from_str("a")), &vec![PortIndex::new(0)])
//             .unwrap();
//         circ.append_op(Op::Rz(Param::new("-a")), &vec![PortIndex::new(0)])
//             .unwrap();
//         circ.append_op(Op::CX, &vec![PortIndex::new(0), PortIndex::new(1)])
//             .unwrap();
//         circ.append_op(Op::CX, &vec![PortIndex::new(0), PortIndex::new(1)])
//             .unwrap();
//         circ.append_op(Op::Rx(Param::new("2.0")), &vec![PortIndex::new(1)])
//             .unwrap();

//         let circ2 = remove_redundancies(circ);

//         let _reser: SerialCircuit<Param> = circ2.into();

//         assert_eq!(_reser.commands.len(), 0);
//         // Rx(2pi) introduces a phase
//         assert_eq!(_reser.phase, Expression::new("1.0"));
//     }
// }

pub fn decompose_custom<'c>(circ: &'c Circuit) -> impl Iterator<Item = CircuitRewrite> + '_ {
    circ.dag.node_indices().filter_map(|n| {
        let op = &circ.dag.node_weight(n).unwrap().op;
        if let Op::Custom(x) = op {
            Some(CircuitRewrite::new(
                BoundedSubgraph::from_node(&circ.dag, n),
                x.to_circuit().expect("Circuit generation failed.").into(),
                0.0,
            ))
        } else {
            None
        }
    })
}

#[cfg(feature = "pyo3")]
use pyo3::prelude::pyfunction;

#[cfg_attr(feature = "pyo3", pyfunction)]
pub fn decompose_custom_pass(circ: Circuit) -> (Circuit, bool) {
    let (circ, suc) = apply_exhaustive(circ, |c| decompose_custom(c).collect()).unwrap();
    let circ = circ.remove_noop();
    (circ, suc)
}
