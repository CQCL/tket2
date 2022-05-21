pub mod classical;
// pub mod redundancy;
pub mod squash;
mod pattern;

use crate::circuit::circuit::{Circuit, CircuitRewrite};

/// Repeatedly apply all available rewrites reported by finder closure until no more are found.
///
/// # Errors
///
/// This function will return an error if rewrite application fails.
pub fn apply_exhaustive<F>(mut circ: Circuit, finder: F) -> Result<(Circuit, bool), String>
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
pub fn apply_greedy<F>(mut circ: Circuit, finder: F) -> Result<(Circuit, bool), String>
where
    F: Fn(&Circuit) -> Option<CircuitRewrite>,
{
    let mut success = false;
    loop {
        if let Some(rewrite) = finder(&circ) {
            success = true;
            circ.apply_rewrite(rewrite)?;
        } else {
            break;
        }
    }

    Ok((circ, success))
}

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
