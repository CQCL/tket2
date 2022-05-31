use thiserror::Error;

use crate::{
    circuit::{
        circuit::Circuit,
        dag::{Edge, Vertex},
        operation::{Op, WireType},
    },
    graph::graph::Direction,
};

#[derive(Debug, Error, PartialEq)]
pub enum ValidateError<'a> {
    #[error("Type error at node index {1:?} : {0}.")]
    TypeError(&'a str, Vertex),
    #[error("Missing node at index {0:?}.")]
    NodeMissing(Vertex),
    #[error("Missing edge at index {0:?}.")]
    EdgeMissing(Edge),
    #[error("Unkown signature for op {0:?}.")]
    UnknownSignature(&'a Op),
    #[error("Edge type mismatch at node {0:?}: {1:?} != {2:?}")]
    EdgeTypeError(Vertex, WireType, WireType),
}

pub fn check_types(circ: &Circuit) -> Result<(), ValidateError> {
    let bound = circ.boundary();
    for nid in circ.dag.node_indices().filter(|nid| !bound.contains(nid)) {
        let op = &circ
            .dag
            .node_weight(nid)
            .ok_or(ValidateError::NodeMissing(nid))?
            .op;
        let [insize, outsize] = circ.dag.node_boundary_size(nid);
        let opsig = op.signature().ok_or(ValidateError::UnknownSignature(op))?;
        if insize != (opsig.linear.len() + opsig.nonlinear[0].len())
            || outsize != (opsig.linear.len() + opsig.nonlinear[1].len())
        {
            return Err(ValidateError::TypeError("Signature size mismatch", nid));
        }
        let mut in_edge_iter = circ.dag.node_edges(nid, Direction::Incoming);
        let mut out_edge_iter = circ.dag.node_edges(nid, Direction::Outgoing);
        for (sig_typ, [in_weight, out_weight]) in
            opsig
                .linear
                .iter()
                .zip(
                    (&mut in_edge_iter)
                        .zip(&mut out_edge_iter)
                        .map(|(e_i, e_o)| {
                            [
                                circ.dag
                                    .edge_weight(*e_i)
                                    .ok_or(ValidateError::EdgeMissing(*e_i)),
                                circ.dag
                                    .edge_weight(*e_o)
                                    .ok_or(ValidateError::EdgeMissing(*e_i)),
                            ]
                        }),
                )
        {
            let intype = in_weight?.edge_type;
            if *sig_typ != intype {
                return Err(ValidateError::EdgeTypeError(nid, *sig_typ, intype));
            }
            let outtype = out_weight?.edge_type;
            if *sig_typ != outtype {
                return Err(ValidateError::EdgeTypeError(nid, *sig_typ, outtype));
            }
        }

        for (sig_vec, e_iter) in opsig
            .nonlinear
            .iter()
            .zip([in_edge_iter, out_edge_iter].into_iter())
        {
            for (sig_typ, e) in sig_vec.iter().zip(e_iter) {
                let intype = circ
                    .dag
                    .edge_weight(*e)
                    .ok_or(ValidateError::EdgeMissing(*e))?
                    .edge_type;
                if *sig_typ != intype {
                    return Err(ValidateError::EdgeTypeError(nid, *sig_typ, intype));
                }
            }
        }
    }

    Ok(())
}

pub fn check_soundness(circ: &Circuit) -> Result<(), ValidateError> {
    check_types(circ)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::circuit::{
        circuit::Circuit,
        operation::{Op, WireType},
    };

    use super::*;

    #[test]
    fn test_check_types() {
        let mut circ = Circuit::new();
        let [i, o] = circ.boundary();
        let h = circ.add_vertex(Op::H);

        let e_wrong = circ.tup_add_edge((i, 0), (h, 0), WireType::F64);

        assert!(matches!(
            check_types(&circ),
            Err(ValidateError::TypeError(..))
        ));

        circ.tup_add_edge((h, 0), (o, 0), WireType::Qubit);
        assert_eq!(
            check_types(&circ),
            Err(ValidateError::EdgeTypeError(
                h,
                WireType::Qubit,
                WireType::F64,
            ))
        );

        circ.dag.remove_edge(e_wrong);
        circ.tup_add_edge((i, 0), (h, 0), WireType::Qubit);

        check_types(&circ).unwrap();
    }
}
