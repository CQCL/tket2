use std::collections::BTreeSet;

use thiserror::Error;

use crate::{
    circuit::{
        circuit::Circuit,
        dag::{Edge, TopSorter, Vertex},
        operation::{Op, WireType},
    },
    graph::graph::{Direction, DIRECTIONS},
};

#[derive(Debug, Error, PartialEq)]
pub enum ValidateError<'a> {
    #[error("Type error at node index {1:?} : {0}.")]
    TypeError(&'a str, Vertex),
    #[error("Missing node at index {0:?}.")]
    NodeMissing(Vertex),
    #[error("Missing edge at index {0:?}.")]
    EdgeMissing(Edge),
    #[error("Dangling edge at index {0:?}.")]
    DanglingEdge(Edge),
    #[error("Unkown signature for op {0:?}.")]
    UnknownSignature(&'a Op),
    #[error("Edge type mismatch at node {0:?}: {1:?} != {2:?}")]
    EdgeTypeError(Vertex, WireType, WireType),
    #[error("Cycle detected.")]
    CycleDetected,
    #[error("Some nodes in the circuit were not visited by topsort.")]
    UnvisitedNodes,
    #[error("Some edges in the circuit were not connected to visited nodes.")]
    UnvisitedEdges,
    #[error("Edge {0:?} reports being connected to {1:?} {2:?} but is not found there.")]
    PortMismatch(Edge, (Vertex, usize), Direction),
    #[error("Multiple edges report being connected to port {0:?} {1:?}")]
    DuplicatePort((Vertex, usize), Direction),
}

pub fn check_soundness(circ: &Circuit) -> Result<(), ValidateError> {
    let bound = circ.boundary();

    let mut nodes_visited = 0;
    let mut edges_visited = 0;
    let mut ports: [BTreeSet<_>; 2] = [BTreeSet::new(), BTreeSet::new()];
    for e in circ.dag.edge_indices() {
        for (ps, direction) in ports.iter_mut().zip(DIRECTIONS.iter()) {
            let n = circ
                .dag
                .edge_endpoint(e, *direction)
                .ok_or(ValidateError::DanglingEdge(e))?;
            let p = circ
                .port_of_edge(n, e, *direction)
                .ok_or(ValidateError::NodeMissing(n))?;
            let np = (n, p);
            if !ps.insert(np) {
                return Err(ValidateError::DuplicatePort(np, *direction));
            }
        }
    }
    let mut topwalk = TopSorter::new(
        &circ.dag,
        circ.dag
            .node_indices()
            .filter(|n| circ.node_boundary_size(*n)[0] == 0)
            .collect(),
    );
    for nid in topwalk.by_ref() {
        nodes_visited += 1;
        for dir in DIRECTIONS {
            edges_visited += circ.dag.node_edges(nid, dir).count();
            // for e in circ.dag.node_edges(nid, dir) {
            //     edges_visited += 1;

            // let edgepoints = circ
            //     .edge_endpoints(e)
            //     .ok_or(ValidateError::EdgeMissing(e))?;

            // let np = edgepoints[1 - dir as usize];
            // circ.dag
            //     .edge_at_port(np, dir)
            //     .and_then(|ep| (ep == *e).then(|| ()))
            //     .ok_or(ValidateError::PortMismatch(*e, np, dir))?;
            // }
        }

        let op = &circ
            .dag
            .node_weight(nid)
            .ok_or(ValidateError::NodeMissing(nid))?
            .op;
        let [insize, outsize] = circ.node_boundary_size(nid);

        if matches!(op, Op::Input | Op::Output) {
            assert!(bound.contains(&nid));
            continue;
        }

        let opsig = op.signature().ok_or(ValidateError::UnknownSignature(op))?;
        if insize != (opsig.linear.len() + opsig.nonlinear[0].len())
            || outsize != (opsig.linear.len() + opsig.nonlinear[1].len())
        {
            return Err(ValidateError::TypeError("Signature size mismatch", nid));
        }
        let mut in_edge_iter = circ.dag.node_edges(nid, Direction::Incoming);
        let mut out_edge_iter = circ.dag.node_edges(nid, Direction::Outgoing);

        for (sig_typ, (ine, oute)) in opsig
            .linear
            .iter()
            .zip(in_edge_iter.by_ref().zip(out_edge_iter.by_ref()))
        {
            let in_weight = circ
                .dag
                .edge_weight(ine)
                .ok_or(ValidateError::EdgeMissing(ine))?;
            let out_weight = circ
                .dag
                .edge_weight(oute)
                .ok_or(ValidateError::EdgeMissing(oute))?;

            let intype = in_weight.edge_type;
            if *sig_typ != intype {
                return Err(ValidateError::EdgeTypeError(nid, *sig_typ, intype));
            }
            let outtype = out_weight.edge_type;
            if *sig_typ != outtype {
                return Err(ValidateError::EdgeTypeError(nid, *sig_typ, outtype));
            }
        }

        for (sig_vec, mut e_iter) in opsig
            .nonlinear
            .iter()
            .zip([in_edge_iter, out_edge_iter].into_iter())
        {
            let mut sig_iter = sig_vec.iter();
            loop {
                let (sig_typ, e) = (sig_iter.next(), e_iter.next());

                let (sig_typ, e) = match (sig_typ, e) {
                    (None, None) => break,
                    (None, _) | (_, None) => {
                        return Err(ValidateError::TypeError("Signature size mismatch", nid))
                    }
                    (Some(sig_typ), Some(e)) => (sig_typ, e),
                };
                // for (sig_typ, e) in sig_vec.iter().zip(e_iter) {
                let intype = circ
                    .dag
                    .edge_weight(e)
                    .ok_or(ValidateError::EdgeMissing(e))?
                    .edge_type;
                if *sig_typ != intype {
                    return Err(ValidateError::EdgeTypeError(nid, *sig_typ, intype));
                }
            }
        }
    }

    if nodes_visited != circ.dag.node_count() {
        return Err(ValidateError::UnvisitedNodes);
    }

    if edges_visited != 2 * circ.dag.edge_count() {
        return Err(ValidateError::UnvisitedEdges);
    }

    if !topwalk.edges_remaining().is_empty() {
        return Err(ValidateError::CycleDetected);
    }
    Ok(())
}

#[cfg(test)]
mod tests {

    use rstest::{fixture, rstest};

    use crate::circuit::{
        circuit::{Circuit, UnitID},
        operation::{Op, WireType},
    };

    use super::*;
    use std::error::Error;

    #[test]
    fn test_check_types() -> Result<(), Box<dyn Error>> {
        let mut circ = Circuit::new();
        let [i, o] = circ.boundary();
        let h = circ.add_vertex(Op::H);

        let e_wrong = circ.add_insert_edge((i, 0), (h, 0), WireType::F64)?;

        assert!(matches!(
            check_soundness(&circ),
            Err(ValidateError::TypeError(..))
        ));

        circ.add_insert_edge((h, 0), (o, 0), WireType::Qubit)?;
        assert_eq!(
            check_soundness(&circ),
            Err(ValidateError::EdgeTypeError(
                h,
                WireType::Qubit,
                WireType::F64,
            ))
        );

        circ.dag.remove_edge(e_wrong);
        circ.add_insert_edge((i, 0), (h, 0), WireType::Qubit)?;
        check_soundness(&circ).unwrap();
        Ok(())
    }

    #[fixture]
    fn bell_circ() -> Circuit {
        let mut circ = Circuit::new();
        circ.add_linear_unitid(UnitID::Qubit {
            reg_name: "q".into(),
            index: vec![0],
        });
        circ.add_linear_unitid(UnitID::Qubit {
            reg_name: "q".into(),
            index: vec![1],
        });
        circ.append_op(Op::H, &[0]).unwrap();
        circ.append_op(Op::CX, &[0, 1]).unwrap();

        circ
    }

    #[rstest]
    fn test_deleted_node(bell_circ: Circuit) {
        check_soundness(&bell_circ).unwrap();

        let nodes: Vec<_> = bell_circ.dag.node_indices().collect();
        let edges: Vec<_> = bell_circ.dag.edge_indices().collect();

        let mut c1 = bell_circ.clone();
        c1.dag.remove_edge(edges[0]);

        assert_eq!(
            check_soundness(&c1),
            Err(ValidateError::TypeError(
                "Signature size mismatch",
                nodes[3]
            ))
        );

        let mut c2 = bell_circ.clone();
        assert_eq!(c2.dag.remove_node(nodes[2]).unwrap().op, Op::H);
        assert_eq!(
            check_soundness(&c2),
            Err(ValidateError::DanglingEdge(edges[2]))
        );

        let mut c3 = bell_circ.clone();
        c3.dag.remove_node(nodes[3]);
        println!("{}", c3.dot_string());

        assert_eq!(
            check_soundness(&c3),
            Err(ValidateError::DanglingEdge(edges[0]))
        );
    }
}
