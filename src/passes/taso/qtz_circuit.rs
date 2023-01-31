use std::collections::HashMap;

use portgraph::graph::Direction;
use serde::{Deserialize, Serialize};

use crate::circuit::{
    circuit::Circuit,
    dag::Edge,
    operation::{Op, WireType},
};

#[derive(Debug, Serialize, Deserialize)]
struct RepCircOp {
    opstr: String,
    outputs: Vec<String>,
    inputs: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct RepCirc(Vec<RepCircOp>);

#[derive(Debug, Serialize, Deserialize)]
struct MetaData {
    n_qb: u64,
    n_input_param: u64,
    n_total_param: u64,
    num_gates: u64,
    id: Vec<String>,
    fingerprint: Vec<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct RepCircData {
    meta: MetaData,
    circ: RepCirc,
}

fn map_op(opstr: &str) -> Op {
    // TODO, more
    match opstr {
        "h" => Op::H,
        "rz" => Op::RzF64,
        "cx" => Op::CX,
        x => panic!("unknown op {x}"),
    }
}

fn map_wt(wirestr: &str) -> (WireType, usize) {
    let wt = if wirestr.starts_with('Q') {
        WireType::Qubit
    } else if wirestr.starts_with('P') {
        WireType::Angle
    } else {
        panic!("unknown op {wirestr}");
    };

    (wt, wirestr[1..].parse().unwrap())
}
impl From<RepCircData> for Circuit {
    fn from(RepCircData { circ: rc, meta }: RepCircData) -> Self {
        let mut circ = Circuit::new();
        let mut param_wire_map: Vec<Edge> = (0..meta.n_input_param)
            .map(|_| circ.new_input(WireType::Angle))
            .collect();
        param_wire_map.reverse();
        let mut qubit_wire_map: Vec<Edge> = (0..meta.n_qb)
            .map(|_| circ.new_input(WireType::Qubit))
            .collect();
        qubit_wire_map.reverse(); // inputs added to front of edge list first
        for RepCircOp {
            opstr,
            outputs,
            inputs,
        } in rc.0
        {
            let op = map_op(&opstr);

            let incoming: Vec<_> = inputs
                .into_iter()
                .map(|is| {
                    let (wt, idx) = map_wt(&is);
                    match wt {
                        WireType::Qubit => qubit_wire_map[idx],
                        WireType::Angle => param_wire_map[idx],
                        _ => panic!("unexpected wire type."),
                    }
                })
                .collect();

            let outgoing: Vec<_> = outputs
                .into_iter()
                .map(|os| {
                    let (wt, idx) = map_wt(&os);
                    assert_eq!(wt, WireType::Qubit, "only qubits expected as output");
                    let e = circ.add_edge(wt);

                    qubit_wire_map[idx] = e;
                    e
                })
                .collect();

            circ.add_vertex_with_edges(op, incoming, outgoing);
        }

        circ.dag
            .connect_many(
                circ.boundary()[1],
                qubit_wire_map,
                Direction::Incoming,
                None,
            )
            .unwrap();

        circ
    }
}

pub(super) fn load_ecc_set(path: &str) -> HashMap<String, Vec<Circuit>> {
    let jsons = std::fs::read_to_string(path).unwrap();
    let (_, ecc_map): (Vec<()>, HashMap<String, Vec<RepCircData>>) =
        serde_json::from_str(&jsons).unwrap();

    ecc_map
        .into_values()
        .map(|datmap| {
            let id = datmap[0].meta.id[0].clone();
            let circs = datmap.into_iter().map(|rcd| rcd.into()).collect();
            (id, circs)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::validate::check_soundness;

    use super::*;
    // fn load_representative_set(path: &str) -> HashMap<String, Circuit> {
    //     let jsons = std::fs::read_to_string(path).unwrap();
    //     // read_rep_json(&jsons).unwrap();
    //     let st: Vec<RepCircData> = serde_json::from_str(&jsons).unwrap();
    //     st.into_iter()
    //         .map(|mut rcd| (rcd.meta.id.remove(0), rcd.into()))
    //         .collect()
    // }

    // #[test]
    // fn test_read_rep() {
    //     let rep_map: HashMap<String, Circuit> =
    //         load_representative_set("../../ext/quartz/h_rz_cxrepresentative_set.json");

    //     for c in rep_map.values() {
    //         println!("{}", c.dot_string());
    //         check_soundness(c).unwrap();
    //     }
    // }

    #[test]
    fn test_read_complete() {
        let ecc: HashMap<String, Vec<Circuit>> =
            load_ecc_set("../../ext/quartz/h_rz_cxcomplete_ECC_set.json");

        ecc.values()
            .flatten()
            .for_each(|c| check_soundness(c).unwrap());
    }
}
