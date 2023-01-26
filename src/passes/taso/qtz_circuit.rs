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

fn map_wt(wirestr: &str) -> WireType {
    if wirestr.starts_with('Q') {
        WireType::Qubit
    } else if wirestr.starts_with('P') {
        WireType::Angle
    } else {
        panic!("unknown op {wirestr}");
    }
}
impl From<RepCirc> for Circuit {
    fn from(rc: RepCirc) -> Self {
        let mut circ = Circuit::new();
        let mut wire_map: HashMap<String, Edge> = HashMap::new();
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
                    if let Some(e) = wire_map.get(&is) {
                        *e
                    } else {
                        let e = circ.new_input(map_wt(&is));
                        wire_map.insert(is, e);
                        e
                    }
                })
                .collect();

            let outgoing: Vec<_> = outputs
                .into_iter()
                .map(|os| {
                    let e = circ.add_edge(map_wt(&os));

                    wire_map.insert(os, e);
                    e
                })
                .collect();

            circ.add_vertex_with_edges(op, incoming, outgoing);
        }

        let outputs: Vec<_> = wire_map
            .into_values()
            .filter(|e| circ.edge_endpoints(*e).is_none())
            .collect();

        circ.dag
            .connect_many(circ.boundary()[1], outputs, Direction::Incoming, None)
            .unwrap();

        circ
    }
}

pub(crate) fn load_representative_set(path: &str) -> HashMap<String, Circuit> {
    let jsons = std::fs::read_to_string(path).unwrap();
    // read_rep_json(&jsons).unwrap();
    let st: Vec<RepCircData> = serde_json::from_str(&jsons).unwrap();
    st.into_iter()
        .map(|RepCircData { mut meta, circ }| (meta.id.remove(0), circ.into()))
        .collect()
}

pub(crate) fn load_ecc_set(path: &str) -> HashMap<String, Vec<Circuit>> {
    let jsons = std::fs::read_to_string(path).unwrap();
    let (_, ecc_map): (Vec<()>, HashMap<String, Vec<RepCircData>>) =
        serde_json::from_str(&jsons).unwrap();

    ecc_map
        .into_values()
        .map(|datmap| {
            let id = datmap[0].meta.id[0].clone();
            let circs = datmap.into_iter().map(|rcd| rcd.circ.into()).collect();

            (id, circs)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::validate::check_soundness;

    use super::*;

    #[test]
    fn test_read_rep() {
        let rep_map: HashMap<String, Circuit> =
            load_representative_set("../../ext/quartz/h_rz_cxrepresentative_set.json");

        for c in rep_map.values() {
            check_soundness(c).unwrap();
        }
    }

    #[test]
    fn test_read_complete() {
        let ecc: HashMap<String, Vec<Circuit>> =
            load_ecc_set("../../ext/quartz/h_rz_cxcomplete_ECC_set.json");

        ecc.values()
            .flatten()
            .for_each(|c| check_soundness(c).unwrap());
    }
}
