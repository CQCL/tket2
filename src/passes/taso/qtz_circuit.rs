use std::collections::HashMap;

use hugr::builder::{DFGBuilder, Dataflow, DataflowHugr};
// use crate::circuit::{
//     circuit::Circuit,
//     dag::Edge,
//     operation::{Op, WireType},
// };
use hugr::ops::{LeafOp, OpType as Op};
use hugr::types::{AbstractSignature, Type};
use hugr::Hugr as Circuit;
use serde::{Deserialize, Serialize};

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
    n_qb: usize,
    n_input_param: usize,
    n_total_param: usize,
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
        "h" => LeafOp::H,
        "rz" => LeafOp::RzF64,
        "cx" => LeafOp::CX,
        x => panic!("unknown op {x}"),
    }
    .into()
}

fn map_wt(wirestr: &str) -> (Type, usize) {
    let wt = if wirestr.starts_with('Q') {
        Type::Qubit
    } else if wirestr.starts_with('P') {
        Type::F64.into()
    } else {
        panic!("unknown op {wirestr}");
    };

    (wt, wirestr[1..].parse().unwrap())
}
// TODO change to TryFrom
impl From<RepCircData> for Circuit {
    fn from(RepCircData { circ: rc, meta }: RepCircData) -> Self {
        let qb_types: Vec<Type> = vec![Type::Qubit; meta.n_qb];
        let param_types: Vec<Type> = vec![Type::F64.into(); meta.n_input_param];
        let mut circ = DFGBuilder::new(AbstractSignature::new_df(
            [param_types, qb_types.clone()].concat(),
            qb_types,
        ))
        .unwrap();

        let inputs: Vec<_> = circ.input_wires().collect();
        let (param_wires, qubit_wires) = inputs.split_at(meta.n_input_param);
        // let param_wires: Vec<Wire> = input_iter.f(meta.n_input_param).collect();
        // let qubit_wires: Vec<Wire> = input_iter.collect();

        let mut qubit_wires: Vec<_> = qubit_wires.into();
        let mut param_wires: Vec<_> = param_wires.iter().map(Some).collect();
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
                        Type::Qubit => qubit_wires[idx],
                        Type::Classic(Type::F64) => *param_wires[idx].take().unwrap(),
                        _ => panic!("unexpected wire type."),
                    }
                })
                .collect();
            let output_wires = circ.add_dataflow_op(op, incoming).unwrap().outputs();

            for (os, wire) in outputs.into_iter().zip(output_wires) {
                let (wt, idx) = map_wt(&os);
                assert_eq!(wt, Type::Qubit, "only qubits expected as output");

                qubit_wires[idx] = wire;
            }

            // circ.add_vertex_with_edges(op, incoming, outgoing);
        }
        // circ.dag
        //     .connect_many(
        //         circ.boundary()[1],
        //         qubit_wire_map,
        //         Direction::Incoming,
        //         None,
        //     )
        //     .unwrap();
        circ.finish_hugr_with_outputs(qubit_wires).unwrap()
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
    // use crate::validate::check_soundness;

    use hugr::HugrView;

    use super::*;
    fn load_representative_set(path: &str) -> HashMap<String, Circuit> {
        let jsons = std::fs::read_to_string(path).unwrap();
        // read_rep_json(&jsons).unwrap();
        let st: Vec<RepCircData> = serde_json::from_str(&jsons).unwrap();
        st.into_iter()
            .map(|mut rcd| (rcd.meta.id.remove(0), rcd.into()))
            .collect()
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
    fn test_read_rep() {
        let rep_map: HashMap<String, Circuit> =
            load_representative_set("test_files/h_rz_cxrepresentative_set.json");

        for c in rep_map.values().take(1) {
            println!("{}", c.dot_string());
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
    fn test_read_complete() {
        let _ecc: HashMap<String, Vec<Circuit>> =
            load_ecc_set("test_files/h_rz_cxcomplete_ECC_set.json");

        // ecc.values()
        //     .flatten()
        //     .for_each(|c| check_soundness(c).unwrap());
    }
}
