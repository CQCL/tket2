//! Example of sync protocol in persistent circuits.

use std::collections::BTreeMap;

use hugr::{
    builder::{BuildError, CircuitBuilder, Dataflow, DataflowHugr, FunctionBuilder},
    extension::prelude::qb_t,
    types::Signature,
    Hugr, HugrView, Node,
};
use itertools::Itertools;
use portgraph::NodeIndex;
use tket2::{
    diff::{
        n_empty_queues, CircuitDiff, DetachedQueue, DiffUnitValue, PersistentCircuit, QueueSync,
    },
    extension::TKET2_EXTENSION_ID,
    rewrite::{CircuitRewrite, Subcircuit},
    Circuit, Tk2Op,
};

pub(crate) fn build_simple_circuit<F>(num_qubits: usize, f: F) -> Result<Circuit, BuildError>
where
    F: FnOnce(&mut CircuitBuilder<FunctionBuilder<Hugr>>) -> Result<(), BuildError>,
{
    let qb_row = vec![qb_t(); num_qubits];
    let signature = Signature::new(qb_row.clone(), qb_row).with_extension_delta(TKET2_EXTENSION_ID);
    let mut h = FunctionBuilder::new("main", signature)?;

    let qbs = h.input_wires();

    let mut circ = h.as_circuit(qbs);

    f(&mut circ)?;

    let qbs = circ.finish();

    let hugr = h.finish_hugr_with_outputs(qbs)?;
    Ok(hugr.into())
}

fn print_status<H, S, V>(circs: &[PersistentCircuit<H, S, V>])
where
    H: HugrView<Node = Node>,
{
    for (i, c) in circs.iter().enumerate() {
        println!("In diff {i}: {} diffs", c.n_diffs());
    }
}

fn create_sync_protocols(queues: &[DetachedQueue<Hugr>; 3]) -> [QueueSync<Hugr>; 3] {
    let broadcast_protocol = |master: usize| -> QueueSync<Hugr> {
        let mut outgoing_queues = BTreeMap::new();
        for i in 0..queues.len() {
            if i != master {
                outgoing_queues.insert(i, queues[i].clone());
            }
        }
        let incoming_queue = queues[master].clone();
        QueueSync::new(incoming_queue, outgoing_queues, master)
    };
    let one_to_one_protocol = |i: usize, j: usize| -> QueueSync<Hugr> {
        let outgoing_queues = BTreeMap::from_iter([(j, queues[j].clone())]);
        let incoming_queue = queues[i].clone();
        QueueSync::new(incoming_queue, outgoing_queues, i)
    };
    [
        broadcast_protocol(0),
        one_to_one_protocol(1, 0),
        one_to_one_protocol(2, 0),
    ]
}

fn main() {
    let queues = n_empty_queues::<3, Hugr>();
    let sync_protocols = create_sync_protocols(&queues);

    let mut circs = sync_protocols.map(|s| PersistentCircuit::new(s, DiffUnitValue::default()));

    print_status(&circs);

    println!("Adding base diff to persistent circuit 0");
    let base_circuit = build_simple_circuit(2, |circ| {
        circ.append(Tk2Op::CX, [0, 1])?;
        circ.append(Tk2Op::H, [1])?;
        circ.append(Tk2Op::CX, [0, 1])?;
        circ.append(Tk2Op::CX, [0, 1])?;
        Ok(())
    })
    .unwrap();
    let base_diff = CircuitDiff::try_from_circuit(base_circuit).unwrap();
    circs[0].add_diff(base_diff.clone());

    print_status(&circs);

    println!("Round of syncing");
    for c in circs.iter_mut() {
        c.sync();
    }

    print_status(&circs);

    println!("Applying a rewrite in persistent circuit 1");
    let replacement = {
        let repl_circ = build_simple_circuit(2, |circ| {
            circ.append(Tk2Op::CX, [1, 0])?;
            Ok(())
        })
        .unwrap();
        let subcircuit = Subcircuit::try_from_nodes(
            [
                NodeIndex::new(3).into(),
                NodeIndex::new(4).into(),
                NodeIndex::new(5).into(),
            ],
            &base_diff.as_circuit(),
        )
        .unwrap();
        CircuitRewrite::try_new(&subcircuit, base_diff.as_circuit(), repl_circ).unwrap()
    };
    let diff1 = base_diff.apply_rewrite(replacement).unwrap();
    circs[1].add_diff(diff1.clone());

    print_status(&circs);

    println!("Round of syncing");
    for c in circs.iter_mut() {
        c.sync();
    }

    print_status(&circs);

    println!("Applying a rewrite in persistent circuit 2");
    let replacement = {
        let repl_circ = build_simple_circuit(2, |_| Ok(())).unwrap();
        let subcircuit = Subcircuit::try_from_nodes(
            [NodeIndex::new(5).into(), NodeIndex::new(6).into()],
            &base_diff.as_circuit(),
        )
        .unwrap();
        CircuitRewrite::try_new(&subcircuit, base_diff.as_circuit(), repl_circ).unwrap()
    };
    let diff1 = base_diff.apply_rewrite(replacement).unwrap();
    circs[2].add_diff(diff1.clone());
    print_status(&circs);

    println!("Round of syncing");
    for c in circs.iter_mut() {
        c.sync();
    }

    print_status(&circs);

    dbg!(queues.iter().map(|q| q.borrow().len()).collect_vec());
}
