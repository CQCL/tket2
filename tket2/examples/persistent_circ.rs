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
        n_empty_queues, CircuitDiff, CircuitDiffData, DetachedQueue, DiffUnitValue, DiffValue,
        InvalidNodes, PersistentCircuit, QueueSync, SyncProtocol,
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

fn add_diff_circ0<V: DiffValue<Hugr>, S: SyncProtocol<Hugr>>(
    circ: &mut PersistentCircuit<Hugr, S, V>,
) {
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
    circ.add_diff(base_diff.clone());
}

fn add_diff_circ1<V: DiffValue<Hugr>, S: SyncProtocol<Hugr>>(
    circ: &mut PersistentCircuit<Hugr, S, V>,
    base_diff: &CircuitDiff<Hugr>,
) {
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
    circ.add_diff(diff1.clone());
}

fn add_diff_circ2<V: DiffValue<Hugr>, S: SyncProtocol<Hugr>>(
    circ: &mut PersistentCircuit<Hugr, S, V>,
    base_diff: &CircuitDiff<Hugr>,
) {
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
    let diff2 = base_diff.apply_rewrite(replacement).unwrap();
    circ.add_diff(diff2.clone());
}

#[cfg(feature = "mpi")]
use mpi::main_mpi;
#[cfg(feature = "mpi")]
mod mpi {
    use std::{thread, time::Duration};

    use super::*;
    use ::mpi::traits::Communicator;
    use relrc::mpi::RelRcCommunicator;
    use tket2::diff::{MPIRank, MPISync};

    pub fn main_mpi() {
        println!("Running MPI version");

        let universe = ::mpi::initialize().unwrap();
        let world = universe.world();
        let rank = world.rank().into();
        let size = world.size();

        assert_eq!(size, 3, "This example must be run with 3 processes");

        let processes = (0..3)
            .map(|i| world.process_at_rank(i))
            .collect_array()
            .unwrap();
        let sync_protocol = create_protocol(rank, processes);

        let mut circ =
            PersistentCircuit::<Hugr, _, _>::new(sync_protocol, DiffUnitValue::default());

        if rank == MPIRank(0) {
            add_diff_circ0(&mut circ);
        }

        sync(&mut circ, rank, 1);
        println!("RANK {}: n_diffs = {}", rank.0, circ.n_diffs());
        let base_diff = circ.diff_iter().exactly_one().ok().expect("one diff");

        if rank == MPIRank(1) {
            add_diff_circ1(&mut circ, &base_diff);
        }

        sync(&mut circ, rank, 2);
        println!("RANK {}: n_diffs = {}", rank.0, circ.n_diffs());

        if rank == MPIRank(2) {
            add_diff_circ2(&mut circ, &base_diff);
        }

        sync(&mut circ, rank, 3);
        println!("RANK {}: n_diffs = {}", rank.0, circ.n_diffs());

        println!("success!");
    }

    fn create_protocol<P>(rank: MPIRank, processes: [P; 3]) -> MPISync<P>
    where
        P: RelRcCommunicator<CircuitDiffData<Hugr>, InvalidNodes> + Clone,
    {
        let broadcast_protocol = |master: usize| -> MPISync<P> {
            let mut ps = BTreeMap::new();
            for i in 0..processes.len() {
                if i != master {
                    ps.insert(MPIRank(i as i32), processes[i].clone());
                }
            }
            MPISync::new(ps)
        };
        let to_one_protocol = |j: usize| -> MPISync<P> {
            let mut ps = BTreeMap::new();
            ps.insert(MPIRank(j as i32), processes[j].clone());
            MPISync::new(ps)
        };
        match rank {
            MPIRank(0) => broadcast_protocol(0),
            MPIRank(1) | MPIRank(2) => to_one_protocol(0),
            _ => unreachable!("must have 3 processes"),
        }
    }

    fn sync<P>(
        circ: &mut PersistentCircuit<Hugr, MPISync<P>, DiffUnitValue>,
        rank: MPIRank,
        expected_n_diffs: usize,
    ) where
        P: RelRcCommunicator<CircuitDiffData<Hugr>, InvalidNodes> + Clone,
    {
        while circ.n_diffs() < expected_n_diffs {
            println!("RANK {}: syncing", rank.0);
            circ.sync();
            thread::sleep(Duration::from_secs(1));
        }
        println!("RANK {}: syncing success", rank.0);
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

fn main_no_mpi() {
    let queues = n_empty_queues::<3, Hugr>();
    let sync_protocols = create_sync_protocols(&queues);

    let mut circs = sync_protocols.map(|s| PersistentCircuit::new(s, DiffUnitValue::default()));

    print_status(&circs);

    add_diff_circ0(&mut circs[0]);

    print_status(&circs);

    println!("Round of syncing");
    for c in circs.iter_mut() {
        c.sync();
    }

    print_status(&circs);

    let base_diff = circs[0].diff_iter().exactly_one().ok().expect("one diff");

    add_diff_circ1(&mut circs[1], &base_diff);

    print_status(&circs);

    println!("Round of syncing");
    for c in circs.iter_mut() {
        c.sync();
    }

    print_status(&circs);

    add_diff_circ2(&mut circs[2], &base_diff);

    print_status(&circs);

    println!("Round of syncing");
    for c in circs.iter_mut() {
        c.sync();
    }

    print_status(&circs);
}

#[cfg(not(feature = "mpi"))]
fn main() {
    main_no_mpi();
}

#[cfg(feature = "mpi")]
fn main() {
    main_mpi();
}
