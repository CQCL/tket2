//! A test of the explorer as it would typically be used by a user in practice.

use hugr::{
    builder::{endo_sig, DFGBuilder, Dataflow, DataflowHugr},
    extension::prelude::qb_t,
    ops::OpType,
    Hugr, HugrView,
};
use itertools::Itertools;
use tket2::rewrite_space::CommuteCZFactory;
use tket2::{
    rewrite_space::{CommitFactory, ExploreOptions, RewriteSpace},
    Tk2Op,
};

fn dfg_hugr() -> Hugr {
    // All gates are CZ gates (i.e. they commute with eachother):
    //
    // --o--o-----o--o-----
    //   |  |     |  |
    // --o--+--o--+--o--o--
    //      |  |  |     |
    // -----o--o--o-----o--
    let mut builder = DFGBuilder::new(endo_sig(vec![qb_t(), qb_t(), qb_t()])).unwrap();
    let [q0, q1, q2] = builder.input_wires_arr();
    let cz1 = builder.add_dataflow_op(Tk2Op::CZ, vec![q0, q1]).unwrap();
    let [q0, q1] = cz1.outputs_arr();
    let cz2 = builder.add_dataflow_op(Tk2Op::CZ, vec![q0, q2]).unwrap();
    let [q0, q2] = cz2.outputs_arr();
    let cz3 = builder.add_dataflow_op(Tk2Op::CZ, vec![q1, q2]).unwrap();
    let [q1, q2] = cz3.outputs_arr();
    let cz4 = builder.add_dataflow_op(Tk2Op::CZ, vec![q0, q2]).unwrap();
    let [q0, q2] = cz4.outputs_arr();
    let cz5 = builder.add_dataflow_op(Tk2Op::CZ, vec![q0, q1]).unwrap();
    let [q0, q1] = cz5.outputs_arr();
    let cz6 = builder.add_dataflow_op(Tk2Op::CZ, vec![q1, q2]).unwrap();
    let [q1, q2] = cz6.outputs_arr();
    builder.finish_hugr_with_outputs(vec![q0, q1, q2]).unwrap()
}

#[test]
fn dummy_test_to_save_hugr() {
    // DAN: If you need to produce serialised HUGRs, you can use this test
    // and run it with `cargo test dummy_test_to_save_hugr`.
    let hugr = dfg_hugr();
    let writer = std::fs::File::create("6cz.hugr").unwrap();
    hugr.store(writer, Default::default()).unwrap();
}

#[test]
#[ignore = "takes 7sec in debug"]
fn test_commute_cz() {
    let explorer = CommuteCZFactory;
    let mut space = RewriteSpace::with_base(dfg_hugr());

    let opts = ExploreOptions {
        max_rewrites: 60.into(),
    };
    explorer.explore(&mut space, &opts);

    ////////////////////////////////////////////////
    //  REMOVE below and un-pub state_space       //
    ////////////////////////////////////////////////
    let state_space = space.state_space;
    let empty_commits = state_space
        .all_commit_ids()
        .filter(|&id| state_space.inserted_nodes(id).count() == 0)
        .collect_vec();

    // there should be a combination of three empty commits that are compatible
    // and such that the resulting HUGR is empty
    let mut empty_hugr = None;
    for cs in empty_commits.iter().combinations(3) {
        // for cs in empty_commits.iter().combinations(2) {
        let cs = cs.into_iter().copied();
        if let Ok(hugr) = state_space.try_extract_hugr(cs) {
            empty_hugr = Some(hugr);
        }
    }

    let empty_hugr = empty_hugr.unwrap().to_hugr();

    // The empty hugr should have 7 nodes:
    // module root, funcdef, 2 func IO, DFG root, 2 DFG IO
    assert_eq!(empty_hugr.num_nodes(), 7);
    assert_eq!(
        empty_hugr
            .nodes()
            .filter(|&n| {
                !matches!(
                    empty_hugr.get_optype(n),
                    OpType::Input(_)
                        | OpType::Output(_)
                        | OpType::FuncDefn(_)
                        | OpType::Module(_)
                        | OpType::DFG(_)
                )
            })
            .count(),
        0
    );
}
