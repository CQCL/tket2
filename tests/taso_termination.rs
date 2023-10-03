#[cfg(feature = "portmatching")]
mod require_portmatching {
    use hugr::Hugr;
    use rstest::{fixture, rstest};
    use tket2::{
        json::TKETDecode,
        optimiser::{DefaultTasoOptimiser, TasoOptimiser},
        Circuit,
    };
    use tket_json_rs::circuit_json::SerialCircuit;

    #[fixture]
    fn nam_4_2() -> DefaultTasoOptimiser {
        // TasoOptimiser::default_with_rewriter_binary("test_files/nam_4_2.rwr").unwrap()
        TasoOptimiser::default_with_eccs_json_file("test_files/Nam_4_2_complete_ECC_set.json")
            .unwrap()
    }

    #[fixture]
    fn simple_circ() -> Hugr {
        let json = r#"
        {"bits": [], "commands": [{"args": [["q", [0]], ["q", [2]]], "op": {"type": "CX"}}, {"args": [["q", [0]]], "op": {"params": ["0.1"], "type": "Rz"}}, {"args": [["q", [0]], ["q", [1]]], "op": {"type": "CX"}}, {"args": [["q", [1]]], "op": {"type": "H"}}, {"args": [["q", [1]]], "op": {"params": ["0.2"], "type": "Rz"}}, {"args": [["q", [1]]], "op": {"type": "H"}}, {"args": [["q", [0]], ["q", [1]]], "op": {"type": "CX"}}, {"args": [["q", [0]], ["q", [2]]], "op": {"type": "CX"}}, {"args": [["q", [0]]], "op": {"params": ["-0.1"], "type": "Rz"}}], "created_qubits": [], "discarded_qubits": [], "implicit_permutation": [[["q", [0]], ["q", [0]]], [["q", [1]], ["q", [1]]], [["q", [2]], ["q", [2]]]], "phase": "0.0", "qubits": [["q", [0]], ["q", [1]], ["q", [2]]]}
    "#;
        let ser: SerialCircuit = serde_json::from_str(json).unwrap();
        ser.decode().unwrap()
    }

    #[rstest]
    #[ignore = "Takes 800ms"]
    fn taso_termination(simple_circ: Hugr, nam_4_2: DefaultTasoOptimiser) {
        let opt_circ = nam_4_2.optimise(&simple_circ, None, 1.try_into().unwrap());
        assert_eq!(opt_circ.commands().count(), 11);
    }
}
