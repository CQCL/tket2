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

    /// A set of equivalence circuit classes (ECC)
    ///
    /// This is the complete set of ECCs for 2-qubit circuits with up to
    /// 4 gates, using the NAM gateset (CX, Rz, H).
    #[fixture]
    fn nam_4_2() -> DefaultTasoOptimiser {
        TasoOptimiser::default_with_eccs_json_file("test_files/Nam_4_2_complete_ECC_set.json")
            .unwrap()
    }

    /// The following circuit
    ///          ┌──────────┐                                    ┌───────────┐
    ///q_0: ──■──┤ Rz(π/10) ├──■─────────────────────────■────■──┤ Rz(-π/10) ├
    ///       │  └──────────┘┌─┴─┐┌───┐┌─────────┐┌───┐┌─┴─┐  │  └───────────┘
    ///q_1: ──┼──────────────┤ X ├┤ H ├┤ Rz(π/5) ├┤ H ├┤ X ├──┼───────────────
    ///     ┌─┴─┐            └───┘└───┘└─────────┘└───┘└───┘┌─┴─┐
    ///q_2: ┤ X ├───────────────────────────────────────────┤ X ├─────────────
    ///     └───┘                                           └───┘
    #[fixture]
    fn simple_circ() -> Hugr {
        // The TK1 json of the circuit
        let json = r#"{
            "bits": [],
            "commands": [
                {"args": [["q", [0]], ["q", [2]]], "op": {"type": "CX"}},
                {"args": [["q", [0]]], "op": {"params": ["0.1"], "type": "Rz"}},
                {"args": [["q", [0]], ["q", [1]]], "op": {"type": "CX"}},
                {"args": [["q", [1]]], "op": {"type": "H"}},
                {"args": [["q", [1]]], "op": {"params": ["0.2"], "type": "Rz"}},
                {"args": [["q", [1]]], "op": {"type": "H"}},
                {"args": [["q", [0]], ["q", [1]]], "op": {"type": "CX"}},
                {"args": [["q", [0]], ["q", [2]]], "op": {"type": "CX"}},
                {"args": [["q", [0]]], "op": {"params": ["-0.1"], "type": "Rz"}}],
            "created_qubits": [],
            "discarded_qubits": [],
            "implicit_permutation": [
                [["q", [0]], ["q", [0]]], [["q", [1]], ["q", [1]]], [["q", [2]], ["q", [2]]]
            ],
            "phase": "0.0",
            "qubits": [["q", [0]], ["q", [1]], ["q", [2]]]
        }"#;
        let ser: SerialCircuit = serde_json::from_str(json).unwrap();
        ser.decode().unwrap()
    }

    #[rstest]
    //#[ignore = "Takes 200ms"]
    fn taso_termination(simple_circ: Hugr, nam_4_2: DefaultTasoOptimiser) {
        let opt_circ = nam_4_2.optimise(&simple_circ, None, 1.try_into().unwrap(), false, 10);
        assert_eq!(opt_circ.commands().count(), 11);
    }
}
