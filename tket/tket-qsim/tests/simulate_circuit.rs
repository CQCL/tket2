//! End-to-end tests for HUGR traversal and QSystem lowering equivalence.

use std::collections::HashMap;

use approx::assert_abs_diff_eq;
use hugr::{
    builder::{
        CFGBuilder, Container, DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer,
        HugrBuilder, ModuleBuilder, SubContainer,
    },
    extension::prelude::{Barrier, Noop, bool_t, qb_t},
    hugr::hugrmut::HugrMut,
    ops::{Value, handle::NodeHandle},
    type_row,
    types::Signature,
};
use num_complex::Complex64;
use tket::{
    TketOp,
    extension::{pauli_exp::pauli_exp, rotation::ConstRotation},
    passes::{
        ComposablePass, InlineDFGsPass, InlineFunctionsPass, inline_funcs::InlineFuncsHeuristic,
    },
};
use tket_qsim::{Simulatable, SimulationError, align_global_phase_to, simulate_circuit};
use tket_qsystem::extension::qsystem::{LowerTketToQSystemPass, QSystemPlatform};

fn reference_circuit() -> hugr::Hugr {
    let mut builder = DFGBuilder::new(Signature::new_endo([qb_t(), qb_t()])).unwrap();
    let [q0, q1] = builder.input_wires_arr();
    let [q0] = builder
        .add_dataflow_op(TketOp::H, [q0])
        .unwrap()
        .outputs_arr();
    let [q0, q1] = builder
        .add_dataflow_op(TketOp::CX, [q0, q1])
        .unwrap()
        .outputs_arr();
    let angle = builder.add_constant(Value::extension(ConstRotation::new(0.5).unwrap()));
    let angle = builder.load_const(&angle);
    let [q1] = builder
        .add_dataflow_op(TketOp::Rz, [q1, angle])
        .unwrap()
        .outputs_arr();
    builder.finish_hugr_with_outputs([q0, q1]).unwrap()
}

fn unsupported_operation(hugr: &hugr::Hugr) -> String {
    match simulate_circuit(hugr, &HashMap::new()) {
        Err(SimulationError::UnsupportedOp(op)) => op,
        Err(error) => panic!("expected an unsupported operation, got {error}"),
        Ok(_) => panic!("expected simulation to reject the circuit"),
    }
}

#[test]
fn composed_circuit_matches_pytket_reference() {
    // Equivalent pytket circuit:
    // Circuit(2).H(0).CX(0, 1).Rz(0.5, 1)
    let hugr = reference_circuit();

    let unitary = simulate_circuit(&hugr, &HashMap::new()).unwrap();
    let p = Complex64::new(0.5, 0.5);
    let m = Complex64::new(0.5, -0.5);
    let z = Complex64::new(0.0, 0.0);
    let expected = [
        m, z, m, z, // row 0
        z, p, z, p, // row 1
        z, m, z, -m, // row 2
        p, z, -p, z, // row 3
    ];

    for row in 0..4 {
        for col in 0..4 {
            assert_abs_diff_eq!(
                unitary[(row, col)],
                expected[row * 4 + col],
                epsilon = 1e-12
            );
        }
    }
}

#[test]
fn lowering_to_each_qsystem_platform_preserves_the_unitary() {
    let original = reference_circuit();
    let expected = simulate_circuit(&original, &HashMap::new()).unwrap();

    for platform in [QSystemPlatform::Helios, QSystemPlatform::Sol] {
        let mut lowered = original.clone();
        LowerTketToQSystemPass::new(platform)
            .run(&mut lowered)
            .unwrap();
        // Multi-operation lowerings are represented as calls. The lightweight
        // simulator operates on a flat dataflow graph, so normalize those
        // calls and their nested DFGs before comparing semantics.
        InlineFunctionsPass::default()
            .with_heuristic(InlineFuncsHeuristic::All)
            .run(&mut lowered)
            .unwrap();
        InlineDFGsPass::default().run(&mut lowered).unwrap();
        let actual = simulate_circuit(&lowered, &HashMap::new()).unwrap();
        let actual = align_global_phase_to(&actual, &expected);
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-10);
    }
}

#[test]
fn explicitly_supported_identity_operations_are_allowed() {
    let mut builder = DFGBuilder::new(Signature::new_endo([qb_t()])).unwrap();
    let [q] = builder.input_wires_arr();
    let [q] = builder
        .add_dataflow_op(Noop::new(qb_t()), [q])
        .unwrap()
        .outputs_arr();
    let [q] = builder
        .add_dataflow_op(Barrier::new([qb_t()]), [q])
        .unwrap()
        .outputs_arr();
    let [q] = builder
        .add_dataflow_op(TketOp::H, [q])
        .unwrap()
        .outputs_arr();
    let hugr = builder.finish_hugr_with_outputs([q]).unwrap();

    let actual = simulate_circuit(&hugr, &HashMap::new()).unwrap();
    let expected = TketOp::H.unitary(&[]);
    assert_abs_diff_eq!(actual, expected, epsilon = 1e-12);
}

#[test]
fn measurement_is_rejected() {
    let mut builder = DFGBuilder::new(Signature::new_endo([qb_t()])).unwrap();
    let [q] = builder.input_wires_arr();
    let [q, _result] = builder
        .add_dataflow_op(TketOp::Measure, [q])
        .unwrap()
        .outputs_arr();
    let hugr = builder.finish_hugr_with_outputs([q]).unwrap();

    assert!(unsupported_operation(&hugr).contains("Measure"));
}

#[test]
fn reset_is_rejected() {
    let mut builder = DFGBuilder::new(Signature::new_endo([qb_t()])).unwrap();
    let [q] = builder.input_wires_arr();
    let [q] = builder
        .add_dataflow_op(TketOp::Reset, [q])
        .unwrap()
        .outputs_arr();
    let hugr = builder.finish_hugr_with_outputs([q]).unwrap();

    assert!(unsupported_operation(&hugr).contains("Reset"));
}

#[test]
fn allocation_and_free_are_rejected() {
    let mut builder = DFGBuilder::new(Signature::new_endo([qb_t()])).unwrap();
    let [q] = builder.input_wires_arr();
    let [ancilla] = builder
        .add_dataflow_op(TketOp::QAlloc, [])
        .unwrap()
        .outputs_arr();
    builder.add_dataflow_op(TketOp::QFree, [ancilla]).unwrap();
    let hugr = builder.finish_hugr_with_outputs([q]).unwrap();

    let operation = unsupported_operation(&hugr);
    assert!(operation.contains("QAlloc") || operation.contains("QFree"));
}

#[test]
fn unknown_quantum_extension_operation_is_rejected() {
    let mut builder = DFGBuilder::new(Signature::new_endo([qb_t()])).unwrap();
    let [q] = builder.input_wires_arr();
    let angle = builder.add_constant(Value::extension(ConstRotation::new(0.25).unwrap()));
    let angle = builder.load_const(&angle);
    let [q] = builder
        .add_dataflow_op(pauli_exp(1, &[1]).unwrap(), [q, angle])
        .unwrap()
        .outputs_arr();
    let hugr = builder.finish_hugr_with_outputs([q]).unwrap();

    assert!(unsupported_operation(&hugr).contains("PauliExp"));
}

#[test]
fn function_calls_must_be_inlined_before_simulation() {
    let mut module = ModuleBuilder::new();
    let callee = {
        let mut builder = module
            .define_function("callee", Signature::new_endo([qb_t()]))
            .unwrap();
        let [q] = builder.input_wires_arr();
        let [q] = builder
            .add_dataflow_op(TketOp::H, [q])
            .unwrap()
            .outputs_arr();
        builder.finish_with_outputs([q]).unwrap()
    };
    let main = {
        let mut builder = module
            .define_function("main", Signature::new_endo([qb_t()]))
            .unwrap();
        let [q] = builder.input_wires_arr();
        let [q] = builder
            .call(callee.handle(), &[], [q])
            .unwrap()
            .outputs_arr();
        builder.finish_with_outputs([q]).unwrap()
    };
    let mut hugr = module.finish_hugr().unwrap();
    hugr.set_entrypoint(main.node());

    assert!(unsupported_operation(&hugr).contains("Call"));
}

#[test]
fn nested_dataflow_must_be_inlined_before_simulation() {
    let mut builder = DFGBuilder::new(Signature::new_endo([qb_t()])).unwrap();
    let inputs = builder.input_wires();
    let mut nested = builder
        .dfg_builder(Signature::new_endo([qb_t()]), inputs)
        .unwrap();
    let [q] = nested.input_wires_arr();
    let [q] = nested
        .add_dataflow_op(TketOp::H, [q])
        .unwrap()
        .outputs_arr();
    let [q] = nested.finish_with_outputs([q]).unwrap().outputs_arr();
    let hugr = builder.finish_hugr_with_outputs([q]).unwrap();

    assert!(unsupported_operation(&hugr).contains("DFG"));
}

#[test]
fn conditional_control_flow_is_rejected() {
    let mut builder = DFGBuilder::new(Signature::new([bool_t(), qb_t()], [qb_t()])).unwrap();
    let [condition, q] = builder.input_wires_arr();
    let mut conditional = builder
        .conditional_builder(
            ([type_row![], type_row![]], condition),
            [(qb_t(), q)],
            [qb_t()].into(),
        )
        .unwrap();

    for case_index in 0..2 {
        let case = conditional.case_builder(case_index).unwrap();
        let [q] = case.input_wires_arr();
        case.finish_with_outputs([q]).unwrap();
    }

    let [q] = conditional.finish_sub_container().unwrap().outputs_arr();
    let hugr = builder.finish_hugr_with_outputs([q]).unwrap();

    assert!(unsupported_operation(&hugr).contains("Conditional"));
}

#[test]
fn cfg_entrypoint_is_rejected() {
    let mut cfg = CFGBuilder::new(Signature::new_endo([qb_t()])).unwrap();
    let block = {
        let mut block = cfg
            .entry_builder(vec![type_row![]], [qb_t()].into())
            .unwrap();
        let [q] = block.input_wires_arr();
        let branch = block.make_sum(0, [type_row![]], []).unwrap();
        block.finish_with_outputs(branch, [q]).unwrap()
    };
    cfg.branch(&block, 0, &cfg.exit_block()).unwrap();
    let hugr = cfg.finish_hugr().unwrap();

    assert!(unsupported_operation(&hugr).contains("CFG"));
}
