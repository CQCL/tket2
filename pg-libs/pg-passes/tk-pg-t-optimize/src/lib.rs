//! T-count optimization for Clifford+T Pauli graphs.
//!
//! Bit-vector operations are scalar by default. The `unstable_simd` feature enables
//! portable SIMD and requires a nightly Rust toolchain.

#![cfg_attr(feature = "unstable_simd", feature(portable_simd))]

mod clifford_propagation;
mod hadamard_optimizer;
mod optimizer;
mod simd_vector;
mod t_optimizer;

use tk_pg_core::{PGPass, PauliGraph};

use crate::optimizer::optimize;

/// A T-count optimization pass for rotations, tableaux, and unconditional H gates.
/// Classical operations stay in place, and each quantum section is optimized separately
#[derive(Default)]
pub struct TOptimizationPass {
    ancilla_budget: usize,
    first_bit: usize,
}

impl TOptimizationPass {
    /// Create a pass that uses no ancillas.
    pub fn new() -> Self {
        Self::default()
    }

    /// Reserve the last `budget` input qubits as idle ancillas prepared in zero.
    ///
    /// Using ancillas may introduce measurements, resets, and conditional corrections.
    pub fn with_ancilla_budget(mut self, budget: usize) -> Self {
        self.ancilla_budget = budget;
        self
    }

    /// Start numbering new measurement results at `first_bit` (default: zero).
    /// Set this to the number of existing classical bits to avoid overwriting them.
    pub fn with_first_bit(mut self, first_bit: usize) -> Self {
        self.first_bit = first_bit;
        self
    }

    /// Optimize a graph whose rotation angles are multiples of 0.25 half turns.
    /// Classical operations split the graph into sections. Measurement bits are
    /// numbered from `first_bit` and aren't reused between sections.
    ///
    /// Panics if the input contains unsupported operations or non-idle reserved ancillas.
    pub fn optimize(&self, graph: &PauliGraph) -> PauliGraph {
        optimize(graph, self.ancilla_budget, self.first_bit)
    }
}

impl PGPass for TOptimizationPass {
    fn transform(&self, pauli_graph: &PauliGraph) -> PauliGraph {
        self.optimize(pauli_graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use tk_pg_converter::compare_unitaries_via_tk;
    use tk_pg_core::{ClassicalData, GateData, GateType, Op, Pauli, RotationData};
    use tk_pg_ir_kernels::PGTableau;
    use tk_pg_qm_tableau::Tableau;

    #[test]
    fn keeps_measurement_bits_distinct_across_classical_sections() {
        let section: Vec<_> = [Pauli::Z, Pauli::X, Pauli::Z]
            .into_iter()
            .map(|p| Op::Rotation {
                data: RotationData::new(vec![p, Pauli::I], 0.25),
            })
            .collect();
        let classical = Op::Classical {
            data: ClassicalData::new(vec![7], "payload".into()),
        };
        let mut input = PauliGraph::new(2).with_ops(section.clone());
        input.add_op(classical.clone());
        input.extend(PauliGraph::new(2).with_ops(section));
        let output = TOptimizationPass::new()
            .with_ancilla_budget(1)
            .with_first_bit(8)
            .transform(&input);
        output.try_validate().unwrap();
        let boundary = output
            .get_ops()
            .iter()
            .position(|op| op == &classical)
            .unwrap();
        let mut bits = HashSet::new();
        for ops in [
            &output.get_ops()[..boundary],
            &output.get_ops()[boundary + 1..],
        ] {
            let measurements: Vec<_> = ops
                .iter()
                .filter_map(|op| match op {
                    Op::Gate { data } if data.get_gate_type() == &GateType::Measure => {
                        Some(data.get_args()[1])
                    }
                    Op::Measure { data } => Some(data.get_cbit()),
                    _ => None,
                })
                .collect();
            assert!(!measurements.is_empty());
            for bit in measurements {
                assert!(bit >= 8);
                assert!(
                    bits.insert(bit),
                    "measurement bit was reused across sections"
                );
            }
        }
    }

    #[test]
    fn reduces_t_count_without_ancillas() {
        let mut clifford = Tableau::eye(4);
        clifford.postcompose_op(&Op::Gate {
            data: GateData::new(GateType::H, vec![0]),
        });
        clifford.postcompose_op(&Op::Gate {
            data: GateData::new(GateType::S, vec![0]),
        });
        clifford.postcompose_op(&Op::Gate {
            data: GateData::new(GateType::ZX, vec![0, 1]),
        });
        clifford.postcompose_op(&Op::Gate {
            data: GateData::new(GateType::H, vec![2]),
        });
        clifford.postcompose_op(&Op::Gate {
            data: GateData::new(GateType::ZX, vec![2, 3]),
        });
        clifford.postcompose_op(&Op::Gate {
            data: GateData::new(GateType::S, vec![3]),
        });
        clifford.postcompose_op(&Op::Gate {
            data: GateData::new(GateType::ZX, vec![1, 2]),
        });
        clifford.postcompose_op(&Op::Gate {
            data: GateData::new(GateType::H, vec![1]),
        });

        for (input_t_count, max_t_count) in [(14, 1), (15, 0)] {
            // T rotations on all 15 nonzero four-qubit parities give identity up to phase.
            // Omitting one parity leaves a single inverse T rotation.
            // The Clifford conjugation preserves commutation and introduces X/Y terms.
            let pg = PauliGraph::new(4).with_ops(
                (1..=input_t_count)
                    .flat_map(|parity| {
                        let string = (0..4)
                            .map(|q| {
                                if parity & (1 << q) == 0 {
                                    Pauli::I
                                } else {
                                    Pauli::Z
                                }
                            })
                            .collect();
                        clifford.conjugate(&Op::Rotation {
                            data: RotationData::new(string, 0.25),
                        })
                    })
                    .collect(),
            );

            for pauli in [Pauli::X, Pauli::Y] {
                assert!(pg.get_ops().iter().any(|op| {
                    matches!(op, Op::Rotation { data } if data.get_string().contains(&pauli))
                }));
            }

            let optimized = TOptimizationPass::new().transform(&pg);
            optimized.try_validate().unwrap();
            assert_eq!(optimized.get_n_qubits(), pg.get_n_qubits());
            let rotations: Vec<_> = optimized
                .get_ops()
                .iter()
                .filter_map(|op| match op {
                    Op::Rotation { data } => Some(data),
                    _ => None,
                })
                .collect();
            assert!(rotations.iter().all(|data| data.get_angle().abs() == 0.25));
            assert!(
                rotations.len() <= max_t_count,
                "input T count: {input_t_count}"
            );
            assert!(compare_unitaries_via_tk(&pg, &optimized));

            let classical = Op::Classical {
                data: ClassicalData::new(vec![0], "payload".into()),
            };
            let mut input = pg.clone();
            input.add_op(classical.clone());
            input.extend(pg.clone());
            let optimized = TOptimizationPass::new().transform(&input);
            let boundary = optimized
                .get_ops()
                .iter()
                .position(|op| op == &classical)
                .unwrap();
            for ops in [
                &optimized.get_ops()[..boundary],
                &optimized.get_ops()[boundary + 1..],
            ] {
                let section = PauliGraph::new(4).with_ops(ops.to_vec());
                let t_count = section
                    .get_ops()
                    .iter()
                    .filter(|op| matches!(op, Op::Rotation { .. }))
                    .count();
                assert!(t_count <= max_t_count);
                assert!(compare_unitaries_via_tk(&pg, &section));
            }
        }
    }
}
