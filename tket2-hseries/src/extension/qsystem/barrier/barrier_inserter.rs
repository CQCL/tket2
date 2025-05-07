use hugr::algorithms::replace_types::NodeTemplate;
use hugr::builder::{Container, DataflowHugr};
use hugr::extension::prelude::{qb_t, UnpackTuple};
use hugr::ops::OpTrait;
use hugr::types::{Signature, Type};
use hugr::{
    algorithms::replace_types::ReplaceTypes,
    builder::{DFGBuilder, Dataflow},
    extension::prelude::Barrier,
    hugr::{
        hugrmut::HugrMut,
        patch::{insert_cut::InsertCut, PatchHugrMut},
    },
    Hugr, HugrView, IncomingPort, Node, OutgoingPort, Wire,
};

use crate::extension::qsystem::barrier::barrier_ops::{
    build_runtime_barrier_op, BarrierOperationFactory,
};
use crate::extension::qsystem::lower::insert_function;
use crate::extension::qsystem::LowerTk2Error;

type Target = (Node, IncomingPort);

/// Responsible for inserting runtime barriers into the HUGR
pub struct BarrierInserter {
    /// Factory for creating barrier operations
    op_factory: BarrierOperationFactory,
}

impl BarrierInserter {
    /// Create a new BarrierInserter instance
    pub fn new() -> Self {
        Self {
            op_factory: BarrierOperationFactory::new(),
        }
    }

    /// Filter out types in the generic barrier that contain qubits
    fn filter_qubit_containers<H: HugrMut<Node = Node>>(
        &mut self,
        hugr: &H,
        barrier: &Barrier,
        node: Node,
    ) -> Vec<(Type, Target)> {
        barrier
            .type_row
            .iter()
            .enumerate()
            .filter_map(|(i, typ)| {
                let wc = self.op_factory.type_analyzer().unpack_type(typ);

                if wc.is_qb_container() {
                    let port = OutgoingPort::from(i);
                    let target = hugr
                        .single_linked_input(node, port)
                        .expect("linearity violation.");
                    Some((typ.clone(), target))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Try to apply a simplified barrier for the special case of a single array of qubits
    fn try_array_barrier_shortcut(
        &mut self,
        hugr: &mut impl HugrMut<Node = Node>,
        parent: Node,
        typ: &Type,
        target: Target,
    ) -> Option<Result<(), LowerTk2Error>> {
        // Check if this is an array of qubits
        let size = self.op_factory.type_analyzer().is_qubit_array(typ)?;

        // Build and insert the barrier
        Some(match build_runtime_barrier_op(size) {
            Ok(barr_hugr) => {
                let insert = InsertCut::new(parent, vec![target], barr_hugr);
                insert.apply_hugr_mut(hugr).map(|_| ()).map_err(Into::into)
            }
            Err(err) => Err(err.into()),
        })
    }

    /// Construct the endofunction HUGR to unpack types, apply the runtime barrier across qubits, and repack
    fn build_packing_hugr(&mut self, container_row: Vec<Type>) -> Result<Hugr, LowerTk2Error> {
        // Create a signature for an endofunction on the container row
        let mut dfg_b = DFGBuilder::new(Signature::new_endo(container_row.clone()))?;

        // Pack the container row in to a tuple to use the tuple unpacking logic
        let tuple_type = Type::new_tuple(container_row.clone());

        let input = dfg_b.input();
        let tuple = dfg_b.make_tuple(input.outputs())?;

        // Unpack the tuple into wires
        let unpacked_wires = self
            .op_factory
            .unpack_container(&mut dfg_b, &tuple_type, tuple)?;

        // Tag the qubit wires
        let tagged_wires: Vec<(bool, Wire)> = unpacked_wires
            .into_iter()
            .map(|wire| {
                let node_sig = dfg_b
                    .hugr()
                    .get_optype(wire.node())
                    .dataflow_signature()
                    .unwrap();
                (node_sig.port_type(wire.source()) == Some(&qb_t()), wire)
            })
            .collect();

        // Extract just the qubit wires
        let qubit_wires: Vec<_> = tagged_wires
            .iter()
            .filter(|(is_qb, _)| *is_qb)
            .map(|(_, w)| *w)
            .collect();

        // Call the runtime barrier on all the qubit wires
        let mut barrier_outputs = self
            .op_factory
            .build_runtime_barrier(&mut dfg_b, qubit_wires)?;

        // Replace the qubit wires with the runtime barrier outputs
        let repack_wires = tagged_wires
            .into_iter()
            .map(|(is_qb, w)| {
                if is_qb {
                    barrier_outputs
                        .next()
                        .expect("Not enough runtime barrier outputs.")
                } else {
                    w
                }
            })
            .collect();

        // Repack the wires into a tuple
        let repacked_tuple =
            self.op_factory
                .repack_container(&mut dfg_b, &tuple_type, repack_wires)?;

        // Separate back into a row
        let new_container_wires = dfg_b
            .add_dataflow_op(
                UnpackTuple::new(container_row.clone().into()),
                [repacked_tuple],
            )?
            .outputs();

        let h = dfg_b.finish_hugr_with_outputs(new_container_wires)?;
        Ok(h)
    }

    /// Insert a runtime barrier after a Barrier in the Hugr
    pub fn insert_runtime_barrier(
        &mut self,
        hugr: &mut impl HugrMut<Node = Node>,
        barrier_node: Node,
        barrier: Barrier,
    ) -> Result<(), LowerTk2Error> {
        // Find all qubit containing types in the barrier
        let filtered_qbs = self.filter_qubit_containers(hugr, &barrier, barrier_node);

        if filtered_qbs.is_empty() {
            return Ok(());
        }

        let parent = hugr
            .get_parent(barrier_node)
            .expect("Barrier can't be root.");

        // Handle the special case of a single array of qubits
        if let [(typ, target)] = filtered_qbs.as_slice() {
            if let Some(result) = self.try_array_barrier_shortcut(hugr, parent, typ, *target) {
                return result;
            }
        }

        // For the general case, build unpacking
        let (types, targets) = filtered_qbs.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
        let insert_hugr = self.build_packing_hugr(types)?;

        // Insert the barrier into the HUGR
        let inserter = InsertCut::new(parent, targets, insert_hugr);
        inserter.apply_hugr_mut(hugr)?;
        Ok(())
    }

    /// Register function replacements for all temporary operations
    pub fn register_operation_replacements(
        self,
        hugr: &mut impl HugrMut<Node = Node>,
        lowerer: &mut ReplaceTypes,
    ) {
        // Use the map of cached functions to register replacements
        for (op, func_def) in self.op_factory.funcs {
            let func_node = insert_function(hugr, func_def.clone());
            lowerer.replace_op(op.extension_op(), NodeTemplate::Call(func_node, vec![]));
        }
    }
}

impl Default for BarrierInserter {
    fn default() -> Self {
        Self::new()
    }
}
