//! Modifier for dataflow blocks.
use std::{collections::HashMap, iter, mem};

use hugr::{
    builder::{ConditionalBuilder, Container, DFGBuilder, Dataflow, FunctionBuilder, HugrBuilder, SubContainer, TailLoopBuilder},
    core::HugrNode,
    extension::prelude::qb_t,
    hugr::hugrmut::HugrMut,
    ops::{Conditional, DataflowOpTrait, OpType, TailLoop, DFG},
    std_extensions::collections::array::ArrayOpBuilder,
    types::{FuncTypeBase, PolyFuncType, TypeRow},
    Hugr, HugrView, IncomingPort, OutgoingPort, PortIndex, Wire,
};
use hugr_core::hugr::internal::PortgraphNodeMap;
use itertools::Itertools;
use petgraph::visit::{Topo, Walker};

use super::{DirWire, ModifierError, ModifierResolver, ModifierResolverErrors, PortExt};

impl<N: HugrNode> ModifierResolver<N> {
    /// Modifies the body of a dataflow graph.
    /// We use the topological order of the circuit.
    pub fn modify_dfg_body(
        &mut self,
        h: &impl HugrMut<Node = N>,
        n: N,
        new_dfg: &mut impl Dataflow,
    ) -> Result<(), ModifierResolverErrors<N>> {
        let mut corresp_map = HashMap::new();
        let mut controls = self.init_control_from_input(h, n, new_dfg)?;
        mem::swap(self.corresp_map(), &mut corresp_map);
        mem::swap(self.controls(), &mut controls);

        // Modify the input/output nodes beforehand.
        self.modify_in_out_node(h, n, new_dfg)?;
        // Modify the children nodes.
        self.modify_dfg_children(h, n, new_dfg)?;

        self.wire_control_to_output(h, n, new_dfg)?;
        self.connect_all(h, new_dfg, n)?;
        mem::swap(self.controls(), &mut controls);
        mem::swap(self.corresp_map(), &mut corresp_map);

        Ok(())
    }

    fn modify_dfg_children(
        &mut self,
        h: &impl HugrMut<Node = N>,
        n: N,
        new_dfg: &mut impl Dataflow,
    ) -> Result<(), ModifierResolverErrors<N>> {
        // Visit the nodes in topological order.
        let (region_graph, node_map) = h.region_portgraph(n);
        let mut topo: Vec<_> = Topo::new(&region_graph).iter(&region_graph).collect();
        // Reverse the topological order if dagger is applied.
        if self.modifiers.dagger {
            topo.reverse();
        }
        for old_n_id in topo {
            let old_n = node_map.from_portgraph(old_n_id);
            self.modify_op(h, old_n, new_dfg)?;
        }
        Ok(())
    }

    fn modify_in_out_node(
        &mut self,
        h: &impl HugrMut<Node = N>,
        n: N,
        new_dfg: &mut impl Dataflow,
    ) -> Result<(), ModifierResolverErrors<N>> {
        let [old_in, old_out] = h.get_io(n).unwrap();
        let [new_in, new_out] = new_dfg.io();
        let optype = h.get_optype(n);
        match optype {
            OpType::FuncDefn(_) | OpType::DFG(_) => {
                let FuncTypeBase { input, output } = match optype {
                    OpType::FuncDefn(fndefn) => fndefn.signature().body(),
                    OpType::DFG(dfg) => &dfg.signature(),
                    _ => unreachable!(),
                };
                let offset = if matches!(optype, OpType::FuncDefn(_)) {
                    self.modifiers.accum_ctrl.len()
                } else {
                    self.modifiers.control
                };

                for (i, ref in_out_ty) in input.iter().zip_longest(output.iter()).enumerate() {
                    // If dagger and the i-th input/output are quantum types, swap them.
                    if self.need_swap(in_out_ty.as_deref()) {
                        let old_in_wire = (old_in, OutgoingPort::from(i)).into();
                        let old_out_wire = (old_out, IncomingPort::from(i)).into();
                        let new_in_wire = (new_out, IncomingPort::from(i).shift(offset)).into();
                        let new_out_wire = (new_in, OutgoingPort::from(i).shift(offset)).into();
                        self.map_insert(old_in_wire, new_in_wire)?;
                        self.map_insert(old_out_wire, new_out_wire)?;
                        continue;
                    }
                    if in_out_ty.has_left() {
                        let old_in_wire = (old_in, OutgoingPort::from(i)).into();
                        let new_in_wire = (new_in, OutgoingPort::from(i).shift(offset)).into();
                        self.map_insert(old_in_wire, new_in_wire)?;
                    }
                    if in_out_ty.has_right() {
                        let old_out_wire = (old_out, IncomingPort::from(i)).into();
                        let new_out_wire = (new_out, IncomingPort::from(i).shift(offset)).into();
                        self.map_insert(old_out_wire, new_out_wire)?;
                    }
                }
            }
            OpType::TailLoop(tail_loop) => {
                let just_input_num = tail_loop.just_inputs.len();
                let offset = self.modifiers.control;
                for port in h.node_outputs(old_in) {
                    let new_port = if port.index() < just_input_num {
                        port
                    } else {
                        port.shift(offset)
                    };
                    self.map_insert((old_in, port).into(), DirWire::from((new_in, new_port)))?;
                    println!(
                        "Inserted input mapping: {} -> {}",
                        DirWire::from((old_in, port)),
                        DirWire::from((new_in, new_port))
                    );
                }
                for port in h.node_inputs(old_out) {
                    let new_port = if port.index() == 0 {
                        port
                    } else {
                        port.shift(offset)
                    };
                    self.map_insert((old_out, port).into(), DirWire::from((new_out, new_port)))?
                }
            }
            OpType::Case(_) => {
                return Err(ModifierResolverErrors::Unreachable(
                    "IO of Case node has to be modified directly while modifying Conditional."
                        .to_string(),
                ));
            }
            optype => {
                return Err(ModifierResolverErrors::Unreachable(format!(
                    "Cannot modify the IO of the node with OpType: {}",
                    optype
                )));
            }
        }

        Ok(())
    }

    fn init_control_from_input(
        &mut self,
        h: &impl HugrMut<Node = N>,
        n: N,
        new_dfg: &mut impl Dataflow,
    ) -> Result<Vec<Wire>, ModifierResolverErrors<N>> {
        let controls = match h.get_optype(n) {
            OpType::FuncDefn(_fndefn) => {
                // if controls needs to be unpacked from arraies
                let mut controls = Vec::new();
                let mut inputs = new_dfg.input_wires();
                for (i, size) in self.modifiers.accum_ctrl.iter().enumerate() {
                    if *size == 0 {
                        // if size is 0, connect directly to the output.
                        let zero_array = inputs.next().unwrap();
                        let out_node = new_dfg.io()[1];
                        new_dfg.hugr_mut().connect(
                            zero_array.node(),
                            zero_array.source(),
                            out_node,
                            i,
                        );
                    } else {
                        let ctrl_arr = inputs.next().unwrap();
                        controls.extend(new_dfg.add_array_unpack(
                            qb_t(),
                            *size as u64,
                            ctrl_arr,
                        )?);
                    }
                }
                controls
            }
            OpType::DFG(_) => new_dfg.input_wires().take(self.modifiers.control).collect(),
            OpType::TailLoop(tail_loop) => {
                let just_input_num = tail_loop.just_inputs.len();
                new_dfg
                    .input_wires()
                    .skip(just_input_num)
                    .take(self.modifiers.control)
                    .collect()
            }
            OpType::Case(_) => return Err(ModifierResolverErrors::Unreachable(
                "Control qubits of Case node have to be initialized directly while modifying Conditional."
                    .to_string(),
            )),
            optype => {
                return Err(ModifierResolverErrors::Unreachable(format!(
                    "Cannot set control qubit of the node with OpType: {}",
                    optype
                )));
            }
        };
        Ok(controls)
    }

    fn wire_control_to_output(
        &mut self,
        h: &impl HugrMut<Node = N>,
        n: N,
        new_dfg: &mut impl Dataflow,
    ) -> Result<(), ModifierResolverErrors<N>> {
        let out_node = new_dfg.io()[1];
        let modifiers = &self.modifiers;
        let controls = self.controls_ref();

        match h.get_optype(n) {
            OpType::FuncDefn(_) => {
                let mut offset = 0;
                for (index, size) in modifiers.accum_ctrl.iter().enumerate() {
                    if *size == 0 {
                        continue;
                    }
                    let wire = new_dfg
                        .add_new_array(qb_t(), controls[offset..offset + size].iter().cloned())?;
                    offset += size;
                    new_dfg
                        .hugr_mut()
                        .connect(wire.node(), wire.source(), out_node, index);
                }
            }
            OpType::DFG(_) | OpType::Case(_) => {
                for (i, ctrl) in controls.iter().enumerate() {
                    new_dfg
                        .hugr_mut()
                        .connect(ctrl.node(), ctrl.source(), out_node, i);
                }
            }
            OpType::TailLoop(_) => {
                for (i, ctrl) in controls.iter().enumerate() {
                    new_dfg
                        .hugr_mut()
                        .connect(ctrl.node(), ctrl.source(), out_node, i + 1);
                }
            }
            optype => {
                return Err(ModifierResolverErrors::Unreachable(format!(
                    "Cannot wire outputs of control qubit in the node of OpType: {}",
                    optype
                )))
            }
        }
        Ok(())
    }

    /// Generates a new function modified by the combined modifier.
    pub fn modify_fn(
        &mut self,
        h: &impl HugrMut<Node = N>,
        func: N,
        signature: PolyFuncType,
    ) -> Result<Hugr, ModifierResolverErrors<N>> {
        // Old function definition
        let OpType::FuncDefn(old_fn_defn) = h.get_optype(func) else {
            return Err(ModifierResolverErrors::Unreachable(
                "Cannot modify a non-function node.".to_string(),
            ));
        };
        let old_fn_name = old_fn_defn.func_name();

        // New modified function definition
        let mut new_fn =
            FunctionBuilder::new(format!("__modified__{}", old_fn_name), signature).unwrap();

        self.modify_dfg_body(h, func, &mut new_fn)?;

        println!(
            "modified (before check):\n{}",
            new_fn.hugr_mut().mermaid_string()
        );

        let new_fn = new_fn
            .finish_hugr()
            .map_err(|e| ModifierResolverErrors::BuildError(e.into()))?;

        Ok(new_fn)
    }

    pub(super) fn modify_dfg(
        &mut self,
        h: &impl HugrMut<Node = N>,
        n: N,
        dfg: &DFG,
        new_parent_dfg: &mut impl Dataflow,
    ) -> Result<(), ModifierResolverErrors<N>> {
        let mut signature = dfg.signature.clone();
        // Build a new DFG with modified body.
        self.modify_signature(&mut signature, true);
        let mut builder = DFGBuilder::new(signature.clone()).unwrap();
        self.modify_dfg_body(h, n, &mut builder)?;
        let new_dfg_hugr = builder
            .finish_hugr()
            .map_err(|e| ModifierResolverErrors::ValidationError(e.to_string()))?;
        let new_dfg_node = new_parent_dfg.add_hugr(new_dfg_hugr).inserted_entrypoint;

        // connect the controls and register the IOs
        for (i, c) in self.controls().iter_mut().enumerate() {
            new_parent_dfg
                .hugr_mut()
                .connect(c.node(), c.source(), new_dfg_node, i);
            *c = Wire::new(new_dfg_node, i);
        }
        let offset = self.modifiers.control;
        for (i, in_out_type) in signature
            .input
            .iter()
            .zip_longest(signature.output.iter())
            .enumerate()
        {
            let swap = self.need_swap(in_out_type.as_deref());
            if in_out_type.has_left() {
                let old_in = (n, IncomingPort::from(i)).into();
                let mut new_in = DirWire::from((new_dfg_node, IncomingPort::from(i + offset)));
                if swap {
                    new_in = new_in.reverse();
                }
                self.map_insert(old_in, new_in)?;
            }
            if in_out_type.has_right() {
                let old_out = (n, OutgoingPort::from(i)).into();
                let mut new_out = DirWire::from((new_dfg_node, OutgoingPort::from(i + offset)));
                if swap {
                    new_out = new_out.reverse();
                }
                self.map_insert(old_out, new_out)?;
            }
        }
        // TODO: StateOrder

        Ok(())
    }

    pub(super) fn modify_tail_loop(
        &mut self,
        h: &impl HugrMut<Node = N>,
        n: N,
        tail_loop: &TailLoop,
        new_dfg: &mut impl Dataflow,
    ) -> Result<(), ModifierResolverErrors<N>> {
        let just_input_num = tail_loop.just_inputs.len();
        let just_output_num = tail_loop.just_outputs.len();

        // TailLoop cannot be daggered as long as it is not the one generated from Power modifier.
        // Every TailLoop that is generated from Power cannot have `just_outputs`.
        if self.modifiers.dagger && !tail_loop.just_outputs.is_empty() {
            let optype = h.get_optype(n);
            return Err(ModifierError::ModifierNotApplicable(n, optype.clone()).into());
        }
        // TODO: Handle the case when TailLoop is generated from Power modifier.
        // Currently, it is not implemented.
        if self.modifiers.dagger {
            todo!("Dagger for TailLoop is not supported yet.");
        }

        // Build a new TailLoop with modified body.
        let mut builder = TailLoopBuilder::new(
            tail_loop.just_inputs.clone(),
            tail_loop
                .rest
                .extend(iter::repeat(&qb_t()).take(self.modifiers.control)),
            tail_loop.just_outputs.clone(),
        )?;
        self.modify_dfg_body(h, n, &mut builder)?;
        let tail_loop_hugr = builder
            .finish_hugr()
            .map_err(|e| ModifierResolverErrors::ValidationError(e.to_string()))?;
        let new_tail_loop = new_dfg.add_hugr(tail_loop_hugr).inserted_entrypoint;

        // connect the controls and register IOs
        let offset = self.modifiers.control;
        for (i, ctrl) in self.controls().iter_mut().enumerate() {
            new_dfg.hugr_mut().connect(
                ctrl.node(),
                ctrl.source(),
                new_tail_loop,
                i + just_input_num,
            );
            *ctrl = Wire::new(new_tail_loop, i + just_output_num);
        }
        for port in h.node_inputs(n) {
            let new_port = if port.index() < just_input_num {
                port
            } else {
                port.shift(offset)
            };
            self.map_insert((n, port).into(), (new_tail_loop, new_port).into())?;
        }
        for port in h.node_outputs(n) {
            let new_port = if port.index() < just_output_num {
                port
            } else {
                port.shift(offset)
            };
            self.map_insert((n, port).into(), (new_tail_loop, new_port).into())?
        }

        Ok(())
    }

    pub(super) fn modify_conditional(
        &mut self,
        h: &impl HugrMut<Node = N>,
        n: N,
        conditional: &Conditional,
        new_dfg: &mut impl Dataflow,
    ) -> Result<(), ModifierResolverErrors<N>> {
        let offset = self.modifiers.control;

        // Build a new Conditional with modified body.
        let control_types: TypeRow = iter::repeat(qb_t()).take(offset).collect::<Vec<_>>().into();
        let mut builder = ConditionalBuilder::new(
            conditional.sum_rows.clone(),
            control_types.extend(conditional.other_inputs.iter()),
            control_types.extend(conditional.outputs.iter()),
        )?;

        // remember the current control qubits
        let controls = self.controls().clone();

        for (i, case_node) in h.children(n).enumerate() {
            let tag_wire_num = conditional.sum_rows[i].len();
            let mut case_builder = builder.case_builder(i).unwrap();

            // Set the controls and corresp_map
            let mut corresp_map = HashMap::new();
            let controls = case_builder
                .input_wires()
                .skip(tag_wire_num)
                .take(offset)
                .collect();
            mem::swap(self.corresp_map(), &mut corresp_map);
            *self.controls() = controls;

            // Modify the IOs
            let [old_in, old_out] = h.get_io(case_node).unwrap();
            let [new_in, new_out] = case_builder.io();

            // Modify the input/output nodes beforehand.
            for i in 0..tag_wire_num {
                let old_port = OutgoingPort::from(i);
                let new_port = OutgoingPort::from(i);
                self.map_insert((old_in, old_port).into(), (new_in, new_port).into())?
            }
            for (i, in_out_type) in conditional
                .other_inputs
                .iter()
                .zip_longest(conditional.outputs.iter())
                .enumerate()
            {
                if self.need_swap(in_out_type.as_deref()) {
                    let old_in_wire = (old_in, OutgoingPort::from(i + tag_wire_num)).into();
                    let old_out_wire = (old_out, IncomingPort::from(i)).into();
                    let new_in_wire = (new_out, IncomingPort::from(i + offset)).into();
                    let new_out_wire =
                        (new_in, OutgoingPort::from(i + offset + tag_wire_num)).into();
                    self.map_insert(old_in_wire, new_in_wire)?;
                    self.map_insert(old_out_wire, new_out_wire)?;
                    continue;
                }
                if in_out_type.has_left() {
                    let old_in_wire = (old_in, OutgoingPort::from(i + tag_wire_num)).into();
                    let new_in_wire = (
                        new_in,
                        OutgoingPort::from(i).shift(i + offset + tag_wire_num),
                    )
                        .into();
                    self.map_insert(old_in_wire, new_in_wire)?;
                }
                if in_out_type.has_right() {
                    let old_out_wire = (old_out, IncomingPort::from(i)).into();
                    let new_out_wire = (new_out, IncomingPort::from(i + offset)).into();
                    self.map_insert(old_out_wire, new_out_wire)?;
                }
            }

            // Modify the children.
            self.modify_dfg_children(h, case_node, &mut case_builder)?;

            // Set the controls and corresp_map back
            self.wire_control_to_output(h, case_node, &mut case_builder)?;
            self.connect_all(h, &mut case_builder, case_node)?;
            mem::swap(self.corresp_map(), &mut corresp_map);

            let _ = case_builder
                .finish_sub_container()
                .map_err(|e| ModifierResolverErrors::ValidationError(e.to_string()))?;
        }

        // insert the conditional
        let new_conditional_hugr = builder
            .finish_hugr()
            .map_err(|e| ModifierResolverErrors::ValidationError(e.to_string()))?;
        let new_conditional = new_dfg.add_hugr(new_conditional_hugr).inserted_entrypoint;

        // connect the controls and register the IOs
        *self.controls() = Vec::new();
        for (i, ctrl) in controls.into_iter().enumerate() {
            new_dfg
                .hugr_mut()
                .connect(ctrl.node(), ctrl.source(), new_conditional, i + 1);
            self.controls().push(Wire::new(new_conditional, i));
        }
        self.map_insert(
            (n, IncomingPort::from(0)).into(),
            (new_conditional, IncomingPort::from(0)).into(),
        )?;
        for (i, in_out_type) in conditional
            .other_inputs
            .iter()
            .zip_longest(conditional.outputs.iter())
            .enumerate()
        {
            if self.need_swap(in_out_type.as_deref()) {
                let old_in_wire = (n, OutgoingPort::from(i)).into();
                let old_out_wire = (n, IncomingPort::from(i + 1)).into();
                let new_in_wire = (new_conditional, IncomingPort::from(i + offset + 1)).into();
                let new_out_wire = (new_conditional, OutgoingPort::from(i + offset)).into();
                self.map_insert(old_in_wire, new_in_wire)?;
                self.map_insert(old_out_wire, new_out_wire)?;
                continue;
            }
            if in_out_type.has_left() {
                let old_in_wire = (n, OutgoingPort::from(i)).into();
                let new_in_wire = (new_conditional, OutgoingPort::from(i + offset)).into();
                self.map_insert(old_in_wire, new_in_wire)?;
            }
            if in_out_type.has_right() {
                let old_out_wire = (n, IncomingPort::from(i + 1)).into();
                let new_out_wire = (new_conditional, IncomingPort::from(i + offset + 1)).into();
                self.map_insert(old_out_wire, new_out_wire)?;
            }
        }
        // StateOrder
        // for i in conditional.other_inputs.len() + 1..h.num_inputs(n) {
        //     let old_port = OutgoingPort::from(i);
        //     let new_port = OutgoingPort::from(i + offset);
        //     self.map_insert((n, old_port).into(), (new_conditional, new_port).into())?
        // }
        // for i in conditional.outputs.len()..h.num_outputs(n) {
        //     let old_port = IncomingPort::from(i);
        //     let new_port = IncomingPort::from(i + offset);
        //     self.map_insert((n, old_port).into(), (new_conditional, new_port).into())?
        // }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use std::{fs::File, io::Write, path::Path};

    use hugr::{
        algorithms::{dead_code, ComposablePass},
        builder::{Dataflow, DataflowSubContainer, HugrBuilder, ModuleBuilder, SubContainer},
        envelope::{EnvelopeConfig, EnvelopeFormat},
        extension::{prelude::qb_t, ExtensionRegistry},
        ops::{
            handle::{FuncID, NodeHandle},
            CallIndirect, ExtensionOp,
        },
        std_extensions::collections::array::{array_type, ArrayOpBuilder},
        types::{Signature, Term}, Hugr,
    };

    use crate::{
        extension::{
            bool::BOOL_EXTENSION,
            rotation::{ConstRotation, ROTATION_EXTENSION},
            TKET_EXTENSION,
        },
        rich_circuit::*,
    };
    use crate::{
        // extension::{debug::StateResult, rotation::ConstRotation},
        rich_circuit::modifier_resolver::*,
    };

    fn foo_dfg(module: &mut ModuleBuilder<Hugr>, t_num: usize) -> FuncID<true> {
        let foo_sig = Signature::new_endo(iter::repeat(qb_t()).take(t_num).collect::<Vec<_>>());
        let mut func = module.define_function("foo", foo_sig.clone()).unwrap();
        let mut inputs: Vec<_> = func.input_wires().collect();
        inputs[0] = func
            .add_dataflow_op(TketOp::X, vec![inputs[0]])
            .unwrap()
            .out_wire(0);
        let targ1 = &mut inputs[0];
        *targ1 = {
            let dfg = func.dfg_builder_endo(vec![(qb_t(), *targ1)]).unwrap();
            let inputs = dfg.input_wires();
            dfg.finish_with_outputs(inputs).unwrap()
        }
        .out_wire(0);
        func.finish_with_outputs(inputs).unwrap().handle().clone()
    }

    fn foo_tail_loop(module: &mut ModuleBuilder<Hugr>, t_num: usize) -> FuncID<true> {
        let foo_sig = Signature::new_endo(iter::repeat(qb_t()).take(t_num).collect::<Vec<_>>());
        let mut func = module.define_function("foo", foo_sig.clone()).unwrap();
        let theta = {
            let angle = ConstRotation::new(0.5).unwrap();
            func.add_load_value(angle)
        };
        let target_type = iter::repeat(qb_t()).take(t_num).collect::<Vec<_>>();
        let loop_inputs: Vec<(_, _)> = target_type
            .iter()
            .cloned()
            .zip(func.input_wires())
            .collect();
        let tail_loop = {
            let mut builder = func
                .tail_loop_builder([(rotation_type(), theta)], loop_inputs, type_row![])
                .unwrap();
            let mut inputs = builder.input_wires();
            let angle = inputs.next().unwrap();
            let qubit = inputs.next().unwrap();
            let rotated = builder
                .add_dataflow_op(TketOp::Rx, vec![qubit, angle])
                .unwrap()
                .out_wire(0);
            let sum_just_input = builder
                .make_sum(0, vec![rotation_type().into(), type_row![]], vec![angle])
                .unwrap();
            let outputs = [rotated].into_iter().chain(inputs);
            builder
                .finish_with_outputs(sum_just_input, outputs)
                .unwrap()
        };
        let outputs = tail_loop.outputs();
        func.finish_with_outputs(outputs).unwrap().handle().clone()
    }

    fn foo_conditional(module: &mut ModuleBuilder<Hugr>, t_num: usize) -> FuncID<true> {
        let foo_sig = Signature::new_endo(iter::repeat(qb_t()).take(t_num).collect::<Vec<_>>());
        let mut func = module.define_function("foo", foo_sig.clone()).unwrap();
        let theta = {
            let angle = ConstRotation::new(0.5).unwrap();
            func.add_load_value(angle)
        };
        let mut inputs = func.input_wires().collect::<Vec<_>>();
        inputs[0] = func
            .add_dataflow_op(TketOp::X, vec![inputs[0]])
            .unwrap()
            .out_wire(0);
        let sum_bool = func
            .make_sum(
                1,
                [type_row![], vec![rotation_type().into()].into()],
                vec![theta],
            )
            .unwrap();
        let mut cond_builder = func
            .conditional_builder(
                ([type_row![], vec![rotation_type().into()].into()], sum_bool),
                iter::repeat(qb_t()).take(t_num).zip(inputs),
                iter::repeat(qb_t()).take(t_num).collect::<Vec<_>>().into(),
            )
            .unwrap();
        let _case1 = {
            let case = cond_builder.case_builder(0).unwrap();
            let inputs = case.input_wires();
            let outputs = [].into_iter().chain(inputs);
            case.finish_with_outputs(outputs).unwrap()
        };
        let _case2 = {
            let mut case = cond_builder.case_builder(1).unwrap();
            let mut inputs = case.input_wires();
            let theta = inputs.next().unwrap();
            let mut q = inputs.next().unwrap();
            q = case
                .add_dataflow_op(TketOp::Rz, vec![q, theta])
                .unwrap()
                .out_wire(0);
            let outputs = [q].into_iter().chain(inputs);
            case.finish_with_outputs(outputs).unwrap()
        };
        let conditional = cond_builder.finish_sub_container().unwrap();
        let outputs = conditional.outputs();
        func.finish_with_outputs(outputs).unwrap().handle().clone()
    }

    #[rstest::rstest]
    #[case::dfg(1, 2, foo_dfg, "dfg", false)]
    #[case::dfg_dagger(1, 2, foo_dfg, "dfg", true)]
    #[case::tail_loop(1, 1, foo_tail_loop, "tail_loop", false)]
    #[case::conditional(1, 1, foo_conditional, "conditional", false)]
    #[case::conditional_dagger(1, 1, foo_conditional, "conditional", true)]
    fn test_modifier_resolver_optypes(
        #[case] t_num: usize,
        #[case] c_num: u64,
        #[case] foo: fn(&mut ModuleBuilder<Hugr>, usize) -> FuncID<true>,
        #[case] name: &str,
        #[case] dagger: bool,
    ) {
        use hugr::builder::Container;

        let mut module = ModuleBuilder::new();
        let call_sig = Signature::new_endo(
            [array_type(c_num, qb_t())]
                .into_iter()
                .chain(iter::repeat(qb_t()).take(t_num))
                .collect::<Vec<_>>(),
        );
        let main_sig = Signature::new(
            type_row![],
            vec![array_type(c_num, qb_t())]
                .into_iter()
                .chain(iter::repeat(qb_t()).take(t_num))
                .collect::<Vec<_>>(),
        );

        let dagger_op: ExtensionOp = {
            MODIFIER_EXTENSION
                .instantiate_extension_op(
                    &DAGGER_OP_ID,
                    [
                        iter::repeat(qb_t().into())
                            .take(t_num)
                            .collect::<Vec<_>>()
                            .into(),
                        vec![].into(),
                    ],
                )
                .unwrap()
        };

        let control_op: ExtensionOp = {
            MODIFIER_EXTENSION
                .instantiate_extension_op(
                    &CONTROL_OP_ID,
                    [
                        Term::BoundedNat(c_num),
                        iter::repeat(qb_t().into())
                            .take(t_num)
                            .collect::<Vec<_>>()
                            .into(),
                        vec![].into(),
                    ],
                )
                .unwrap()
        };

        let foo = foo(&mut module, t_num);

        let _main = {
            let mut func = module.define_function("main", main_sig).unwrap();
            let mut call = func.load_func(&foo, &[]).unwrap();
            if dagger {
                call = func
                    .add_dataflow_op(dagger_op, vec![call])
                    .unwrap()
                    .out_wire(0);
            }
            call = func
                .add_dataflow_op(control_op, vec![call])
                .unwrap()
                .out_wire(0);

            let mut controls = Vec::new();
            for _ in 0..c_num {
                controls.push(
                    func.add_dataflow_op(TketOp::QAlloc, vec![])
                        .unwrap()
                        .out_wire(0),
                );
            }

            let mut targ = Vec::new();
            for _ in 0..t_num {
                targ.push(
                    func.add_dataflow_op(TketOp::QAlloc, vec![])
                        .unwrap()
                        .out_wire(0),
                )
            }

            let control_arr = func.add_new_array(qb_t(), controls).unwrap();
            let fn_outs = func
                .add_dataflow_op(
                    CallIndirect {
                        signature: call_sig,
                    },
                    [call, control_arr].into_iter().chain(targ.into_iter()),
                )
                .unwrap()
                .outputs();

            func.finish_with_outputs(fn_outs).unwrap()
        };

        println!("Before modification:\n{}", module.hugr().mermaid_string());
        let mut h = module.finish_hugr().unwrap();
        h.validate().unwrap();
        println!("Before modification:\n{}", h.mermaid_string());

        let entrypoint = h.entrypoint().clone();
        resolve_modifier_with_entrypoints(&mut h, vec![entrypoint].into_iter()).unwrap();
        dead_code::DeadCodeElimPass::default()
            .with_entry_points(vec![_main.node()])
            .run(&mut h)
            .unwrap();
        println!("After modification\n{}", h.mermaid_string());
        h.validate().unwrap();
        {
            let f = File::create(Path::new(&format!("test_{}.mermaid", name))).unwrap();
            let mut writer = std::io::BufWriter::new(f);
            write!(writer, "{}", h.mermaid_string()).unwrap();
        }
        let env_format = EnvelopeFormat::PackageJson;
        let env_conf: EnvelopeConfig = EnvelopeConfig::new(env_format);
        let iter: Vec<Arc<Extension>> = vec![
            ROTATION_EXTENSION.to_owned(),
            TKET_EXTENSION.to_owned(),
            BOOL_EXTENSION.to_owned(),
        ];
        let regist: ExtensionRegistry = ExtensionRegistry::new(iter);
        let f = File::create(Path::new(&format!("test_{}.json", name))).unwrap();
        let writer = std::io::BufWriter::new(f);
        h.store_with_exts(writer, env_conf, &regist).unwrap();
        // println!(
        //     "hugr\n{}",
        //     h.store_str_with_exts(env_conf, &regist).unwrap()
        // );
    }
}
