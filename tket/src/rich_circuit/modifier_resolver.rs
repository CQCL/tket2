//! Try to delete modifier by applying the modifier to each component.
//!

use std::{
    collections::HashMap,
    iter,
    mem::{self, swap},
};

use derive_more::Error;
use hugr::{
    builder::{BuildError, Dataflow, FunctionBuilder, HugrBuilder},
    core::HugrNode,
    extension::{prelude::qb_t, simple_op::MakeExtensionOp},
    hugr::{hugrmut::HugrMut, patch::replace::ReplaceError},
    ops::{LoadFunction, OpType},
    types::{FuncTypeBase, PolyFuncType},
    Hugr, HugrView, Node, Wire,
};
use hugr_core::hugr::internal::PortgraphNodeMap;
use itertools::Itertools;
use petgraph::visit::{Topo, Walker};

use crate::{
    rich_circuit::{CombinedModifier, Modifier},
    Tk2Op,
};

/// Error that can occur when resolving modifiers.
#[derive(Debug, Error, derive_more::Display)]
pub enum ModifierError<N = Node> {
    /// The node is not a modifier
    #[display("Node to modify {_0} expected to be a modifier but actually {_1}")]
    NotModifier(N, OpType),
    /// The node cannot be modified.
    #[display("Modification by {_0:?} is not defined for the node {_1}")]
    Unimplemented(Modifier, OpType),
    /// No caller of this modified function exists.
    #[display("No caller of the modified function exists for node {_0}")]
    NoCaller(N),
    /// No target of this modifer exists.
    #[display("No caller of the modified function exists for node {_0}")]
    NoTarget(N),
    /// Not the first modifier in a chain.
    #[display("Node {_0} is not the first modifier in a chain. It is called by {_0}")]
    NotInitialModifier(N, OpType),
    /// Modifier applied to a node that cannot be modified.
    #[display("Modifier {_0} cannot be applied to the node {_1}")]
    ModifierNotApplicable(N, OpType),
}

/// Possible errors that can occur during the modifier resolution process.
#[derive(Debug, derive_more::Display)]
pub enum ModifierResolverErrors<N = Node> {
    /// Cannot modify the node.
    ModifierError(ModifierError<N>),
    /// Error during the replacement process.
    ReplaceError(ReplaceError),
    /// Error during the DFG build process.
    BuildError(BuildError),
    /// Unreachable error variant.
    Unreachable(String),
    // /// Validation error.
    // ValidationError(ValidationError<N>)
}
impl<N> From<ModifierError<N>> for ModifierResolverErrors<N> {
    fn from(err: ModifierError<N>) -> Self {
        ModifierResolverErrors::ModifierError(err)
    }
}
impl<N> From<ReplaceError> for ModifierResolverErrors<N> {
    fn from(err: ReplaceError) -> Self {
        ModifierResolverErrors::ReplaceError(err)
    }
}
impl<N> From<BuildError> for ModifierResolverErrors<N> {
    fn from(err: BuildError) -> Self {
        ModifierResolverErrors::BuildError(err)
    }
}

/// A container for modifier resolving.
pub struct ModifierResolver<N = Node> {
    node: N,
    // TODO:
    // Should keep track of the collection of modifiers that are applied to the same function.
    // This will prevent the duplicated generation of Controlled-functions.
}

impl<N: HugrNode> ModifierResolver<N> {
    fn verify(&self, h: &impl HugrView<Node = N>) -> Result<(), ModifierError<N>> {
        // Check if the node is a modifier, modifying an operation.
        let optype = h.get_optype(self.node);
        if Modifier::from_optype(optype).is_none() {
            return Err(ModifierError::NotModifier(self.node, optype.clone()));
        }
        // Check if this is the first modifier in a chain.
        if let Some((caller, _)) = h.linked_inputs(self.node, 0).exactly_one().ok() {
            let optype = h.get_optype(caller);
            if Modifier::from_optype(optype).is_some() {
                return Err(ModifierError::NotInitialModifier(caller, optype.clone()));
            }
        } else {
            return Err(ModifierError::NoCaller(self.node));
        }
        Ok(())
    }

    fn try_rewrite(self, h: &mut impl HugrMut<Node = N>) -> Result<(), ModifierResolverErrors<N>> {
        // Verify that the rewrite can be applied.
        self.verify(h)?;

        // the ports that takes inputs from the modified function.
        let modified_fn_loader: Vec<(_, Vec<_>)> = h
            .node_outputs(self.node)
            .map(|p| (p, h.linked_inputs(self.node, p).collect()))
            .collect();

        // The final target of modifiers to apply.
        let mut targ = self.node;
        // Collection of modifiers to apply.
        let mut modifiers: Vec<Modifier> = Vec::new();
        let mut modifier_and_targ: Vec<N> = Vec::new();
        loop {
            modifier_and_targ.push(targ);
            let optype = h.get_optype(targ);
            match Modifier::from_optype(optype) {
                Some(modifier) => modifiers.push(modifier),
                // Found the target
                None => break,
            }
            targ = h
                .all_linked_outputs(targ)
                .exactly_one()
                .ok()
                .map(|(n, _)| n)
                .ok_or(ModifierError::NoTarget(self.node))?;
        }
        println!("Evaluating {}, Current target: {}", self.node, targ);

        // Calculate the accumulated modifier.
        let combined_modifier: CombinedModifier = modifiers.into();

        let optype = h.get_optype(targ).clone();
        // The function to apply the modifier to.
        let (func, load) = match optype {
            OpType::Input(_) => return Err(ModifierError::NoTarget(self.node).into()),
            // If the target is a function, we need to create a new dataflow block of it.
            OpType::LoadFunction(load) => {
                let (fn_node, _) = h.all_linked_outputs(targ).exactly_one().map_err(|_| {
                    ModifierResolverErrors::Unreachable(
                        "Loading multiple or no function.".to_string(),
                    )
                })?;
                let fn_optype = h.get_optype(fn_node);
                let OpType::FuncDefn(_) = fn_optype else {
                    return Err({
                        println!("error happened here!");
                        ModifierError::ModifierNotApplicable(self.node, fn_optype.clone()).into()
                    });
                };
                // Note: I was thinking of getting a function that is uniquely referenced
                // so that we can apply the modifier to it while rewriting it.
                // But it seems like that HugrMut does not have a interface to get the information
                // of nodes with mutable reference.
                //
                // TODO: We want some machinery to prevent generating a lot of controlled-U
                // for the same function U.
                //
                // if h.num_outputs(fn_node) != 1 {
                //     fn_node = todo!("generate another function by cloning the function body.")
                // }
                (fn_node, load)
            }
            _ => {
                // TODO: Handle modifiers for other op types.
                // For example, tail loops, or conditionals.
                println!("error happened here 2!");
                return Err(ModifierError::ModifierNotApplicable(self.node, optype.clone()).into());
            }
        };

        // generate modified function
        let modified_sig = combined_modifier.modify_signature(&load.func_sig);
        let modified_fn = combined_modifier.modify_fn(h, func, modified_sig.clone())?;
        let modified_fn = h
            .insert_hugr(h.module_root(), modified_fn)
            .inserted_entrypoint;

        // insert the new LoadFunction node to load the modified function
        let load = LoadFunction::try_new(modified_sig, load.type_args).map_err(BuildError::from)?;
        let new_load = h.add_node_after(self.node, load);
        h.connect(modified_fn, 0, new_load, 0);

        // delete the modifiers, and change the function to be loaded
        for mod_or_targ in modifier_and_targ {
            h.remove_node(mod_or_targ);
        }
        for (out_port, inputs) in modified_fn_loader {
            // Connect the inputs to the modified function.
            for (recv, recv_port) in inputs {
                h.connect(new_load, out_port, recv, recv_port);
            }
        }

        Ok(())
    }
}

/// Resolve modifiers in a circuit by applying them to each entry point.
pub fn resolve_modifier_with_entrypoints(
    h: &mut impl HugrMut<Node = Node>,
    entry_points: impl Iterator<Item = Node>,
) -> Result<(), ModifierResolverErrors<Node>> {
    use ModifierResolverErrors::*;

    for entry_point in entry_points {
        let descendants = h.descendants(entry_point).collect::<Vec<_>>();
        for node in descendants {
            let resolver = ModifierResolver { node };
            // Verify the resolver can be applied.
            match resolver.try_rewrite(h) {
                Ok(_) => (),
                // If not resolvable, just skip.
                Err(ModifierError(e)) => {
                    println!("Not modifiable {}: Reason: {}", node, e);
                    continue;
                }
                // Others will be the actual error.
                e => return e,
            }
        }
    }
    Ok(())
}

impl CombinedModifier {
    /// Takes a signature and modifies it according to the combined modifier.
    pub fn modify_signature(&self, signature: &PolyFuncType) -> PolyFuncType {
        let mut signature = signature.clone();
        if !self.dagger && self.control == 0 {
            return signature;
        }
        let FuncTypeBase { input, output } = signature.body_mut();

        if self.dagger {
            swap(input, output);
        }

        let n = self.control;
        input.to_mut().splice(0..0, iter::repeat(qb_t()).take(n));
        output.to_mut().splice(0..0, iter::repeat(qb_t()).take(n));

        signature
    }

    /// Generates a new function modified by the combined modifier.
    pub fn modify_fn<N: HugrNode>(
        &self,
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
        // Control qubits. These are passed around each operation in the function, so it's mutable.
        let mut controls: Vec<_> = new_fn.input_wires().take(self.control).collect();
        // let control_output: Vec<_> = new_fn.output_wires().take(self.control).collect();

        // Old function body in topological order.
        let (region_graph, node_map) = h.region_portgraph(func);
        let topo = Topo::new(&region_graph);

        // A map that corresponds the node in the old function body to the node in the new one.
        let mut corresp_map: HashMap<Wire<N>, Wire> = HashMap::new();

        // Iterate through the old function body in topological order.
        for old_n_id in topo.iter(&region_graph) {
            let old_n = node_map.from_portgraph(old_n_id);
            self.modify_op(h, old_n, &mut controls, &mut corresp_map, &mut new_fn)?;
        }

        let new_fn = new_fn
            .finish_hugr()
            .map_err(|e| ModifierResolverErrors::BuildError(e.into()))?;

        Ok(new_fn)
    }

    // TODO: We take arbitral topological order of the circuit so that we can plug the control qubits
    // and pass around them in that order. However, this is not ideal, as it may produce an inefficient order.
    fn modify_op<N: HugrNode>(
        &self,
        h: &impl HugrMut<Node = N>,
        n: N,
        controls: &mut Vec<Wire>,
        corresp_map: &mut HashMap<Wire<N>, Wire>,
        new_fn: &mut FunctionBuilder<Hugr>,
    ) -> Result<(), ModifierResolverErrors<N>> {
        let optype = h.get_optype(n);
        match optype {
            // Connect the inputs
            OpType::Input(_) => {
                for (old, new) in iter::zip(
                    h.node_outputs(n).map(|p| Wire::new(n, p)),
                    new_fn.input_wires().skip(self.control),
                ) {
                    corresp_map.insert(old, new);
                }
            }
            // If it's an output, connect the wires.
            OpType::Output(_) => {
                // control wires are the head of the output wires.
                let mut outputs = Vec::new();
                println!("length of control: {}", controls.len());
                mem::swap(controls, &mut outputs);

                // register the corresponding output wires to the old ones.
                for (output, port) in h.all_linked_outputs(n) {
                    // old one
                    let old = Wire::new(output, port);
                    let new = corresp_map
                        .get(&old)
                        .ok_or(ModifierResolverErrors::Unreachable(
                            "No correspondence for the output wire".to_string(),
                        ))?;
                    outputs.push(new.clone());
                }
                new_fn.set_outputs(outputs)?;
            }
            OpType::FuncDefn(_) | OpType::FuncDecl(_) => {
                return Err(ModifierResolverErrors::Unreachable(
                    "Function definition or declaration appears inside modified function"
                        .to_string(),
                ));
            }
            OpType::ExtensionOp(_) => {
                self.modify_extension_op(h, n, controls, corresp_map, optype, new_fn)?;
            }
            _ => todo!(),
        }
        Ok(())
    }

    fn modify_extension_op<N: HugrNode>(
        &self,
        h: &impl HugrMut<Node = N>,
        n: N,
        controls: &mut Vec<Wire>,
        corresp_map: &mut HashMap<Wire<N>, Wire>,
        optype: &OpType,
        new_fn: &mut FunctionBuilder<Hugr>,
    ) -> Result<(), ModifierResolverErrors<N>> {
        use Tk2Op::*;

        if controls.len() != self.control {
            return Err(ModifierResolverErrors::Unreachable(
                "Control qubits are not set correctly.".to_string(),
            ));
        }

        // If Tk2Op, return the modified operation.
        println!("I'm modifying Node: {:?}", n);
        if let Some(op) = Tk2Op::from_optype(optype) {
            if self.control != 0 || self.dagger {
                if !op.is_quantum() {
                    return Err(ModifierError::ModifierNotApplicable(n, op.into()).into());
                }
            }
            match op {
                H => todo!(),
                X => {
                    let old_inputs = h.all_linked_outputs(n).map(|(n, p)| Wire::new(n, p));
                    // There might be a port of StateOrder.
                    let old_outputs = h.node_outputs(n).map(|p| Wire::new(n, p));

                    let mut new_inputs = Vec::new();
                    mem::swap(controls, &mut new_inputs);
                    for old_input in old_inputs {
                        let new_in = corresp_map.get(&old_input).ok_or(
                            ModifierResolverErrors::Unreachable(
                                "No correspondence for the input wire".to_string(),
                            ),
                        )?;
                        new_inputs.push(new_in.clone());
                    }

                    let new_op = match self.control {
                        0 => X,
                        1 => CX,
                        2 => Toffoli,
                        _ => todo!(),
                    };
                    let new_op_out = new_fn.add_dataflow_op(new_op, new_inputs)?;
                    for i in 0..self.control {
                        controls.push(new_op_out.out_wire(i));
                    }
                    for (old_out, new_out) in
                        iter::zip(old_outputs, new_op_out.outputs().skip(self.control))
                    {
                        corresp_map.insert(old_out, new_out).map_or(Ok(()), |_| {
                            Err(ModifierResolverErrors::Unreachable(
                                "Output wire already registered".to_string(),
                            ))
                        })?
                    }
                }
                Y => todo!(),
                Z => todo!(),
                Rx => todo!(),
                Ry => todo!(),
                Rz => todo!(),
                CX => todo!(),
                CY => todo!(),
                CZ => todo!(),
                CRz => todo!(),
                Toffoli => todo!(),
                T => todo!(),
                Tdg => todo!(),
                S => todo!(),
                Sdg => todo!(),
                Measure => todo!(),
                MeasureFree => todo!(),
                QAlloc => todo!(),
                TryQAlloc => todo!(),
                QFree => todo!(),
                Reset => todo!(),
                V => todo!(),
                Vdg => todo!(),
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use hugr::{
        builder::{Dataflow, DataflowSubContainer, HugrBuilder, ModuleBuilder},
        extension::prelude::qb_t,
        ops::CallIndirect,
        types::{Signature, Term},
    };

    use crate::rich_circuit::modifier_resolver::*;
    use crate::rich_circuit::*;

    #[test]
    fn test_modification() {
        let mut module = ModuleBuilder::new();
        let sig = Signature::new(vec![qb_t()], vec![qb_t()]);
        let fn_sig = Signature::new(vec![qb_t(), qb_t()], vec![qb_t(), qb_t()]);

        let control_op = MODIFIER_EXTENSION
            .instantiate_extension_op(
                &CONTROL_OP_ID,
                [
                    Term::new_list([qb_t().into()]),
                    Term::new_list([qb_t().into()]),
                ],
            )
            .unwrap();
        let dagger_op = MODIFIER_EXTENSION
            .instantiate_extension_op(
                &DAGGER_OP_ID,
                [
                    Term::new_list([qb_t().into(), qb_t().into()]),
                    Term::new_list([qb_t().into(), qb_t().into()]),
                ],
            )
            .unwrap();

        let foo = {
            let mut func = module.define_function("foo", sig.clone()).unwrap();
            let [in1] = func.input_wires_arr();
            let xgate = func.add_dataflow_op(Tk2Op::X, vec![in1]).unwrap();
            func.finish_with_outputs(xgate.outputs()).unwrap()
        };

        let _main = {
            let mut func = module.define_function("main", fn_sig.clone()).unwrap();
            let [in1, in2] = func.input_wires_arr();
            let loaded = func.load_func(foo.handle(), &[]).unwrap();
            let controlled = func
                .add_dataflow_op(control_op, vec![loaded])
                .unwrap()
                .out_wire(0);
            let daggered = func
                .add_dataflow_op(dagger_op, vec![controlled])
                .unwrap()
                .out_wire(0);
            let [out1, out2] = func
                .add_dataflow_op(CallIndirect { signature: fn_sig }, vec![daggered, in1, in2])
                .unwrap()
                .outputs_arr();
            func.finish_with_outputs(vec![out1, out2]).unwrap()
        };

        let mut h = module.finish_hugr().unwrap();

        let entrypoint = h.entrypoint().clone();
        resolve_modifier_with_entrypoints(&mut h, vec![entrypoint].into_iter()).unwrap();
        println!("{}", h.mermaid_string());
    }
}
