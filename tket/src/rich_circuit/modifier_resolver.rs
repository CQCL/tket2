//! Try to delete modifier by applying the modifier to each component.
//!

use std::{cmp::min, collections::HashMap, iter, mem};

use derive_more::Error;
use hugr::{
    builder::{BuildError, ConditionalBuilder, Container, Dataflow, FunctionBuilder, HugrBuilder},
    core::HugrNode,
    extension::{prelude::qb_t, simple_op::MakeExtensionOp},
    hugr::{hugrmut::HugrMut, patch::replace::ReplaceError},
    ops::{Conditional, Const, DataflowOpTrait, DataflowParent, LoadFunction, OpType},
    std_extensions::arithmetic::float_ops::FloatOps,
    types::{FuncTypeBase, PolyFuncType},
    Hugr, HugrView, IncomingPort, Node, OutgoingPort, Port, PortIndex, Wire,
};
use hugr_core::hugr::internal::PortgraphNodeMap;
use itertools::{Either, Itertools};
use petgraph::{
    visit::{Topo, Walker},
    Direction::{Incoming, Outgoing},
};

use crate::{extension::rotation::RotationOp, rich_circuit::Modifier, Tk2Op};

/// An accumulated modifier that combines control, dagger, and power modifiers.
pub struct CombinedModifier {
    control: usize,
    dagger: bool,
    #[allow(dead_code)]
    power: usize,
}

impl Default for CombinedModifier {
    fn default() -> Self {
        CombinedModifier {
            control: 0,
            dagger: false,
            power: 1,
        }
    }
}

impl From<Vec<Modifier>> for CombinedModifier {
    fn from(modifiers: Vec<Modifier>) -> Self {
        let mut control = 0;
        let mut dagger = false;
        let mut power = 0;

        for modifier in modifiers {
            match modifier {
                Modifier::ControlModifier => control += 1,
                Modifier::DaggerModifier => dagger = true,
                Modifier::PowerModifier => power += 1,
            }
        }

        CombinedModifier {
            control,
            dagger,
            power,
        }
    }
}

/// A wire of both direction.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DirWire<N = Node>(N, Port);

impl<N: HugrNode> std::fmt::Display for DirWire<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dir = match self.1.as_directed() {
            Either::Left(_) => "In",
            Either::Right(_) => "Out",
        };
        write!(f, "DirWire({}, {}({}))", self.0, dir, self.1.index())
    }
}

impl<N> DirWire<N> {
    pub fn new(node: N, port: Port) -> Self {
        DirWire(node, port)
    }

    pub fn reverse(self) -> Self {
        let index = self.1.index();
        let port = match self.1.as_directed() {
            Either::Left(_in) => OutgoingPort::from(index).into(),
            Either::Right(_out) => IncomingPort::from(index).into(),
        };
        DirWire::new(self.0, port)
    }
}
// impl<N: HugrNode> DirWire<N> {
//     pub fn as_wire(self) -> Result<Wire<N>, HugrError> {
//         let out_port = self.1.as_outgoing()?;
//         Ok(Wire::new(self.0, out_port))
//     }
// }

impl<N: HugrNode> From<Wire<N>> for DirWire<N> {
    fn from(wire: Wire<N>) -> Self {
        DirWire(wire.node(), wire.source().into())
    }
}
impl<N: HugrNode> From<(N, OutgoingPort)> for DirWire<N> {
    fn from((node, port): (N, OutgoingPort)) -> Self {
        DirWire(node, port.into())
    }
}
impl<N: HugrNode> From<(N, IncomingPort)> for DirWire<N> {
    fn from((node, port): (N, IncomingPort)) -> Self {
        DirWire(node, port.into())
    }
}

trait PortExt {
    fn shift(self, offset: usize) -> Self;
}
impl PortExt for Port {
    fn shift(self, offset: usize) -> Self {
        Port::new(self.direction(), self.index() + offset)
    }
}
impl PortExt for IncomingPort {
    fn shift(self, offset: usize) -> Self {
        IncomingPort::from(self.index() + offset)
    }
}
impl PortExt for OutgoingPort {
    fn shift(self, offset: usize) -> Self {
        OutgoingPort::from(self.index() + offset)
    }
}

/// A container for modifier resolving.
pub struct ModifierResolver<N = Node> {
    node: Option<N>,
    modifiers: CombinedModifier,
    corresp_map: HashMap<DirWire<N>, DirWire>,
    controls: Vec<Wire>,
    // TODO:
    // Should keep track of the collection of modifiers that are applied to the same function.
    // This will prevent the duplicated generation of Controlled-functions.
}

impl<N> ModifierResolver<N> {
    /// Create a new modifier resolver.
    pub fn new() -> Self {
        ModifierResolver {
            node: None,
            modifiers: CombinedModifier::default(),
            corresp_map: HashMap::new(),
            controls: Vec::new(),
        }
    }
}

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

impl<N: HugrNode> ModifierResolver<N> {
    fn node(&self) -> N {
        self.node
            .expect("ModifierResolver is not initialized with a node")
    }
    fn controls(&mut self) -> &mut Vec<Wire> {
        &mut self.controls
    }
    fn corresp_map(&mut self) -> &mut HashMap<DirWire<N>, DirWire> {
        &mut self.corresp_map
    }

    /// normal insert method.
    fn map_insert(
        &mut self,
        old: DirWire<N>,
        new: DirWire, // new: impl Into<Either<IncomingPort, OutgoingPort>>,
    ) -> Result<(), ModifierResolverErrors<N>> {
        // TODO: don't clone
        println!("Insert {} -> {}", old, new);
        self.corresp_map()
            .insert(old.clone(), new)
            .map_or(Ok(()), |former| {
                Err(ModifierResolverErrors::Unreachable(format!(
                    "Output wire already registered. Former {}, Latter {}",
                    former, old
                )))
            })
    }
    fn map_get(&self, key: &DirWire<N>) -> Result<&DirWire, ModifierResolverErrors<N>> {
        self.corresp_map
            .get(key)
            .ok_or(ModifierResolverErrors::Unreachable(format!(
                "No correspondence for the input wire. Input: {}",
                key
            )))
    }

    /// Add a node to the builder, plugging the control qubits to the first n-inputs and outputs.
    fn add_node_control(&mut self, new_fn: &mut impl Dataflow, op: impl Into<OpType>) -> Node {
        let node = new_fn.add_child_node(op);
        for i in 0..self.modifiers.control {
            let in_node = self.controls[i].node();
            // I assume that the control qubtits are always the first n-wires.
            new_fn.hugr_mut().connect(in_node, i, node, i);
            self.controls[i] = Wire::new(node, i);
        }
        node
    }

    /// This function adds a node to the builder, swapping the given wires
    /// if `self.modifiers.dagger` is true.
    /// The rev_if_dag is supposed to be a partial bijection between ports.
    /// This function also adds control qubits as the input wires.
    fn add_node_reversible(
        &mut self,
        new_fn: &mut impl Dataflow,
        op: impl Into<OpType>,
        h: &impl HugrMut<Node = N>,
        old_n: N,
        rev_if_dag: Vec<(Port, Port)>,
    ) -> Result<Node, ModifierResolverErrors<N>> {
        let node = self.add_node_control(new_fn, op);
        for port in h.all_node_ports(old_n) {
            let dir_wire = DirWire::new(old_n, port);
            let new_port = if self.modifiers.dagger {
                rev_if_dag
                    .iter()
                    .find_map(|(from, to)| {
                        if *from == port {
                            Some(to.clone())
                        } else {
                            None
                        }
                    })
                    .unwrap_or_else(|| port)
            } else {
                port
            }
            .shift(self.modifiers.control);
            self.map_insert(dir_wire, DirWire(node, new_port))?;
        }
        // for i in 0..self.modifiers.control {
        //     let in_node = self.controls[i].node();
        //     // I assume that the control qubtits are always the first n-wires.
        //     new_fn.hugr_mut().connect(in_node, i, node, i);
        //     self.controls[i] = Wire::new(node, i);
        // }
        Ok(node)
    }

    /// This function adds a node to the builder, that swaps the inputs and outputs
    /// of n-th port of the node.
    fn add_node_reversible_simple(
        &mut self,
        new_fn: &mut impl Dataflow,
        op: impl Into<OpType>,
        h: &impl HugrMut<Node = N>,
        old_n: N,
        rev_if_dag: Vec<usize>,
    ) -> Result<Node, ModifierResolverErrors<N>> {
        let v = rev_if_dag
            .into_iter()
            .flat_map(|i| {
                vec![
                    (Port::new(Outgoing.into(), i), Port::new(Incoming.into(), i)),
                    (Port::new(Incoming.into(), i), Port::new(Outgoing.into(), i)),
                ]
            })
            .collect::<Vec<_>>();
        self.add_node_reversible(new_fn, op, h, old_n, v)
    }

    /// This function adds a node to the builder, that does not affected by the modifiers.
    fn add_node_no_modification(
        &mut self,
        new_fn: &mut impl Dataflow,
        op: impl Into<OpType>,
        h: &impl HugrMut<Node = N>,
        old_n: N,
    ) -> Result<Node, ModifierResolverErrors<N>> {
        let node = new_fn.add_child_node(op);
        for port in h.all_node_ports(old_n) {
            let dir_wire = DirWire::new(old_n, port);
            self.map_insert(dir_wire, DirWire(node, port))?;
        }
        Ok(node)
    }

    /// connects all the wires in the builder.
    pub fn connect_all(
        &mut self,
        h: &impl HugrView<Node = N>,
        new_dfg: &mut impl Dataflow,
        parent: N,
    ) -> Result<(), ModifierResolverErrors<N>> {
        for p in self.corresp_map().iter() {
            println!("  Wire mapping: {} -> {}", p.0, p.1);
        }
        for out_node in h.children(parent) {
            for out_port in h.node_outputs(out_node) {
                for (in_node, in_port) in h.linked_inputs(out_node, out_port) {
                    let a = self.map_get(&(in_node, in_port).into())?;
                    let b = self.map_get(&(out_node, out_port).into())?;
                    let (n_o, p_o, n_i, p_i) = match (a.1.as_directed(), b.1.as_directed()) {
                        (Either::Right(p_o), Either::Left(p_i)) => (a.0, p_o, b.0, p_i),
                        (Either::Left(p_i), Either::Right(p_o)) => (b.0, p_o, a.0, p_i),
                        _ => {
                            return Err(ModifierResolverErrors::Unreachable(
                                "Cannot connect the wires with the same direction.".to_string(),
                            ))
                        }
                    };
                    new_dfg.hugr_mut().connect(n_o, p_o, n_i, p_i)
                }
            }
        }
        Ok(())
    }
}

impl<N: HugrNode> ModifierResolver<N> {
    fn verify(&self, h: &impl HugrView<Node = N>) -> Result<(), ModifierError<N>> {
        // Check if the node is a modifier, modifying an operation.
        let optype = h.get_optype(self.node());
        if Modifier::from_optype(optype).is_none() {
            return Err(ModifierError::NotModifier(self.node(), optype.clone()));
        }
        // Check if this is the first modifier in a chain.
        if let Some((caller, _)) = h.linked_inputs(self.node(), 0).exactly_one().ok() {
            let optype = h.get_optype(caller);
            if Modifier::from_optype(optype).is_some() {
                return Err(ModifierError::NotInitialModifier(caller, optype.clone()));
            }
        } else {
            return Err(ModifierError::NoCaller(self.node()));
        }
        Ok(())
    }

    fn try_rewrite(
        &mut self,
        h: &mut impl HugrMut<Node = N>,
    ) -> Result<(), ModifierResolverErrors<N>> {
        // Verify that the rewrite can be applied.
        self.verify(h)?;

        // the ports that takes inputs from the modified function.
        let modified_fn_loader: Vec<(_, Vec<_>)> = h
            .node_outputs(self.node())
            .map(|p| (p, h.linked_inputs(self.node(), p).collect()))
            .collect();

        // The final target of modifiers to apply.
        let mut targ = self.node();
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
                .ok_or(ModifierError::NoTarget(self.node()))?;
        }
        println!("Evaluating {}, Current target: {}", self.node(), targ);

        // Calculate the accumulated modifier.
        self.modifiers = modifiers.into();

        let optype = h.get_optype(targ).clone();
        // The function to apply the modifier to.
        let (func, load) = match optype {
            OpType::Input(_) => return Err(ModifierError::NoTarget(self.node()).into()),
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
                        ModifierError::ModifierNotApplicable(self.node(), fn_optype.clone()).into()
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
                return Err(
                    ModifierError::ModifierNotApplicable(self.node(), optype.clone()).into(),
                );
            }
        };

        // generate modified function
        let mut modified_sig = load.func_sig.clone();
        self.modify_signature(&mut modified_sig);
        println!(
            "Modifying function {} with signature {:?}",
            func, modified_sig
        );
        let mut corresp_map = HashMap::new();
        mem::swap(self.corresp_map(), &mut corresp_map);
        let modified_fn = self.modify_fn(h, func, modified_sig.clone())?;
        mem::swap(self.corresp_map(), &mut corresp_map);
        let modified_fn = h
            .insert_hugr(h.module_root(), modified_fn)
            .inserted_entrypoint;

        // insert the new LoadFunction node to load the modified function
        let load = LoadFunction::try_new(modified_sig, load.type_args).map_err(BuildError::from)?;
        let new_load = h.add_node_after(self.node(), load);
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

    /// Takes a signature and modifies it according to the combined modifier.
    pub fn modify_signature(&mut self, signature: &mut PolyFuncType) {
        if !self.modifiers.dagger && self.modifiers.control == 0 {
            return;
        }
        let FuncTypeBase { input, output } = signature.body_mut();

        // Even if dagger is applied, we do not change the signature.
        // if self.modifiers.dagger { }

        let n = self.modifiers.control;
        input.to_mut().splice(0..0, iter::repeat(qb_t()).take(n));
        output.to_mut().splice(0..0, iter::repeat(qb_t()).take(n));
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
        // Control qubits. These are passed around each operation in the function, so it's mutable.
        // let control_output: Vec<_> = new_fn.output_wires().take(self.modifiers.control).collect();

        self.modify_dfg_elements(h, func, &mut new_fn)?;

        println!(
            "modified (before connect):\n{}",
            new_fn.hugr_mut().mermaid_string()
        );
        // Add the edges
        self.connect_all(h, &mut new_fn, func)?;
        println!(
            "modified (before check):\n{}",
            new_fn.hugr_mut().mermaid_string()
        );
        let new_fn = new_fn
            .finish_hugr()
            .map_err(|e| ModifierResolverErrors::BuildError(e.into()))?;

        Ok(new_fn)
    }

    /// Modifies the dataflow graph elements of the function.
    /// We use the topological order of the circuit.
    /// TODO
    /// I need to reverse the order of the circuits when dagger is applied.
    pub fn modify_dfg_elements(
        &mut self,
        h: &impl HugrMut<Node = N>,
        n: N,
        new_dfg: &mut impl Dataflow,
    ) -> Result<(), ModifierResolverErrors<N>> {
        let mut controls: Vec<_> = new_dfg.input_wires().take(self.modifiers.control).collect();
        let _ = mem::swap(self.controls(), &mut controls);
        let (region_graph, node_map) = h.region_portgraph(n);
        let mut topo: Vec<_> = Topo::new(&region_graph).iter(&region_graph).collect();

        if self.modifiers.dagger {
            // Reverse the topological order if dagger is applied.
            topo.reverse();
        }

        for old_n_id in topo {
            let old_n = node_map.from_portgraph(old_n_id);
            self.modify_op(h, old_n, new_dfg)?;
        }

        // swap IO if dagger is applied
        if self.modifiers.dagger {
            let [in_node, out_node] = h.get_io(n).unwrap();
            let opty = h.get_optype(n);
            let FuncTypeBase { input, output } = match opty {
                OpType::DFG(dfg) => dfg.signature().into_owned(),
                OpType::Case(case) => case.inner_signature().into_owned(),
                OpType::FuncDefn(fndefn) => fndefn.signature().body().clone(),
                OpType::TailLoop(tail_loop) => tail_loop.signature().into_owned(),
                _ => {
                    return Err(ModifierResolverErrors::Unreachable(format!(
                        "Cannot modify the IO of the node with OpType: {}",
                        opty
                    )));
                }
            };
            for i in 0..min(input.len(), output.len()) {
                if input[i] == qb_t() && output[i] == qb_t() {
                    let old_in = (in_node, OutgoingPort::from(i)).into();
                    let old_out = (out_node, IncomingPort::from(i)).into();
                    let in_val = self.corresp_map().remove(&old_in);
                    let out_val = self.corresp_map().remove(&old_out);
                    match (in_val, out_val) {
                        (Some(in_val), Some(out_val)) => {
                            self.corresp_map().insert(old_in, out_val);
                            self.corresp_map().insert(old_out, in_val);
                        }
                        _ => {
                            return Err(ModifierResolverErrors::Unreachable(format!(
                                "Failed to find {}-th IO wires for swap",
                                i
                            )));
                        }
                    }
                }
            }
        }

        mem::swap(self.controls(), &mut controls);
        let out_node = new_dfg.io()[1];
        for (index, wire) in controls.iter().enumerate() {
            new_dfg
                .hugr_mut()
                .connect(wire.node(), wire.source(), out_node, index);
        }

        Ok(())
    }

    // TODO: We take arbitral topological order of the circuit so that we can plug the control qubits
    // and pass around them in that order. However, this is not ideal, as it may produce an inefficient order.
    fn modify_op(
        &mut self,
        h: &impl HugrMut<Node = N>,
        n: N,
        new_fn: &mut impl Dataflow,
    ) -> Result<(), ModifierResolverErrors<N>> {
        let optype = h.get_optype(n);
        println!("Modifying Op: {}, OpType: {:?}", n, optype);
        match optype {
            OpType::Input(_) => {
                for (old, new) in iter::zip(
                    h.node_outputs(n).map(|p| Wire::new(n, p)),
                    new_fn.input_wires().skip(self.modifiers.control),
                ) {
                    println!("Input Node: old: {}, new: {}", old, new);
                    self.map_insert(old.into(), new.into())?;
                }
            }
            OpType::Output(_) => {
                // register the corresponding output wires to the old ones.
                let new_out_node = new_fn.io()[1];
                for (num, old_port) in h.node_inputs(n).enumerate() {
                    self.map_insert(
                        (n, old_port).into(),
                        (
                            new_out_node,
                            IncomingPort::from(self.modifiers.control + num),
                        )
                            .into(),
                    )?
                }
            }
            OpType::ExtensionOp(_) => {
                self.modify_extension_op(h, n, optype, new_fn)?;
            }
            OpType::DFG(_) | OpType::TailLoop(_) | OpType::Conditional(_) => {
                // let (region_graph, node_map) = h.region_portgraph(n);
                // let topo = Topo::new(&region_graph);
                todo!()
                // let mut new_dfg = if matches!(optype, OpType::DFG(_)) {
                //     DFGBuilder::new(signature)

                //     optype};
                // for old_n_id in topo.iter(&region_graph) {
                //     let old_n = node_map.from_portgraph(old_n_id);
                //     self.modify_op(h, old_n, &mut controls, &mut corresp_map, &mut new_dfg)?;
                // }
            }
            OpType::Conditional(conditional) => {
                let Conditional {
                    sum_rows,
                    mut other_inputs,
                    mut outputs,
                } = conditional.clone();
                let c = self.modifiers.control;
                other_inputs
                    .to_mut()
                    .splice(0..0, iter::repeat(qb_t()).take(c));
                outputs.to_mut().splice(0..0, iter::repeat(qb_t()).take(c));
                let _new_conditional = ConditionalBuilder::new(sum_rows, other_inputs, outputs)?;
                // TODO
                todo!()
            }
            OpType::Call(_) => todo!(),
            OpType::CallIndirect(_) => todo!(),
            OpType::LoadFunction(_) => todo!(),

            OpType::Case(_) => todo!(),
            OpType::Const(constant) => {
                self.modify_constant(n, constant.clone(), new_fn)?;
            }
            OpType::LoadConstant(_) | OpType::OpaqueOp(_) => {
                self.modify_dataflow_op(h, n, optype, new_fn)?;
            }
            OpType::Tag(_) => todo!(),
            // Not supported
            OpType::FuncDefn(_) | OpType::FuncDecl(_) | OpType::Module(_) => {
                return Err(ModifierResolverErrors::Unreachable(format!(
                    "Invalid node found inside modified function (OpType = {})",
                    optype.clone()
                )))
            }
            OpType::AliasDecl(_) | OpType::AliasDefn(_) | OpType::ExitBlock(_) | OpType::CFG(_) => {
                return Err(ModifierError::ModifierNotApplicable(n, optype.clone()).into());
            }
            OpType::DataflowBlock(_) => {
                return Err(ModifierResolverErrors::Unreachable(
                    "DataflowBlock cannot be modified.".to_string(),
                ))
            }
            _ => {
                return Err(ModifierResolverErrors::Unreachable(
                    "Unknown operation type found in the modifier resolver.".to_string(),
                ))
            }
        }
        Ok(())
    }

    fn modify_constant(
        &mut self,
        n: N,
        constant: Const,
        new_fn: &mut impl Dataflow,
    ) -> Result<(), ModifierResolverErrors<N>> {
        let output = new_fn.add_child_node(constant);
        self.map_insert(Wire::new(n, 0).into(), Wire::new(output, 0).into())
    }

    /// Copy the dataflow operation to the new function.
    /// These are the operations that are not modified by the modifier.
    fn modify_dataflow_op(
        &mut self,
        h: &impl HugrMut<Node = N>,
        n: N,
        optype: &OpType,
        new_fn: &mut impl Dataflow,
    ) -> Result<(), ModifierResolverErrors<N>> {
        self.add_node_no_modification(new_fn, optype.clone(), h, n)
            .map(|_| ())
    }

    fn modify_extension_op(
        &mut self,
        h: &impl HugrMut<Node = N>,
        n: N,
        optype: &OpType,
        new_fn: &mut impl Dataflow,
    ) -> Result<(), ModifierResolverErrors<N>> {
        use Tk2Op::*;

        let num = self.modifiers.control;
        if self.controls().len() != num {
            println!(
                "Control qubits are not set correctly: {:?}.len() != {} when modifying {}",
                self.controls(),
                num,
                n
            );
            return Err(ModifierResolverErrors::Unreachable(
                "Control qubits are not set correctly.".to_string(),
            ));
        }

        // If Tk2Op, return the modified operation.
        println!("I'm modifying Node: {:?}", n);
        if let Some(op) = Tk2Op::from_optype(optype) {
            if self.modifiers.control != 0 || self.modifiers.dagger {
                if !op.is_quantum() {
                    return Err(ModifierError::ModifierNotApplicable(n, op.into()).into());
                }
            }
            match op {
                X | CX | Toffoli | Y | CY | Z | CZ => {
                    let (gate, c_gate, cc_gate) = if X == op || CX == op || Toffoli == op {
                        (X, CX, Some(Toffoli))
                    } else if Y == op || CY == op {
                        (Y, CY, None)
                    } else if Z == op || CZ == op {
                        (Z, CZ, None)
                    } else {
                        unreachable!()
                    };
                    let original_control = match op {
                        X | Y | Z => 0,
                        CX | CY | CZ => 1,
                        Toffoli => 2,
                        _ => unreachable!(),
                    };

                    let new_op = match original_control + self.modifiers.control {
                        0 => gate,
                        1 => c_gate,
                        2 => cc_gate.unwrap_or_else(|| todo!()),
                        _ => todo!(),
                    };
                    let qubit_index = (0..(original_control + 1)).collect();
                    self.add_node_reversible_simple(new_fn, new_op, h, n, qubit_index)?;
                }
                Rz | CRz => {
                    let ctrl = if Rz == op { 0 } else { 1 };

                    let new_op = match ctrl + self.modifiers.control {
                        0 => Rz,
                        1 => CRz,
                        _ => todo!(),
                    };
                    let new = self.add_node_control(new_fn, new_op);

                    if !self.modifiers.dagger {
                        for port in h.all_node_ports(n) {
                            self.map_insert(
                                DirWire::new(n, port),
                                DirWire::new(new, port.shift(self.modifiers.control)),
                            )?;
                        }
                    } else {
                        // If dagered
                        let halfturn = new_fn.add_child_node(RotationOp::to_halfturns);
                        self.map_insert(
                            (n, IncomingPort::from(ctrl + 1)).into(),
                            (halfturn, IncomingPort::from(0)).into(),
                        )?;
                        let reversed_float = new_fn
                            .add_dataflow_op(FloatOps::fneg, vec![Wire::new(halfturn, 0)])
                            .map(|out| out.out_wire(0))?;
                        let reversed = new_fn
                            .add_dataflow_op(
                                RotationOp::from_halfturns_unchecked,
                                vec![reversed_float],
                            )
                            .map(|out| out.out_wire(0))?;
                        for port in h.node_inputs(n) {
                            if port.index() <= ctrl {
                                let dir_wire: DirWire =
                                    (new, port.shift(self.modifiers.control)).into();
                                self.map_insert((n, port).into(), dir_wire.reverse())?;
                            } else if port.index() == ctrl + 1 {
                                new_fn.hugr_mut().connect(
                                    reversed.node(),
                                    reversed.source(),
                                    new,
                                    port.shift(self.modifiers.control),
                                )
                            }
                        }
                        for port in h.node_outputs(n) {
                            let index = self.modifiers.control
                                + port.index()
                                + if port.index() <= ctrl { 0 } else { 1 };
                            self.map_insert(
                                (n, port).into(),
                                (new, IncomingPort::from(index)).into(),
                            )?;
                        }
                    }
                }
                H => todo!(),
                Rx => todo!(),
                Ry => todo!(),
                T => todo!(),
                Tdg => todo!(),
                S => todo!(),
                Sdg => todo!(),
                V => todo!(),
                Vdg => todo!(),
                Measure => todo!(),
                MeasureFree => todo!(),
                QAlloc => todo!(),
                TryQAlloc => todo!(),
                QFree => todo!(),
                Reset => todo!(),
            }
        } else {
            // Some other Hugr extension operation.
            // Here, we do not know what is the modified version.
            // We try to place the previous operation.
            return self.modify_dataflow_op(h, n, optype, new_fn);
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
        let mut resolver = ModifierResolver::new();
        for node in descendants {
            resolver.node = Some(node);
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

impl CombinedModifier {}

#[cfg(test)]
mod test {
    use hugr::{
        builder::{Dataflow, DataflowSubContainer, HugrBuilder, ModuleBuilder},
        extension::prelude::qb_t,
        ops::CallIndirect,
        types::{Signature, Term},
    };

    use crate::rich_circuit::*;
    use crate::{extension::rotation::ConstRotation, rich_circuit::modifier_resolver::*};

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
            let theta = {
                let angle = ConstRotation::new(0.5).unwrap();
                func.add_load_value(angle)
            };
            let rota = func
                .add_dataflow_op(Tk2Op::Rz, vec![in1, theta])
                .unwrap()
                .out_wire(0);
            let xgate = func.add_dataflow_op(Tk2Op::X, vec![rota]).unwrap();
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
        println!("Before modification:\n{}", h.mermaid_string());

        let entrypoint = h.entrypoint().clone();
        resolve_modifier_with_entrypoints(&mut h, vec![entrypoint].into_iter()).unwrap();
        println!("After modification\n{}", h.mermaid_string());
    }
}
