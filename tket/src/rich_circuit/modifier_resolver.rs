//! Try to delete modifier by applying the modifier to each component.
//!
pub mod global_phase_modify;
pub mod tket_op_modify;
use derive_more::Error;
use hugr::{
    builder::{
        BuildError, ConditionalBuilder, Container, Dataflow, FunctionBuilder, HugrBuilder,
        SubContainer, TailLoopBuilder,
    },
    core::HugrNode,
    extension::{prelude::qb_t, simple_op::MakeExtensionOp},
    hugr::{hugrmut::HugrMut, patch::replace::ReplaceError},
    ops::{
        handle::NodeHandle, Conditional, Const, DataflowOpTrait, DataflowParent, ExtensionOp,
        LoadFunction, OpType, TailLoop, DFG,
    },
    std_extensions::collections::array::{array_type, ArrayOpBuilder},
    types::{FuncTypeBase, PolyFuncType, Signature, TypeRow},
    Hugr, HugrView, IncomingPort, Node, OutgoingPort, Port, PortIndex, Wire,
};
use hugr_core::hugr::internal::PortgraphNodeMap;
use itertools::{Either, Itertools};
use petgraph::visit::{Topo, Walker};
use std::{cmp::min, collections::HashMap, iter, mem};

use crate::{
    rich_circuit::{modifier_resolver::global_phase_modify::delete_phase, GlobalPhase, Modifier},
    TketOp,
};

/// An accumulated modifier that combines control, dagger, and power modifiers.
pub struct CombinedModifier {
    control: usize,
    accum_ctrl: Vec<usize>,
    dagger: bool,
    #[allow(dead_code)]
    power: bool,
}

impl Default for CombinedModifier {
    fn default() -> Self {
        CombinedModifier {
            control: 0,
            accum_ctrl: vec![],
            dagger: false,
            power: false,
        }
    }
}

impl CombinedModifier {
    /// Add a modifier
    pub fn push(&mut self, ext_op: &ExtensionOp) {
        match Modifier::from_extension_op(ext_op) {
            Ok(Modifier::ControlModifier) => {
                let ctrl = ext_op.args()[0].as_nat().unwrap() as usize;
                self.control += ctrl;
                self.accum_ctrl.push(ctrl);
            }
            Ok(Modifier::DaggerModifier) => self.dagger = !self.dagger,
            Ok(Modifier::PowerModifier) => self.power = !self.power,
            Err(_) => {}
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

    #[allow(dead_code)]
    pub fn reverse(self) -> Self {
        let index = self.1.index();
        let port = match self.1.as_directed() {
            Either::Left(_in) => OutgoingPort::from(index).into(),
            Either::Right(_out) => IncomingPort::from(index).into(),
        };
        DirWire::new(self.0, port)
    }
}

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
impl<N: HugrNode> TryFrom<DirWire<N>> for Wire<N> {
    type Error = hugr::hugr::HugrError;

    fn try_from(value: DirWire<N>) -> Result<Self, Self::Error> {
        let out_port = value.1.as_outgoing()?;
        Ok(Wire::new(value.0, out_port))
    }
}
impl<N: HugrNode> TryFrom<DirWire<N>> for (N, IncomingPort) {
    type Error = hugr::hugr::HugrError;

    fn try_from(value: DirWire<N>) -> Result<Self, Self::Error> {
        let in_port = value.1.as_incoming()?;
        Ok((value.0, in_port))
    }
}

fn connect<N>(
    new_dfg: &mut impl Dataflow,
    w1: &DirWire<Node>,
    w2: &DirWire<Node>,
) -> Result<(), ModifierResolverErrors<N>> {
    let (n_o, p_o, n_i, p_i) = match (w1.1.as_directed(), w2.1.as_directed()) {
        (Either::Right(p_o), Either::Left(p_i)) => (w1.0, p_o, w2.0, p_i),
        (Either::Left(p_i), Either::Right(p_o)) => (w2.0, p_o, w1.0, p_i),
        _ => {
            return Err(ModifierResolverErrors::Unreachable(format!(
                "Cannot connect the wires with the same direction: {} -> {}",
                w1, w2
            )))
        }
    };
    Ok(new_dfg.hugr_mut().connect(n_o, p_o, n_i, p_i))
}

/// Connect a wire to a node by its number, returning the other side of the connection.
fn connect_by_num(
    new_dfg: &mut impl Dataflow,
    dw: &DirWire<Node>,
    node: Node,
    num: usize,
) -> DirWire<Node> {
    let dw_node = dw.0;
    match dw.1.as_directed() {
        Either::Left(incoming) => {
            new_dfg.hugr_mut().connect(node, num, dw_node, incoming);
            (node, IncomingPort::from(num)).into()
        }
        Either::Right(outgoing) => {
            new_dfg.hugr_mut().connect(dw_node, outgoing, node, num);
            (node, OutgoingPort::from(num)).into()
        }
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
impl<N> PortExt for DirWire<N> {
    fn shift(self, offset: usize) -> Self {
        DirWire(self.0, self.1.shift(offset))
    }
}

/// A vector of ports for each node.
/// The `if_rev` vector is used to swap the wires if the dagger is applied.
pub struct PortVector<N = Node> {
    incoming: Vec<DirWire<N>>,
    outgoing: Vec<DirWire<N>>,
}
impl<N: HugrNode> PortVector<N> {
    fn port_vector(
        n: N,
        inputs: impl Iterator<Item = usize>,
        outputs: impl Iterator<Item = usize>,
    ) -> Self {
        let incoming = inputs.map(|p| (n, IncomingPort::from(p)).into()).collect();
        let outgoing = outputs.map(|p| (n, OutgoingPort::from(p)).into()).collect();
        PortVector { incoming, outgoing }
    }
    fn port_vector_rev(
        n: N,
        inputs: impl Iterator<Item = usize>,
        outputs: impl Iterator<Item = usize>,
        iter: impl Iterator<Item = usize>,
    ) -> Self {
        let iter = iter.collect::<Vec<_>>();
        let incoming = inputs
            .map(|p| {
                if iter.contains(&p) {
                    (n, OutgoingPort::from(p)).into()
                } else {
                    (n, IncomingPort::from(p)).into()
                }
            })
            .collect();
        let outgoing = outputs
            .map(|p| {
                if iter.contains(&p) {
                    (n, IncomingPort::from(p)).into()
                } else {
                    (n, OutgoingPort::from(p)).into()
                }
            })
            .collect();
        PortVector { incoming, outgoing }
    }
}

/// A container for modifier resolving.
pub struct ModifierResolver<N = Node> {
    node: Option<N>,
    modifiers: CombinedModifier,
    corresp_map: HashMap<DirWire<N>, Vec<DirWire>>,
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

    fn with_ancilla<T>(
        &mut self,
        wire: &mut Wire<Node>,
        ancilla: &mut Vec<Wire<Node>>,
        f: impl FnOnce(&mut Self, &mut Vec<Wire<Node>>) -> T,
    ) -> T {
        ancilla.push(*wire);
        let r = f(self, ancilla);
        *wire = ancilla.pop().unwrap();
        r
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
    /// Validation error.
    ValidationError(String),
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
    fn controls_ref(&self) -> &Vec<Wire> {
        &self.controls
    }
    fn corresp_map(&mut self) -> &mut HashMap<DirWire<N>, Vec<DirWire>> {
        &mut self.corresp_map
    }

    fn pop_control(&mut self) -> Option<Wire<Node>> {
        if let Some(c) = self.controls().pop() {
            self.modifiers.control -= 1;
            Some(c)
        } else {
            None
        }
    }

    fn push_control(&mut self, c: Wire<Node>) {
        self.controls().push(c);
        self.modifiers.control += 1;
    }

    /// normal insert method.
    fn map_insert(
        &mut self,
        old: DirWire<N>,
        new: DirWire, // new: impl Into<Either<IncomingPort, OutgoingPort>>,
    ) -> Result<(), ModifierResolverErrors<N>> {
        self.corresp_map()
            .insert(old.clone(), vec![new.clone()])
            .map_or(Ok(()), |former| {
                Err(ModifierResolverErrors::Unreachable(format!(
                    "Wire already registered for node {}. Former [{},...], Latter {}.",
                    old, former[0], new
                )))
            })
    }

    fn map_get(&self, key: &DirWire<N>) -> Result<&Vec<DirWire>, ModifierResolverErrors<N>> {
        self.corresp_map
            .get(key)
            .ok_or(ModifierResolverErrors::Unreachable(format!(
                "No correspondence for the input wire. Input: {}",
                key
            )))
    }

    fn port_vector_dagger(
        &self,
        n: Node,
        inputs: impl Iterator<Item = usize>,
        outputs: impl Iterator<Item = usize>,
        iter: impl Iterator<Item = usize>,
    ) -> PortVector<Node> {
        if self.modifiers.dagger {
            PortVector::port_vector_rev(n, inputs, outputs, iter)
        } else {
            PortVector::port_vector(n, inputs, outputs)
        }
    }

    fn add_edge_from_pv(
        &mut self,
        h: &impl HugrMut<Node = N>,
        n: N,
        pv: PortVector<Node>,
    ) -> Result<(), ModifierResolverErrors<N>> {
        let PortVector { incoming, outgoing } = pv;
        for (old_in, new) in (0..h.num_inputs(n)).map(IncomingPort::from).zip(incoming) {
            self.map_insert((n, old_in).into(), new)?
        }
        for (old_out, new) in (0..h.num_outputs(n)).map(OutgoingPort::from).zip(outgoing) {
            self.map_insert((n, old_out).into(), new)?
        }
        Ok(())
    }

    /// Add a node to the builder, plugging the control qubits to the first n-inputs and outputs.
    fn add_node_control(&mut self, new_dfg: &mut impl Dataflow, op: impl Into<OpType>) -> Node {
        let node = new_dfg.add_child_node(op);
        for (i, ctrl) in self.controls().iter_mut().enumerate() {
            new_dfg
                .hugr_mut()
                .connect(ctrl.node(), ctrl.source(), node, i);
            *ctrl = Wire::new(node, i);
        }
        node
    }

    /// This function adds a node to the builder, that does not affected by the modifiers.
    fn add_node_no_modification(
        &mut self,
        new_dfg: &mut impl Dataflow,
        op: impl Into<OpType>,
        h: &impl HugrMut<Node = N>,
        old_n: N,
    ) -> Result<Node, ModifierResolverErrors<N>> {
        let node = new_dfg.add_child_node(op);
        for port in h.all_node_ports(old_n) {
            self.map_insert(DirWire(old_n, port), DirWire(node, port))?;
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
        for (a, bs) in self.corresp_map().iter() {
            for b in bs {
                println!("  Wire mapping: {} -> {}", a, b);
            }
        }
        // println!(" hugr: \n{}", new_dfg.hugr().mermaid_string());
        for out_node in h.children(parent) {
            for out_port in h.node_outputs(out_node) {
                for (in_node, in_port) in h.linked_inputs(out_node, out_port) {
                    for a in self.map_get(&(in_node, in_port).into())? {
                        for b in self.map_get(&(out_node, out_port).into())? {
                            connect(new_dfg, a, b)?;
                        }
                    }
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
        let mut modifiers = CombinedModifier::default();
        let mut modifier_and_targ: Vec<N> = Vec::new();
        loop {
            modifier_and_targ.push(targ);
            let optype = h.get_optype(targ);
            match Modifier::from_optype(optype) {
                Some(_) => modifiers.push(optype.as_extension_op().unwrap()),
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
        self.modifiers = modifiers;

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
        self.modify_signature(modified_sig.body_mut(), false);
        println!(
            "Modifying function {} with signature {:?}",
            func, modified_sig
        );
        let modified_fn = self.modify_fn(h, func, modified_sig.clone())?;
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
    pub fn modify_signature(&self, signature: &mut Signature, flatten: bool) {
        let FuncTypeBase { input, output } = signature;

        if flatten {
            let n = self.modifiers.control;
            input.to_mut().splice(0..0, iter::repeat(qb_t()).take(n));
            output.to_mut().splice(0..0, iter::repeat(qb_t()).take(n));
        } else {
            for ctrls in &self.modifiers.accum_ctrl {
                let n = ctrls.clone() as u64;
                input.to_mut().insert(0, array_type(n, qb_t()));
                output.to_mut().insert(0, array_type(n, qb_t()));
            }
        }
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

    // TODO: We take arbitral topological order of the circuit so that we can plug the control qubits
    // and pass around them in that order. However, this is not ideal, as it may produce an inefficient order.
    fn modify_op(
        &mut self,
        h: &impl HugrMut<Node = N>,
        n: N,
        new_dfg: &mut impl Dataflow,
    ) -> Result<(), ModifierResolverErrors<N>> {
        let optype = h.get_optype(n);
        println!("Modifying Op: {}, OpType: {:?}", n, optype);
        match optype {
            // Skip input/output nodes: it should be handled by its parent as it sets control qubits.
            OpType::Input(_) | OpType::Output(_) => {}
            OpType::ExtensionOp(_) => {
                self.modify_extension_op(h, n, optype, new_dfg)?;
            }
            OpType::DFG(dfg) => self.modify_dfg(h, n, dfg.clone(), new_dfg)?,
            OpType::TailLoop(tail_loop) => self.modify_tail_loop(h, n, tail_loop, new_dfg)?,
            OpType::Conditional(conditional) => {
                self.modify_conditional(h, n, conditional, new_dfg)?
            }
            OpType::Call(_) => todo!(),
            OpType::CallIndirect(_) => todo!(),
            OpType::LoadFunction(_) => todo!(),

            OpType::Case(_) => todo!(),
            OpType::Const(constant) => {
                self.modify_constant(n, constant.clone(), new_dfg)?;
            }
            OpType::LoadConstant(_) | OpType::OpaqueOp(_) => {
                self.modify_dataflow_op(h, n, optype, new_dfg)?
            }
            OpType::Tag(_) => self.modify_dataflow_op(h, n, optype, new_dfg)?,
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
                let offset = if matches!(optype, OpType::FuncDefn(_)) {
                    self.modifiers.accum_ctrl.len()
                } else {
                    self.modifiers.control
                };
                for port in h.node_outputs(old_in) {
                    self.map_insert(
                        (old_in, port).into(),
                        DirWire::from((new_in, port)).shift(offset),
                    )?;
                    println!(
                        "Inserted input mapping: {} -> {}",
                        DirWire::from((old_in, port)),
                        DirWire::from((new_in, port)).shift(offset)
                    );
                }
                for port in h.node_inputs(old_out) {
                    self.map_insert(
                        (old_out, port).into(),
                        DirWire::from((new_out, port)).shift(offset),
                    )?
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

        // FIXME: very hacky way to swap the IO wires if dagger is applied.
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
            // TODO: replace `== qb_t()` to appropriate type check.
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
            OpType::Case(_) => todo!(),
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

    fn modify_constant(
        &mut self,
        n: N,
        constant: Const,
        new_dfg: &mut impl Dataflow,
    ) -> Result<(), ModifierResolverErrors<N>> {
        let output = new_dfg.add_child_node(constant);
        self.map_insert(Wire::new(n, 0).into(), Wire::new(output, 0).into())
    }

    /// Copy the dataflow operation to the new function.
    /// These are the operations that are not modified by the modifier.
    fn modify_dataflow_op(
        &mut self,
        h: &impl HugrMut<Node = N>,
        n: N,
        optype: &OpType,
        new_dfg: &mut impl Dataflow,
    ) -> Result<(), ModifierResolverErrors<N>> {
        self.add_node_no_modification(new_dfg, optype.clone(), h, n)
            .map(|_| ())
    }

    fn modify_extension_op(
        &mut self,
        h: &impl HugrMut<Node = N>,
        n: N,
        optype: &OpType,
        new_dfg: &mut impl Dataflow,
    ) -> Result<(), ModifierResolverErrors<N>> {
        if self.controls().len() != self.modifiers.control {
            return Err(ModifierResolverErrors::Unreachable(
                "Control qubits are not set correctly.".to_string(),
            ));
        }

        // If TketOp, return the modified operation.
        println!("I'm modifying Node: {:?}", n);
        if let Some(op) = TketOp::from_optype(optype) {
            let pv = self.modify_tket_op(n, op, new_dfg, &mut vec![])?;
            self.add_edge_from_pv(h, n, pv)
        } else if GlobalPhase::from_optype(optype).is_some() {
            let inputs = self.modify_global_phase(n, new_dfg, &mut vec![])?;
            self.corresp_map().insert(
                (n, IncomingPort::from(0)).into(),
                inputs.into_iter().map(Into::into).collect(),
            );
            Ok(())
        } else {
            // Some other Hugr extension operation.
            // Here, we do not know what is the modified version.
            // We try to place the previous operation.
            self.modify_dataflow_op(h, n, optype, new_dfg)
        }
    }

    fn modify_dfg(
        &mut self,
        h: &impl HugrMut<Node = N>,
        n: N,
        mut dfg: DFG,
        new_parent_dfg: &mut impl Dataflow,
    ) -> Result<(), ModifierResolverErrors<N>> {
        // Build a new DFG with modified body.
        self.modify_signature(&mut dfg.signature, true);
        // FIXME: This uses a hack: dfg_builder does not check the types or number of input_wires, so when no input_wires are given,
        // it just creates a DFG without connecting any inputs.
        let mut new_dfg = new_parent_dfg.dfg_builder(dfg.signature, vec![])?;
        self.modify_dfg_body(h, n, &mut new_dfg)?;
        let new_dfg = new_dfg
            .finish_sub_container()
            .map_err(ModifierResolverErrors::BuildError)?;
        let new_dfg_node = new_dfg.node();

        // connect the controls and register the IOs
        for (i, c) in self.controls().iter_mut().enumerate() {
            new_parent_dfg
                .hugr_mut()
                .connect(c.node(), c.source(), new_dfg_node, i);
            *c = Wire::new(new_dfg_node, i);
        }
        for port in h.all_node_ports(n) {
            self.map_insert(
                DirWire(n, port),
                DirWire(new_dfg_node, port).shift(self.modifiers.control),
            )?;
        }

        Ok(())
    }

    fn modify_tail_loop(
        &mut self,
        h: &impl HugrMut<Node = N>,
        n: N,
        tail_loop: &TailLoop,
        new_dfg: &mut impl Dataflow,
    ) -> Result<(), ModifierResolverErrors<N>> {
        let just_input_num = tail_loop.just_inputs.len();
        let just_output_num = tail_loop.just_outputs.len();

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

    fn modify_conditional(
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
            let tag_type_num = conditional.sum_rows[i].len();
            let mut case_builder = builder.case_builder(i).unwrap();

            // Set the controls and corresp_map
            let mut corresp_map = HashMap::new();
            let controls = case_builder
                .input_wires()
                .skip(tag_type_num)
                .take(offset)
                .collect();
            mem::swap(self.corresp_map(), &mut corresp_map);
            *self.controls() = controls;

            // Modify the IOs
            let [old_in, old_out] = h.get_io(case_node).unwrap();
            let [new_in, new_out] = case_builder.io();
            // Modify the input/output nodes beforehand.
            for port in h.node_outputs(old_in) {
                let new_port = if port.index() < tag_type_num {
                    port
                } else {
                    port.shift(offset)
                };
                self.map_insert((old_in, port).into(), DirWire::from((new_in, new_port)))?;
            }
            for port in h.node_inputs(old_out) {
                self.map_insert(
                    (old_out, port).into(),
                    DirWire::from((new_out, port.shift(offset))),
                )?
            }

            // FIXME: This implementation does not support dagger
            // Need to do the same stuff as in modify_in_out_node.

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
        for port in h.node_inputs(n) {
            let new_port = if port.index() == 0 {
                port
            } else {
                port.shift(offset)
            };
            self.map_insert((n, port).into(), (new_conditional, new_port).into())?;
        }
        for port in h.node_outputs(n) {
            let new_port = port.shift(offset);
            self.map_insert((n, port).into(), (new_conditional, new_port).into())?
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

    let entry_points = entry_points.collect::<Vec<_>>();

    for entry_point in entry_points.clone() {
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

    delete_phase(h, entry_points.into_iter())?;

    Ok(())
}

#[cfg(test)]
mod test {
    use std::{fs::File, io::Write, path::Path};

    use hugr::{
        algorithms::{dead_code, ComposablePass},
        builder::{Dataflow, DataflowSubContainer, HugrBuilder, ModuleBuilder},
        envelope::{EnvelopeConfig, EnvelopeFormat},
        extension::{prelude::qb_t, ExtensionRegistry},
        ops::{
            handle::{FuncID, NodeHandle},
            CallIndirect, ExtensionOp,
        },
        std_extensions::collections::array::{array_type, ArrayOpBuilder},
        types::{Signature, Term, Type},
    };

    use crate::{
        extension::{
            bool::BOOL_EXTENSION,
            rotation::{ConstRotation, RotationOp, ROTATION_EXTENSION},
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
                iter::repeat(qb_t()).take(t_num).zip(func.input_wires()),
                iter::repeat(qb_t()).take(t_num).collect::<Vec<_>>().into(),
            )
            .unwrap();
        let _case1 = {
            let mut case = cond_builder.case_builder(0).unwrap();
            let mut inputs = case.input_wires();
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
    #[case::dfg(1, 2, foo_dfg, "dfg")]
    #[case::tail_loop(1, 1, foo_tail_loop, "tail_loop")]
    #[case::conditional(1, 1, foo_conditional, "conditional")]
    fn test_modifier_resolver_optypes(
        #[case] t_num: usize,
        #[case] c_num: u64,
        #[case] foo: fn(&mut ModuleBuilder<Hugr>, usize) -> FuncID<true>,
        #[case] name: &str,
    ) {
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
