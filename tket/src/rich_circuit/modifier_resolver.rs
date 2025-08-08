//! Try to delete modifier by applying the modifier to each component.
//!

pub mod tket2_modify;
use derive_more::Error;
use hugr::{
    builder::{BuildError, ConditionalBuilder, Container, Dataflow, FunctionBuilder, HugrBuilder},
    core::HugrNode,
    extension::{prelude::qb_t, simple_op::MakeExtensionOp},
    hugr::{hugrmut::HugrMut, patch::replace::ReplaceError},
    ops::{Conditional, Const, DataflowOpTrait, DataflowParent, LoadFunction, OpType},
    types::{FuncTypeBase, PolyFuncType},
    Hugr, HugrView, IncomingPort, Node, OutgoingPort, Port, PortIndex, Wire,
};
use hugr_core::hugr::internal::PortgraphNodeMap;
use itertools::{Either, Itertools};
use petgraph::visit::{Topo, Walker};
use std::{cmp::min, collections::HashMap, iter, mem};

use crate::{rich_circuit::Modifier, Tk2Op};

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
            .insert(old.clone(), new.clone())
            .map_or(Ok(()), |former| {
                Err(ModifierResolverErrors::Unreachable(format!(
                    "Wire already registered for node {}. Former {}, Latter {}.",
                    old, former, new
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

    fn add_edge(
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
    fn add_node_control(&mut self, new_fn: &mut impl Dataflow, op: impl Into<OpType>) -> Node {
        let node = new_fn.add_child_node(op);
        for (i, ctrl) in self.controls().iter_mut().enumerate() {
            new_fn
                .hugr_mut()
                .connect(ctrl.node(), ctrl.source(), node, i);
            *ctrl = Wire::new(node, i);
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
        Ok(node)
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
                    connect(new_dfg, a, b)?;
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
        if self.controls().len() != self.modifiers.control {
            return Err(ModifierResolverErrors::Unreachable(
                "Control qubits are not set correctly.".to_string(),
            ));
        }

        // If Tk2Op, return the modified operation.
        println!("I'm modifying Node: {:?}", n);
        if let Some(op) = Tk2Op::from_optype(optype) {
            let pv = self.modify_tket2_op(n, op, new_fn, &mut vec![])?;
            self.add_edge(h, n, pv)
        } else {
            // Some other Hugr extension operation.
            // Here, we do not know what is the modified version.
            // We try to place the previous operation.
            return self.modify_dataflow_op(h, n, optype, new_fn);
        }
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
        envelope::{EnvelopeConfig, EnvelopeFormat},
        extension::{prelude::{bool_t, qb_t}, ExtensionRegistry},
        ops::{CallIndirect, ExtensionOp},
        types::{Signature, Term},
    };

    use crate::{extension::{bool::BOOL_EXTENSION, rotation::{rotation_type, ROTATION_EXTENSION}, TKET2_EXTENSION}, rich_circuit::*};
    use crate::{extension::rotation::ConstRotation, rich_circuit::modifier_resolver::*};

    #[test]
    fn test_control_simple() {
        let mut module = ModuleBuilder::new();
        let foo_sig = Signature::new(vec![qb_t(), qb_t()], vec![qb_t(), qb_t()]);
        let main_sig = Signature::new(vec![qb_t(), qb_t(), qb_t()], vec![qb_t(), qb_t(), qb_t()]);

        let control_op = MODIFIER_EXTENSION
            .instantiate_extension_op(
                &CONTROL_OP_ID,
                [
                    Term::new_list([qb_t().into(), qb_t().into()]),
                    Term::new_list([qb_t().into(), qb_t().into()]),
                ],
            )
            .unwrap();

        // fn foo {
        //     -- • -- Y ---
        //        |
        //     -- X -- Z --
        // }
        let foo = {
            let mut func = module.define_function("foo", foo_sig.clone()).unwrap();
            let [in1, in2] = func.input_wires_arr();
            let [w1, w2] = func
                .add_dataflow_op(Tk2Op::CX, vec![in1, in2])
                .unwrap()
                .outputs_arr();
            let o1 = func
                .add_dataflow_op(Tk2Op::Y, vec![w1])
                .unwrap()
                .out_wire(0);
            let o2 = func
                .add_dataflow_op(Tk2Op::Z, vec![w2])
                .unwrap()
                .out_wire(0);
            func.finish_with_outputs(vec![o1, o2]).unwrap()
        };

        let _main = {
            let mut func = module.define_function("main", main_sig.clone()).unwrap();
            let [in1, in2, in3] = func.input_wires_arr();
            let loaded = func.load_func(foo.handle(), &[]).unwrap();
            let controlled = func
                .add_dataflow_op(control_op, vec![loaded])
                .unwrap()
                .out_wire(0);
            let [out1, out2, out3] = func
                .add_dataflow_op(
                    CallIndirect {
                        signature: main_sig,
                    },
                    vec![controlled, in1, in2, in3],
                )
                .unwrap()
                .outputs_arr();
            func.finish_with_outputs(vec![out1, out2, out3]).unwrap()
        };

        let mut h = module.finish_hugr().unwrap();
        println!("Before modification:\n{}", h.mermaid_string());

        let entrypoint = h.entrypoint().clone();
        resolve_modifier_with_entrypoints(&mut h, vec![entrypoint].into_iter()).unwrap();
        println!("After modification\n{}", h.mermaid_string());
    }

    #[test]
    fn test_control_ry_s() {
        let mut module = ModuleBuilder::new();
        let foo_sig = Signature::new(vec![qb_t(), qb_t()], vec![qb_t(), qb_t()]);
        let main_sig = Signature::new(vec![qb_t(), qb_t(), qb_t()], vec![qb_t(), qb_t(), qb_t()]);
        let call_sig = Signature::new(vec![qb_t(), qb_t(), qb_t()], vec![qb_t(), qb_t(), qb_t()]);

        let control_op = MODIFIER_EXTENSION
            .instantiate_extension_op(
                &CONTROL_OP_ID,
                [
                    Term::new_list([qb_t().into(), qb_t().into()]),
                    Term::new_list([qb_t().into(), qb_t().into()]),
                ],
            )
            .unwrap();

        // fn foo {
        //     -- • -- S ---
        //        |
        //     -- X -- Ry --
        //             ||
        //      0.5 ====
        // }
        let foo = {
            let mut func = module.define_function("foo", foo_sig.clone()).unwrap();
            let [mut i1, mut i2] = func.input_wires_arr();
            [i1, i2] = func
                .add_dataflow_op(Tk2Op::CX, vec![i1, i2])
                .unwrap()
                .outputs_arr();
            i1 = func
                .add_dataflow_op(Tk2Op::S, vec![i1])
                .unwrap()
                .out_wire(0);
            let theta = {
                let angle = ConstRotation::new(0.5).unwrap();
                func.add_load_value(angle)
            };
            i2 = func
                .add_dataflow_op(Tk2Op::Ry, vec![i2, theta])
                .unwrap()
                .out_wire(0);
            func.finish_with_outputs(vec![i1, i2]).unwrap()
        };

        let _main = {
            let mut func = module.define_function("main", main_sig.clone()).unwrap();
            let loaded = func.load_func(foo.handle(), &[]).unwrap();
            let controlled = func
                .add_dataflow_op(control_op, vec![loaded])
                .unwrap()
                .out_wire(0);
            // let theta = func.add_load_value(ConstRotation::new(0.75).unwrap());
            let mut inputs = vec![controlled];
            inputs.extend(func.input_wires());
            // inputs.push(theta);
            let outs = func
                .add_dataflow_op(
                    CallIndirect {
                        signature: call_sig,
                    },
                    inputs,
                )
                .unwrap()
                .outputs();
            func.finish_with_outputs(outs).unwrap()
        };

        let mut h = module.finish_hugr().unwrap();
        println!("Before modification:\n{}", h.mermaid_string());

        let entrypoint = h.entrypoint().clone();
        resolve_modifier_with_entrypoints(&mut h, vec![entrypoint].into_iter()).unwrap();
        println!("After modification\n{}", h.mermaid_string());
    }

    #[test]
    fn test_simple_dagger() {
        // Reversed Rz(θ) gate

        let mut module = ModuleBuilder::new();
        let foo_sig = Signature::new(vec![qb_t(), rotation_type()], vec![qb_t()]);
        let fn_sig = Signature::new(vec![qb_t()], vec![qb_t()]);

        let dagger_op = MODIFIER_EXTENSION
            .instantiate_extension_op(
                &DAGGER_OP_ID,
                [
                    Term::new_list([qb_t().into(), rotation_type().into()]),
                    Term::new_list([qb_t().into()]),
                ],
            )
            .unwrap();

        // fn foo {
        //     -- Rz -- S --
        //        ||
        //     ====
        // }
        let foo = {
            let mut func = module.define_function("foo", foo_sig.clone()).unwrap();
            let [in1, in2] = func.input_wires_arr();
            let rxgate = func
                .add_dataflow_op(Tk2Op::Rz, vec![in1, in2])
                .unwrap()
                .out_wire(0);
            let sgate = func.add_dataflow_op(Tk2Op::S, vec![rxgate]).unwrap();
            func.finish_with_outputs(sgate.outputs()).unwrap()
        };

        let _main = {
            let mut func = module.define_function("main", fn_sig.clone()).unwrap();
            let [in1] = func.input_wires_arr();
            let loaded = func.load_func(foo.handle(), &[]).unwrap();
            let theta = {
                let angle = ConstRotation::new(0.25).unwrap();
                func.add_load_value(angle)
            };
            let daggered = func
                .add_dataflow_op(dagger_op, vec![loaded])
                .unwrap()
                .out_wire(0);
            let [out1] = func
                .add_dataflow_op(
                    CallIndirect { signature: foo_sig },
                    vec![daggered, in1, theta],
                )
                .unwrap()
                .outputs_arr();
            func.finish_with_outputs(vec![out1]).unwrap()
        };

        let mut h = module.finish_hugr().unwrap();
        println!("Before modification:\n{}", h.mermaid_string());
        let entrypoint = h.entrypoint().clone();
        resolve_modifier_with_entrypoints(&mut h, vec![entrypoint].into_iter()).unwrap();
        println!("After modification\n{}", h.mermaid_string());
    }

    #[test]
    fn test_combined() {
        let mut module = ModuleBuilder::new();
        let foo_sig = Signature::new(
            vec![qb_t(), qb_t(), qb_t(), rotation_type()],
            vec![qb_t(), qb_t(), qb_t()],
        );
        let call_sig = Signature::new(
            vec![qb_t(), qb_t(), qb_t(), qb_t(), rotation_type()],
            vec![qb_t(), qb_t(), qb_t(), qb_t()],
        );
        let main_sig = Signature::new(
            vec![qb_t(), qb_t(), qb_t(), qb_t()],
            vec![qb_t(), qb_t(), qb_t(), qb_t()],
        );

        let control_op = MODIFIER_EXTENSION
            .instantiate_extension_op(
                &CONTROL_OP_ID,
                [
                    Term::new_list([
                        qb_t().into(),
                        qb_t().into(),
                        qb_t().into(),
                        rotation_type().into(),
                    ]),
                    Term::new_list([qb_t().into(), qb_t().into(), qb_t().into()]),
                ],
            )
            .unwrap();
        let dagger_op = MODIFIER_EXTENSION
            .instantiate_extension_op(
                &DAGGER_OP_ID,
                [
                    Term::new_list([
                        qb_t().into(),
                        qb_t().into(),
                        qb_t().into(),
                        qb_t().into(),
                        rotation_type().into(),
                    ]),
                    Term::new_list([qb_t().into(), qb_t().into(), qb_t().into(), qb_t().into()]),
                ],
            )
            .unwrap();

        // fn foo {
        //     -------- Ry -- V --
        //              ||
        //        0.5 ===
        //
        //     --- H --- S -------
        //
        //     --- Z ---- Rx -----
        //                ||
        //     ============
        // }
        let foo = {
            let mut func = module.define_function("foo", foo_sig.clone()).unwrap();
            let [mut in1, mut in2, mut in3, in4] = func.input_wires_arr();
            let theta = func.add_load_value(ConstRotation::new(0.46).unwrap());
            in1 = func
                .add_dataflow_op(Tk2Op::Ry, vec![in1, theta])
                .unwrap()
                .out_wire(0);
            in1 = func
                .add_dataflow_op(Tk2Op::V, vec![in1])
                .unwrap()
                .out_wire(0);
            in2 = func
                .add_dataflow_op(Tk2Op::H, vec![in2])
                .unwrap()
                .out_wire(0);
            in2 = func
                .add_dataflow_op(Tk2Op::S, vec![in2])
                .unwrap()
                .out_wire(0);
            in3 = func
                .add_dataflow_op(Tk2Op::Z, vec![in3])
                .unwrap()
                .out_wire(0);
            in3 = func
                .add_dataflow_op(Tk2Op::Rx, vec![in3, in4])
                .unwrap()
                .out_wire(0);
            func.finish_with_outputs(vec![in1, in2, in3]).unwrap()
        };

        let _main = {
            let mut func = module.define_function("main", main_sig.clone()).unwrap();
            let loaded = func.load_func(foo.handle(), &[]).unwrap();
            let controlled = func
                .add_dataflow_op(control_op, vec![loaded])
                .unwrap()
                .out_wire(0);
            let daggered = func
                .add_dataflow_op(dagger_op, vec![controlled])
                .unwrap()
                .out_wire(0);
            let theta = func.add_load_value(ConstRotation::new(0.75).unwrap());
            let mut inputs = vec![daggered];
            inputs.extend(func.input_wires());
            inputs.push(theta);
            let outs = func
                .add_dataflow_op(
                    CallIndirect {
                        signature: call_sig,
                    },
                    inputs,
                )
                .unwrap()
                .outputs();
            func.finish_with_outputs(outs).unwrap()
        };

        let mut h = module.finish_hugr().unwrap();
        println!("Before modification:\n{}", h.mermaid_string());

        let entrypoint = h.entrypoint().clone();
        resolve_modifier_with_entrypoints(&mut h, vec![entrypoint].into_iter()).unwrap();
        println!("After modification\n{}", h.mermaid_string());
    }

    #[test]
    fn test_cccx() {
        let mut module = ModuleBuilder::new();
        let t_num = 4;
        let c_num = 1;
        let targs = iter::repeat(qb_t()).take(t_num).collect::<Vec<_>>();
        let foo_sig = Signature::new_endo(targs);
        let qubits = iter::repeat(qb_t()).take(c_num + t_num).collect::<Vec<_>>();
        let call_sig = Signature::new_endo(qubits);
        let main_sig = call_sig.clone();

        fn control_op(num: usize) -> ExtensionOp {
            let term_list: Vec<Term> = iter::repeat(qb_t().into()).take(num).collect();
            MODIFIER_EXTENSION
                .instantiate_extension_op(
                    &CONTROL_OP_ID,
                    [Term::new_list(term_list.clone()), Term::new_list(term_list)],
                )
                .unwrap()
        }

        // fn foo {
        //     ----•--------
        //         |
        //     ----•--------
        //         |
        //     ----X--------
        // }
        let foo = {
            let mut func = module.define_function("foo", foo_sig.clone()).unwrap();
            let mut inputs: Vec<Wire> = func.input_wires().collect();
            let (i1, i2, i3) = inputs.iter_mut().take(3).collect_tuple().unwrap();
            // let theta = func.add_load_value(ConstRotation::new(0.46).unwrap());
            [*i1, *i2, *i3] = func
                .add_dataflow_op(Tk2Op::Toffoli, vec![*i1, *i2, *i3])
                .unwrap()
                .outputs_arr();
            func.finish_with_outputs(inputs).unwrap()
        };

        let _main = {
            let mut func = module.define_function("main", main_sig).unwrap();
            let mut call = func.load_func(foo.handle(), &[]).unwrap();
            for i in 0..c_num {
                call = func
                    .add_dataflow_op(control_op(t_num + i), vec![call])
                    .unwrap()
                    .out_wire(0);
            }
            let mut inputs = vec![call];
            inputs.extend(func.input_wires());
            let outs = func
                .add_dataflow_op(
                    CallIndirect {
                        signature: call_sig,
                    },
                    inputs,
                )
                .unwrap()
                .outputs();
            func.finish_with_outputs(outs).unwrap()
        };

        let mut h = module.finish_hugr().unwrap();
        println!("Before modification:\n{}", h.mermaid_string());

        let entrypoint = h.entrypoint().clone();
        resolve_modifier_with_entrypoints(&mut h, vec![entrypoint].into_iter()).unwrap();
        println!("After modification\n{}", h.mermaid_string());
    }

    #[test]
    fn test_multi_control_ancilla() {
        let mut module = ModuleBuilder::new();
        let t_num = 4;
        let c_num = 2;
        let targs = iter::repeat(qb_t()).take(t_num).collect::<Vec<_>>();
        let foo_sig = Signature::new_endo(targs);
        let qubits = iter::repeat(qb_t()).take(c_num + t_num).collect::<Vec<_>>();
        let call_sig = Signature::new_endo(qubits);
        let main_sig = call_sig.clone();

        fn control_op(num: usize) -> ExtensionOp {
            let term_list: Vec<Term> = iter::repeat(qb_t().into()).take(num).collect();
            MODIFIER_EXTENSION
                .instantiate_extension_op(
                    &CONTROL_OP_ID,
                    [Term::new_list(term_list.clone()), Term::new_list(term_list)],
                )
                .unwrap()
        }

        // fn foo {
        //     ----•----
        //         |
        //     ----•----
        //         |
        //     ----X----
        // }
        let foo = {
            let mut func = module.define_function("foo", foo_sig.clone()).unwrap();
            let mut inputs: Vec<Wire> = func.input_wires().collect();
            let (i1, i2, i3) = inputs.iter_mut().take(3).collect_tuple().unwrap();
            // let theta = func.add_load_value(ConstRotation::new(0.46).unwrap());
            [*i1, *i2, *i3] = func
                .add_dataflow_op(Tk2Op::Toffoli, vec![*i1, *i2, *i3])
                .unwrap()
                .outputs_arr();
            // [*i1, *i2] = func
            //     .add_dataflow_op(Tk2Op::CZ, vec![*i1, *i2])
            //     .unwrap()
            //     .outputs_arr();
            // in1 = func
            //     .add_dataflow_op(Tk2Op::V, vec![in1])
            //     .unwrap()
            //     .out_wire(0);
            // in2 = func
            //     .add_dataflow_op(Tk2Op::H, vec![in2])
            //     .unwrap()
            //     .out_wire(0);
            // in2 = func
            //     .add_dataflow_op(Tk2Op::S, vec![in2])
            //     .unwrap()
            //     .out_wire(0);
            // in3 = func
            //     .add_dataflow_op(Tk2Op::Z, vec![in3])
            //     .unwrap()
            //     .out_wire(0);
            // in3 = func
            //     .add_dataflow_op(Tk2Op::Rx, vec![in3, in4])
            //     .unwrap()
            //     .out_wire(0);
            func.finish_with_outputs(inputs).unwrap()
        };

        let _main = {
            let mut func = module.define_function("main", main_sig).unwrap();
            let mut call = func.load_func(foo.handle(), &[]).unwrap();
            for i in 0..c_num {
                call = func
                    .add_dataflow_op(control_op(t_num + i), vec![call])
                    .unwrap()
                    .out_wire(0);
            }
            let mut inputs = vec![call];
            inputs.extend(func.input_wires());
            let outs = func
                .add_dataflow_op(
                    CallIndirect {
                        signature: call_sig,
                    },
                    inputs,
                )
                .unwrap()
                .outputs();
            func.finish_with_outputs(outs).unwrap()
        };

        let mut h = module.finish_hugr().unwrap();
        h.validate().unwrap();
        println!("Before modification:\n{}", h.mermaid_string());


        let entrypoint = h.entrypoint().clone();
        resolve_modifier_with_entrypoints(&mut h, vec![entrypoint].into_iter()).unwrap();
        println!("After modification\n{}", h.mermaid_string());
        let env_format = EnvelopeFormat::PackageJson;
        let env_conf: EnvelopeConfig = EnvelopeConfig::new(env_format);
        let iter: Vec<Arc<Extension>> = vec![ROTATION_EXTENSION.to_owned(), TKET2_EXTENSION.to_owned(), BOOL_EXTENSION.to_owned()];
        let regist: ExtensionRegistry = ExtensionRegistry::new(iter);
        println!("hugr\n{}", h.store_str_with_exts(env_conf, &regist).unwrap());
    }
}
