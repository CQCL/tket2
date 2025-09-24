//! Try to delete modifier by applying the modifier to each component.
//!
pub mod array_modify;
pub mod call_modify;
pub mod dfg_modify;
pub mod global_phase_modify;
pub mod tket_op_modify;
use derive_more::Error;
use hugr::{
    builder::{BuildError, CFGBuilder, Container, Dataflow, SubContainer},
    core::HugrNode,
    extension::{prelude::qb_t, simple_op::MakeExtensionOp},
    hugr::hugrmut::HugrMut,
    ops::{Const, ExtensionOp, OpType, CFG},
    std_extensions::collections::array::array_type,
    type_row,
    types::{EdgeKind, FuncTypeBase, Signature, Type},
    HugrView, IncomingPort, Node, OutgoingPort, Port, PortIndex, Wire,
};
use itertools::{Either, Itertools};
use std::{
    collections::{HashMap, VecDeque},
    iter, mem,
};

use super::{
    dagger::contain_quantum_type, modifier_resolver::global_phase_modify::delete_phase,
    GlobalPhase, Modifier,
};
use crate::TketOp;

/// An accumulated modifier that combines control, dagger, and power modifiers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    #[allow(dead_code)]
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
    new_dfg: &mut impl Container,
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
    modifiers: CombinedModifier,
    corresp_map: HashMap<DirWire<N>, Vec<DirWire>>,
    controls: Vec<Wire>,
    worklist: VecDeque<N>,
    call_map: HashMap<N, (Node, IncomingPort)>,
    // TODO:
    // Should keep track of the collection of modifiers that are applied to the same function.
    // This will prevent the duplicated generation of Controlled-functions.
    // Some HashMap should be held here so that we remember such information.
    _modified_functions: HashMap<N, (CombinedModifier, Node)>,
}

impl<N> ModifierResolver<N> {
    /// Create a new modifier resolver.
    pub fn new() -> Self {
        ModifierResolver {
            modifiers: CombinedModifier::default(),
            corresp_map: HashMap::default(),
            controls: Vec::default(),
            worklist: VecDeque::default(),
            call_map: HashMap::default(),
            _modified_functions: HashMap::default(),
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
    /// No caller of this modified function exists.
    #[display("No caller of the modified function exists for node {_0}")]
    NoCaller(N),
    /// No target of this modifer exists.
    #[display("No caller of the modified function exists for node {_0}")]
    NoTarget(N),
    /// Not the first modifier in a chain.
    #[display("Node {_0} is not the first modifier in a chain. It is called by {_0}")]
    NotInitialModifier(N, OpType),
    /// The modifier cannot be applied to the node.
    #[display("Modifier cannot be applied to the node {_0} of type {_1}")]
    ModifierNotApplicable(N, OpType),
}

impl<N> ModifierError<N> {
    fn node(self) -> N {
        match self {
            ModifierError::NotModifier(n, _)
            | ModifierError::NoCaller(n)
            | ModifierError::NoTarget(n)
            | ModifierError::NotInitialModifier(n, _)
            | ModifierError::ModifierNotApplicable(n, _) => n,
        }
    }
}

/// Possible errors that can occur during the modifier resolution process.
#[derive(Debug, derive_more::Display)]
pub enum ModifierResolverErrors<N = Node> {
    /// Cannot modify the node.
    ModifierError(ModifierError<N>),
    /// Error during the DFG build process.
    BuildError(BuildError),
    /// Unreachable error variant.
    Unreachable(String),
    /// Modifier applied to a node that cannot be modified.
    #[display("Modifier {_0} applied to the node {_1} cannot be modified")]
    UnResolvable(N, String, OpType),
    /// The node cannot be modified.
    #[display("Modification by {_0:?} is not defined for the node {_1}")]
    Unimplemented(Modifier, OpType),
}
impl<N> From<ModifierError<N>> for ModifierResolverErrors<N> {
    fn from(err: ModifierError<N>) -> Self {
        ModifierResolverErrors::ModifierError(err)
    }
}
impl<N> From<BuildError> for ModifierResolverErrors<N> {
    fn from(err: BuildError) -> Self {
        ModifierResolverErrors::BuildError(err)
    }
}

// Utility functions for ModifierResolver
impl<N: HugrNode> ModifierResolver<N> {
    fn modifiers_mut(&mut self) -> &mut CombinedModifier {
        &mut self.modifiers
    }
    fn modifiers(&self) -> &CombinedModifier {
        &self.modifiers
    }
    fn control_num(&self) -> usize {
        self.modifiers.control
    }
    fn controls(&mut self) -> &mut Vec<Wire> {
        &mut self.controls
    }
    fn controls_ref(&self) -> &Vec<Wire> {
        &self.controls
    }
    fn worklist(&mut self) -> &mut VecDeque<N> {
        &mut self.worklist
    }
    fn corresp_map(&mut self) -> &mut HashMap<DirWire<N>, Vec<DirWire>> {
        &mut self.corresp_map
    }
    fn call_map(&mut self) -> &mut HashMap<N, (Node, IncomingPort)> {
        &mut self.call_map
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

    /// Register a correspondence from old to new wire.
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

    /// Remember that old wire has no correspondence.
    fn map_insert_none(&mut self, old: DirWire<N>) -> Result<(), ModifierResolverErrors<N>> {
        self.corresp_map().insert(old.clone(), vec![]);
        // .map_or(Ok(()), |former| {
        //     Err(ModifierResolverErrors::Unreachable(format!(
        //         "Failed to forget port {}. [{},...] are already registered.",
        //         old, former[0]
        //     )))
        // })
        Ok(())
    }

    fn map_get(&self, key: &DirWire<N>) -> Result<&Vec<DirWire>, ModifierResolverErrors<N>> {
        self.corresp_map
            .get(key)
            .ok_or(ModifierResolverErrors::Unreachable(format!(
                "No correspondence for the wire: {}",
                key
            )))
    }

    fn forget_node(
        &mut self,
        h: &impl HugrView<Node = N>,
        n: N,
    ) -> Result<(), ModifierResolverErrors<N>> {
        for port in h.all_node_ports(n) {
            let dw = DirWire(n, port);
            self.map_insert_none(dw)?;
        }
        Ok(())
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
    fn add_node_control(&mut self, new_dfg: &mut impl Container, op: impl Into<OpType>) -> Node {
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
        new_dfg: &mut impl Container,
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
        new_dfg: &mut impl Container,
        parent: N,
    ) -> Result<(), ModifierResolverErrors<N>> {
        for (a, bs) in self.corresp_map().iter() {
            for b in bs {
                println!("  Wire mapping: {} -> {}", a, b);
            }
        }
        println!("before connect_all:\n{}", new_dfg.hugr().mermaid_string());
        for out_node in h.children(parent) {
            for out_port in h.node_outputs(out_node) {
                // TODO: ad hoc solution: ignore all StateOrder connections.
                if let Some(EdgeKind::StateOrder) = h.get_optype(out_node).port_kind(out_port) {
                    continue;
                }
                for (in_node, in_port) in h.linked_inputs(out_node, out_port) {
                    for a in self.map_get(&(in_node, in_port).into())? {
                        for b in self.map_get(&(out_node, out_port).into())? {
                            connect(new_dfg, a, b).map_err(|e| {
                                let (subgraph, result) = h.extract_hugr(parent);
                                println!("{}", subgraph.mermaid_string());
                                e
                            })?
                        }
                    }
                }
            }
        }
        // FIXME: StateOrder is not preserved here.
        Ok(())
    }
}

impl<N: HugrNode> ModifierResolver<N> {
    // FIXME: Shouldn't we check that there is a caller of the modified function?
    // We don't want to modify a function that is loaded and modified but never called.
    // When more than one modifier is chained, after the last modifier is resolved,
    // we delete the last modifier node, but the previous modifiers are not deleted.
    // If the second last modifier was only called by the last modifier, that will not be called anymore.
    fn verify(&self, h: &impl HugrView<Node = N>, n: N) -> Result<(), ModifierError<N>> {
        // Check if the node is a modifier, modifying an operation.
        let optype = h.get_optype(n);
        if Modifier::from_optype(optype).is_none() {
            return Err(ModifierError::NotModifier(n, optype.clone()));
        }
        // Check if this is the first modifier in a chain.
        if let Some((caller, _)) = h.linked_inputs(n, 0).exactly_one().ok() {
            let optype = h.get_optype(caller);
            if Modifier::from_optype(optype).is_some() {
                return Err(ModifierError::NotInitialModifier(caller, optype.clone()));
            }
        } else {
            return Err(ModifierError::NoCaller(n));
        }
        Ok(())
    }

    fn try_rewrite(
        &mut self,
        h: &mut impl HugrMut<Node = N>,
        n: N,
    ) -> Result<(), ModifierResolverErrors<N>> {
        // Verify that the rewrite can be applied.
        self.verify(h, n)?;

        // the ports that takes inputs from the modified function.
        let modified_fn_loader: Vec<(_, Vec<_>)> = h
            .node_outputs(n)
            .map(|p| (p, h.linked_inputs(n, p).collect()))
            .collect();

        // Modify the chain of modifiers.
        // Make sure that the modifiers are initially empty.
        let mut modifiers = CombinedModifier::default();
        mem::swap(self.modifiers_mut(), &mut modifiers);
        let new_load = self.apply_modifier_chain_to_loaded_fn(h, n)?;
        mem::swap(self.modifiers_mut(), &mut modifiers);

        // Connect the modified function to the inputs
        for (out_port, inputs) in modified_fn_loader {
            for (recv, recv_port) in inputs {
                h.disconnect(recv, recv_port);
                h.connect(new_load, out_port, recv, recv_port);
            }
        }

        Ok(())
    }

    /// Takes a signature and modifies it according to the combined modifier.
    pub fn modify_signature(&self, signature: &mut Signature, flatten: bool) {
        let FuncTypeBase { input, output } = signature;

        if flatten {
            let n = self.control_num();
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

    // We take arbitral topological order of the circuit so that we can plug the control qubits
    // and pass around them in that order. This might not be ideal, as it may produce an inefficient order.
    fn modify_op(
        &mut self,
        h: &mut impl HugrMut<Node = N>,
        n: N,
        new_dfg: &mut impl Dataflow,
    ) -> Result<(), ModifierResolverErrors<N>> {
        let optype = &h.get_optype(n).clone();
        match optype {
            // Skip input/output nodes: it should be handled by its parent as it sets control qubits.
            OpType::Input(_) | OpType::Output(_) => {}

            // CFG
            OpType::CFG(cfg) => self.modify_cfg(h, n, cfg, new_dfg)?,

            // DFGs
            OpType::DFG(dfg) => self.modify_dfg(h, n, dfg, new_dfg)?,
            OpType::TailLoop(tail_loop) => self.modify_tail_loop(h, n, tail_loop, new_dfg)?,
            OpType::Conditional(conditional) => {
                self.modify_conditional(h, n, conditional, new_dfg)?
            }

            // Function calls
            OpType::Call(_) => self.modify_call(h, n, optype, new_dfg)?,
            OpType::CallIndirect(indir_call) => {
                self.modify_indirect_call(h, n, indir_call, new_dfg)?
            }
            OpType::LoadFunction(load) => self.modify_load_function(h, n, load, new_dfg)?,

            // Operations
            OpType::ExtensionOp(_) => {
                self.modify_extension_op(h, n, optype, new_dfg)?;
            }
            OpType::Const(constant) => {
                self.modify_constant(n, constant, new_dfg)?;
            }
            OpType::LoadConstant(_) | OpType::OpaqueOp(_) | OpType::Tag(_) => {
                self.modify_dataflow_op(h, n, optype, new_dfg)?
            }

            // Unreachable
            OpType::FuncDefn(_) | OpType::FuncDecl(_) | OpType::Module(_) => {
                return Err(ModifierResolverErrors::Unreachable(format!(
                    "Invalid node found inside modified function (OpType = {})",
                    optype.clone()
                )))
            }
            OpType::Case(_) => {
                return Err(ModifierResolverErrors::Unreachable(
                    "Case cannot be directly modified.".to_string(),
                ))
            }
            // Q. `ExitBlock | DataflowBlock` perhaps should be here?

            // Not resolvable
            OpType::AliasDecl(_)
            | OpType::AliasDefn(_)
            | OpType::ExitBlock(_)
            | OpType::DataflowBlock(_) => {
                return Err(ModifierResolverErrors::UnResolvable(
                    n,
                    "Unmodifiable node found".to_string(),
                    optype.clone(),
                )
                .into());
            }
            _ => {
                // Q. Maybe we should just ignore unknown operations?
                return Err(ModifierResolverErrors::Unreachable(
                    "Unknown operation type found in the modifier resolver.".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// This function registers the correspondence of the ports of the old node to the new node.
    /// If the dagger is not applied, the ports are mapped directly.
    /// If the dagger is applied, the quantum input/output ports are swapped.
    /// Inputs:
    /// * `n`: the old node
    /// * `node`: the new node
    /// * `inputs`/`outputs`: the types of the input/output ports of the old node
    /// * `input_offset`/`output_offset`: the offset of the ports of the old and new node
    ///   - e.g., for IndirectCall, the first input port is the loaded function, which we want to ignore here.
    ///     So the `input_offset` is 1.
    /// * `new_offset`: the offset of the ports of the new node, especially the number of control qubits.
    ///
    /// The order of the resulting ports is determined as follows:
    /// - Every ports are devided into quantum ports and non-quantum ports.
    /// - Until the first quantum port is reached, the non-quantum ports are wired in order.
    /// - When a quantum port is reached for both inputs and outputs,
    ///   if the dagger is applied, the quantum input is wired to the output,
    ///   and the quantum output is wired to the input until they reaches the next non-quantum port.
    /// - This is repeated until all ports are wired.
    ///
    /// For example, if the input types are `[qubit, int, qubit, qubit, int]` and
    /// the output types are `[qubit, array[qubit, _]]`,
    /// and the dagger is applied, the wiring is as follows:
    /// - input: [out0:qubit, in1:int, out1:array[qubit, _], in4:int]
    /// - output: [in0:qubit, in2:qubit, in3:qubit]
    ///
    /// FIXME: This reverses everything that can contain qubits, which might not be intended.
    /// TODO: Handle state order edges.
    fn wire_node_inout<'a>(
        &mut self,
        n: N,
        node: Node,
        inputs: impl Iterator<Item = &'a Type>,
        outputs: impl Iterator<Item = &'a Type>,
        input_offset: usize,
        output_offset: usize,
        new_offset: usize,
    ) -> Result<(), ModifierResolverErrors<N>> {
        self.wire_inout(
            n,
            n,
            node,
            node,
            inputs,
            outputs,
            input_offset,
            output_offset,
            new_offset,
        )
    }

    fn wire_inout<'a>(
        &mut self,
        old_in: N,
        old_out: N,
        new_in: Node,
        new_out: Node,
        mut inputs: impl Iterator<Item = &'a Type>,
        mut outputs: impl Iterator<Item = &'a Type>,
        input_offset: usize,
        output_offset: usize,
        new_offset: usize,
    ) -> Result<(), ModifierResolverErrors<N>> {
        let mut old_in_wire = (old_in, IncomingPort::from(input_offset)).into();
        let mut old_out_wire = (old_out, OutgoingPort::from(output_offset)).into();
        let mut new_in_wire = (new_in, IncomingPort::from(input_offset + new_offset)).into();
        let mut new_out_wire = (new_out, OutgoingPort::from(output_offset + new_offset)).into();
        let mut in_ty = inputs.next();
        let mut out_ty = outputs.next();

        loop {
            // Wire inputs until the first quantum type
            while let Some(ty) = in_ty {
                if contain_quantum_type(ty) {
                    break;
                }
                self.map_insert(old_in_wire, new_in_wire)?;
                old_in_wire = old_in_wire.shift(1);
                new_in_wire = new_in_wire.shift(1);
                in_ty = inputs.next();
            }

            // Wire outputs until the first quantum type
            while let Some(ty) = out_ty {
                if contain_quantum_type(ty) {
                    break;
                }
                self.map_insert(old_out_wire, new_out_wire)?;
                old_out_wire = old_out_wire.shift(1);
                new_out_wire = new_out_wire.shift(1);
                out_ty = outputs.next();
            }

            // If both are quantum types
            while let Some(ty) = in_ty {
                if !contain_quantum_type(ty) {
                    break;
                }
                let new_in = if !self.modifiers.dagger {
                    let new_in = new_in_wire;
                    new_in_wire = new_in_wire.shift(1);
                    new_in
                } else {
                    let new_in = new_out_wire;
                    new_out_wire = new_out_wire.shift(1);
                    new_in
                };
                self.map_insert(old_in_wire, new_in)?;
                old_in_wire = old_in_wire.shift(1);
                in_ty = inputs.next();
            }
            while let Some(ty) = out_ty {
                if !contain_quantum_type(ty) {
                    break;
                }
                let new_out = if !self.modifiers.dagger {
                    let new_out = new_out_wire;
                    new_out_wire = new_out_wire.shift(1);
                    new_out
                } else {
                    let new_out = new_in_wire;
                    new_in_wire = new_in_wire.shift(1);
                    new_out
                };
                self.map_insert(old_out_wire, new_out)?;
                old_out_wire = old_out_wire.shift(1);
                out_ty = outputs.next();
            }

            if in_ty.is_none() && out_ty.is_none() {
                break;
            }
        }
        Ok(())
    }

    fn modify_constant(
        &mut self,
        n: N,
        constant: &Const,
        new_dfg: &mut impl Container,
    ) -> Result<(), ModifierResolverErrors<N>> {
        let output = new_dfg.add_child_node(constant.clone());
        self.map_insert(Wire::new(n, 0).into(), Wire::new(output, 0).into())
    }

    /// Copy the dataflow operation to the new function.
    /// These are the operations that are not modified by the modifier.
    fn modify_dataflow_op(
        &mut self,
        h: &impl HugrMut<Node = N>,
        n: N,
        optype: &OpType,
        new_dfg: &mut impl Container,
    ) -> Result<(), ModifierResolverErrors<N>> {
        self.add_node_no_modification(new_dfg, optype.clone(), h, n)?;
        Ok(())
    }

    fn modify_extension_op(
        &mut self,
        h: &impl HugrMut<Node = N>,
        n: N,
        optype: &OpType,
        new_dfg: &mut impl Dataflow,
    ) -> Result<(), ModifierResolverErrors<N>> {
        if self.controls().len() != self.control_num() {
            return Err(ModifierResolverErrors::Unreachable(
                "Control qubits are not set correctly.".to_string(),
            ));
        }

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
        } else if Modifier::from_optype(optype).is_some() {
            // TODO: check if this is ok.
            self.forget_node(h, n)
        } else if self.modify_array_op(h, n, optype, new_dfg)? {
            Ok(())
        } else if self.try_array_convert(h, n, optype, new_dfg)? {
            Ok(())
        } else {
            // Some other Hugr extension operation.
            // Here, we do not know what is the modified version.
            // We try to place the original operation.
            self.modify_dataflow_op(h, n, optype, new_dfg)
        }
    }

    /// Temporary implementation of CFG modification.
    /// This function expects that the CFG contains only one block.
    /// If not, it returns an error.
    fn modify_cfg(
        &mut self,
        h: &mut impl HugrMut<Node = N>,
        n: N,
        cfg: &CFG,
        new_dfg: &mut impl Container,
    ) -> Result<(), ModifierResolverErrors<N>> {
        // Check if the CFG contains only one block.
        let children: Vec<N> = h
            .children(n)
            .filter(|child| h.get_optype(*child).is_dataflow_block())
            .collect();
        if children.len() != 1 {
            return Err(ModifierResolverErrors::UnResolvable(
                n,
                "CFG with more than one node found.".to_string(),
                cfg.clone().into(),
            ));
        }
        let old_bb = children[0];

        let mut signature = cfg.signature.clone();
        self.modify_signature(&mut signature, true);
        let mut new_cfg = CFGBuilder::new(signature.clone())?;
        let mut new_bb = new_cfg.entry_builder([type_row![]], signature.output.clone())?;
        self.modify_dfg_body(h, old_bb, &mut new_bb)?;
        let bb_id = new_bb.finish_sub_container()?;
        new_cfg.branch(&bb_id, 0, &new_cfg.exit_block())?;

        let new = self.insert_sub_dfg(new_dfg, new_cfg)?;

        // connect the controls and register the IOs
        for (i, c) in self.controls().iter_mut().enumerate() {
            new_dfg.hugr_mut().connect(c.node(), c.source(), new, i);
            *c = Wire::new(new, i);
        }
        let offset = self.control_num();
        self.wire_node_inout(
            n,
            new,
            signature.input.iter(),
            signature.output.iter(),
            0,
            0,
            offset,
        )?;

        Ok(())
    }
}

/// Resolve modifiers in a circuit by applying them to each entry point.
//
// Shouldn't we use a worklist of nodes?
// As we may want to change the order of resolving modifiers
// but might want to rollback if the second last one is called in a different path,
// this may be needed.
pub fn resolve_modifier_with_entrypoints(
    h: &mut impl HugrMut<Node = Node>,
    entry_points: impl IntoIterator<Item = Node>,
) -> Result<(), ModifierResolverErrors<Node>> {
    use ModifierResolverErrors::*;

    println!("before modification:\n{}", h.mermaid_string());

    let entry_points: Vec<_> = entry_points.into_iter().collect();

    let mut resolver = ModifierResolver::new();
    let mut worklist = entry_points.into_iter().collect::<VecDeque<_>>();
    let mut visited = vec![];
    while let Some(node) = worklist.pop_front() {
        if !h.contains_node(node) || visited.contains(&node) {
            continue;
        }
        worklist.extend(h.children(node).filter(|n| !visited.contains(n)));
        worklist.extend(h.all_neighbours(node).filter(|n| !visited.contains(n)));
        visited.push(node);
        match resolver.try_rewrite(h, node) {
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

    // Global phase is no more needed after resolving modifiers.
    // delete_phase(h, entry_points.clone().into_iter())?;
    delete_phase(h, vec![h.module_root()])?;

    println!("After modifier resolution:\n{}", h.mermaid_string());
    // TODO: ad hoc cleanup of remaining modifer nodes.
    let entry_points = vec![h.module_root()];
    for entry_point in entry_points.clone() {
        let descendants = h.descendants(entry_point).collect::<Vec<_>>();
        for node in descendants {
            if !h.contains_node(node) {
                continue;
            }
            let optype = h.get_optype(node);
            if Modifier::from_optype(optype).is_some() {
                let mut l = vec![node];
                while let Some(n) = l.pop() {
                    l.extend(h.output_neighbours(n));
                    h.remove_node(n);
                }
            }
        }
    }

    print!("Resulting circuit:\n{}", h.mermaid_string());

    {
        use hugr::{
            envelope::{EnvelopeConfig, EnvelopeFormat},
            extension::ExtensionRegistry,
        };
        use std::{fs::File, path::Path};

        use crate::{
            extension::{bool::BOOL_EXTENSION, rotation::ROTATION_EXTENSION, TKET_EXTENSION},
            rich_circuit::*,
        };

        let env_format = EnvelopeFormat::PackageJson;
        let env_conf: EnvelopeConfig = EnvelopeConfig::new(env_format);
        let iter: Vec<Arc<Extension>> = vec![
            ROTATION_EXTENSION.to_owned(),
            TKET_EXTENSION.to_owned(),
            BOOL_EXTENSION.to_owned(),
        ];
        let regist: ExtensionRegistry = ExtensionRegistry::new(iter);
        let f = File::create(Path::new("HUGR.json")).unwrap();
        let writer = std::io::BufWriter::new(f);
        h.extract_hugr(h.module_root())
            .0
            .store_with_exts(writer, env_conf, &regist)
            .unwrap();
    }

    h.validate()
        .map_err(|e| ModifierResolverErrors::BuildError(e.into()))?;
    print!("Validated!");

    Ok(())
}
