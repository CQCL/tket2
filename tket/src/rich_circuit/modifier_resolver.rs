//! Try to delete modifier by applying the modifier to each component.
//!
pub mod call_modify;
pub mod dfg_modify;
pub mod global_phase_modify;
pub mod tket_op_modify;
use derive_more::Error;
use hugr::{
    builder::{BuildError, Dataflow},
    core::HugrNode,
    extension::{prelude::qb_t, simple_op::MakeExtensionOp},
    hugr::{hugrmut::HugrMut, patch::replace::ReplaceError},
    ops::{Const, ExtensionOp, LoadFunction, OpType},
    std_extensions::collections::array::array_type,
    types::{FuncTypeBase, Signature, Type},
    HugrView, IncomingPort, Node, OutgoingPort, Port, PortIndex, Wire,
};
use itertools::{Either, EitherOrBoth, Itertools};
use std::{collections::HashMap, iter, mem};

use super::{
    dagger::is_quantum_type, modifier_resolver::global_phase_modify::delete_phase, GlobalPhase,
    Modifier,
};
use crate::TketOp;

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
    // Some HashMap should be held here so that we remember such information.
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
        println!(" hugr: \n{}", new_dfg.hugr().mermaid_string());
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
        // FIXME: StateOrder is not preserved here.
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
                // TODO: We want some machinery to prevent generating a lot of controlled-U
                // for the same function U.
                (fn_node, load)
            }
            _ => {
                // TODO: Handle modifiers provided from other nodes.
                // For example, conditionals?
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

    // We take arbitral topological order of the circuit so that we can plug the control qubits
    // and pass around them in that order. This might not be ideal, as it may produce an inefficient order.
    fn modify_op(
        &mut self,
        h: &impl HugrMut<Node = N>,
        n: N,
        new_dfg: &mut impl Dataflow,
    ) -> Result<(), ModifierResolverErrors<N>> {
        let optype = h.get_optype(n);
        match optype {
            // Skip input/output nodes: it should be handled by its parent as it sets control qubits.
            OpType::Input(_) | OpType::Output(_) => {}

            // DFGs
            OpType::DFG(dfg) => self.modify_dfg(h, n, dfg, new_dfg)?,
            OpType::TailLoop(tail_loop) => self.modify_tail_loop(h, n, tail_loop, new_dfg)?,
            OpType::Conditional(conditional) => {
                self.modify_conditional(h, n, conditional, new_dfg)?
            }

            // Function calls
            OpType::Call(call) => self.modify_call(h, n, call, new_dfg)?,
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

            // Not applicable
            OpType::AliasDecl(_)
            | OpType::AliasDefn(_)
            | OpType::ExitBlock(_)
            | OpType::CFG(_)
            | OpType::DataflowBlock(_) => {
                return Err(ModifierError::ModifierNotApplicable(n, optype.clone()).into());
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

    fn need_swap(&self, t: EitherOrBoth<&Type>) -> bool {
        self.modifiers.dagger
            && matches!(t.both(), Some((t1, t2)) if is_quantum_type(t1) && is_quantum_type(t2))
    }

    fn modify_constant(
        &mut self,
        n: N,
        constant: &Const,
        new_dfg: &mut impl Dataflow,
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
        new_dfg: &mut impl Dataflow,
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
        if self.controls().len() != self.modifiers.control {
            return Err(ModifierResolverErrors::Unreachable(
                "Control qubits are not set correctly.".to_string(),
            ));
        }

        // If TketOp, return the modified operation.
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
            // We try to place the original operation.
            self.modify_dataflow_op(h, n, optype, new_dfg)
        }
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

    // Global phase is no more needed after resolving modifiers.
    delete_phase(h, entry_points.into_iter())?;

    Ok(())
}
