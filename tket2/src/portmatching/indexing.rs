//! Indexing schemes for pattern matching with portmatching.
//!
//! Indexing schemes assign a unique variable name (keys) for every element
//! in the hugr that is being matched. We currently match dataflow wires
//! (uniquely identified by their outgoing port) and nodes.
//!
//! The important part of indexing schemes is that the variable names are
//! "canonical" in some sense: that is, multiple patterns that match on an
//! overlapping part of the hugr should as much as possible be using the same
//! variable names to refer to the same nodes and wires. This maximises the
//! overlaps between sets of constraints, defined using these variable names,
//! across patterns.
//!
//! ## Current limitations
//! Currently, the indexing scheme assumes that every wire is connected to  a
//! unique outgoing port. This is the case in DFGs, but may not handle arbitrary
//! CFGs. There is in principle no reason the scheme could not be extended to
//! handle more general graphs, as well as hierarchy.

use derive_more::{Display, Error, From, TryInto};
use hugr::{HugrView, PortIndex};
use itertools::{Either, Itertools};
pub(super) use map::HugrBindMap;
use portmatching::{self as pm, BindMap};

use crate::Circuit;

mod map;
mod path;
mod unary_packed;
pub(super) use path::HugrPath;

////////////////////////////////////////////////////////////////////////////////
//////////////////// Variable Naming scheme used for Hugrs /////////////////////
////////////////////////////////////////////////////////////////////////////////

/// Symbols used as keys that bind to hugr entities at pattern matching time.
///
/// A "match" is then a mapping from these keys to values in the hugr being
/// matched.
#[derive(
    Clone,
    Copy,
    Debug,
    Hash,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    serde::Serialize,
    serde::Deserialize,
    From,
    TryInto,
)]
pub enum HugrVariableID {
    /// A variable that binds to a hugr node.
    #[from]
    #[try_into]
    Op(HugrNodeID),
    /// A variable that binds to a copyable hugr wire.
    ///
    /// As a convention, we identify a wire by the outgoing port it is attached
    /// to, but any other canonical choice would work too.
    CopyableWire(HugrPortID),
    /// A variable that binds to a linear hugr wire.
    ///
    /// It is useful to distinguish between linear and copyable wires, as
    /// constraints on linear types have stronger guarantees (e.g. there may
    /// be at most one incoming port connected to it).
    ///
    /// As a convention, we identify a wire by the outgoing port it is attached
    /// to, but any other canonical choice would work too.
    LinearWire(HugrPortID),
}

impl TryFrom<HugrVariableID> for HugrPortID {
    type Error = UnexpectedVariableType;

    fn try_from(value: HugrVariableID) -> Result<Self, Self::Error> {
        use HugrVariableID::*;
        match value {
            Op(_) => Err(UnexpectedVariableType::Node),
            CopyableWire(port) | LinearWire(port) => Ok(port),
        }
    }
}

/// The hugr entities that variables can be bound to, either a node or a wire.
/// Wires are represented by the unique outgoing port on that wire.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum HugrVariableValue {
    /// The value of a HugrVariableID::Node variable.
    Node(hugr::Node),
    /// The value of a HugrVariableID::{Copyable, Linear}Wire variable,
    /// identified by the outgoing port.
    Wire(hugr::Wire),
}

impl HugrVariableValue {
    /// Create a new wire value from a node and an outgoing port.
    pub fn new_wire_from_outgoing(node: hugr::Node, port: impl Into<hugr::OutgoingPort>) -> Self {
        Self::Wire(hugr::Wire::new(node, port))
    }

    /// Create a new wire value from a node and an incoming port.
    ///
    /// This requires a reference to the underlying hugr to find the unique
    /// outgoing port of the wire.
    pub fn new_wire_from_incoming(
        node: hugr::Node,
        port: impl Into<hugr::IncomingPort>,
        hugr: &impl HugrView,
    ) -> Self {
        let (out_node, out_port) = hugr
            .single_linked_output(node, port)
            .expect("a unique source for every wire");
        Self::new_wire_from_outgoing(out_node, out_port)
    }
}

impl From<(hugr::Node, hugr::OutgoingPort)> for HugrVariableValue {
    fn from((node, port): (hugr::Node, hugr::OutgoingPort)) -> Self {
        HugrVariableValue::new_wire_from_outgoing(node, port)
    }
}

impl From<hugr::Wire> for HugrVariableValue {
    fn from(wire: hugr::Wire) -> Self {
        HugrVariableValue::new_wire_from_outgoing(wire.node(), wire.source())
    }
}

impl From<hugr::Node> for HugrVariableValue {
    fn from(node: hugr::Node) -> Self {
        HugrVariableValue::Node(node)
    }
}

/// A port variable ID, given by a node ID and a port offset.
#[derive(
    Clone, Copy, Debug, Eq, PartialOrd, Ord, Hash, PartialEq, serde::Serialize, serde::Deserialize,
)]
pub struct HugrPortID {
    /// The node that the port belongs to.
    pub node: HugrNodeID,
    /// The port index.
    pub port: hugr::Port,
}

impl HugrPortID {
    /// Create a new port ID from a hugr node and a port.
    pub fn new(node: HugrNodeID, port: hugr::Port) -> Self {
        Self { node, port }
    }

    /// Find the wire the port is on given `bindings` that contains a binding
    /// for self.node in the hugr.
    fn bound_wire(
        &self,
        bindings: &HugrBindMap,
        hugr: &impl HugrView,
    ) -> Option<HugrVariableValue> {
        let node = bindings.get_node(self.node)?;
        match self.port.as_directed() {
            Either::Left(in_port) if hugr.num_inputs(node) > in_port.index() => {
                let (node, out_port) = hugr.single_linked_output(node, in_port)?;
                Some(HugrVariableValue::new_wire_from_outgoing(node, out_port))
            }
            Either::Right(out_port) if hugr.num_outputs(node) > out_port.index() => {
                Some(HugrVariableValue::new_wire_from_outgoing(node, out_port))
            }
            _ => None,
        }
    }
}

/// A node variable ID, given based on a unique path from the root node.
#[derive(
    Clone,
    Copy,
    Debug,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
    From,
)]
pub struct HugrNodeID {
    /// The path from the root node to the node.
    ///
    /// An empty path is the root node
    path_from_root: HugrPath,
}

impl HugrNodeID {
    pub(super) fn new(path_from_root: HugrPath) -> Self {
        Self { path_from_root }
    }

    #[cfg(test)]
    pub(super) fn root() -> Self {
        Self::new(HugrPath::empty())
    }

    fn is_root(&self) -> bool {
        self.path_from_root.is_empty()
    }
}

/// Conversion error of HugrVariableValue to Hugr value
#[derive(Debug, Error, Display)]
pub enum UnexpectedValueType {
    /// Unexpected value type: Node
    #[display("unexpected value type: Node")]
    Node,
    /// Unexpected value type: OutgoingPort
    #[display("unexpected value type: OutgoingPort")]
    OutgoingPort,
}

/// Conversion error of HugrVariableID to a variable type
#[derive(Debug, Error, Display)]
pub enum UnexpectedVariableType {
    /// Unexpected value type: Node
    #[display("unexpected variable type: Node")]
    Node,
    /// Unexpected value type: CopyableWire
    #[display("unexpected variable type: CopyableWire")]
    CopyableWire,
    /// Unexpected value type: LinearWire
    #[display("unexpected variable type: LinearWire")]
    LinearWire,
}

impl TryFrom<HugrVariableValue> for (hugr::Node, hugr::OutgoingPort) {
    type Error = UnexpectedValueType;

    fn try_from(value: HugrVariableValue) -> Result<Self, Self::Error> {
        match value {
            HugrVariableValue::Node(..) => Err(UnexpectedValueType::Node),
            HugrVariableValue::Wire(wire) => Ok((wire.node(), wire.source())),
        }
    }
}

impl TryFrom<HugrVariableValue> for hugr::Node {
    type Error = UnexpectedValueType;

    fn try_from(value: HugrVariableValue) -> Result<Self, Self::Error> {
        match value {
            HugrVariableValue::Node(node) => Ok(node),
            HugrVariableValue::Wire(..) => Err(UnexpectedValueType::OutgoingPort),
        }
    }
}

impl TryFrom<HugrVariableValue> for hugr::Wire {
    type Error = UnexpectedValueType;

    fn try_from(value: HugrVariableValue) -> Result<Self, Self::Error> {
        match value {
            HugrVariableValue::Node(..) => Err(UnexpectedValueType::Node),
            HugrVariableValue::Wire(wire) => Ok(wire),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
////////////// Indexing scheme: resolve variable IDs to values /////////////////
////////////////////////////////////////////////////////////////////////////////

/// An indexing scheme for hugrs
///
/// This relies on a variable naming scheme based on paths through the hugr,
/// starting from some root node. Such a path is given by a sequence of
/// port offsets, indicating how the hugr is traversed. It supports copyable
/// wires, in which case the path may not be unique and thus resolve to multiple
/// nodes.
///
/// ## How variable binding works
/// The indexing scheme binds variables to nodes and wires in a hugr. It
/// proceeds recursively:
///  - port variable are bound based on the binding of the node they are
///    attached to,
///  - node variables are bound based on the binding of their parent node, i.e.
///    the previous node on the variale path from the root to the node,
///  - the root node can be bound to any node in the hugr,
///
/// Any binding may fail if the specified node or port does not exist in a
/// given hugr.
///
/// ## Non-unique variable bindings
/// Given a binding for the previous node variable, a node variable can have as
/// many possible bindings as there are wires connected to the previous node at
/// the specified port. E.g if the variable path specifies that the last node
/// is reached from its parent node through OutgoingPort(1), then the node can
/// be bound to any node that is connected to the parent node at that port. If
/// the wire is linear or the specified port is an incoming port, then the
/// binding is unique (if it exists).
///
/// ## Current limitations
/// Currently, the indexing scheme assumes that every incoming wire is connected to  a
/// unique outgoing port. This is the case in DFGs, but may not handle arbitrary
/// CFGs.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct HugrIndexingScheme;

/// For every variable, this specifies the variables that must be bound first,
/// before attempting to bind it.
impl pm::IndexingScheme for HugrIndexingScheme {
    type BindMap = HugrBindMap;
    type Key = <HugrBindMap as BindMap>::Key;
    type Value = <HugrBindMap as BindMap>::Value;

    fn required_bindings(&self, key: &HugrVariableID) -> Vec<HugrVariableID> {
        use HugrVariableID::*;

        match key {
            Op(node) if node.is_root() => {
                // The root node can be bound to any node.
                vec![]
            }
            Op(node) => {
                let parent_path = node.path_from_root.parent().expect("not empty");
                vec![HugrNodeID::new(parent_path).into()]
            }
            CopyableWire(port) | LinearWire(port) => vec![port.node.into()],
        }
    }
}

/// How variable names are resolved to hugr nodes and wires for particular
/// circuits.
///
/// This is currently limited to [Circuit]s, but could in the future be extended
/// to any [HugrView].
impl<H: HugrView> pm::IndexedData<HugrVariableID> for Circuit<H> {
    type IndexingScheme = HugrIndexingScheme;
    type Value = <HugrIndexingScheme as pm::IndexingScheme>::Value;
    type BindMap = <HugrIndexingScheme as pm::IndexingScheme>::BindMap;

    fn bind_options(
        &self,
        key: &HugrVariableID,
        known_bindings: &HugrBindMap,
    ) -> Vec<HugrVariableValue> {
        use HugrVariableID::*;
        match key {
            Op(node) if node.is_root() => {
                // Every hugr node is a valid binding for the root node.
                self.commands().map(|cmd| cmd.node().into()).collect()
            }
            Op(node) => {
                let (parent_path, port, _) = node.path_from_root.split_back().expect("not empty");
                let parent_node_id = HugrNodeID::new(parent_path);
                let Some(parent) = known_bindings.get_node(parent_node_id) else {
                    return vec![];
                };
                follow_port(parent, port, self.hugr()).map_into().collect()
            }
            CopyableWire(port) | LinearWire(port) => {
                let Some(val) = port.bound_wire(known_bindings, self.hugr()) else {
                    return vec![];
                };
                vec![val]
            }
        }
    }
}

/// Iterator over the nodes reachable from `parent` through `port`
fn follow_port(
    parent: hugr::Node,
    port: hugr::Port,
    hugr: &impl HugrView,
) -> impl Iterator<Item = hugr::Node> + '_ {
    (port.index() < hugr.num_ports(parent, port.direction()))
        .then(|| hugr.linked_ports(parent, port).map(|(n, _)| n))
        .into_iter()
        .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extension::rotation::rotation_type;
    use crate::Tk2Op;
    use hugr::builder::{DFGBuilder, Dataflow, DataflowHugr};
    use hugr::extension::prelude::qb_t;
    use hugr::ops::handle::NodeHandle;
    use hugr::types::Signature;
    use hugr::{Direction, Port};
    use portmatching::{IndexedData, IndexingScheme};

    fn create_test_circuit() -> (Circuit<impl HugrView>, hugr::Node) {
        let mut hrz =
            DFGBuilder::new(Signature::new(vec![qb_t(), rotation_type()], vec![qb_t()])).unwrap();
        let [q_in, angle_in] = hrz.input_wires_arr();

        let res = hrz.add_dataflow_op(Tk2Op::H, [q_in]).unwrap();
        let q = res.out_wire(0);

        let res = hrz.add_dataflow_op(Tk2Op::Rz, [q, angle_in]).unwrap();
        let fst_rz = res.node();
        let q = res.out_wire(0);

        let res = hrz.add_dataflow_op(Tk2Op::Rz, [q, angle_in]).unwrap();
        let q = res.out_wire(0);

        let res = hrz.finish_hugr_with_outputs([q]).unwrap().into();
        (res, fst_rz)
    }

    #[test]
    fn test_bind_options() {
        let (circuit, root_choice) = create_test_circuit();
        let mut known_bindings = HugrBindMap::new();
        let scheme = HugrIndexingScheme;

        // Test for root node
        assert!(scheme
            .required_bindings(&HugrVariableID::Op(HugrNodeID::root()))
            .is_empty());
        let root_options =
            circuit.bind_options(&HugrVariableID::Op(HugrNodeID::root()), &known_bindings);
        assert_eq!(root_options.len(), 3); // H gate, and 2x Rz nodes

        // Bind the root node
        known_bindings
            .bind(
                HugrVariableID::Op(HugrNodeID::root()),
                HugrVariableValue::Node(root_choice),
            )
            .unwrap();

        // Test for incoming port
        let ports = [
            (Port::new(Direction::Outgoing, 0), 0usize),
            (Port::new(Direction::Incoming, 1), 0),
            (Port::new(Direction::Outgoing, 1), 1),
        ];

        // The first two ports along the path should be unique
        let input_node = circuit.input_node();
        let optypes = [
            Tk2Op::Rz.into(),
            circuit.hugr().get_optype(input_node).clone(),
        ];
        for i in 1..=2 {
            let path = HugrPath::try_from(&ports[..i]).unwrap();
            let var: HugrVariableID = HugrNodeID {
                path_from_root: path,
            }
            .into();
            assert_eq!(scheme.required_bindings(&var).len(), 1);
            let options = circuit.bind_options(&var, &known_bindings);
            let sole_option = options.into_iter().exactly_one().unwrap();
            let &HugrVariableValue::Node(n) = &sole_option else {
                panic!("Expected node");
            };
            assert_eq!(circuit.hugr().get_optype(n), &optypes[i - 1]);
            known_bindings.bind(var, sole_option).unwrap();
        }

        // The last port along the path should have two options
        // (the two uses of the rotation angle)
        let path = HugrPath::try_from(&ports as &[_]).unwrap();
        let var: HugrVariableID = HugrNodeID::new(path).into();
        let options = circuit.bind_options(&var, &known_bindings);
        assert_eq!(options.len(), 2);
        for option in &options {
            let &HugrVariableValue::Node(n) = option else {
                panic!("Expected node");
            };
            assert_eq!(circuit.hugr().get_optype(n), &Tk2Op::Rz.into());
            if n != root_choice {
                // Bind the second Rz node
                known_bindings.bind(var, *option).unwrap();
            }
        }
        assert!(known_bindings.get_node(HugrNodeID::new(path)).is_some());

        // Test for ports
        let ports = [
            Port::new(Direction::Outgoing, 0),
            Port::new(Direction::Incoming, 0),
        ];
        for port in ports {
            let port_var = HugrVariableID::CopyableWire(HugrPortID {
                node: HugrNodeID::new(path),
                port,
            });
            let sole_option = circuit
                .bind_options(&port_var, &known_bindings)
                .into_iter()
                .exactly_one()
                .expect("expected exactly one option");
            match port.direction() {
                Direction::Incoming => {
                    let &HugrVariableValue::Wire(wire) = &sole_option else {
                        panic!("Expected wire");
                    };
                    // (n, p) is opposite `port`, so at the previous rotation
                    assert_eq!(wire.node(), root_choice);
                    assert_eq!(wire.source(), 0.into());
                }
                Direction::Outgoing => {
                    let &HugrVariableValue::Wire(wire) = &sole_option else {
                        panic!("Expected outgoing port");
                    };
                    assert_eq!(circuit.hugr().get_optype(wire.node()), &Tk2Op::Rz.into());
                    assert_eq!(wire.source(), port.as_outgoing().unwrap());
                    assert_ne!(wire.node(), root_choice);
                    assert_eq!(
                        circuit
                            .hugr()
                            .linked_inputs(wire.node(), wire.source())
                            .count(),
                        1
                    );
                }
            }
        }
    }
}
