//! Indexing schemes for pattern matching with portmatching.
//!
//! Indexing schemes assign a unique variable names (keys) for every element
//! in the hugr that is being matched. We currently match dataflow wires
//! (uniquely identified by their outgoing port) and nodes.
//!
//! The important part of indexing schemes is that the variable names are
//! "canonical" in some sense: that is, multiple patterns that overlap should
//! as much as possible be using the same variable names. Thus, the overlap
//! between constraints between different patterns is maximised.

use derive_more::{Display, Error, From};
use hugr::{HugrView, PortIndex};
use itertools::{Either, Itertools};
pub(super) use map::HugrBindMap;
use portmatching::{self as pm, BindMap};

use crate::Circuit;

mod map;
mod path;
pub(super) use path::HugrPath;

////////////////////////////////////////////////////////////////////////////////
//////////////////// Variable Naming scheme used for Hugrs /////////////////////
////////////////////////////////////////////////////////////////////////////////

/// Symbols used as keys that bind to hugr entities at pattern matching time.
///
/// A "match" is then a mapping from these keys to values in the hugr being
/// matched.
#[derive(
    Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum HugrVariableID {
    /// A variable that binds to a hugr node.
    Op(HugrNodeID),
    /// A variable that binds to a copyable hugr value.
    ///
    /// We identify a wire by the "smallest" port it is attached to, according
    /// to some canonical port ordering.
    CopyableWire(HugrPortID),
    /// A variable that binds to a linear hugr value.
    ///
    /// It is useful to distinguish between linear and copyable wires, as
    /// constraints on linear types have stronger guarantees (e.g. there may
    /// be at most one incoming port connected to it).
    ///
    /// We identify a wire by the "smallest" port it is attached to, according
    /// to some canonical port ordering.
    LinearWire(HugrPortID),
}

impl From<HugrNodeID> for HugrVariableID {
    fn from(node: HugrNodeID) -> Self {
        Self::Op(node)
    }
}

impl HugrVariableID {
    fn path(&self) -> HugrPath {
        match self {
            HugrVariableID::Op(node) => node.path_from_root,
            HugrVariableID::CopyableWire(port) => port.node.path_from_root,
            HugrVariableID::LinearWire(port) => port.node.path_from_root,
        }
    }
}

/// The values that variables can be bound to, either a node or a wire,
/// represented by the unique outgoing port on that wire.
///
/// This representation assumes that there is always a unique outgoing port for
/// every wire.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum HugrVariableValue {
    /// The value of a HugrVariableID::Node variable.
    Node(hugr::Node),
    /// The value of a HugrVariableID::{Copyable, Linear}Wire variable,
    /// identified by the outgoing port.
    Wire(hugr::Wire),
}

impl HugrVariableValue {
    pub(crate) fn new_wire_from_source(
        node: hugr::Node,
        port: impl Into<hugr::OutgoingPort>,
    ) -> Self {
        Self::Wire(hugr::Wire::new(node, port))
    }

    pub(crate) fn new_wire_from_sink(
        node: hugr::Node,
        port: impl Into<hugr::IncomingPort>,
        hugr: &impl HugrView,
    ) -> Self {
        let (out_node, out_port) = find_source(node, port.into(), hugr).unwrap();
        Self::new_wire_from_source(out_node, out_port)
    }

    pub(crate) fn new_node(node: hugr::Node) -> Self {
        Self::Node(node)
    }
}

pub(super) fn find_source(
    node: hugr::Node,
    port: hugr::IncomingPort,
    hugr: &impl HugrView,
) -> Option<(hugr::Node, hugr::OutgoingPort)> {
    if hugr.num_inputs(node) <= port.index() {
        return None;
    }
    let res = hugr
        .linked_outputs(node, port)
        .exactly_one()
        .ok()
        .expect("indexing does not handle wires with zero or multiple outputs");
    Some(res)
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
    pub fn new(node: HugrNodeID, port: hugr::Port) -> Self {
        Self { node, port }
    }

    /// Resolve a port ID given `bindings` that contains a binding for the node.
    ///
    /// Returns a HugrVariableValue::OutgoingPort variant. If the port is an
    /// incoming port, then the connected outgoing port is returned.
    fn resolve(&self, bindings: &HugrBindMap, hugr: &impl HugrView) -> Option<HugrVariableValue> {
        let node = bindings.get_node(self.node)?;
        match self.port.as_directed() {
            Either::Left(in_port) => {
                let (node, out_port) = find_source(node, in_port, hugr)?;
                Some(HugrVariableValue::new_wire_from_source(node, out_port))
            }
            Either::Right(out_port) if hugr.num_outputs(node) > out_port.index() => {
                Some(HugrVariableValue::new_wire_from_source(node, out_port))
            }
            _ => None,
        }
    }
}

/// A node variable ID, given based on a unique path from the root node.
#[derive(
    Clone, Copy, Debug, Eq, PartialOrd, Ord, Hash, PartialEq, serde::Serialize, serde::Deserialize,
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

    pub(super) fn root() -> Self {
        Self::new(HugrPath::empty())
    }
}

impl From<(hugr::Node, hugr::OutgoingPort)> for HugrVariableValue {
    fn from((node, port): (hugr::Node, hugr::OutgoingPort)) -> Self {
        HugrVariableValue::new_wire_from_source(node, port)
    }
}

impl From<hugr::Node> for HugrVariableValue {
    fn from(node: hugr::Node) -> Self {
        HugrVariableValue::Node(node)
    }
}

/// Conversion error of HugrVariableValue to native Hugr types
#[derive(Debug, Error, Display)]
pub enum UnexpectedValueType {
    /// Unexpected value type: Node
    #[display("unexpected value type: Node")]
    Node,
    /// Unexpected value type: OutgoingPort
    #[display("unexpected value type: OutgoingPort")]
    OutgoingPort,
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

////////////////////////////////////////////////////////////////////////////////
////////////// Indexing scheme: resolve variable IDs to values /////////////////
////////////////////////////////////////////////////////////////////////////////

/// An indexing scheme for hugrs
#[derive(Clone, Debug, Default)]
pub struct HugrIndexingScheme;

impl pm::IndexingScheme for HugrIndexingScheme {
    type BindMap = HugrBindMap;
    type Key = <HugrBindMap as BindMap>::Key;
    type Value = <HugrBindMap as BindMap>::Value;

    fn required_bindings(&self, key: &HugrVariableID) -> Vec<HugrVariableID> {
        use HugrVariableID::*;

        match key {
            Op(node) => {
                if let Some(parent_path) = node.path_from_root.parent() {
                    vec![HugrNodeID::new(parent_path).into()]
                } else {
                    // The root node can be bound to any node.
                    vec![]
                }
            }
            CopyableWire(port) | LinearWire(port) => vec![port.node.into()],
        }
    }
}

impl<H: HugrView> pm::IndexedData<HugrVariableID> for Circuit<H> {
    type IndexingScheme = HugrIndexingScheme;
    type Value = <HugrIndexingScheme as pm::IndexingScheme>::Value;
    type BindMap = <HugrIndexingScheme as pm::IndexingScheme>::BindMap;

    fn list_bind_options(
        &self,
        key: &HugrVariableID,
        known_bindings: &HugrBindMap,
    ) -> Vec<HugrVariableValue> {
        use HugrVariableID::*;
        match key {
            Op(node) => {
                if let Some((parent_path, port, _)) = node.path_from_root.uncons() {
                    let parent_node_id = HugrNodeID::new(parent_path);
                    let Some(parent) = known_bindings.get_node(parent_node_id) else {
                        return vec![];
                    };
                    follow_port(parent, port, self.hugr()).map_into().collect()
                } else {
                    // Every hugr node is a valid binding for the root node.
                    let nodes = self.commands().map(|cmd| cmd.node());
                    nodes.map(HugrVariableValue::Node).map_into().collect()
                }
            }
            CopyableWire(port) | LinearWire(port) => {
                let Some(val) = port.resolve(known_bindings, self.hugr()) else {
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
    if hugr.num_ports(parent, port.direction()) <= port.index() {
        return None.into_iter().flatten();
    }
    let linked_ports = hugr.linked_ports(parent, port);
    Some(linked_ports.map(|(n, _)| n)).into_iter().flatten()
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
    fn test_list_bind_options() {
        let (circuit, root_choice) = create_test_circuit();
        let mut known_bindings = HugrBindMap::new();
        let scheme = HugrIndexingScheme;

        // Test for root node
        assert!(scheme
            .required_bindings(&HugrVariableID::Op(HugrNodeID::root()))
            .is_empty());
        let root_options =
            circuit.list_bind_options(&HugrVariableID::Op(HugrNodeID::root()), &known_bindings);
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
            let path = HugrPath::try_from(&ports[..i] as &[_]).unwrap();
            let var: HugrVariableID = HugrNodeID {
                path_from_root: path,
            }
            .into();
            assert_eq!(scheme.required_bindings(&var).len(), 1);
            let options = circuit.list_bind_options(&var, &known_bindings);
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
        let options = circuit.list_bind_options(&var, &known_bindings);
        assert_eq!(options.len(), 2);
        for option in &options {
            let &HugrVariableValue::Node(n) = option else {
                panic!("Expected node");
            };
            assert_eq!(circuit.hugr().get_optype(n), &Tk2Op::Rz.into());
            if n != root_choice {
                // Bind the second Rz node
                known_bindings.bind(var, option.clone()).unwrap();
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
                .list_bind_options(&port_var, &known_bindings)
                .into_iter()
                .exactly_one()
                .expect("expected exactly one option");
            match port.direction() {
                Direction::Incoming => {
                    let &HugrVariableValue::Wire(wire) = &sole_option else {
                        panic!("Expected wire");
                    };
                    // (n, p) is opposite `port`, so at the previous rotation
                    println!("1");
                    assert_eq!(wire.node(), root_choice);
                    assert_eq!(wire.source(), 0.into());
                }
                Direction::Outgoing => {
                    let &HugrVariableValue::Wire(wire) = &sole_option else {
                        panic!("Expected outgoing port");
                    };
                    println!("2");
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
