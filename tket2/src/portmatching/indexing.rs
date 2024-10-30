//! Indexing schemes for pattern matching with portmatching.
//!
//! Indexing schemes assign a unique variable name for every variable (i.e every
//! port and node in the hugr). This is used by the portmatcher to express
//! constraints to be checked whilst matching.

use std::collections::BTreeMap;

use derive_more::{Display, Error, From};
use hugr::{HugrView, PortIndex};
use itertools::{Either, Itertools};
use portmatching as pm;

use crate::Circuit;

mod path;
pub(super) use path::{HugrPath, HugrPathBuilder};

////////////////////////////////////////////////////////////////////////////////
//////////////////// Variable Naming scheme used for Hugrs /////////////////////
////////////////////////////////////////////////////////////////////////////////

/// Variables refer to either a node or a port in the hugr.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, From, serde::Serialize, serde::Deserialize)]
pub enum HugrVariableID {
    /// A variable that binds to a node.
    Node(HugrNodeID),
    /// A variable that binds to a port.
    Port(HugrPortID),
}

impl HugrVariableID {
    fn path(&self) -> HugrPath {
        match self {
            HugrVariableID::Node(node) => node.path_from_root,
            HugrVariableID::Port(port) => port.node.path_from_root,
        }
    }
}

/// The value of a variable in the indexing scheme, either a node or a port.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum HugrVariableValue {
    /// The value of a HugrVariableID::Node variable.
    Node(hugr::Node),
    /// The value of a HugrVariableID::Port(HugrPortID::Incoming) variable.
    IncomingPort(hugr::Node, hugr::IncomingPort),
    /// The value of a HugrVariableID::Port(HugrPortID::Outgoing) variable.
    OutgoingPort(hugr::Node, hugr::OutgoingPort),
}

/// A map to store bindings for variables in a hugr.
pub type HugrBindMap = BTreeMap<HugrVariableID, HugrVariableValue>;

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

    /// Resolve an outgoing port ID given `bindings` that specify the opposite port.
    ///
    /// Returns a HugrVariableValue::OutgoingPort variant. Calling this on an
    /// incoming port ID will result in a panic.
    fn resolve(&self, bindings: &HugrBindMap, hugr: &impl HugrView) -> Option<HugrVariableValue> {
        let &HugrVariableValue::Node(node) = bindings.get(&self.node.into())? else {
            panic!("expected node to be bound");
        };
        match self.port.as_directed() {
            Either::Left(in_port) if hugr.num_inputs(node) > in_port.index() => {
                Some(HugrVariableValue::IncomingPort(node, in_port))
            }
            Either::Right(out_port) if hugr.num_outputs(node) > out_port.index() => {
                Some(HugrVariableValue::OutgoingPort(node, out_port))
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

impl From<(hugr::Node, hugr::Port)> for HugrVariableValue {
    fn from((node, port): (hugr::Node, hugr::Port)) -> Self {
        match port.as_directed() {
            Either::Left(in_port) => HugrVariableValue::IncomingPort(node, in_port),
            Either::Right(out_port) => HugrVariableValue::OutgoingPort(node, out_port),
        }
    }
}

impl From<(hugr::Node, hugr::IncomingPort)> for HugrVariableValue {
    fn from((node, port): (hugr::Node, hugr::IncomingPort)) -> Self {
        HugrVariableValue::IncomingPort(node, port)
    }
}

impl From<(hugr::Node, hugr::OutgoingPort)> for HugrVariableValue {
    fn from((node, port): (hugr::Node, hugr::OutgoingPort)) -> Self {
        HugrVariableValue::OutgoingPort(node, port)
    }
}

impl From<hugr::Node> for HugrVariableValue {
    fn from(node: hugr::Node) -> Self {
        HugrVariableValue::Node(node)
    }
}

impl PartialOrd for HugrVariableID {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HugrVariableID {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let self_path = self.path();
        let other_path = other.path();

        // Compare paths, break ties with variable type
        self_path.cmp(&other_path).then_with(|| {
            use HugrVariableID::*;
            match (self, other) {
                (Node(..), Port(..)) => std::cmp::Ordering::Less,
                (Port(..), Node(..)) => std::cmp::Ordering::Greater,
                (Port(self_port), Port(other_port)) => self_port.cmp(&other_port),
                (Node(..), Node(..)) => std::cmp::Ordering::Equal,
            }
        })
    }
}

/// Conversion error of HugrVariableValue to native Hugr types
#[derive(Debug, Error, Display)]
pub enum UnexpectedValueType {
    /// Unexpected value type: Node
    #[display("unexpected value type: Node")]
    Node,
    /// Unexpected value type: IncomingPort
    #[display("unexpected value type: IncomingPort")]
    IncomingPort,
    /// Unexpected value type: OutgoingPort
    #[display("unexpected value type: OutgoingPort")]
    OutgoingPort,
}

impl TryFrom<HugrVariableValue> for (hugr::Node, hugr::Port) {
    type Error = UnexpectedValueType;

    fn try_from(value: HugrVariableValue) -> Result<Self, Self::Error> {
        match value {
            HugrVariableValue::Node(..) => Err(UnexpectedValueType::Node),
            HugrVariableValue::IncomingPort(node, in_port) => Ok((node, in_port.into())),
            HugrVariableValue::OutgoingPort(node, out_port) => Ok((node, out_port.into())),
        }
    }
}

impl TryFrom<HugrVariableValue> for (hugr::Node, hugr::IncomingPort) {
    type Error = UnexpectedValueType;

    fn try_from(value: HugrVariableValue) -> Result<Self, Self::Error> {
        match value {
            HugrVariableValue::Node(..) => Err(UnexpectedValueType::Node),
            HugrVariableValue::IncomingPort(node, in_port) => Ok((node, in_port)),
            HugrVariableValue::OutgoingPort(..) => Err(UnexpectedValueType::OutgoingPort),
        }
    }
}

impl TryFrom<HugrVariableValue> for (hugr::Node, hugr::OutgoingPort) {
    type Error = UnexpectedValueType;

    fn try_from(value: HugrVariableValue) -> Result<Self, Self::Error> {
        match value {
            HugrVariableValue::Node(..) => Err(UnexpectedValueType::Node),
            HugrVariableValue::IncomingPort(..) => Err(UnexpectedValueType::IncomingPort),
            HugrVariableValue::OutgoingPort(node, out_port) => Ok((node, out_port)),
        }
    }
}

impl TryFrom<HugrVariableValue> for hugr::Node {
    type Error = UnexpectedValueType;

    fn try_from(value: HugrVariableValue) -> Result<Self, Self::Error> {
        match value {
            HugrVariableValue::Node(node) => Ok(node),
            HugrVariableValue::IncomingPort(..) => Err(UnexpectedValueType::IncomingPort),
            HugrVariableValue::OutgoingPort(..) => Err(UnexpectedValueType::OutgoingPort),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
////////////// Indexing scheme: resolve variable IDs to values /////////////////
////////////////////////////////////////////////////////////////////////////////

/// An indexing scheme for hugrs that does not handle hierarchy.
#[derive(Clone, Debug, Default)]
pub struct FlatHugrIndexingScheme;

impl pm::IndexingScheme for FlatHugrIndexingScheme {
    type BindMap = HugrBindMap;

    fn required_bindings(&self, key: &HugrVariableID) -> Vec<HugrVariableID> {
        match key {
            HugrVariableID::Node(node) => {
                if let Some(parent_path) = node.path_from_root.parent() {
                    vec![HugrNodeID::new(parent_path).into()]
                } else {
                    // The root node can be bound to any node.
                    vec![]
                }
            }
            HugrVariableID::Port(port) => vec![port.node.into()],
        }
    }
}

impl<H: HugrView> pm::IndexedData for Circuit<H> {
    type IndexingScheme = FlatHugrIndexingScheme;

    fn list_bind_options(
        &self,
        key: &HugrVariableID,
        known_bindings: &HugrBindMap,
    ) -> Vec<HugrVariableValue> {
        match key {
            &HugrVariableID::Node(node) => {
                if let Some((parent_path, port, _)) = node.path_from_root.uncons() {
                    let parent_node_id = HugrNodeID::new(parent_path);
                    let Some(&HugrVariableValue::Node(parent)) =
                        known_bindings.get(&parent_node_id.into())
                    else {
                        return vec![];
                    };
                    follow_port(parent, port, self.hugr()).map_into().collect()
                } else {
                    // Every hugr node is a valid binding for the root node.
                    let nodes = self.commands().map(|cmd| cmd.node());
                    nodes.map(HugrVariableValue::Node).map_into().collect()
                }
            }
            HugrVariableID::Port(port) => port
                .resolve(known_bindings, self.hugr())
                .into_iter()
                .collect(),
        }
    }
}

fn follow_port(
    parent: hugr::Node,
    port: hugr::Port,
    hugr: &impl HugrView,
) -> impl Iterator<Item = hugr::Node> + '_ {
    let linked_ports = match port.as_directed() {
        Either::Left(in_port) if hugr.num_inputs(parent) > in_port.index() => {
            Some(hugr.linked_ports(parent, in_port))
        }
        Either::Right(out_port) if hugr.num_outputs(parent) > out_port.index() => {
            Some(hugr.linked_ports(parent, out_port))
        }
        _ => None,
    };
    linked_ports.into_iter().flatten().map(|(n, _)| n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extension::rotation::ROTATION_TYPE;
    use crate::Tk2Op;
    use hugr::builder::{DFGBuilder, Dataflow, DataflowHugr};
    use hugr::extension::prelude::QB_T;
    use hugr::ops::handle::NodeHandle;
    use hugr::types::Signature;
    use hugr::{Direction, Port};
    use portmatching::{IndexedData, IndexingScheme};

    fn create_test_circuit() -> (Circuit<impl HugrView>, hugr::Node) {
        let mut hrz =
            DFGBuilder::new(Signature::new(vec![QB_T, ROTATION_TYPE], vec![QB_T])).unwrap();
        let [q_in, angle_in] = hrz.input_wires_arr();

        let res = hrz.add_dataflow_op(Tk2Op::H, [q_in]).unwrap();
        let q = res.out_wire(0);

        let res = hrz.add_dataflow_op(Tk2Op::Rz, [q, angle_in]).unwrap();
        let fst_rz = res.node();
        let q = res.out_wire(0);

        let res = hrz.add_dataflow_op(Tk2Op::Rz, [q, angle_in]).unwrap();
        let q = res.out_wire(0);

        let res = hrz
            .finish_hugr_with_outputs([q], &crate::extension::REGISTRY)
            .unwrap()
            .into();
        (res, fst_rz)
    }

    #[test]
    fn test_list_bind_options() {
        let (circuit, root_choice) = create_test_circuit();
        let mut known_bindings = HugrBindMap::new();
        let scheme = FlatHugrIndexingScheme;

        // Test for root node
        assert!(scheme
            .required_bindings(&HugrVariableID::Node(HugrNodeID::root()))
            .is_empty());
        let root_options =
            circuit.list_bind_options(&HugrVariableID::Node(HugrNodeID::root()), &known_bindings);
        assert_eq!(root_options.len(), 3); // H gate, and 2x Rz nodes

        // Bind the root node
        known_bindings.insert(
            HugrVariableID::Node(HugrNodeID::root()),
            HugrVariableValue::Node(root_choice),
        );

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
            known_bindings.insert(var, sole_option);
        }

        // The last port along the path should have two options
        // (the two uses of the rotation angle)
        let path = HugrPath::try_from(&ports as &[_]).unwrap();
        let var: HugrVariableID = HugrNodeID {
            path_from_root: path,
        }
        .into();
        let options = circuit.list_bind_options(&var, &known_bindings);
        assert_eq!(options.len(), 2);
        for option in &options {
            let &HugrVariableValue::Node(n) = option else {
                panic!("Expected incoming port");
            };
            assert_eq!(circuit.hugr().get_optype(n), &Tk2Op::Rz.into());
            if n != root_choice {
                // Bind the second Rz node
                known_bindings.insert(var, option.clone());
            }
        }

        // Test for ports
        let ports = [
            Port::new(Direction::Outgoing, 1),
            Port::new(Direction::Incoming, 0),
        ];
        for port in ports {
            let port_var = HugrVariableID::Port(HugrPortID {
                node: HugrNodeID::new(path),
                port,
            });
            let sole_option = circuit
                .list_bind_options(&port_var, &known_bindings)
                .into_iter()
                .exactly_one()
                .expect("expected exactly one option");
            let (n, p): (hugr::Node, hugr::Port) = match port.direction() {
                Direction::Incoming => {
                    let &HugrVariableValue::IncomingPort(n, p) = &sole_option else {
                        panic!("Expected incoming port");
                    };
                    (n, p.into())
                }
                Direction::Outgoing => {
                    let &HugrVariableValue::OutgoingPort(n, p) = &sole_option else {
                        panic!("Expected outgoing port");
                    };
                    (n, p.into())
                }
            };
            assert_eq!(circuit.hugr().get_optype(n), &Tk2Op::Rz.into());
            assert_eq!(p, port);
            assert_ne!(n, root_choice);
        }
    }

    #[test]
    fn test_var_ord() {
        let path = HugrPath::try_from(&[(Port::new(Direction::Incoming, 0), 0)]).unwrap();
        let node = HugrNodeID::new(path);
        let inp = HugrPortID {
            node,
            port: Port::new(Direction::Incoming, 0),
        };
        let out = HugrPortID {
            node,
            port: Port::new(Direction::Outgoing, 0),
        };

        assert!(inp < out);

        let var_in: HugrVariableID = inp.into();
        let var_out: HugrVariableID = out.into();
        assert!(var_in < var_out);
    }
}
