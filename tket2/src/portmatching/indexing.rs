//! Indexing schemes for pattern matching with portmatching.
//!
//! Indexing schemes assign a unique variable name for every variable (i.e every
//! port and node in the hugr). This is used by the portmatcher to express
//! constraints to be checked whilst matching.

use std::collections::BTreeMap;

use derive_more::{Display, Error, From};
use hugr::HugrView;
use itertools::{Either, Itertools};
use portmatching as pm;

use crate::Circuit;

mod path;
pub(super) use path::{HugrPath, HugrPathBuilder};

////////////////////////////////////////////////////////////////////////////////
//////////////////// Variable Naming scheme used for Hugrs /////////////////////
////////////////////////////////////////////////////////////////////////////////

/// Variables refer to either a node or a port in the hugr.
#[derive(
    Clone,
    Copy,
    Debug,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    From,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum HugrVariableID {
    /// A variable that binds to a node.
    Node(HugrNodeID),
    /// A variable that binds to a port.
    Port(HugrPortID),
}

impl HugrVariableID {
    /// Resolve the variable ID to a unique value in the hugr given `bindings`.
    ///
    /// Any non-incoming port variable and non-root node variable can be
    /// resolved uniquely (if it exists) to a value. Calling `resolve` on an
    /// incoming port variable will panic.
    ///
    /// This assumes that the required incoming port bindings are in `bindings`.
    fn resolve(&self, bindings: &HugrBindMap, hugr: &impl HugrView) -> Option<HugrVariableValue> {
        match self {
            HugrVariableID::Node(node) => node.resolve(bindings),
            HugrVariableID::Port(port) => port.resolve(bindings, hugr),
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

impl HugrVariableValue {
    fn node(&self) -> hugr::Node {
        match *self {
            HugrVariableValue::Node(node) => node,
            HugrVariableValue::IncomingPort(node, _) => node,
            HugrVariableValue::OutgoingPort(node, _) => node,
        }
    }
}

/// A map to store bindings for variables in a hugr.
pub type HugrBindMap = BTreeMap<HugrVariableID, HugrVariableValue>;

/// A port variable ID, given based on the unique IDs given to incoming ports.
///
/// - An incoming port ID is given by a path from a root node to an incoming port.
/// - An outgoing port ID is given by the opposite incoming port ID. This defines
///   the outgoing port uniquely as there is a one-to-many outgoing port to
///   incoming port relationship.
#[derive(
    Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize,
)]
pub enum HugrPortID {
    /// A port variable that binds to an outgoing port.
    Outgoing {
        /// The path from the root node to an incoming port opposite.
        opposite_port: HugrPath,
    },
    /// A port variable that binds to an incoming port.
    Incoming {
        /// The path from the root node to the incoming port.
        path_from_root: HugrPath,
    },
}

impl HugrPortID {
    /// Resolve an outgoing port ID given `bindings` that specify the opposite port.
    ///
    /// Returns a HugrVariableValue::OutgoingPort variant. Calling this on an
    /// incoming port ID will result in a panic.
    fn resolve(&self, bindings: &HugrBindMap, hugr: &impl HugrView) -> Option<HugrVariableValue> {
        match self {
            &HugrPortID::Outgoing { opposite_port } => {
                let opp_var = HugrPortID::new_incoming(opposite_port).into();
                let &HugrVariableValue::IncomingPort(opp_node, opp_port) =
                    bindings.get(&opp_var)?
                else {
                    panic!("expected opposite port to be incoming");
                };
                // Currently, silently fail if there is more than one output
                let (node, port) = hugr.single_linked_output(opp_node, opp_port)?;
                HugrVariableValue::OutgoingPort(node, port).into()
            }
            HugrPortID::Incoming { .. } => {
                panic!("Incoming port IDs do not resolve uniquely to a value")
            }
        }
    }

    pub(crate) fn new_incoming(path_from_root: HugrPath) -> Self {
        Self::Incoming { path_from_root }
    }

    pub(crate) fn new_outgoing(opposite_port: HugrPath) -> Self {
        Self::Outgoing { opposite_port }
    }
}

/// A node variable ID, given based on a unique ID given to one of its ports.
///
/// The root is a special case that must be handled separately, as it might not
/// have any ports.
#[derive(
    Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize,
)]
pub enum HugrNodeID {
    /// Variable that binds to the root node.
    Root,
    /// Variable that binds to a non-root node.
    NonRoot {
        /// A port that is incident to the node.
        incident_port: HugrPortID,
    },
}

impl HugrNodeID {
    /// Resolve a non-root node ID to a node in the hugr given `bindings`.
    ///
    /// Root IDs cannot be resolved uniquely to a value. Calling this on a
    /// HugrNodeID::Root will result in a panic.
    fn resolve(&self, bindings: &HugrBindMap) -> Option<HugrVariableValue> {
        match self {
            HugrNodeID::Root => panic!("Root node IDs do not resolve uniquely to a value"),
            HugrNodeID::NonRoot { incident_port } => {
                let node = bindings.get(&(*incident_port).into())?.node();
                HugrVariableValue::Node(node).into()
            }
        }
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
            HugrVariableID::Node(node) => match node {
                // The root node can be bound to any node.
                HugrNodeID::Root => vec![],
                // Otherwise require the incident port to be bound.
                &HugrNodeID::NonRoot { incident_port } => vec![incident_port.into()],
            },
            HugrVariableID::Port(port) => match port {
                &HugrPortID::Outgoing { opposite_port } => {
                    // Require the opposite port to be bound.
                    let port_id = HugrPortID::new_incoming(opposite_port);
                    vec![port_id.into()]
                }
                HugrPortID::Incoming { path_from_root } => {
                    // Require the parent of the incoming port to be bound.
                    if let Some(parent_path) = path_from_root.parent() {
                        let port_id = HugrPortID::new_incoming(parent_path);
                        vec![port_id.into()]
                    } else {
                        vec![HugrNodeID::Root.into()]
                    }
                }
            },
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
            HugrVariableID::Node(HugrNodeID::Root) => {
                // Every hugr node is a valid binding for the root node.
                let nodes = self.commands().map(|cmd| cmd.node());
                nodes.map(HugrVariableValue::Node).map_into().collect()
            }
            HugrVariableID::Port(HugrPortID::Incoming { path_from_root }) => {
                let ports = path_from_root.list_bind_options(known_bindings, self.hugr());
                ports
                    .into_iter()
                    .map(|(n, p)| HugrVariableValue::IncomingPort(n, p))
                    .collect()
            }
            // Otherwise, resolves uniquely
            key => Vec::from_iter(key.resolve(known_bindings, self.hugr())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extension::rotation::ROTATION_TYPE;
    use crate::Tk2Op;
    use hugr::builder::{DFGBuilder, Dataflow, DataflowHugr};
    use hugr::extension::prelude::QB_T;
    use hugr::ops::handle::NodeHandle;
    use hugr::ops::OpType;
    use hugr::types::Signature;
    use hugr::{Direction, IncomingPort, OutgoingPort, Port};
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
            .required_bindings(&HugrVariableID::Node(HugrNodeID::Root))
            .is_empty());
        let root_options =
            circuit.list_bind_options(&HugrVariableID::Node(HugrNodeID::Root), &known_bindings);
        assert_eq!(root_options.len(), 3); // H gate, and 2x Rz nodes

        // Bind the root node
        known_bindings.insert(
            HugrVariableID::Node(HugrNodeID::Root),
            HugrVariableValue::Node(root_choice),
        );

        // Test for incoming port
        let ports = [
            (Port::new(Direction::Outgoing, 0), 0usize),
            (Port::new(Direction::Incoming, 1), 0),
            (Port::new(Direction::Outgoing, 1), 1),
        ];

        // The first two ports along the path should be unique
        for i in 1..=2 {
            let path = HugrPath::try_from(&ports[..i] as &[_]).unwrap();
            let var: HugrVariableID = HugrPortID::Incoming {
                path_from_root: path,
            }
            .into();
            assert_eq!(scheme.required_bindings(&var).len(), 1);
            let options = circuit.list_bind_options(&var, &known_bindings);
            let sole_option = options.into_iter().exactly_one().unwrap();
            let &HugrVariableValue::IncomingPort(n, p) = &sole_option else {
                panic!("Expected incoming port");
            };
            assert_eq!(circuit.hugr().get_optype(n), &Tk2Op::Rz.into());
            assert_eq!(p, IncomingPort::from(i - 1));
            known_bindings.insert(var, sole_option);
        }

        // The last port along the path should have two options
        // (the two uses of the rotation angle)
        let path = HugrPath::try_from(&ports as &[_]).unwrap();
        let var: HugrVariableID = HugrPortID::Incoming {
            path_from_root: path,
        }
        .into();
        let options = circuit.list_bind_options(&var, &known_bindings);
        assert_eq!(options.len(), 2);
        for option in &options {
            let &HugrVariableValue::IncomingPort(n, p) = option else {
                panic!("Expected incoming port");
            };
            assert_eq!(circuit.hugr().get_optype(n), &Tk2Op::Rz.into());
            assert_eq!(p, IncomingPort::from(1));
        }
        known_bindings.insert(var, options[0].clone());

        // Test for an outgoing port
        let outgoing = HugrVariableID::Port(HugrPortID::Outgoing {
            opposite_port: path,
        });
        let out_options = circuit.list_bind_options(&outgoing, &known_bindings);
        assert_eq!(out_options.len(), 1);
        let &HugrVariableValue::OutgoingPort(n, p) =
            &out_options.into_iter().exactly_one().unwrap()
        else {
            panic!("Expected outgoing port");
        };
        assert!(matches!(circuit.hugr().get_optype(n), &OpType::Input(_)));
        assert_eq!(p, OutgoingPort::from(1));

        // Test for a non-root node
        let node_var = HugrNodeID::NonRoot {
            incident_port: HugrPortID::new_incoming(path.parent().unwrap()),
        }
        .into();
        let options = circuit.list_bind_options(&node_var, &known_bindings);
        let sole_option = options.into_iter().exactly_one().unwrap();
        let HugrVariableValue::Node(node) = sole_option else {
            panic!("Expected node");
        };
        assert_eq!(circuit.hugr().get_optype(node), &Tk2Op::Rz.into());
        assert_ne!(node, root_choice);
    }
}
