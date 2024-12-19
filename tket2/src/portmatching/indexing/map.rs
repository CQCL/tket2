use std::{
    borrow::Borrow,
    collections::{BTreeMap, BTreeSet},
};

use delegate::delegate;
use portmatching::{self as pm, indexing::Binding, BindMap};

use super::{HugrNodeID, HugrVariableID, HugrVariableValue};

/// A map to store bindings for variables in a hugr.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct HugrBindMap(BTreeMap<HugrVariableID, Option<HugrVariableValue>>);

impl HugrBindMap {
    delegate! {
        to self.0 {
            pub(crate) fn len(&self) -> usize;
            pub(crate) fn is_empty(&self) -> bool;
        }
    }

    pub(crate) fn new() -> Self {
        Self(BTreeMap::new())
    }

    pub(crate) fn get_node(&self, node: HugrNodeID) -> Option<hugr::Node> {
        match self.0.get_binding(&HugrVariableID::Op(node)).borrowed() {
            Binding::Bound(&HugrVariableValue::Node(node)) => Some(node),
            Binding::Unbound | Binding::Failed => None,
            Binding::Bound(_) => panic!("invalid value type in HugrBindMap"),
        }
    }

    // pub(crate) fn get_port(&self, port: HugrPortID) -> Option<(hugr::Node, hugr::OutgoingPort)> {
    //     match self.0.get_binding(&port).borrowed() {
    //         Binding::Bound(&HugrVariableValue::OutgoingPort(node, port)) => Some((node, port)),
    //         Binding::Unbound | Binding::Failed => None,
    //         Binding::Bound(_) => panic!("invalid value type in HugrBindMap"),
    //     }
    // }
}

impl pm::BindMap for HugrBindMap {
    type Key = HugrVariableID;
    type Value = HugrVariableValue;

    delegate! {
        to self.0 {
            fn get_binding(&self, var: &Self::Key) -> Binding<impl Borrow<Self::Value> + '_>;
            fn retain_keys(&mut self, keys: &BTreeSet<Self::Key>);
            fn bind(
                &mut self,
                var: Self::Key,
                val: Self::Value,
            ) -> Result<(), portmatching::indexing::BindVariableError>;
            fn bind_failed(&mut self, var: Self::Key);
        }
    }
}
