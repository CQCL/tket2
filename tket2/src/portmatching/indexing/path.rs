//! Path encodings for the indexing scheme.

use std::fmt::Debug;

use delegate::delegate;
use derive_more::Into;
use hugr::{Direction, PortIndex};

use super::unary_packed::{PackOverflow, UnaryPacked};

/// A path traversing a HUGR, relative to a start (root) node.
///
/// This identifies a [`hugr::Node`] relative to the choice of a root
/// node by specifying the path from the root to the node. Note that the path
/// may not specify the node uniquely in the existence of multi-ports, i.e.
/// multiple wires connected to the same port. In this case, it is up to the
/// matcher to make sure every valid binding is considered, or choose a
/// canonical one.
///
/// ## Path encoding
/// The path is encoded as a sequence of tuples `(hugr::Port, i)`. The sequence
/// of ports indicate the path from the root to the node, while the indices `i`
/// are integers. They are assigned values 0, 1, ... as required to distinguish
/// multiple (distinct) nodes that may be reached by the same port sequence,
/// which occurs when multiple wires are connected to the same port.
#[derive(
    Clone, Copy, Eq, Hash, PartialEq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct HugrPath(UnaryPacked<u64>);

impl HugrPath {
    delegate! {
        to self.0 {
            /// Whether the path is empty.
            pub fn is_empty(&self) -> bool;
        }
    }

    /// The empty path.
    pub fn empty() -> Self {
        Self(UnaryPacked::empty())
    }

    /// Append a port and index to the path.
    pub fn push(&mut self, port: hugr::Port, index: usize) -> Result<(), PackOverflow> {
        let is_outgoing = port.direction() == Direction::Outgoing;
        self.0.push_back(is_outgoing as u32)?;
        self.0.push_back(port.index() as u32)?;
        self.0.push_back(index as u32)?;
        Ok(())
    }

    /// Return the parent path and the last tuple (port, index).
    ///
    /// If self is empty, return None.
    pub(super) fn split_back(mut self) -> Option<(Self, hugr::Port, usize)> {
        let front_index = self.0.pop_back()?;
        let front_port_offset = self.0.pop_back().unwrap();
        let front_is_outgoing = self.0.pop_back().unwrap();

        let port = hugr::Port::new(
            if front_is_outgoing == 0 {
                Direction::Incoming
            } else {
                Direction::Outgoing
            },
            front_port_offset as usize,
        );
        let index = front_index as usize;

        (self, port, index).into()
    }

    pub(super) fn parent(&self) -> Option<Self> {
        Some(self.split_back()?.0)
    }
}

impl TryFrom<&[(hugr::Port, usize)]> for HugrPath {
    type Error = PackOverflow;

    fn try_from(value: &[(hugr::Port, usize)]) -> Result<Self, Self::Error> {
        let mut path = HugrPath::empty();
        for &(port, index) in value.iter() {
            path.push(port, index)?;
        }
        Ok(path)
    }
}

impl<const N: usize> TryFrom<&[(hugr::Port, usize); N]> for HugrPath {
    type Error = PackOverflow;

    fn try_from(value: &[(hugr::Port, usize); N]) -> Result<Self, Self::Error> {
        value.as_slice().try_into()
    }
}

impl Debug for HugrPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut tuples = vec![];
        let mut curr = *self;
        while let Some((parent, port, index)) = curr.split_back() {
            tuples.push((port, index));
            curr = parent;
        }
        tuples.reverse();
        f.debug_tuple("HugrPath").field(&tuples).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hugr::Direction;
    use itertools::Itertools;

    #[test]
    fn test_hugr_path_parent() {
        let ports = [
            (hugr::Port::new(Direction::Incoming, 0), 2),
            (hugr::Port::new(Direction::Outgoing, 1), 0),
            (hugr::Port::new(Direction::Incoming, 2), 1),
        ];
        let mut exp_paths = Vec::new();
        let mut path = HugrPath::empty();
        for &(port, index) in ports.iter() {
            path.push(port, index).unwrap();
            exp_paths.push(path);
        }
        let (exp_grandparent, exp_parent, path) = exp_paths.into_iter().collect_tuple().unwrap();

        let (parent, port, index) = path.split_back().expect("invalid parent");
        assert_eq!(parent, exp_parent);
        assert_eq!((port, index), ports[2]);

        let (grandparent, port, index) = parent.split_back().expect("invalid grandparent");
        assert_eq!(grandparent, exp_grandparent);
        assert_eq!((port, index), ports[1]);

        let (empty, port, index) = grandparent.split_back().expect("non-empty root");
        assert_eq!(empty, HugrPath::empty());
        assert_eq!((port, index), ports[0]);
    }

    #[test]
    fn test_path_cmp() {
        let path1 = vec![(hugr::Port::new(Direction::Incoming, 2), 2)];
        let path2 = vec![
            (hugr::Port::new(Direction::Incoming, 3), 3),
            (hugr::Port::new(Direction::Incoming, 3), 3),
        ];
        let a = HugrPath::try_from(path1.as_slice()).unwrap();
        let b = HugrPath::try_from(path2.as_slice()).unwrap();
        assert!(a < b);
    }

    #[test]
    fn test_debug_fmt() {
        let path = vec![
            (hugr::Port::new(Direction::Incoming, 3), 3),
            (hugr::Port::new(Direction::Outgoing, 1), 2),
        ];
        insta::assert_debug_snapshot!(path);
    }
}
