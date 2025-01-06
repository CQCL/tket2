//! Path encodings for the indexing scheme.

use std::{cmp::Ordering, fmt::Debug};

use derive_more::Into;
use hugr::{Direction, PortIndex};

use derive_more::{Display, Error};

use super::unary_packed::{PackOverflow, UnaryPacked};

const MAX_BITS: usize = usize::BITS as usize;

/// A compact encoding for a path from a root node to an incoming port.
///
/// This compactly defines an incoming port relative to the choice of a root
/// node by specifying the path from the root to the port. The path may not
/// specify the port uniquely in the case of multi-ports. In this case, the
/// matcher will choose one/explore every valid binding.
///
/// ## Path encoding
/// The path is encoded as a sequence of tuples `(hugr::Port, i)` where i is a
/// non-negative integer. The incoming port that is associated with a sequence
/// `[(p1, i1), .., (pn, in)]` can be determined inductively:
///  - `[(p1, i1)]` is an incoming port on the edge attached to `p1` at the root
///    node. The index `i1` has no semantic meaning but serves to distinguish
///    between multiple incoming ports on the same node. Which index is associated
///    to which incoming port may be chosen arbitrarily.
///  - to find the incoming port for `[(p1, i1), .., (pn, in)]`, we first associate
///    `[(p1, i1), .., (pn-1, in-1)]` with a node: if p1 is an outgoing port, then
///    the node is the node that the port `[(p1, i1), .., (pn-1, in-1)]` is
///    incident to. Otherwise, it is the node at the other end of the edge attached
///    to port `[(p1, i1), .., (pn-1, in-1)]` (this is unique as we assume outgoing
///    ports are unique). The port associated with `[(p1, i1), .., (pn, in)]`
///    is then one of the incoming ports on the edge at `pn` at the associated node.
///
/// Note that if we associate `[]` with the root node, then the second criterion
/// above is sufficient as a definition.
///
/// ## Bit encoding
/// We encode each tuple `(p, i)` as a bitstring
/// ```plaintext
/// +------ one bit --------+-- (p.offset + 1) bits ---+-- (i + 1) bits --+
/// | matches!(p, Outgoing) | p.offset (unary flipped) |     i (unary)    |
/// +-----------------------+--------------------------+------------------+
/// ```
/// The first bit indicates whether `p` is an incoming or outgoing port, the
/// other two fields are the offset and index. We use unary encoding for the
/// integers, i.e. 0 -> 0, 1 -> 10, 2 -> 110, etc. This makes small integers very
/// compact to encode, without fixing a maximum value.
/// The rest of the bits are zeroed, so that the last bit of `i` is also
/// the first bit of the zero padding (which can be omitted if running out of
/// space).
///
/// Unary flipped means that the binary encoding of the integer is flipped.
/// Flipping one of the unary encodings means that a tuple can never be all
/// 0, thus differentiating between tuples and zero padding.
#[derive(
    Clone, Copy, Eq, Hash, PartialEq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct HugrPath(UnaryPacked);

impl HugrPath {
    /// The empty path.
    pub fn empty() -> Self {
        Self(UnaryPacked::empty())
    }

    #[must_use]
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
    pub(super) fn split_back(&self) -> Option<(Self, hugr::Port, usize)> {
        let mut self_copy = *self;
        let front_index = self_copy.0.pop_back()?;
        let front_port_offset = self_copy.0.pop_back().unwrap();
        let front_is_outgoing = self_copy.0.pop_back().unwrap();

        let port = hugr::Port::new(
            if front_is_outgoing == 0 {
                Direction::Incoming
            } else {
                Direction::Outgoing
            },
            front_port_offset as usize,
        );
        let index = front_index as usize;

        (self_copy, port, index).into()
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
            exp_paths.push(path.clone());
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
