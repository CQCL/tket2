//! Path encodings for the indexing scheme.

use std::{cmp::Ordering, fmt::Debug};

use derive_more::Into;
use hugr::{Direction, PortIndex};

use derive_more::{Display, Error};

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
#[derive(Clone, Copy, Eq, Hash, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HugrPath(usize);

/// An error that occurs when a path is too long to encode in a HugrPath.
#[derive(Error, Debug, Display)]
#[display("path is too long to encode in HugrPath")]
pub struct PathEncodeOverflow;

impl HugrPath {
    /// The empty path.
    pub(super) fn empty() -> Self {
        Self(0)
    }

    /// Return the parent path and the last tuple (port, index).
    ///
    /// If self is empty, return None.
    pub(super) fn uncons(&self) -> Option<(Self, hugr::Port, usize)> {
        let mut starts = self.tuple_starts();
        let last_start = starts.pop()?;
        let parent = {
            if last_start == 0 {
                Self::empty()
            } else {
                // Remove all but the `last_start` MSBs
                let mask = n_ones_msb(last_start);
                HugrPath(self.0 & mask)
            }
        };
        let (port, index) = self.read_tuple(last_start);
        (parent, port, index).into()
    }

    fn read_tuple(&self, start: usize) -> (hugr::Port, usize) {
        let dir = if ith_msb(self.0, start) {
            Direction::Outgoing
        } else {
            Direction::Incoming
        };
        let offset = read_unary(self.0, start + 1, true);
        let index = read_unary(self.0, start + offset + 2, false);
        (hugr::Port::new(dir, offset), index)
    }

    pub(super) fn parent(&self) -> Option<Self> {
        Some(self.uncons()?.0)
    }

    /// Returns the start bit position of each tuple in the path.
    fn tuple_starts(&self) -> Vec<usize> {
        let mut starts = vec![];
        let mut val = self.0;

        let mut bit_pos = 0;
        while val > 0 {
            starts.push(bit_pos);

            // consume incoming/outgoing bit
            val <<= 1;
            bit_pos += 1;

            // consume offset (regex: 0*1)
            bit_pos += consume(false, &mut val);
            val <<= 1;
            bit_pos += 1;

            // consume index (regex: 1*0)
            bit_pos += consume(true, &mut val);
            val <<= 1;
            bit_pos += 1;
        }
        starts
    }

    fn len(&self) -> usize {
        let mut len = 0;
        let mut val = self.0;

        let mut bit_pos = 0;
        while val > 0 {
            len += 1;

            // consume incoming/outgoing bit
            val <<= 1;
            bit_pos += 1;

            // consume offset (regex: 0*1)
            bit_pos += consume(false, &mut val);
            val <<= 1;
            bit_pos += 1;

            // consume index (regex: 1*0)
            bit_pos += consume(true, &mut val);
            val <<= 1;
            bit_pos += 1;
        }
        len
    }
}

impl TryFrom<&[(hugr::Port, usize)]> for HugrPath {
    type Error = PathEncodeOverflow;

    fn try_from(value: &[(hugr::Port, usize)]) -> Result<Self, Self::Error> {
        let mut builder = HugrPathBuilder::new();
        for &(port, index) in value.iter() {
            builder.push(port, index)?;
        }
        Ok(builder.finish())
    }
}

impl<const N: usize> TryFrom<&[(hugr::Port, usize); N]> for HugrPath {
    type Error = PathEncodeOverflow;

    fn try_from(value: &[(hugr::Port, usize); N]) -> Result<Self, Self::Error> {
        value.as_slice().try_into()
    }
}

impl PartialOrd for HugrPath {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HugrPath {
    fn cmp(&self, other: &Self) -> Ordering {
        let len_self = self.len();
        let len_other = other.len();
        if len_self != len_other {
            len_self.cmp(&len_other)
        } else {
            let key =
                |path: &HugrPath| path.uncons().map(|(path, port, index)| (port, index, path));
            key(self).cmp(&key(other))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct HugrPathBuilder {
    encoded: usize,
    bits_used: usize,
}

impl HugrPathBuilder {
    pub(crate) fn new() -> Self {
        HugrPathBuilder {
            encoded: 0,
            bits_used: 0,
        }
    }

    fn write_bit(&mut self, bit: bool) -> Result<(), PathEncodeOverflow> {
        if bit {
            if self.bits_used >= MAX_BITS {
                return Err(PathEncodeOverflow);
            }
            self.encoded |= 1 << (MAX_BITS - self.bits_used - 1);
        }
        self.bits_used += 1;
        Ok(())
    }

    fn encode_unary(&mut self, value: usize, flipped: bool) -> Result<(), PathEncodeOverflow> {
        let (on_bit, off_bit) = if flipped {
            (false, true)
        } else {
            (true, false)
        };

        for _ in 0..value {
            self.write_bit(on_bit)?;
        }
        self.write_bit(off_bit)
    }

    pub(crate) fn push(
        &mut self,
        port: hugr::Port,
        index: usize,
    ) -> Result<(), PathEncodeOverflow> {
        // Encode port direction (1 bit)
        let direction_bit = port.direction() == Direction::Outgoing;
        self.write_bit(direction_bit)?;

        // Encode port offset (unary flipped)
        self.encode_unary(port.index(), true)?;

        // Encode index (unary)
        self.encode_unary(index, false)?;

        Ok(())
    }

    pub(crate) fn finish(self) -> HugrPath {
        HugrPath(self.encoded)
    }
}

// Left shit as long as the MSB is bit
fn consume(bit: bool, val: &mut usize) -> usize {
    // Whether MSB is == bit
    let cond = |val: usize| {
        if val == 0 {
            return false;
        }
        if (val >> 63) > 0usize {
            bit
        } else {
            !bit
        }
    };
    let mut bit_pos = 0;
    while cond(*val) {
        *val <<= 1;
        bit_pos += 1;
    }
    bit_pos
}

/// Returns a number whose binary representation is `n` ones at the MSB.
fn n_ones_msb(n: usize) -> usize {
    let val = usize::MAX;
    let shift = MAX_BITS - n;
    (val >> shift) << shift
}

fn ith_msb(val: usize, i: usize) -> bool {
    assert!(i < MAX_BITS);
    let mask = 1 << (MAX_BITS - 1 - i);
    val & mask > 0
}

fn read_unary(val: usize, start_i: usize, flipped: bool) -> usize {
    let on_bit = !flipped;
    let mut count = 0;
    while start_i + count < MAX_BITS && ith_msb(val, start_i + count) == on_bit {
        count += 1;
    }
    count
}

impl Debug for HugrPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut tuples = vec![];
        let mut curr = *self;
        while let Some((parent, port, index)) = curr.uncons() {
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
    fn test_hugr_path_builder() {
        let mut builder = HugrPathBuilder::new();

        // Push an incoming port with index 0 and child index 2
        builder
            .push(hugr::Port::new(Direction::Incoming, 0), 2)
            .unwrap();

        // Push an outgoing port with index 1 and child index 0
        builder
            .push(hugr::Port::new(Direction::Outgoing, 1), 0)
            .unwrap();

        // Push another incoming port with index 2 and child index 1
        builder
            .push(hugr::Port::new(Direction::Incoming, 2), 1)
            .unwrap();

        let path = builder.finish();

        // The expected encoding should be:
        // 0 (incoming) + 1 (port index 0) + 110 (child index 2)
        // + 1 (outgoing) + 01 (port index 1) + 0 (child index 0)
        // + 0 (incoming) + 001 (port index 2) + 10 (child index 1)
        // = 011101010000110
        let exp = 0b011101010000110 << 49;

        assert_eq!(path.0, exp);
    }

    #[test]
    fn test_hugr_path_builder_overflow() {
        let mut builder = HugrPathBuilder::new();

        // Try to push exactly MAX_BITS worth of data
        builder
            .push(hugr::Port::new(Direction::Incoming, 59), 3)
            .unwrap();
        // the last bit is a zero, so this does not overflow
        assert_eq!(builder.bits_used, MAX_BITS + 1);
        let val = builder.finish();
        assert_eq!(val.0, 0b1111);

        let mut builder = HugrPathBuilder::new();
        // Try to push more than MAX_BITS worth of data
        builder
            .push(hugr::Port::new(Direction::Incoming, 60), 3)
            .unwrap_err();
    }

    #[test]
    fn test_hugr_path_parent() {
        let ports = [
            (hugr::Port::new(Direction::Incoming, 0), 2),
            (hugr::Port::new(Direction::Outgoing, 1), 0),
            (hugr::Port::new(Direction::Incoming, 2), 1),
        ];
        let mut exp_paths = Vec::new();
        let mut builder = HugrPathBuilder::new();
        for &(port, index) in ports.iter() {
            builder.push(port, index).unwrap();
            exp_paths.push(builder.clone().finish());
        }
        let (exp_grandparent, exp_parent, path) = exp_paths.into_iter().collect_tuple().unwrap();

        let (parent, port, index) = path.uncons().expect("invalid parent");
        assert_eq!(parent, exp_parent);
        assert_eq!((port, index), ports[2]);

        let (grandparent, port, index) = parent.uncons().expect("invalid grandparent");
        assert_eq!(grandparent, exp_grandparent);
        assert_eq!((port, index), ports[1]);

        let (empty, port, index) = grandparent.uncons().expect("non-empty root");
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
}
