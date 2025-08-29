//! Core type definitions for TKET resource tracking.
//!
//! This module defines the fundamental types used to track resources and
//! copyable values throughout a HUGR circuit, including resource identifiers,
//! positions, and the mapping structures that associate them with operations.

use std::ops::RangeInclusive;

use cgmath::Zero;
use derive_more::derive::From;
use hugr::{
    core::HugrNode, types::Signature, Direction, IncomingPort, OutgoingPort, Port, PortIndex, Wire,
};
use itertools::Itertools;
use num_rational::Rational64;

/// Unique identifier for a linear resource.
///
/// ResourceIds are assigned in increasing order starting from scope inputs,
/// following the canonical topological ordering. They cannot be constructed
/// manually and must be obtained through a ResourceAllocator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResourceId(usize);

impl ResourceId {
    /// Create a new ResourceId.
    ///
    /// ResourceIds should typically be obtained from [`ResourceAllocator`].
    /// Only use this in testing.
    pub(super) fn new(id: usize) -> Self {
        Self(id)
    }

    /// Get the underlying usize value of this ResourceId.
    pub fn as_usize(self) -> usize {
        self.0
    }
}

/// Position of an operation along a resource path.
///
/// Positions are rational numbers that order operations along resource paths.
/// Initially assigned as contiguous integers, they may become non-integer
/// when operations are inserted or removed.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Position(pub(crate) Rational64);

impl std::fmt::Debug for Position {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Position({})", self.0)
    }
}

impl Position {
    /// Create a new integer Position.
    ///
    /// This method should only be called by allocators and tests.
    #[allow(unused)]
    pub(super) fn new_integer(numer: i64) -> Self {
        Self(Rational64::from_integer(numer))
    }

    /// Get position as f64, rounded to the given precision.
    pub fn to_f64(&self, precision: usize) -> f64 {
        let big = self.0 * Rational64::from_integer(10).pow(precision as i32);
        big.round().to_integer() as f64 / 10f64.powi(precision as i32)
    }

    /// Increment the position by 1.
    pub fn increment(&self) -> Self {
        Self(self.0 + 1)
    }

    /// Rescale the position such that any number in the old range (including
    /// ends) is within the new range (excluding ends).
    pub(crate) fn rescale(
        &self,
        old_range: RangeInclusive<Position>,
        new_range: RangeInclusive<Position>,
    ) -> Self {
        let Self(pos) = self;
        let old_range_size = Rational64::from_integer(2) + old_range.end().0 - old_range.start().0;
        let new_range_size = new_range.end().0 - new_range.start().0;

        if old_range_size == Rational64::zero() {
            return *new_range.start();
        }

        let new_pos = new_range.start().0
            + (Rational64::from_integer(1) + pos - old_range.start().0) / old_range_size
                * new_range_size;
        Self(new_pos)
    }
}

/// A value associated with a dataflow port, identified either by a resource ID
/// (for linear values) or by its wire (for copyable values).
///
/// This can currently be converted to and from [`hugr::CircuitUnit`], but
/// linear wires are assigned to resources with typed resource IDs instead of
/// integers.
///
/// Equivalence with [`hugr::CircuitUnit`] is not guaranteed in the future: we
/// may expand expressivity, e.g. identifying copyable units by their ASTs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, From)]
pub enum CircuitUnit<N: HugrNode> {
    /// A linear resource.
    Resource(#[from] ResourceId),
    /// A copyable value.
    Copyable(#[from] Wire<N>),
}

impl<N: HugrNode> From<CircuitUnit<N>> for hugr::CircuitUnit<N> {
    fn from(value: CircuitUnit<N>) -> Self {
        match value {
            CircuitUnit::Resource(resource_id) => Self::Linear(resource_id.as_usize()),
            CircuitUnit::Copyable(wire) => Self::Wire(wire),
        }
    }
}

impl<N: HugrNode> From<hugr::CircuitUnit<N>> for CircuitUnit<N> {
    fn from(value: hugr::CircuitUnit<N>) -> Self {
        match value {
            hugr::CircuitUnit::Wire(wire) => CircuitUnit::Copyable(wire),
            hugr::CircuitUnit::Linear(resource_id) => {
                CircuitUnit::Resource(ResourceId::new(resource_id))
            }
        }
    }
}

impl<N: HugrNode> CircuitUnit<N> {
    pub(super) fn map_node<N2: HugrNode>(self, map_fn: impl FnOnce(N) -> N2) -> CircuitUnit<N2> {
        match self {
            CircuitUnit::Resource(resource_id) => CircuitUnit::Resource(resource_id),
            CircuitUnit::Copyable(wire) => {
                CircuitUnit::Copyable(Wire::new(map_fn(wire.node()), wire.source()))
            }
        }
    }

    /// Returns true if this is a resource value.
    pub fn is_resource(&self) -> bool {
        matches!(self, CircuitUnit::Resource(..))
    }

    /// Returns true if this is a copyable value.
    pub fn is_copyable(&self) -> bool {
        matches!(self, CircuitUnit::Copyable(..))
    }

    /// Extract the ResourceId and Position if this is a resource.
    pub fn as_resource(&self) -> Option<ResourceId> {
        match self {
            CircuitUnit::Resource(id) => Some(*id),
            CircuitUnit::Copyable(..) => None,
        }
    }

    /// Extract the wire if this is a copyable value.
    pub fn as_copyable_wire(&self) -> Option<Wire<N>> {
        match self {
            CircuitUnit::Resource(..) => None,
            CircuitUnit::Copyable(wire) => Some(*wire),
        }
    }

    /// Sentinel value used for uninitialized ports.
    pub(super) fn sentinel() -> Self {
        CircuitUnit::Resource(ResourceId::new(usize::MAX))
    }

    pub(super) fn is_sentinel(&self) -> bool {
        self == &Self::sentinel()
    }
}

/// Map from port indices to values.
///
/// For an operation with n_in incoming ports and n_out outgoing ports:
/// - Port i (incoming) maps to index i
/// - Port j (outgoing) maps to index n_in + j
#[derive(Debug, Clone)]
pub(super) struct PortMap<T> {
    vec: Vec<T>,
    num_inputs: usize,
}

impl<T> PortMap<T> {
    pub(super) fn with_default(default: T, signature: &Signature) -> Self
    where
        T: Clone,
    {
        let num_inputs = signature.input_count();
        let num_outputs = signature.output_count();

        debug_assert!(
            signature.input_ports().all(|p| p.index() < num_inputs),
            "dataflow in ports are not in range 0..num_inputs"
        );
        debug_assert!(
            signature.output_ports().all(|p| p.index() < num_outputs),
            "dataflow out ports are not in range 0..num_outputs"
        );

        Self {
            vec: vec![default; num_inputs + num_outputs],
            num_inputs,
        }
    }

    pub(super) fn map<U>(self, mut map_fn: impl FnMut(T) -> U) -> PortMap<U> {
        PortMap {
            vec: self.vec.into_iter().map(|t| map_fn(t)).collect(),
            num_inputs: self.num_inputs,
        }
    }

    fn index(&self, port: impl Into<Port>) -> usize {
        let port = port.into();
        match port.direction() {
            Direction::Incoming => port.index(),
            Direction::Outgoing => self.num_inputs + port.index(),
        }
    }

    pub(super) fn get(&self, port: impl Into<Port>) -> &T {
        let port = port.into();
        let index = self.index(port);
        &self.vec[index]
    }

    pub(super) fn get_slice(&self, dir: Direction) -> &[T] {
        match dir {
            Direction::Incoming => &self.vec[..self.num_inputs],
            Direction::Outgoing => &self.vec[self.num_inputs..],
        }
    }

    pub(super) fn set(&mut self, port: impl Into<Port>, value: impl Into<T>) {
        let port = port.into();
        let index = self.index(port);
        self.vec[index] = value.into();
    }

    #[allow(unused)]
    pub(super) fn iter(&self) -> impl Iterator<Item = (Port, &T)> {
        let (inp_slice, out_slice) = self.vec.split_at(self.num_inputs);
        let inp_ports = (0..).map(IncomingPort::from).map_into();
        let out_ports = (0..).map(OutgoingPort::from).map_into();
        let inp = inp_ports.zip(inp_slice);
        let out = out_ports.zip(out_slice);
        inp.chain(out)
    }

    #[allow(unused)]
    pub(super) fn values(&self) -> impl Iterator<Item = &T> {
        self.vec.iter()
    }
}

/// Allocator for ResourceIds that ensures they are assigned in increasing
/// order.
#[derive(Debug, Clone)]
pub struct ResourceAllocator {
    next_id: usize,
}

impl ResourceAllocator {
    /// Create a new ResourceAllocator starting from ID 0.
    pub fn new() -> Self {
        Self { next_id: 0 }
    }

    /// Allocate the next available ResourceId.
    pub fn allocate(&mut self) -> ResourceId {
        let id = ResourceId::new(self.next_id);
        self.next_id += 1;
        id
    }
}

impl Default for ResourceAllocator {
    fn default() -> Self {
        Self::new()
    }
}
