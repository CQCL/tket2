//! A 2d array-like representation of simple quantum circuits.

mod hash;
mod match_op;
mod rewrite;

pub use rewrite::{BoxedStaticRewrite, StaticRewrite};

use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
    fmt, mem,
    rc::Rc,
};

use hugr::{Direction, HugrView, Port, PortIndex};
pub(crate) use match_op::MatchOp;

use derive_more::{From, Into};
use serde::{Deserialize, Deserializer};
use thiserror::Error;

use crate::{
    circuit::{units::filter, CircuitCostTrait, RemoveEmptyWire},
    Circuit, CircuitMutError,
};

/// A circuit with a fixed number of qubits numbered from 0 to `num_qubits - 1`.
#[derive(Clone, Default, serde::Serialize)]
pub struct StaticSizeCircuit {
    /// All quantum operations on qubits.
    qubit_ops: Vec<Vec<Rc<MatchOp>>>,
    /// Map operations to their locations in `qubit_ops`.
    #[serde(skip)]
    op_locations: BTreeMap<MatchOpPtr, Vec<OpLocation>>,
}

type MatchOpPtr = *const MatchOp;

/// The location of an operation in a `StaticSizeCircuit`.
///
/// Given by the qubit index and the position within that qubit's op list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OpLocation {
    /// The index of the qubit the operation acts on.
    pub qubit: StaticQubitIndex,
    /// The index of the operation in the qubit's operation list.
    pub op_idx: usize,
}

impl OpLocation {
    pub(crate) fn try_add_op_idx(self, op_idx: isize) -> Option<Self> {
        Some(Self {
            op_idx: self.op_idx.checked_add_signed(op_idx)?,
            ..self
        })
    }
}

impl StaticSizeCircuit {
    /// Create an empty `StaticSizeCircuit` with the given number of qubits.
    pub fn with_qubit_count(qubit_count: usize) -> Self {
        Self {
            qubit_ops: vec![Vec::new(); qubit_count],
            op_locations: BTreeMap::new(),
        }
    }

    /// Returns the number of qubits in the circuit.
    pub fn qubit_count(&self) -> usize {
        self.qubit_ops.len()
    }

    pub fn n_ops(&self) -> usize {
        self.op_locations.len()
    }

    /// Returns an iterator over the qubits in the circuit.
    pub fn qubits_iter(&self) -> impl ExactSizeIterator<Item = StaticQubitIndex> + '_ {
        (0..self.qubit_count()).map(StaticQubitIndex)
    }

    /// Returns the operations on a given qubit.
    pub fn qubit_ops(&self, qubit: StaticQubitIndex) -> &[Rc<MatchOp>] {
        &self.qubit_ops[qubit.0]
    }

    pub(crate) fn get_rc(&self, loc: OpLocation) -> Option<&Rc<MatchOp>> {
        self.qubit_ops.get(loc.qubit.0)?.get(loc.op_idx)
    }

    pub fn get(&self, loc: OpLocation) -> Option<&MatchOp> {
        self.get_rc(loc).map(|op| op.as_ref())
    }

    pub fn get_ptr(&self, loc: OpLocation) -> Option<MatchOpPtr> {
        self.get_rc(loc).map(Rc::as_ptr)
    }

    fn exists(&self, loc: OpLocation) -> bool {
        self.qubit_ops
            .get(loc.qubit.0)
            .map_or(false, |ops| ops.get(loc.op_idx).is_some())
    }

    /// The port of the operation that `loc` is at.
    pub(crate) fn qubit_port(&self, loc: OpLocation) -> usize {
        let op = self.get_rc(loc).unwrap();
        self.op_locations(op)
            .iter()
            .position(|l| l == &loc)
            .unwrap()
    }

    /// Get an equivalent location for the op at `loc` but at `port`.
    ///
    /// Every op corresponds to as many locations as it has qubits. This
    /// function returns the location of the op at `loc` but at `port`.
    pub fn equivalent_location(&self, loc: OpLocation, port: usize) -> Option<OpLocation> {
        let op = self.get_rc(loc)?;
        self.op_locations(op).get(port).copied()
    }

    pub fn all_locations(&self) -> impl Iterator<Item = OpLocation> + '_ {
        self.qubits_iter().flat_map(|qb| self.qubit_locations(qb))
    }

    pub fn qubit_locations(&self, qb: StaticQubitIndex) -> impl Iterator<Item = OpLocation> {
        (0..self.qubit_ops(qb).len()).map(move |op_idx| OpLocation { qubit: qb, op_idx })
    }

    /// Returns the location and port of the operation linked to the given
    /// operation at the given port.
    pub fn linked_op(&self, loc: OpLocation, port: Port) -> Option<(Port, OpLocation)> {
        let loc = self.equivalent_location(loc, port.index())?;
        match port.direction() {
            Direction::Outgoing => {
                let next_loc = OpLocation {
                    qubit: loc.qubit,
                    op_idx: loc.op_idx + 1,
                };
                if self.exists(next_loc) {
                    let index = self.qubit_port(next_loc);
                    Some((Port::new(Direction::Incoming, index), next_loc))
                } else {
                    None
                }
            }
            Direction::Incoming => {
                if loc.op_idx == 0 {
                    None
                } else {
                    let prev_loc = OpLocation {
                        qubit: loc.qubit,
                        op_idx: loc.op_idx - 1,
                    };
                    let index = self.qubit_port(prev_loc);
                    Some((Port::new(Direction::Outgoing, index), prev_loc))
                }
            }
        }
    }

    pub(crate) fn op_locations(&self, op: &Rc<MatchOp>) -> &[OpLocation] {
        self.op_locations[&Rc::as_ptr(op)].as_slice()
    }

    fn append_op(&mut self, op: MatchOp, qubits: impl IntoIterator<Item = StaticQubitIndex>) {
        let qubits = qubits.into_iter();
        let op = Rc::new(op);
        let op_ptr = Rc::as_ptr(&op);
        for qubit in qubits {
            if qubit.0 >= self.qubit_count() {
                panic!(
                    "Cannot add op on qubit {qubit:?} to circuit with {} qubits",
                    self.qubit_count()
                );
            }
            let op_idx = self.qubit_ops[qubit.0].len();
            self.qubit_ops[qubit.0].push(op.clone());
            self.op_locations
                .entry(op_ptr)
                .or_default()
                .push(OpLocation { qubit, op_idx });
        }
    }

    #[allow(unused)]
    fn all_ops_iter(&self) -> impl Iterator<Item = &Rc<MatchOp>> {
        self.qubit_ops.iter().flat_map(|ops| ops.iter())
    }

    /// Add `other` on a new set of qubits.
    pub fn add_subcircuit(
        &mut self,
        other: &Self,
        ops: &BTreeSet<MatchOpPtr>,
    ) -> BTreeMap<OpLocation, OpLocation> {
        // The new qubits to append to self
        let mut append_qubit_ops = Vec::new();
        // The current new qubit being added
        let mut new_qubit_ops = Vec::new();
        // Map from locations in `other` to new locations in (future) `self`
        let mut op_location_map = BTreeMap::new();

        for qb in other.qubits_iter() {
            for other_loc in other.qubit_locations(qb) {
                let op = other.get_rc(other_loc).unwrap();
                let new_qubit_idx = StaticQubitIndex(self.qubit_count() + append_qubit_ops.len());
                let new_op_idx = new_qubit_ops.len();
                if ops.contains(&Rc::as_ptr(op)) {
                    new_qubit_ops.push(op.clone());
                    op_location_map.insert(
                        other_loc,
                        OpLocation {
                            qubit: new_qubit_idx,
                            op_idx: new_op_idx,
                        },
                    );
                } else {
                    if !new_qubit_ops.is_empty() {
                        append_qubit_ops.push(mem::take(&mut new_qubit_ops));
                    }
                }
            }
            if !new_qubit_ops.is_empty() {
                append_qubit_ops.push(mem::take(&mut new_qubit_ops));
            }
        }

        self.qubit_ops.extend(append_qubit_ops);
        dbg!(&op_location_map);
        self.op_locations
            .extend(other.op_locations.iter().filter_map(|(op, locs)| {
                Some((
                    *op,
                    locs.iter()
                        .map(|loc| op_location_map.get(loc).cloned())
                        .collect::<Option<Vec<_>>>()?,
                ))
            }));
        op_location_map
    }

    pub(crate) fn merge_qubits(
        &mut self,
        StaticQubitIndex(mut left_qubit): StaticQubitIndex,
        StaticQubitIndex(right_qubit): StaticQubitIndex,
    ) -> StaticQubitIndex {
        assert_ne!(left_qubit, right_qubit);

        let left_offset = self.qubit_ops[left_qubit].len();

        // Append right qubit ops to left qubit
        let right_ops = mem::take(&mut self.qubit_ops[right_qubit]);
        self.qubit_ops[left_qubit].extend(right_ops);
        self.qubit_ops.remove(right_qubit);

        if left_qubit > right_qubit {
            left_qubit -= 1;
        }

        // Update op locations
        // TODO: check if this preserves convexity
        for ops in self.op_locations.values_mut() {
            for op in ops.iter_mut() {
                match op.qubit.0.cmp(&right_qubit) {
                    Ordering::Equal => {
                        op.qubit = StaticQubitIndex(left_qubit);
                        op.op_idx += left_offset;
                    }
                    Ordering::Greater => {
                        op.qubit = StaticQubitIndex(op.qubit.0 - 1);
                    }
                    _ => {}
                }
            }
        }
        StaticQubitIndex(left_qubit)
    }
}

/// A qubit index within a `StaticSizeCircuit`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, From, Into)]
pub struct StaticQubitIndex(pub(crate) usize);

// TODO: this is unsafe but was added for ECCRewriter to work.
impl<H: HugrView> From<H> for StaticSizeCircuit {
    fn from(hugr: H) -> Self {
        let circuit = Circuit::from(hugr);
        (&circuit).try_into().unwrap()
    }
}

impl<H: HugrView> TryFrom<&Circuit<H>> for StaticSizeCircuit {
    type Error = StaticSizeCircuitError;

    fn try_from(circuit: &Circuit<H>) -> Result<Self, Self::Error> {
        let mut res = Self::with_qubit_count(circuit.qubit_count());
        for cmd in circuit.commands() {
            let qubits = cmd
                .units(Direction::Incoming)
                .map(|unit| {
                    let Some((qb, _, _)) = filter::filter_qubit(unit) else {
                        return Err(StaticSizeCircuitError::NonQubitInput);
                    };
                    Ok(qb)
                })
                .collect::<Result<Vec<_>, _>>()?;
            if cmd.units(Direction::Outgoing).count() != qubits.len() {
                return Err(StaticSizeCircuitError::InvalidCircuit);
            }
            let op = cmd.optype().clone().into();
            res.append_op(op, qubits.into_iter().map(|u| StaticQubitIndex(u.index())));
        }
        Ok(res)
    }
}

/// Errors that can occur when converting a `Circuit` to a `StaticSizeCircuit`.
#[derive(Debug, Error)]
pub enum StaticSizeCircuitError {
    /// An input to a gate was not a qubit.
    #[error("Only qubits are supported as inputs")]
    NonQubitInput,

    /// The given tket2 circuit cannot be expressed as a StaticSizeCircuit.
    #[error("The given tket2 circuit cannot be expressed as a StaticSizeCircuit")]
    InvalidCircuit,
}

impl fmt::Debug for StaticSizeCircuit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StaticSizeCircuit")
            .field("qubit_ops", &self.qubit_ops)
            .finish()
    }
}

impl PartialEq for StaticSizeCircuit {
    fn eq(&self, other: &Self) -> bool {
        self.qubit_ops == other.qubit_ops
    }
}

impl Eq for StaticSizeCircuit {}

impl<'de> Deserialize<'de> for StaticSizeCircuit {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct StaticSizeCircuitHelper {
            qubit_ops: Vec<Vec<Rc<MatchOp>>>,
        }

        let helper = StaticSizeCircuitHelper::deserialize(deserializer)?;
        let mut op_locations = BTreeMap::<MatchOpPtr, Vec<OpLocation>>::new();

        for (qubit, ops) in helper.qubit_ops.iter().enumerate() {
            for (op_idx, op) in ops.iter().enumerate() {
                let op_ptr = Rc::as_ptr(op);
                op_locations.entry(op_ptr).or_default().push(OpLocation {
                    qubit: StaticQubitIndex(qubit),
                    op_idx,
                });
            }
        }

        Ok(StaticSizeCircuit {
            qubit_ops: helper.qubit_ops,
            op_locations,
        })
    }
}

impl CircuitCostTrait for StaticSizeCircuit {
    fn circuit_cost<F, C>(&self, op_cost: F) -> C
    where
        Self: Sized,
        C: std::iter::Sum,
        F: Fn(&hugr::ops::OpType) -> C,
    {
        todo!()
    }

    fn nodes_cost<F, C>(&self, nodes: impl IntoIterator<Item = hugr::Node>, op_cost: F) -> C
    where
        C: std::iter::Sum,
        F: Fn(&hugr::ops::OpType) -> C,
    {
        todo!()
    }
}

impl RemoveEmptyWire for StaticSizeCircuit {
    fn remove_empty_wire(&mut self, input_port: usize) -> Result<(), CircuitMutError> {
        todo!()
    }

    fn empty_wires(&self) -> Vec<usize> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use hugr::Port;
    use portgraph::PortOffset;
    use rstest::rstest;

    use super::StaticSizeCircuit;
    use crate::ops::Tk2Op;
    use crate::static_circ::OpLocation;
    use crate::utils::build_simple_circuit;

    #[test]
    fn test_convert_to_static_size_circuit() {
        // Create a circuit with 2 qubits, a CX gate, and two H gates
        let circuit = build_simple_circuit(2, |circ| {
            circ.append(Tk2Op::H, [0])?;
            circ.append(Tk2Op::CX, [0, 1])?;
            circ.append(Tk2Op::H, [1])?;
            Ok(())
        })
        .unwrap();

        // Convert the circuit to StaticSizeCircuit
        let static_circuit: StaticSizeCircuit = (&circuit).try_into().unwrap();

        // Check the conversion
        assert_eq!(static_circuit.qubit_count(), 2);
        assert_eq!(static_circuit.qubit_ops(0.into()).len(), 2); // H gate on qubit 0
        assert_eq!(static_circuit.qubit_ops(1.into()).len(), 2); // CX and H gate on qubit 1
    }

    #[rstest]
    #[case(PortOffset::Outgoing(0), None)]
    #[case(PortOffset::Incoming(1), None)]
    #[case(
        PortOffset::Outgoing(1),
        Some((PortOffset::Incoming(0).into(), OpLocation {
            qubit: 1.into(),
            op_idx: 1,
        }))
    )]
    #[case(
        PortOffset::Incoming(0),
        Some((PortOffset::Outgoing(0).into(), OpLocation {
            qubit: 0.into(),
            op_idx: 0,
        }))
    )]
    fn test_linked_op(#[case] port: PortOffset, #[case] expected_loc: Option<(Port, OpLocation)>) {
        let circuit = build_simple_circuit(2, |circ| {
            circ.append(Tk2Op::H, [0])?;
            circ.append(Tk2Op::CX, [0, 1])?;
            circ.append(Tk2Op::H, [1])?;
            Ok(())
        })
        .unwrap();
        // Convert the circuit to StaticSizeCircuit
        let static_circuit: StaticSizeCircuit = (&circuit).try_into().unwrap();

        // Define the location of the CX gate
        let cx_location = OpLocation {
            qubit: 0.into(),
            op_idx: 1,
        };

        // Define the port for the CX gate
        let cx_port = port.into();

        // Get the linked operation for the CX gate
        let linked_op_location = static_circuit.linked_op(cx_location, cx_port);

        // Check if the linked operation is correct
        assert_eq!(linked_op_location, expected_loc);
    }

    #[test]
    fn test_add_subcircuit() {
        // Create a main circuit
        let main_circuit = build_simple_circuit(3, |circ| {
            circ.append(Tk2Op::H, [0])?;
            circ.append(Tk2Op::CX, [0, 1])?;
            Ok(())
        })
        .unwrap();
        let mut main_circuit: StaticSizeCircuit = (&main_circuit).try_into().unwrap();

        // Create a subcircuit
        let sub_circuit = build_simple_circuit(2, |circ| {
            circ.append(Tk2Op::X, [0])?;
            circ.append(Tk2Op::X, [0])?;
            circ.append(Tk2Op::X, [0])?;
            circ.append(Tk2Op::Y, [1])?;
            Ok(())
        })
        .unwrap();
        let sub_circuit: StaticSizeCircuit = (&sub_circuit).try_into().unwrap();

        // Define the nodes to add from the subcircuit
        let nodes = BTreeSet::from_iter(
            [
                OpLocation {
                    qubit: 0.into(),
                    op_idx: 0,
                },
                OpLocation {
                    qubit: 0.into(),
                    op_idx: 2,
                },
                OpLocation {
                    qubit: 1.into(),
                    op_idx: 0,
                },
            ]
            .into_iter()
            .map(|loc| sub_circuit.get_ptr(loc).unwrap()),
        );

        // Add the subcircuit to the main circuit
        let node_map = main_circuit.add_subcircuit(&sub_circuit, &nodes);

        // Check if the nodes were added correctly
        assert_eq!(node_map.len(), 3);

        // Three original qubits + first new qubit split into two + second new qubit
        assert_eq!(main_circuit.qubit_count(), 6);

        // Check if the operations were added to the main circuit
        assert_eq!(main_circuit.qubit_ops(3.into()).len(), 1); // First X gate
        assert_eq!(main_circuit.qubit_ops(4.into()).len(), 1); // Second X gate
        assert_eq!(main_circuit.qubit_ops(5.into()).len(), 1); // Y gate

        insta::assert_debug_snapshot!(main_circuit);
    }
}
