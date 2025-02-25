//! History of circuit transformations.

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    hash::Hash,
};

use derive_where::derive_where;
use hugr::{hugr::hugrmut::HugrMut, Hugr, HugrView, Node, Port};
use itertools::Itertools;
use relrc::RelRcGraph;

use crate::{circuit::HashError, Circuit};

use super::{CircuitDiff, CircuitDiffData, CircuitDiffError, CircuitDiffPtr, InvalidNodes, Owned};

/// A set of compatible diffs
#[derive_where(Clone)]
pub struct CircuitHistory<H> {
    /// The root of the history that dominates all other diffs
    pub(super) root: CircuitDiff<H>,
    /// The set of diffs in the history, stored as a relrc graph
    pub(super) diffs: RelRcGraph<CircuitDiffData<H>, InvalidNodes>,
}

impl<H: HugrView<Node = Node>> CircuitHistory<H> {
    /// Create a (trivial) history from a circuit
    ///
    /// This will fail if the conversion from circuit to diff fails.
    pub fn try_from_circuit(circuit: Circuit<H>) -> Result<Self, HashError> {
        let diff = CircuitDiff::try_from_circuit(circuit)?;
        Ok(Self::from_diff(diff))
    }

    /// Create a new history for a single diff
    ///
    /// This will include `diff` and all its ancestors
    pub fn from_diff(diff: CircuitDiff<H>) -> Self {
        let graph = RelRcGraph::from_sinks(vec![diff.0]);
        let root = get_root(&graph).expect("no unique root found").into();
        Self { root, diffs: graph }
    }

    /// Create the joint history of multiple diffs
    ///
    /// This will fail if any of the diffs are incompatible, returning a
    /// [`CircuitDiffError::ConflictingDiffs`] in that case.
    pub fn try_from_diffs(
        diffs: impl IntoIterator<Item = CircuitDiff<H>>,
    ) -> Result<Self, CircuitDiffError> {
        let mut histories = diffs.into_iter().map(Self::from_diff);
        histories
            .try_fold(None, |mut acc: Option<Self>, diff| {
                if let Some(acc) = acc.as_mut() {
                    acc.merge(diff)?;
                } else {
                    acc = Some(diff);
                }
                Ok(acc)
            })
            .and_then(|res| res.ok_or(CircuitDiffError::EmptyHistory))
    }

    /// Merge two histories
    ///
    /// This will fail if the histories are incompatible, returning a
    /// [`CircuitDiffError::ConflictingDiffs`] in that case.
    pub fn merge(&mut self, other: Self) -> Result<(), CircuitDiffError> {
        if !self.root.0.ptr_eq(&other.root.0) {
            // TODO: we currently disallow merging histories with distinct roots
            // in the future, this could instead compute the dominator node of
            // the two histories and merge the histories at that node
            return Err(CircuitDiffError::DistinctRoots);
        }
        self.diffs.merge(other.diffs, |_, self_edges, other_edges| {
            let self_invalidated = self_edges.iter().flat_map(|e| &e.value().0);
            let other_invalidated = other_edges.iter().flat_map(|e| &e.value().0);
            let mut invalidated = self_invalidated.chain(other_invalidated);
            if invalidated.all_unique() {
                Ok(())
            } else {
                Err(CircuitDiffError::ConflictingDiffs)
            }
        })
    }

    /// Check if a diff is in the history
    pub fn contains_diff(&self, diff: &CircuitDiff<H>) -> bool {
        self.diffs.all_nodes().contains(&(&diff.0).into())
    }

    /// Check if a diff is the root of the history
    pub fn is_root(&self, diff: &CircuitDiff<H>) -> bool {
        self.root.0.ptr_eq(&diff.0)
    }

    /// Check if a port is valid in the history
    ///
    /// A port is valid if a) its owner is in the history, b) the port is not
    /// invalidated by any child diff in the history and c) it is not
    /// the input or output port of a non-root diff.
    pub fn is_valid_port(&self, node_port: &Owned<H, (Node, Port)>) -> bool {
        let diff = &node_port.owner;
        let node = node_port.data.0;
        // check a) the owner is in the history
        if !self.contains_diff(&diff) {
            return false;
        }
        // check c) it is not the input or output port of a non-root diff
        if !self.is_root(diff) && diff.io_nodes().contains(&node) {
            return false;
        }
        // check b) the port is not invalidated by any child diff
        let mut out_edges = diff.0.all_outgoing().into_iter().filter(|e| {
            let target = e.target();
            self.contains_diff(&CircuitDiff(target.clone()))
        });
        out_edges.all(|e| !e.value().0.contains(&node))
    }

    /// Get the valid ports opposite to a given port in `self`
    ///
    /// The set of edges given by [`Self::linked_ports`] always defines a valid
    /// Hugr.
    pub fn linked_ports(
        &self,
        node_port: Owned<H, (Node, Port)>,
    ) -> impl Iterator<Item = Owned<H, (Node, Port)>> + Clone + '_ {
        let valid_diff = |diff: &CircuitDiff<H>| self.contains_diff(diff);
        self.equivalent_ports(node_port, valid_diff)
            .flat_map(|node_port| {
                let (node, port) = node_port.data;
                let diff = node_port.owner;
                let to_owned = |data| Owned {
                    owner: diff.clone(),
                    data,
                };
                diff.as_hugr()
                    .linked_ports(node, port)
                    .map(to_owned)
                    .collect_vec()
            })
            .filter(|node_port| self.is_valid_port(&node_port))
            .unique_by(|node_port| (node_port.owner.as_ptr(), node_port.data))
    }

    /// All ports equivalent to a given port.
    ///
    /// The returned ports may not be in `self`. Will traverse all ancestors
    /// and descendants of `node_port` for as long as `valid_diff` returns
    /// true.
    pub fn equivalent_ports(
        &self,
        node_port: Owned<H, (Node, Port)>,
        valid_diff: impl FnMut(&CircuitDiff<H>) -> bool + Clone,
    ) -> impl Iterator<Item = Owned<H, (Node, Port)>> + Clone {
        let current = VecDeque::from([node_port]);
        EquivalentPortsIter {
            visited: BTreeSet::new(),
            current,
            valid_diff,
        }
    }

    /// Build a hugr equivalent to the history
    pub fn extract_hugr(&self) -> Hugr {
        use super::experimental::ExperimentalHugrWrapper;
        let exp_self = ExperimentalHugrWrapper(self.clone());
        let mut hugr = Hugr::new(exp_self.get_optype(exp_self.root()).clone());
        let hugr_root = hugr.root();

        let mut node_map = BTreeMap::new();

        // Add input/output node
        let [in_node, out_node] = exp_self.get_io(exp_self.root()).unwrap();
        let new_in_node =
            hugr.add_node_with_parent(hugr_root, exp_self.get_optype(in_node).clone());
        node_map.insert(in_node, new_in_node);
        let new_out_node =
            hugr.add_node_with_parent(hugr_root, exp_self.get_optype(out_node).clone());
        node_map.insert(out_node, new_out_node);

        // Add all other nodes
        for node in exp_self.nodes() {
            if ![in_node, out_node, exp_self.root()].contains(&node) {
                let new_node =
                    hugr.add_node_with_parent(hugr_root, exp_self.get_optype(node).clone());
                node_map.insert(node, new_node);
            }
        }

        // Add all edges
        for (&old_out_node, &new_out_node) in node_map.iter() {
            for out_port in exp_self.node_outputs(old_out_node) {
                for (old_in_node, in_port) in exp_self.linked_inputs(old_out_node, out_port) {
                    let new_in_node = node_map[&old_in_node];
                    hugr.connect(new_out_node, out_port, new_in_node, in_port);
                }
            }
        }

        hugr
    }
}

/// Return the unique node with no parents in `graph`
fn get_root<N: Hash, E: Hash>(graph: &relrc::RelRcGraph<N, E>) -> Option<relrc::RelRc<N, E>> {
    let nodes = graph.all_nodes().iter().map(|&n| graph.get_node_rc(n));
    nodes
        .filter(|n| n.all_parents().next().is_none())
        .exactly_one()
        .ok()
}

/// Iterator over all equivalent ports in a history
#[derive_where(Clone; F)]
pub struct EquivalentPortsIter<H, F> {
    visited: BTreeSet<(CircuitDiffPtr<H>, Node, Port)>,
    current: VecDeque<Owned<H, (Node, Port)>>,
    valid_diff: F,
}

fn get_key<H>(node_port: &Owned<H, (Node, Port)>) -> (CircuitDiffPtr<H>, Node, Port) {
    (node_port.owner.as_ptr(), node_port.data.0, node_port.data.1)
}

impl<H: HugrView<Node = Node>, F: FnMut(&CircuitDiff<H>) -> bool> Iterator
    for EquivalentPortsIter<H, F>
{
    type Item = Owned<H, (Node, Port)>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let node_port = self.current.pop_front()?;
            if self.visited.insert(get_key(&node_port)) {
                let Owned { owner, data } = &node_port;
                let neighbours = owner
                    .equivalent_children_ports(data.0, data.1)
                    .chain(owner.equivalent_parent_ports(data.0, data.1));
                self.current.extend(
                    neighbours
                        .filter(|p| !self.visited.contains(&get_key(p)))
                        .filter(|p| (self.valid_diff)(&p.owner)),
                );
                return Some(node_port);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use hugr::{
        builder::{DFGBuilder, Dataflow, HugrBuilder},
        extension::prelude::qb_t,
        hugr::views::SiblingSubgraph,
        types::Signature,
        Hugr, HugrView, Node, SimpleReplacement,
    };
    use itertools::Itertools;
    use portgraph::NodeIndex;
    use rstest::{fixture, rstest};

    use crate::{
        circuit::CircuitHash,
        diff::{experimental::ExperimentalHugrWrapper, CircuitDiff, CircuitHistory},
        extension::rotation::{rotation_type, RotationOp},
        rewrite::CircuitRewrite,
        Circuit, Tk2Op,
    };

    struct CircuitAndThreeRewrites {
        /// A test circuit
        circuit: Circuit<Hugr>,
        /// Three replacements to be applied as diffs
        diff_replacements: [CircuitRewrite; 3],
        /// Three replacements to be applied in the "traditional" way that are
        /// equivalent to the diff replacements
        flat_replacements: [CircuitRewrite; 3],
    }

    #[fixture]
    fn circuit_and_three_rewrites() -> CircuitAndThreeRewrites {
        // Create the main circuit with [rz(a), rz(b), h, rx(a+b), h]
        let sum_ab;
        let circuit = {
            let mut builder = DFGBuilder::new(Signature::new(
                vec![qb_t(), rotation_type(), rotation_type()], // qb, a, b
                vec![qb_t()],
            ))
            .unwrap();

            let [qb, a, b] = builder.input_wires_arr();

            // Gate 1: rz(a)
            let op1 = builder.add_dataflow_op(Tk2Op::Rz, vec![qb, a]).unwrap();
            let [qb1] = op1.outputs_arr();

            // Gate 2: rz(b)
            let op2 = builder.add_dataflow_op(Tk2Op::Rz, vec![qb1, b]).unwrap();
            let [qb2] = op2.outputs_arr();

            // Gate 3: h
            let op3 = builder.add_dataflow_op(Tk2Op::H, vec![qb2]).unwrap();
            let [qb3] = op3.outputs_arr();

            // Gate 4: rx(a+b)
            let op_add = builder
                .add_dataflow_op(RotationOp::radd, vec![a, b])
                .unwrap();
            sum_ab = op_add.outputs().next();
            let op4 = builder
                .add_dataflow_op(Tk2Op::Rx, vec![qb3, sum_ab.unwrap()])
                .unwrap();
            let [qb4] = op4.outputs_arr();

            // Gate 5: h
            let op5 = builder.add_dataflow_op(Tk2Op::H, vec![qb4]).unwrap();
            let [qb_final] = op5.outputs_arr();

            builder.set_outputs(vec![qb_final]).unwrap();
            let hugr = builder.finish_hugr().unwrap();
            let root = hugr.root();
            Circuit::new(hugr, root)
        };

        // Create the three replacements
        let mut flat_replacements = Vec::with_capacity(3);
        let mut diff_replacements = Vec::with_capacity(3);

        // 1. Merge rz(a) rz(b) into rz(a+b)
        let replacement: CircuitRewrite = {
            let subgraph = SiblingSubgraph::try_from_nodes(
                (3..=4).map(NodeIndex::new).map_into().collect_vec(),
                circuit.hugr(),
            )
            .unwrap();
            let repl = {
                let mut builder = DFGBuilder::new(Signature::new(
                    vec![qb_t(), rotation_type(), rotation_type()],
                    vec![qb_t()],
                ))
                .unwrap();
                let [qb, a, b] = builder.input_wires_arr();
                let op_add = builder
                    .add_dataflow_op(RotationOp::radd, vec![a, b])
                    .unwrap();
                let [sum] = op_add.outputs_arr();
                let op_rz = builder.add_dataflow_op(Tk2Op::Rz, vec![qb, sum]).unwrap();
                let [qb_out] = op_rz.outputs_arr();
                builder.set_outputs(vec![qb_out]).unwrap();
                builder.finish_hugr().unwrap()
            };
            let repl_circ = Circuit::new(&repl, repl.root());
            let nu_inp = repl
                .all_linked_inputs(repl_circ.input_node())
                .zip(circuit.hugr().all_linked_inputs(circuit.input_node()))
                .collect();
            let nu_out = {
                let last_node: Node = NodeIndex::new(4).into();
                (circuit.hugr().linked_inputs(last_node, 0))
                    .zip(repl.node_inputs(repl_circ.output_node()))
                    .collect()
            };
            SimpleReplacement::new(subgraph, repl, nu_inp, nu_out).into()
        };
        diff_replacements.push(replacement.clone());
        flat_replacements.push(replacement);

        // 2. Replace h rx(x) h with rz(x)
        let replacement: CircuitRewrite = {
            let subgraph = SiblingSubgraph::try_from_nodes(
                [5, 9, 10].map(NodeIndex::new).map(Into::into),
                circuit.hugr(),
            )
            .unwrap();
            let repl = {
                let mut builder =
                    DFGBuilder::new(Signature::new(vec![qb_t(), rotation_type()], vec![qb_t()]))
                        .unwrap();
                let [qb, x] = builder.input_wires_arr();
                let op_rz = builder.add_dataflow_op(Tk2Op::Rz, vec![qb, x]).unwrap();
                let [qb_out] = op_rz.outputs_arr();
                builder.set_outputs(vec![qb_out]).unwrap();
                builder.finish_hugr().unwrap()
            };
            let repl_circ = Circuit::new(&repl, repl.root());
            let nu_inp = {
                let sum_ab_target = circuit
                    .hugr()
                    .single_linked_input(sum_ab.unwrap().node(), sum_ab.unwrap().source())
                    .unwrap();
                let first_node: Node = NodeIndex::new(5).into();
                repl.all_linked_inputs(repl_circ.input_node())
                    .zip([(first_node, 0.into()), sum_ab_target])
                    .collect()
            };
            let nu_out = HashMap::from_iter([((circuit.output_node(), 0.into()), 0.into())]);
            SimpleReplacement::new(subgraph, repl, nu_inp, nu_out).into()
        };
        diff_replacements.push(replacement.clone());
        flat_replacements.push(replacement);

        // 3. Merge rz(x) rz(y) into rz(x + y)
        let replacement: CircuitRewrite = {
            let subgraph = {
                let mut circuit = circuit.clone();
                flat_replacements[0].clone().apply(&mut circuit).unwrap();
                flat_replacements[1].clone().apply(&mut circuit).unwrap();
                SiblingSubgraph::try_from_nodes(
                    [15, 16].map(NodeIndex::new).map(Into::into),
                    circuit.hugr(),
                )
                .unwrap()
            };
            let repl = {
                let mut builder = DFGBuilder::new(Signature::new(
                    vec![qb_t(), rotation_type(), rotation_type()],
                    vec![qb_t()],
                ))
                .unwrap();
                let [qb, x, y] = builder.input_wires_arr();
                let op_add = builder
                    .add_dataflow_op(RotationOp::radd, vec![x, y])
                    .unwrap();
                let [sum] = op_add.outputs_arr();
                let op_rz = builder.add_dataflow_op(Tk2Op::Rz, vec![qb, sum]).unwrap();
                let [qb_out] = op_rz.outputs_arr();
                builder.set_outputs(vec![qb_out]).unwrap();
                builder.finish_hugr().unwrap()
            };
            let repl_circ = Circuit::new(&repl, repl.root());
            let nu_inp = {
                let first_rz: Node = NodeIndex::new(15).into();
                let second_rz: Node = NodeIndex::new(16).into();
                repl.all_linked_inputs(repl_circ.input_node())
                    .zip([
                        (first_rz, 0.into()),
                        (first_rz, 1.into()),
                        (second_rz, 1.into()),
                    ])
                    .collect()
            };
            let nu_out = HashMap::from_iter([((circuit.output_node(), 0.into()), 0.into())]);
            SimpleReplacement::new(subgraph, repl, nu_inp, nu_out).into()
        };
        diff_replacements.push(replacement.clone());
        flat_replacements.push(replacement); // dummy

        let diff_replacements: [CircuitRewrite; 3] = diff_replacements.try_into().unwrap();
        let flat_replacements: [CircuitRewrite; 3] = flat_replacements.try_into().unwrap();
        CircuitAndThreeRewrites {
            circuit,
            diff_replacements,
            flat_replacements,
        }
    }

    fn compare_hashes(circuit: &Circuit<Hugr>, diffs: impl IntoIterator<Item = CircuitDiff>) {
        let flat_hash = circuit.circuit_hash().unwrap();
        let history = CircuitHistory::try_from_diffs(diffs).unwrap();
        let exp_history: ExperimentalHugrWrapper<_> = history.into();
        let diff_hash = exp_history.circuit_hash().unwrap();
        assert_eq!(flat_hash, diff_hash);
    }

    #[rstest]
    fn test_history(circuit_and_three_rewrites: CircuitAndThreeRewrites) {
        let CircuitAndThreeRewrites {
            mut circuit,
            diff_replacements: [diff_rw1, diff_rw2, _diff_rw3],
            flat_replacements: [flat_rw1, flat_rw2, flat_rw3],
        } = circuit_and_three_rewrites;

        // let history = CircuitHistory::try_from_circuit(circuit.clone()).unwrap();
        let diff = CircuitDiff::try_from_circuit(circuit.clone()).unwrap();

        flat_rw1.apply(&mut circuit).unwrap();
        let new_diff1 = diff.apply_rewrite(diff_rw1).unwrap();
        compare_hashes(&circuit, [new_diff1.clone()]);

        flat_rw2.apply(&mut circuit).unwrap();
        let new_diff2 = diff.apply_rewrite(diff_rw2).unwrap();
        compare_hashes(&circuit, [new_diff1.clone(), new_diff2.clone()]);

        flat_rw3.apply(&mut circuit).unwrap();
        println!("{}", circuit.mermaid_string());
        // println!(
        //     "{}",
        //     CircuitHistory::try_from_diffs([new_diff1.clone(), new_diff2.clone()])
        //         .unwrap()
        //         .extract_hugr()
        //         .mermaid_string()
        // );
        // let new_diff3 = diff.apply_rewrite(diff_rw3).unwrap();
        // compare_hashes(
        //     &circuit,
        //     [new_diff1.clone(), new_diff2.clone(), new_diff3.clone()],
        // );
        unimplemented!("finish with a rewrite that overlaps with the previous diffs");
    }

    #[rstest]
    fn test_extract_hugr(circuit_and_three_rewrites: CircuitAndThreeRewrites) {
        let CircuitAndThreeRewrites {
            circuit,
            diff_replacements: [diff_rw1, diff_rw2, _],
            ..
        } = circuit_and_three_rewrites;

        let diff = CircuitDiff::try_from_circuit(circuit).unwrap();
        let new_diff1 = diff.apply_rewrite(diff_rw1).unwrap();
        let new_diff2 = diff.apply_rewrite(diff_rw2).unwrap();

        let history = CircuitHistory::try_from_diffs([new_diff1, new_diff2]).unwrap();
        insta::assert_snapshot!(history.extract_hugr().mermaid_string());
    }
}
