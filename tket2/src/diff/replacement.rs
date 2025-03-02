//! Replacement operations on circuit diffs

use std::collections::BTreeSet;

use hugr::{
    hugr::{
        rewrite::{HostPort, ReplacementPort},
        views::PetgraphWrapper,
    },
    HugrView, Node, Port, SimpleReplacement,
};
use itertools::{izip, Itertools};
use petgraph::visit::{depth_first_search, Control};

use crate::{rewrite::CircuitRewrite, Circuit};

use super::{
    port_to_wire, CircuitDiff, CircuitDiffData, CircuitDiffError, ParentWire, WireEquivalence,
};

impl CircuitDiff {
    /// Apply a simple replacement.
    ///
    /// The result is returned as a new diff that is a child of the
    /// current diff.
    pub fn apply_rewrite(&self, replacement: CircuitRewrite) -> Result<Self, CircuitDiffError> {
        let replacement: SimpleReplacement = replacement.into();
        replacement.is_valid_rewrite(self.as_hugr())?;

        let equivalent_wires = self.compute_wire_equivalence(&replacement)?;
        let invalid_nodes = equivalent_wires.invalidated_nodes(&[self.as_hugr()])?;

        let hugr_root = replacement.replacement().root();
        let replacement_circuit = Circuit::try_new(replacement.into_replacement(), hugr_root)?;
        let data = CircuitDiffData {
            circuit: replacement_circuit.try_into()?,
            equivalent_wires,
        };

        let parent = (
            self.clone(),
            invalid_nodes.into_iter().exactly_one().expect("one parent"),
        );

        let new_child = Self::with_parents(data, [parent]);

        Ok(new_child)
    }

    fn compute_wire_equivalence(
        &self,
        replacement: &SimpleReplacement,
    ) -> Result<WireEquivalence, CircuitDiffError> {
        let mut equivalent_wires = WireEquivalence::new();
        let to_parent_wire = |port: HostPort<Port>| -> Result<ParentWire, CircuitDiffError> {
            let HostPort(node, port) = port;
            let wire = port_to_wire(node, port, self.as_hugr())?;
            let parent_wire = ParentWire {
                incoming_index: 0, // `self` is the unique parent
                wire,
            };
            Ok(parent_wire)
        };

        // 1. Add equivalences for replacement input wires
        for (src, tgt) in replacement.incoming_boundary(self.as_hugr()) {
            let src_parent_wire = to_parent_wire(src.into())?;
            let ReplacementPort(tgt_node, tgt_port) = tgt;
            let tgt_child_wire = port_to_wire(tgt_node, tgt_port, replacement.replacement())?;
            let ret = equivalent_wires
                .input_to_parent
                .insert(tgt_child_wire, src_parent_wire);
            assert!(ret.is_none(), "input wire already exists");
        }

        // 2. Add equivalences for replacement output wires
        for (src, tgt) in replacement.outgoing_boundary(self.as_hugr()) {
            let ReplacementPort(src_node, src_port) = src;
            let src_child_wire = port_to_wire(src_node, src_port, replacement.replacement())?;
            let tgt_parent_wire = to_parent_wire(tgt.into())?;
            equivalent_wires
                .output_to_parent
                .entry(src_child_wire)
                .or_default()
                .insert(tgt_parent_wire);
        }

        // 3. Add equivalences for wires that are both input and output wires
        let [_, rep_output] = replacement
            .get_replacement_io()
            .expect("replacement is a DFG");
        for (src, tgt) in replacement.host_to_host_boundary(self.as_hugr()) {
            let ReplacementPort(_, incoming) = replacement
                .map_host_output(tgt)
                .expect("invalid host to host edge");
            let src_parent_wire = to_parent_wire(src.into())?;
            let tgt_parent_wire = to_parent_wire(tgt.into())?;
            let io_child_wire = port_to_wire(rep_output, incoming, replacement.replacement())?;
            let ret = equivalent_wires
                .input_to_parent
                .insert(io_child_wire, src_parent_wire);
            assert!(
                ret.is_none() || ret == Some(src_parent_wire),
                "input wire already exists"
            );
            equivalent_wires
                .output_to_parent
                .entry(io_child_wire)
                .or_default()
                .insert(tgt_parent_wire);
        }

        Ok(equivalent_wires)
    }
}

impl WireEquivalence {
    /// The set of wires that are invalidated by a rewrite.
    ///
    /// A wire is invalidated if
    ///  - the replacement defines a new value that replaces it
    ///  - the replacement uses the value AND it is a linear value
    fn invalidated_nodes(
        &self,
        parents: &[&impl HugrView<Node = Node>],
    ) -> Result<Vec<BTreeSet<Node>>, CircuitDiffError> {
        let mut invalidation_start = vec![BTreeSet::default(); parents.len()];
        let mut invalidation_end = vec![BTreeSet::default(); parents.len()];

        for parent_wire in self.input_to_parent.values() {
            let ParentWire {
                wire,
                incoming_index,
            } = *parent_wire;
            let hugr = &parents[incoming_index];
            if hugr
                .get_optype(wire.node())
                .port_kind(wire.source())
                .unwrap()
                .is_linear()
            {
                let (node, _) = hugr
                    .single_linked_input(wire.node(), wire.source())
                    .expect("linear wire");
                invalidation_start[incoming_index].insert(node);
            }
        }

        for parent_wires in self.output_to_parent.values() {
            for parent_wire in parent_wires {
                let ParentWire {
                    wire,
                    incoming_index,
                } = *parent_wire;
                invalidation_end[incoming_index].insert(wire.node());
            }
        }

        let mut all_invalidated_nodes = Vec::with_capacity(parents.len());

        for (starts, ends, hugr) in izip!(invalidation_start, invalidation_end, parents) {
            let petgraph_hugr = PetgraphWrapper::from(hugr);
            let mut invalid_nodes =
                BTreeSet::from_iter(starts.iter().copied().chain(ends.iter().copied()));

            use petgraph::visit::DfsEvent::*;
            depth_first_search(petgraph_hugr, starts, |event| match event {
                Discover(node, _) => {
                    if ends.contains(&node) {
                        return Ok(Control::<()>::Prune);
                    }
                    invalid_nodes.insert(node);
                    Ok(Control::Continue)
                }
                BackEdge(_, _) => Err(CircuitDiffError::Cycle),
                _ => Ok(Control::Continue),
            })?;

            all_invalidated_nodes.push(invalid_nodes);
        }

        Ok(all_invalidated_nodes)
    }
}

#[cfg(test)]
mod test {
    // tests copied and adapted from SimpleReplacement tests

    use hugr::{
        builder::{endo_sig, inout_sig, DFGBuilder, Dataflow, DataflowHugr},
        extension::prelude::bool_t,
        hugr::views::SiblingSubgraph,
        ops::{handle::NodeHandle, DataflowOpTrait, OpTag, OpTrait},
        std_extensions::logic::LogicOp,
        types::Signature,
        Hugr, HugrView, IncomingPort, Node,
    };
    use itertools::Itertools;
    use rstest::{fixture, rstest};
    use std::collections::HashMap;

    use crate::{diff::CircuitDiff, utils::build_simple_circuit, Circuit, Tk2Op};

    use super::SimpleReplacement;

    /// Creates a hugr like the following:
    /// --   H   --
    /// -- [DFG] --
    /// where [DFG] is:
    /// ┌───┐     ┌───┐
    /// ┤ H ├──■──┤ H ├
    /// ├───┤┌─┴─┐├───┤
    /// ┤ H ├┤ X ├┤ H ├
    /// └───┘└───┘└───┘
    #[fixture]
    fn simple_hugr() -> Circuit {
        build_simple_circuit(2, |circ| {
            circ.append(Tk2Op::H, [0])?;
            circ.append(Tk2Op::H, [1])?;
            circ.append(Tk2Op::CX, [0, 1])?;
            circ.append(Tk2Op::H, [0])?;
            circ.append(Tk2Op::H, [1])?;
            Ok(())
        })
        .unwrap()
    }

    /// Creates a hugr with a DFG root like the following:
    /// ┌───┐
    /// ┤ H ├──■──
    /// ├───┤┌─┴─┐
    /// ┤ H ├┤ X ├
    /// └───┘└───┘
    #[fixture]
    fn dfg_hugr() -> Circuit {
        build_simple_circuit(2, |circ| {
            circ.append(Tk2Op::H, [0])?;
            circ.append(Tk2Op::H, [1])?;
            circ.append(Tk2Op::CX, [0, 1])?;
            Ok(())
        })
        .unwrap()
    }

    /// Creates a hugr with a DFG root like the following:
    /// ─────
    /// ┌───┐
    /// ┤ H ├
    /// └───┘
    #[fixture]
    fn dfg_hugr2() -> Circuit {
        build_simple_circuit(2, |circ| {
            circ.append(Tk2Op::H, [1])?;
            Ok(())
        })
        .unwrap()
    }

    /// A hugr with a DFG root mapping bool_t() to (bool_t(), bool_t())
    ///                     ┌─────────┐
    ///                ┌────┤ (1) NOT ├──
    ///  ┌─────────┐   │    └─────────┘
    /// ─┤ (0) NOT ├───┤
    ///  └─────────┘   │    ┌─────────┐
    ///                └────┤ (2) NOT ├──
    ///                     └─────────┘
    /// This can be replaced with an empty hugr coping the input to both outputs.
    ///
    /// Returns the hugr and the nodes of the NOT gates, in order.
    #[fixture]
    fn dfg_hugr_copy_bools() -> (Hugr, Vec<Node>) {
        let mut dfg_builder =
            DFGBuilder::new(inout_sig(vec![bool_t()], vec![bool_t(), bool_t()])).unwrap();
        let [b] = dfg_builder.input_wires_arr();

        let not_inp = dfg_builder.add_dataflow_op(LogicOp::Not, vec![b]).unwrap();
        let [b] = not_inp.outputs_arr();

        let not_0 = dfg_builder.add_dataflow_op(LogicOp::Not, vec![b]).unwrap();
        let [b0] = not_0.outputs_arr();
        let not_1 = dfg_builder.add_dataflow_op(LogicOp::Not, vec![b]).unwrap();
        let [b1] = not_1.outputs_arr();

        (
            dfg_builder.finish_hugr_with_outputs([b0, b1]).unwrap(),
            vec![not_inp.node(), not_0.node(), not_1.node()],
        )
    }

    /// A hugr with a DFG root mapping bool_t() to (bool_t(), bool_t())
    ///                     ┌─────────┐
    ///                ┌────┤ (4) NOT ├──
    ///  ┌─────────┐   │    └─────────┘
    /// ─┤ (3) NOT ├───┤
    ///  └─────────┘   │
    ///                └─────────────────
    ///
    /// This can be replaced with a single NOT op, coping the input to the first output.
    ///
    /// Returns the hugr and the nodes of the NOT ops, in order.
    #[fixture]
    fn dfg_hugr_half_not_bools() -> (Hugr, Vec<Node>) {
        let mut dfg_builder =
            DFGBuilder::new(inout_sig(vec![bool_t()], vec![bool_t(), bool_t()])).unwrap();
        let [b] = dfg_builder.input_wires_arr();

        let not_inp = dfg_builder.add_dataflow_op(LogicOp::Not, vec![b]).unwrap();
        let [b] = not_inp.outputs_arr();

        let not_0 = dfg_builder.add_dataflow_op(LogicOp::Not, vec![b]).unwrap();
        let [b0] = not_0.outputs_arr();
        let b1 = b;

        (
            dfg_builder.finish_hugr_with_outputs([b0, b1]).unwrap(),
            vec![not_inp.node(), not_0.node()],
        )
    }

    #[rstest]
    /// Replace the
    ///      ┌───┐
    /// ──■──┤ H ├
    /// ┌─┴─┐├───┤
    /// ┤ X ├┤ H ├
    /// └───┘└───┘
    /// part of
    /// ┌─────┐       ┌─────┐
    /// ┤ H(3)├───■───┤ H(6)├
    /// ├─────┤┌──┴──┐├─────┤
    /// ┤ H(4)├┤ X(5)├┤ H(7)├
    /// └─────┘└─────┘└─────┘
    /// with
    /// ┌───┐
    /// ┤ H ├──■──
    /// ├───┤┌─┴─┐
    /// ┤ H ├┤ X ├
    /// └───┘└───┘
    fn test_simple_replacement(simple_hugr: Circuit, dfg_hugr: Circuit) {
        let h = simple_hugr.hugr();
        // 1. Locate the CX and its successor H's in h
        let h_node_cx: Node = h
            .nodes()
            .find(|node: &Node| *h.get_optype(*node) == Tk2Op::CX.into())
            .unwrap();
        let (h_node_h0, h_node_h1) = h.output_neighbours(h_node_cx).collect_tuple().unwrap();
        let s: Vec<Node> = vec![h_node_cx, h_node_h0, h_node_h1].into_iter().collect();
        // 2. Construct a new DFG-rooted hugr for the replacement
        let n = dfg_hugr.hugr();
        // 3. Construct the input and output matchings
        // 3.1. Locate the CX and its predecessor H's in n
        let n_node_cx = n
            .nodes()
            .find(|node: &Node| *n.get_optype(*node) == Tk2Op::CX.into())
            .unwrap();
        let (n_node_h0, n_node_h1) = n.input_neighbours(n_node_cx).collect_tuple().unwrap();
        // 3.2. Locate the ports we need to specify as "glue" in n
        let n_port_0 = n.node_inputs(n_node_h0).next().unwrap();
        let n_port_1 = n.node_inputs(n_node_h1).next().unwrap();
        let (n_cx_out_0, n_cx_out_1) = n.node_outputs(n_node_cx).take(2).collect_tuple().unwrap();
        let n_port_2 = n.linked_inputs(n_node_cx, n_cx_out_0).next().unwrap().1;
        let n_port_3 = n.linked_inputs(n_node_cx, n_cx_out_1).next().unwrap().1;
        // 3.3. Locate the ports we need to specify as "glue" in h
        let (h_port_0, h_port_1) = h.node_inputs(h_node_cx).take(2).collect_tuple().unwrap();
        let h_h0_out = h.node_outputs(h_node_h0).next().unwrap();
        let h_h1_out = h.node_outputs(h_node_h1).next().unwrap();
        let (h_outp_node, h_port_2) = h.linked_inputs(h_node_h0, h_h0_out).next().unwrap();
        let h_port_3 = h.linked_inputs(h_node_h1, h_h1_out).next().unwrap().1;
        // 3.4. Construct the maps
        let mut nu_inp: HashMap<(Node, IncomingPort), (Node, IncomingPort)> = HashMap::new();
        let mut nu_out: HashMap<(Node, IncomingPort), IncomingPort> = HashMap::new();
        nu_inp.insert((n_node_h0, n_port_0), (h_node_cx, h_port_0));
        nu_inp.insert((n_node_h1, n_port_1), (h_node_cx, h_port_1));
        nu_out.insert((h_outp_node, h_port_2), n_port_2);
        nu_out.insert((h_outp_node, h_port_3), n_port_3);
        // 4. Define the replacement
        let r = SimpleReplacement::new(
            SiblingSubgraph::try_from_nodes(s, &h).unwrap(),
            n.clone(),
            nu_inp,
            nu_out,
        );
        let diff = CircuitDiff::try_from_circuit(simple_hugr).unwrap();
        let new_diff = diff.apply_rewrite(r.into()).unwrap();
        // Expect [DFG] to be replaced with:
        // ┌───┐┌───┐
        // ┤ H ├┤ H ├──■──
        // ├───┤├───┤┌─┴─┐
        // ┤ H ├┤ H ├┤ X ├
        // └───┘└───┘└───┘
        insta::assert_debug_snapshot!(new_diff);
    }

    #[rstest]
    /// Replace the
    ///
    /// ──■──
    /// ┌─┴─┐
    /// ┤ X ├
    /// └───┘
    /// part of
    /// ┌──────┐        ┌──────┐
    /// ┤ H(3) ├────■───┤ H(6) ├
    /// ├──────┤┌───┴──┐├──────┤
    /// ┤ H(4) ├┤ X(5) ├┤ H(7) ├
    /// └──────┘└──────┘└──────┘
    /// with
    /// ─────
    /// ┌───┐
    /// ┤ H ├
    /// └───┘
    fn test_simple_replacement_with_empty_wires(simple_hugr: Circuit, dfg_hugr2: Circuit) {
        let h = simple_hugr.hugr();

        // 1. Locate the CX in h
        let h_node_cx: Node = h
            .nodes()
            .find(|node: &Node| *h.get_optype(*node) == Tk2Op::CX.into())
            .unwrap();
        let s: Vec<Node> = vec![h_node_cx].into_iter().collect();
        // 2. Construct a new DFG-rooted hugr for the replacement
        let n = dfg_hugr2.hugr();
        // 3. Construct the input and output matchings
        // 3.1. Locate the Output and its predecessor H in n
        let n_node_output = n
            .nodes()
            .find(|node: &Node| n.get_optype(*node).tag() == OpTag::Output)
            .unwrap();
        let (_n_node_input, n_node_h) = n.input_neighbours(n_node_output).collect_tuple().unwrap();
        // 3.2. Locate the ports we need to specify as "glue" in n
        let (n_port_0, n_port_1) = n
            .node_inputs(n_node_output)
            .take(2)
            .collect_tuple()
            .unwrap();
        let n_port_2 = n.node_inputs(n_node_h).next().unwrap();
        // 3.3. Locate the ports we need to specify as "glue" in h
        let (h_port_0, h_port_1) = h.node_inputs(h_node_cx).take(2).collect_tuple().unwrap();
        let (h_node_h0, h_node_h1) = h.output_neighbours(h_node_cx).collect_tuple().unwrap();
        let h_port_2 = h.node_inputs(h_node_h0).next().unwrap();
        let h_port_3 = h.node_inputs(h_node_h1).next().unwrap();
        // 3.4. Construct the maps
        let mut nu_inp: HashMap<(Node, IncomingPort), (Node, IncomingPort)> = HashMap::new();
        let mut nu_out: HashMap<(Node, IncomingPort), IncomingPort> = HashMap::new();
        nu_inp.insert((n_node_output, n_port_0), (h_node_cx, h_port_0));
        nu_inp.insert((n_node_h, n_port_2), (h_node_cx, h_port_1));
        nu_out.insert((h_node_h0, h_port_2), n_port_0);
        nu_out.insert((h_node_h1, h_port_3), n_port_1);
        // 4. Define the replacement
        let r = SimpleReplacement::new(
            SiblingSubgraph::try_from_nodes(s, &h).unwrap(),
            n.clone(),
            nu_inp,
            nu_out,
        );
        let diff = CircuitDiff::try_from_circuit(simple_hugr).unwrap();
        let new_diff = diff.apply_rewrite(r.into()).unwrap();
        // Expect [DFG] to be replaced with:
        // ┌───┐┌───┐
        // ┤ H ├┤ H ├
        // ├───┤├───┤┌───┐
        // ┤ H ├┤ H ├┤ H ├
        // └───┘└───┘└───┘
        insta::assert_debug_snapshot!(new_diff);
    }

    /// replace CX XC with a swap
    #[test]
    fn test_replace_cx_cross() {
        let circ = build_simple_circuit(2, |circ| {
            circ.append(Tk2Op::CX, [0, 1]).unwrap();
            circ.append(Tk2Op::CX, [1, 0]).unwrap();
            Ok(())
        })
        .unwrap();
        let h = circ.hugr();
        let [input, output] = circ.io_nodes();
        let replacement = h.clone();

        let removal = h
            .nodes()
            .filter(|&n| h.get_optype(n).tag() == OpTag::Leaf)
            .collect_vec();
        let inputs = h
            .node_outputs(input)
            .filter(|&p| {
                h.get_optype(input)
                    .as_input()
                    .unwrap()
                    .signature()
                    .port_type(p)
                    .is_some()
            })
            .map(|p| {
                let link = h.linked_inputs(input, p).next().unwrap();
                (link, link)
            })
            .collect();
        let outputs = h
            .node_inputs(output)
            .filter(|&p| {
                h.get_optype(output)
                    .as_output()
                    .unwrap()
                    .signature()
                    .port_type(p)
                    .is_some()
            })
            .map(|p| ((output, p), p))
            .collect();
        let r = SimpleReplacement::new(
            SiblingSubgraph::try_from_nodes(removal, &h).unwrap(),
            replacement,
            inputs,
            outputs,
        );
        let diff = CircuitDiff::try_from_circuit(circ.clone()).unwrap();
        let new_diff = diff.apply_rewrite(r.into()).unwrap();

        insta::assert_debug_snapshot!(new_diff);
    }

    /// rewrite x -> x & x to x, y -> x & y
    /// tests using the same input twice
    #[test]
    fn test_replace_after_copy() {
        let one_bit = vec![bool_t()];
        let two_bit = vec![bool_t(), bool_t()];

        // x -> x & x
        let mut builder = DFGBuilder::new(endo_sig(one_bit.clone())).unwrap();
        let inw = builder.input_wires().exactly_one().unwrap();
        let outw = builder
            .add_dataflow_op(LogicOp::And, [inw, inw])
            .unwrap()
            .outputs();
        let [input, _] = builder.io();
        let h = builder.finish_hugr_with_outputs(outw).unwrap();

        // x, y -> x & y
        let mut builder = DFGBuilder::new(inout_sig(two_bit, one_bit)).unwrap();
        let inw = builder.input_wires();
        let outw = builder
            .add_dataflow_op(LogicOp::And, inw)
            .unwrap()
            .outputs();
        let [repl_input, repl_output] = builder.io();
        let repl = builder.finish_hugr_with_outputs(outw).unwrap();

        // rewrite all nodes in h
        let removal = h
            .nodes()
            .filter(|&n| h.get_optype(n).tag() == OpTag::Leaf)
            .collect_vec();

        let first_out_p = h.node_outputs(input).next().unwrap();
        let embedded_inputs = h.linked_inputs(input, first_out_p);
        let repl_inputs = repl
            .node_outputs(repl_input)
            .map(|p| repl.linked_inputs(repl_input, p).next().unwrap());
        let inputs = embedded_inputs.zip(repl_inputs).collect();

        let outputs = repl
            .node_inputs(repl_output)
            .filter(|&p| repl.signature(repl_output).unwrap().port_type(p).is_some())
            .map(|p| ((repl_output, p), p))
            .collect();

        let r = SimpleReplacement::new(
            SiblingSubgraph::try_from_nodes(removal, &h).unwrap(),
            repl,
            inputs,
            outputs,
        );
        let circ_root = h.root();
        let circ = Circuit::new(h, circ_root);
        let diff = CircuitDiff::try_from_circuit(circ).unwrap();
        let new_diff = diff.apply_rewrite(r.into()).unwrap();

        insta::assert_debug_snapshot!(new_diff);
    }

    /// Remove all the NOT gates in [`dfg_hugr_copy_bools`] by connecting the input
    /// directly to the outputs.
    ///
    /// https://github.com/CQCL/hugr/issues/1190
    #[rstest]
    fn test_copy_inputs(dfg_hugr_copy_bools: (Hugr, Vec<Node>)) {
        let (hugr, nodes) = dfg_hugr_copy_bools;
        let (input_not, output_not_0, output_not_1) = nodes.into_iter().collect_tuple().unwrap();

        let [_input, output] = hugr.get_io(hugr.root()).unwrap();

        let replacement = {
            let b =
                DFGBuilder::new(Signature::new(vec![bool_t()], vec![bool_t(), bool_t()])).unwrap();
            let [w] = b.input_wires_arr();
            b.finish_hugr_with_outputs([w, w]).unwrap()
        };
        let [_repl_input, repl_output] = replacement.get_io(replacement.root()).unwrap();

        let subgraph =
            SiblingSubgraph::try_from_nodes(vec![input_not, output_not_0, output_not_1], &hugr)
                .unwrap();
        // A map from (target ports of edges from the Input node of `replacement`) to (target ports of
        // edges from nodes not in `removal` to nodes in `removal`).
        let nu_inp = [
            (
                (repl_output, IncomingPort::from(0)),
                (input_not, IncomingPort::from(0)),
            ),
            (
                (repl_output, IncomingPort::from(1)),
                (input_not, IncomingPort::from(0)),
            ),
        ]
        .into_iter()
        .collect();
        // A map from (target ports of edges from nodes in `removal` to nodes not in `removal`) to
        // (input ports of the Output node of `replacement`).
        let nu_out = [
            ((output, IncomingPort::from(0)), IncomingPort::from(0)),
            ((output, IncomingPort::from(1)), IncomingPort::from(1)),
        ]
        .into_iter()
        .collect();

        let r = SimpleReplacement::new(subgraph, replacement, nu_inp, nu_out);
        let circ_root = hugr.root();
        let circ = Circuit::new(hugr, circ_root);
        let diff = CircuitDiff::try_from_circuit(circ).unwrap();
        let new_diff = diff.apply_rewrite(r.into()).unwrap();

        insta::assert_debug_snapshot!(new_diff);
    }

    /// Remove one of the NOT ops in [`dfg_hugr_half_not_bools`] by connecting the input
    /// directly to the output.
    ///
    /// https://github.com/CQCL/hugr/issues/1323
    #[rstest]
    fn test_half_nots(dfg_hugr_half_not_bools: (Hugr, Vec<Node>)) {
        let (hugr, nodes) = dfg_hugr_half_not_bools;
        let (input_not, output_not_0) = nodes.into_iter().collect_tuple().unwrap();

        let [_input, output] = hugr.get_io(hugr.root()).unwrap();

        let (replacement, repl_not) = {
            let mut b =
                DFGBuilder::new(inout_sig(vec![bool_t()], vec![bool_t(), bool_t()])).unwrap();
            let [w] = b.input_wires_arr();
            let not = b.add_dataflow_op(LogicOp::Not, vec![w]).unwrap();
            let [w_not] = not.outputs_arr();
            (b.finish_hugr_with_outputs([w, w_not]).unwrap(), not.node())
        };
        let [_repl_input, repl_output] = replacement.get_io(replacement.root()).unwrap();

        let subgraph =
            SiblingSubgraph::try_from_nodes(vec![input_not, output_not_0], &hugr).unwrap();
        // A map from (target ports of edges from the Input node of `replacement`) to (target ports of
        // edges from nodes not in `removal` to nodes in `removal`).
        let nu_inp = [
            (
                (repl_output, IncomingPort::from(0)),
                (input_not, IncomingPort::from(0)),
            ),
            (
                (repl_not, IncomingPort::from(0)),
                (input_not, IncomingPort::from(0)),
            ),
        ]
        .into_iter()
        .collect();
        // A map from (target ports of edges from nodes in `removal` to nodes not in `removal`) to
        // (input ports of the Output node of `replacement`).
        let nu_out = [
            ((output, IncomingPort::from(0)), IncomingPort::from(0)),
            ((output, IncomingPort::from(1)), IncomingPort::from(1)),
        ]
        .into_iter()
        .collect();

        let r = SimpleReplacement::new(subgraph, replacement, nu_inp, nu_out);
        let circ_root = hugr.root();
        let circ = Circuit::new(hugr, circ_root);
        let diff = CircuitDiff::try_from_circuit(circ).unwrap();
        let new_diff = diff.apply_rewrite(r.into()).unwrap();

        insta::assert_debug_snapshot!(new_diff);
    }
}
