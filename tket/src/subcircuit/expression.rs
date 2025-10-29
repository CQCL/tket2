use std::collections::{BTreeSet, VecDeque};

use hugr::{
    core::HugrNode,
    hugr::views::{sibling_subgraph::InvalidSubgraph, SiblingSubgraph},
    ops::OpTrait,
    Direction, HugrView, IncomingPort, OutgoingPort,
};
use indexmap::IndexSet;
use itertools::{Either, Itertools};
use priority_queue::PriorityQueue;

use crate::resource::ResourceScope;

/// AST of a copyable value, represented as a hugr subgraph.
///
/// The value of the expression is `value` if the expression is
/// [CopyableExpressionAST::Identity], or is given by the first output of the
/// subgraph otherwise. The subgraph of [CopyableExpressionAST::Composite] only
/// contains nodes such that all its inputs and outputs are copyable types;
/// further, all inputs and outputs of the subgraph are in the past of the first
/// output (and hence the subgraph is always convex).
///
/// The subgraph may have other (copyable) outputs, which are ignored in the
/// context of the AST.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CopyableExpressionAST<N = hugr::Node> {
    /// A trivial expression that is just the value at `value`.
    Identity {
        /// The value of the expression, as a value port in the hugr graph.
        value: (N, OutgoingPort),
    },
    /// A composite expression, with AST represented as a hugr subgraph.
    Composite {
        /// The hugr subgraph representing the expression AST.
        ///
        /// The output value is the first output of the subgraph.
        subgraph: SiblingSubgraph<N>,
    },
}

impl<N: HugrNode> CopyableExpressionAST<N> {
    /// Construct an AST for the value at `output` in `circuit`.
    ///
    /// The inputs of the AST will be determined by traversing `circuit`
    /// backwards as follows. Starting from `output`,
    /// - any input port that is contained in `allowed_input_ports` is added as
    ///   an input to the AST;
    /// - any input port that is linked to a node in `allowed_input_nodes` is
    ///   added as an input to the AST;
    /// - any input port that is linked to a node that is not pure copyable
    ///   (i.e. has non-copyable inputs or outputs), or that is not in
    ///   `circuit``, is added as an input to the AST.
    ///
    /// The traversal does not progress beyond an incoming port that is added as
    /// input. The set of inputs are ordered according to the following
    /// ordering:
    /// 1. first by the order in which they appear in `allowed_input_ports`,
    /// 2. then by the order in which their nodes appear in
    ///    `allowed_input_nodes`,
    /// 3. and finally in the order in which they are encountered during
    ///    traversal.
    ///
    /// If `output` is not attached to a pure copyable node, or is in
    /// `allowed_input_nodes`, then the AST is trivial and a
    /// [CopyableExpressionAST::Identity] variant is returned.
    ///
    /// If the subgraph is not convex, then an [InvalidSubgraph::NotConvex]
    /// error is returned.
    ///
    /// ## Panics
    ///
    /// This will panic if `output` is not a value of copyable type in
    /// `circuit`.
    pub fn try_new(
        output: (N, OutgoingPort),
        allowed_input_ports: IndexSet<(N, IncomingPort)>,
        allowed_input_nodes: IndexSet<N>,
        circuit: &ResourceScope<impl HugrView<Node = N>>,
    ) -> Result<Self, InvalidSubgraph> {
        Self::validate_output_copyable(output, circuit);
        let circuit_nodes = circuit.hugr().nodes().collect::<BTreeSet<_>>();
        let admissible_node =
            |node: N| admissible_node(node, &allowed_input_nodes, circuit.hugr(), &circuit_nodes);

        if !admissible_node(output.0) {
            return Ok(Self::Identity { value: output });
        }

        let (inputs, outputs, function_calls, nodes) =
            Self::traverse_expression(output, &allowed_input_ports, circuit, &admissible_node);

        if any_in_future_of(&inputs, &outputs, circuit) {
            return Err(InvalidSubgraph::NotConvex);
        }

        let inputs = sort_inputs(inputs, &allowed_input_ports, &allowed_input_nodes);
        let outputs = outputs.into_iter().collect_vec();
        let nodes = flip_topological_order(nodes);
        let function_calls = group_function_calls(function_calls, circuit);

        let subgraph = SiblingSubgraph::new_unchecked(inputs, outputs, function_calls, nodes);
        Ok(Self::Composite { subgraph })
    }

    fn validate_output_copyable(
        output: (N, OutgoingPort),
        circuit: &ResourceScope<impl HugrView<Node = N>>,
    ) {
        assert!(
            circuit
                .hugr()
                .out_value_types(output.0)
                .filter_map(|(p, t)| (p == output.1).then_some(t))
                .exactly_one()
                .ok()
                .expect("known output port")
                .copyable(),
            "output value must exist in circuit and be copyable"
        );
    }

    fn traverse_expression(
        output: (N, OutgoingPort),
        allowed_input_ports: &IndexSet<(N, IncomingPort)>,
        circuit: &ResourceScope<impl HugrView<Node = N>>,
        admissible_node: impl Fn(N) -> bool,
    ) -> (
        IndexSet<(N, IncomingPort)>,
        IndexSet<(N, OutgoingPort)>,
        Vec<(N, IncomingPort)>,
        Vec<N>,
    ) {
        let mut outputs = IndexSet::<(N, OutgoingPort)>::from_iter([output]);
        let mut inputs = IndexSet::new();
        let mut function_calls = vec![];
        let mut nodes = Vec::new();

        // Queues and set useful during traversal.
        let mut curr_nodes = PriorityQueue::new();
        let prio = |node: N| circuit.get_position(node).expect("known node has position");
        let mut visited = BTreeSet::new();
        curr_nodes.push(output.0, prio(output.0));
        let mut inputs_within_expr = IndexSet::new();

        // Traverse nodes in reverse dataflow order.
        while let Some((node, _)) = curr_nodes.pop() {
            if !visited.insert(node) {
                continue; // been here before
            }

            debug_assert!(
                admissible_node(node),
                "cannot include a non-pure-copyable or allowed input into CopyableExpression"
            );

            // Add node to expression subgraph
            nodes.push(node);

            // Add all node outputs that we have not traversed to `outputs`
            let node_outputs = circuit.hugr().out_value_types(node).map(|(p, _)| (node, p));
            outputs.extend(node_outputs.filter(|&(n, p)| {
                circuit
                    .hugr()
                    .linked_inputs(n, p)
                    .any(|inp| !inputs_within_expr.contains(&inp))
            }));
            for (in_port, t) in circuit.hugr().in_value_types(node) {
                debug_assert!(t.copyable());
                let (prev_node, _out_port) = circuit
                    .hugr()
                    .single_linked_output(node, in_port)
                    .expect("valid dataflow wire");

                if allowed_input_ports.contains(&(node, in_port)) || !admissible_node(prev_node) {
                    // Add in_port to expression inputs
                    inputs.insert((node, in_port));
                } else {
                    // Continue traversing expression backwards
                    inputs_within_expr.insert((node, in_port));
                    curr_nodes.push(prev_node, prio(prev_node));
                }
            }

            // Add function call input if there is one
            let op = circuit.hugr().get_optype(node);
            if op.static_input().is_some_and(|edge| edge.is_function()) {
                function_calls.push((node, op.static_input_port().expect("function input port")));
            }
        }

        (inputs, outputs, function_calls, nodes)
    }

    /// Get the output value of the expression, as a port in the hugr graph.
    pub fn output(&self) -> (N, OutgoingPort) {
        match self {
            Self::Identity { value } => *value,
            Self::Composite { subgraph } => subgraph.outgoing_ports()[0],
        }
    }

    /// The AST as a hugr subgraph, if it is not the identity.
    pub fn as_subgraph(&self) -> Option<&SiblingSubgraph<N>> {
        match self {
            Self::Identity { .. } => None,
            Self::Composite { subgraph } => Some(subgraph),
        }
    }
}

/// Whether any of the `ends` are in the future of any of the `starts`.
fn any_in_future_of<N: HugrNode>(
    ends: &IndexSet<(N, IncomingPort)>,
    starts: &IndexSet<(N, OutgoingPort)>,
    circuit: &ResourceScope<impl HugrView<Node = N>>,
) -> bool {
    let end_nodes = BTreeSet::from_iter(ends.iter().map(|&(n, _)| n));
    let Some(max_pos) = end_nodes
        .iter()
        .map(|&n| circuit.get_position(n).expect("node in circuit"))
        .max()
    else {
        return false;
    };
    let mut visited_nodes = BTreeSet::new();
    let mut is_of_interest =
        |n| circuit.get_position(n).is_some_and(|p| p <= max_pos) && visited_nodes.insert(n);

    let mut curr_nodes = VecDeque::from_iter(
        starts
            .iter()
            .flat_map(|&(n, p)| circuit.hugr().linked_inputs(n, p))
            .filter_map(|(n, _)| is_of_interest(n).then_some(n)),
    );

    while let Some(node) = curr_nodes.pop_front() {
        if end_nodes.contains(&node) {
            return true;
        }
        curr_nodes.extend(
            circuit
                .hugr()
                .output_neighbours(node)
                .filter(|&n| is_of_interest(n)),
        );
    }

    false
}

/// Whether a node only contains copyable inputs and output values.
fn pure_copyable<N: HugrNode>(node: N, hugr: impl HugrView<Node = N>) -> bool {
    let mut all_port_types = Direction::BOTH
        .iter()
        .flat_map(|&dir| hugr.value_types(node, dir));
    all_port_types.all(|(_, t)| t.copyable())
}

fn admissible_node<N: HugrNode>(
    node: N,
    allowed_input_nodes: &IndexSet<N>,
    hugr: &impl HugrView<Node = N>,
    circuit_nodes: &BTreeSet<N>,
) -> bool {
    !allowed_input_nodes.contains(&node)
        && pure_copyable(node, hugr)
        && circuit_nodes.contains(&node)
}

fn flip_topological_order<N>(mut nodes: Vec<N>) -> Vec<N> {
    nodes.reverse();
    nodes
}

fn group_function_calls<N: HugrNode>(
    function_calls: Vec<(N, IncomingPort)>,
    circuit: &ResourceScope<impl HugrView<Node = N>>,
) -> Vec<Vec<(N, IncomingPort)>> {
    function_calls
        .into_iter()
        .into_group_map_by(|&(node, port)| {
            circuit
                .hugr()
                .single_linked_output(node, port)
                .expect("valid function call")
        })
        .into_values()
        .collect_vec()
}

fn sort_inputs<N: HugrNode>(
    inputs: IndexSet<(N, IncomingPort)>,
    allowed_input_ports: &IndexSet<(N, IncomingPort)>,
    allowed_input_nodes: &IndexSet<N>,
) -> Vec<Vec<(N, IncomingPort)>> {
    inputs
        .into_iter()
        .sorted_by_key(|&(node, port)| {
            if let Some(pos) = allowed_input_ports.get_index_of(&(node, port)) {
                Either::Left(pos)
            } else {
                Either::Right(allowed_input_nodes.get_index_of(&node))
            }
        })
        .map(|np| vec![np])
        .collect_vec()
}

#[cfg(test)]
mod tests {
    use std::iter;

    use crate::{extension::rotation::RotationOp, Circuit, TketOp};

    use super::*;
    use hugr::{
        builder::{Dataflow, DataflowHugr, FunctionBuilder, HugrBuilder},
        extension::prelude::{bool_t, qb_t},
        hugr::views::SiblingSubgraph,
        std_extensions::arithmetic::float_types::float64_type,
        types::Signature,
        Hugr, HugrView, IncomingPort, Node, OutgoingPort,
    };
    use rstest::{fixture, rstest};

    #[fixture]
    fn hugr_with_midcircuit_meas() -> Hugr {
        let qb_row = vec![qb_t()];
        let signature = Signature::new_endo(qb_row);
        let mut h = FunctionBuilder::new("main", signature).unwrap();
        let bool_to_float = h
            .module_root_builder()
            .declare(
                "bool_to_float",
                Signature::new(vec![bool_t()], vec![float64_type()]).into(),
            )
            .unwrap();

        let mut circ = h.as_circuit(h.input_wires());
        let [meas] = circ.append_with_outputs_arr(TketOp::Measure, [0]).unwrap();
        let qbs = circ.finish();
        let [float] = h.call(&bool_to_float, &[], [meas]).unwrap().outputs_arr();

        let mut circ = h.as_circuit(qbs);
        let [rot_angle] = circ
            .append_with_outputs_arr(RotationOp::from_halfturns_unchecked, [float])
            .unwrap();
        let [two_rot_angle] = circ
            .append_with_outputs_arr(RotationOp::radd, [rot_angle, rot_angle])
            .unwrap();
        circ.append_and_consume(
            TketOp::Rx,
            [
                hugr::CircuitUnit::Linear(0),
                hugr::CircuitUnit::Wire(two_rot_angle),
            ],
        )
        .unwrap();
        let circ_out = circ.finish();
        h.finish_hugr_with_outputs(circ_out).unwrap()
        // (hugr, (two_rot_angle.node(), two_rot_angle.source()))
    }

    #[rstest]
    #[case::identity_expression(
        // Single-qubit circuit where a measurement output is directly used as AST output
        {
            let qb_row = vec![qb_t()];
            let signature = Signature::new_endo(qb_row);
            let mut h = FunctionBuilder::new("main", signature).unwrap();
            let mut circ = h.as_circuit(h.input_wires());
            let [meas] = circ.append_with_outputs_arr(TketOp::Measure, [0]).unwrap();
            let circ_out = circ.finish();
            let hugr = h.finish_hugr_with_outputs(circ_out).unwrap();
            (hugr, (meas.node(), meas.source()))
        },
        CopyableExpressionAST::Identity {
            value: (Node::from(portgraph::NodeIndex::new(4)), OutgoingPort::from(1)),
        }
    )]
    #[case::midcircuit_meas(
        // Circuit where a measurement output is converted to float then to
        // rotation, multiplied by two and passed to a Rx rotation
        {
            let hugr = hugr_with_midcircuit_meas();
            let rx = hugr
                .nodes()
                .find(|&n| hugr.get_optype(n) == &(TketOp::Rx.into()))
                .unwrap();
            let two_rot_angle_input = (rx, IncomingPort::from(1));
            let two_rot_angle_output = hugr
                .single_linked_output(two_rot_angle_input.0, two_rot_angle_input.1)
                .unwrap();

            (hugr, two_rot_angle_output)
        },
        CopyableExpressionAST::Composite {
            subgraph: SiblingSubgraph::new_unchecked(
                vec![vec![(Node::from(portgraph::NodeIndex::new(6)), IncomingPort::from(0))]],
                vec![(Node::from(portgraph::NodeIndex::new(8)), OutgoingPort::from(0))],
                vec![vec![(Node::from(portgraph::NodeIndex::new(6)), IncomingPort::from(1))]],
                (6..=8).map(|i| Node::from(portgraph::NodeIndex::new(i))).collect(),
            ),
        }
    )]
    fn test_copyable_expression_ast_new(
        #[case] input: (Hugr, (Node, OutgoingPort)),
        #[case] expected: CopyableExpressionAST,
    ) {
        let (hugr, output) = input;

        let circuit = ResourceScope::from_circuit(Circuit::new(hugr));

        let result = CopyableExpressionAST::try_new(
            output,
            iter::empty().collect(),
            iter::empty().collect(),
            &circuit,
        )
        .unwrap();
        assert_eq!(result, expected);
    }

    #[rstest]
    fn test_copyable_expression_ast_new_bounded_inputs() {
        let hugr = hugr_with_midcircuit_meas();

        let rot_cast = hugr
            .nodes()
            .find(|&n| hugr.get_optype(n) == &(RotationOp::from_halfturns_unchecked.into()))
            .unwrap();
        let circuit = ResourceScope::from_circuit(Circuit::new(&hugr));
        let two_rot_angle_output = {
            let rx = hugr
                .nodes()
                .find(|&n| hugr.get_optype(n) == &(TketOp::Rx.into()))
                .unwrap();
            let two_rot_angle_input = (rx, IncomingPort::from(1));

            hugr.single_linked_output(two_rot_angle_input.0, two_rot_angle_input.1)
                .unwrap()
        };

        let expr = CopyableExpressionAST::try_new(
            two_rot_angle_output,
            iter::empty().collect(),
            [rot_cast].into_iter().collect(),
            &circuit,
        )
        .unwrap();

        let subgraph = expr.as_subgraph().expect("non-identity expression");

        assert_eq!(subgraph.nodes(), [Node::from(portgraph::NodeIndex::new(8))]);
        let &[radd] = subgraph.nodes() else {
            panic!("expected radd node")
        };
        assert_eq!(
            subgraph.incoming_ports(),
            &vec![
                vec![(radd, IncomingPort::from(0))],
                vec![(radd, IncomingPort::from(1))]
            ]
        );
        assert_eq!(
            subgraph.outgoing_ports(),
            &vec![(radd, OutgoingPort::from(0))]
        );
    }

    #[test]
    fn test_copyable_expression_ast_new_non_convex() {
        let hugr = hugr_with_midcircuit_meas();

        let radd = hugr
            .nodes()
            .find(|&n| hugr.get_optype(n) == &(RotationOp::radd.into()))
            .unwrap();
        let circuit = ResourceScope::from_circuit(Circuit::new(&hugr));

        let _expr = CopyableExpressionAST::try_new(
            (radd, OutgoingPort::from(0)),
            [(radd, IncomingPort::from(0))].into_iter().collect(),
            iter::empty().collect(),
            &circuit,
        )
        .expect_err("is not convex");
    }
}
