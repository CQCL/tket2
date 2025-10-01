//! Subgraph representation using intervals on resource paths.

use std::cmp::Ordering;

use derive_more::derive::{Display, Error};
use hugr::{core::HugrNode, Direction, HugrView};
use itertools::Itertools;

use super::{Position, ResourceId, ResourceScope};

/// A non-empty interval on a resource path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Interval<N> {
    /// Resource ID of the resource path.
    resource_id: ResourceId,
    /// Start and end positions of the interval (inclusive).
    positions: [Position; 2],
    /// Start and end nodes of the interval (inclusive).
    nodes: [N; 2],
}

impl<N: HugrNode> Interval<N> {
    /// Create an interval for a single node.
    pub fn singleton(
        resource_id: ResourceId,
        node: N,
        scope: &ResourceScope<impl HugrView<Node = N>>,
    ) -> Self {
        let pos = scope
            .get_position(node)
            .expect("node is not on resource path");

        Self {
            resource_id,
            positions: [pos, pos],
            nodes: [node, node],
        }
    }

    /// Create an interval for a range of nodes.
    ///
    /// This will panic if
    ///  - either node is not on the resource path `resource_id`
    ///  - or the start node is after the end node on the resource path.
    ///
    /// Use [`Interval::try_new`] instead for creating an interval with error
    /// handling.
    pub fn new(
        resource_id: ResourceId,
        start_node: N,
        end_node: N,
        scope: &ResourceScope<impl HugrView<Node = N>>,
    ) -> Self {
        Self::try_new(resource_id, start_node, end_node, scope).unwrap()
    }

    /// Create an interval for a range of nodes.
    pub fn try_new(
        resource_id: ResourceId,
        start_node: N,
        end_node: N,
        scope: &ResourceScope<impl HugrView<Node = N>>,
    ) -> Result<Self, InvalidInterval<N>> {
        let start_pos = scope
            .get_position(start_node)
            .ok_or(InvalidInterval::NotOnResourcePath(start_node))?;
        let end_pos = scope
            .get_position(end_node)
            .ok_or(InvalidInterval::NotOnResourcePath(end_node))?;

        if start_pos > end_pos {
            return Err(InvalidInterval::StartAfterEnd(
                start_node,
                end_node,
                resource_id,
            ));
        }

        Ok(Self {
            resource_id,
            positions: [start_pos, end_pos],
            nodes: [start_node, end_node],
        })
    }

    /// Get the resource ID of the interval.
    pub fn resource_id(&self) -> ResourceId {
        self.resource_id
    }

    /// Get the start node of the interval.
    pub fn start_node(&self) -> N {
        self.nodes[0]
    }

    /// Get the end node of the interval.
    pub fn end_node(&self) -> N {
        self.nodes[1]
    }

    /// Extend the interval to include the given node.
    ///
    /// Return the direction the interval was extended in, that is:
    ///  - if `node` was just before the interval, return `Direction::Incoming`
    ///  - if `node` was just after the interval, return `Direction::Outgoing`
    ///  - if `node` was already in the interval, return `None`
    ///
    /// If `node` is not contiguous with the interval, or if `node` is not
    /// on the interval's resource path, return an error.
    pub fn try_extend(
        &mut self,
        node: N,
        scope: &ResourceScope<impl HugrView<Node = N>>,
    ) -> Result<Option<Direction>, InvalidInterval<N>> {
        let Some(pos) = scope.get_position(node) else {
            return Err(InvalidInterval::NotOnResourcePath(node));
        };

        match self.position_in_interval(pos) {
            // pos is already within the interval
            Ordering::Equal => Ok(None),
            // pos is before the interval
            Ordering::Less => {
                let next_node = scope
                    .resource_path_iter(self.resource_id, node, Direction::Outgoing)
                    .nth(1)
                    .expect("same resource ID with larger position exists");
                if next_node == self.nodes[0] {
                    // Success! extend the interval to the left by one node
                    self.positions[0] = pos;
                    self.nodes[0] = node;
                    Ok(Some(Direction::Incoming))
                } else {
                    Err(InvalidInterval::NotContiguous(node))
                }
            }
            // pos is after the interval
            Ordering::Greater => {
                let prev_node = scope
                    .resource_path_iter(self.resource_id, node, Direction::Incoming)
                    .nth(1)
                    .expect("same resource ID with smaller position exists");
                if prev_node == self.nodes[1] {
                    // Success! extend the interval to the right by one node
                    self.positions[1] = pos;
                    self.nodes[1] = node;
                    Ok(Some(Direction::Outgoing))
                } else {
                    Err(InvalidInterval::NotContiguous(node))
                }
            }
        }
    }

    /// Include the given node in the interval.
    ///
    /// Does not check if the node is contiguous with the interval. Use
    /// [`Interval::try_extend`] instead for a safe way to extend an interval.
    #[allow(dead_code)]
    pub(crate) fn add_node_unchecked(&mut self, node: N, pos: Position) {
        if pos < self.positions[0] {
            self.positions[0] = pos;
            self.nodes[0] = node;
        }
        if pos > self.positions[1] {
            self.positions[1] = pos;
            self.nodes[1] = node;
        }
    }

    /// Whether `pos` is smaller, larger or within the interval.
    ///
    /// [Ordering] is used to express the relative position of `pos` with
    /// respect to the interval:
    /// - [`Ordering::Less`] if `pos` is before the interval,
    /// - [`Ordering::Greater`] if `pos` is after the interval, and
    /// - [`Ordering::Equal`] if `pos` is within the interval.
    fn position_in_interval(&self, pos: Position) -> Ordering {
        if pos < self.positions[0] {
            Ordering::Less
        } else if pos > self.positions[1] {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }
}

/// Errors that can occur when extending an interval.
#[derive(Debug, Clone, PartialEq, Display, Error)]
pub enum InvalidInterval<N> {
    /// The node is not contiguous with the interval.
    #[display("node {_0:?} is not contiguous with the interval")]
    NotContiguous(N),
    /// The node is not on the interval's resource path.
    #[display("node {_0:?} is not on the interval's resource path")]
    NotOnResourcePath(N),
    /// The start node is after the end node.
    #[display("start node {_0:?} is after end node {_1:?} on resource path {_2:?}")]
    StartAfterEnd(N, N, ResourceId),
}

impl<H: HugrView> ResourceScope<H> {
    /// Get the nodes in an interval.
    pub fn nodes_in_interval(
        &self,
        interval: Interval<H::Node>,
    ) -> impl Iterator<Item = H::Node> + '_ {
        let [start_node, end_node] = interval.nodes;
        self.resource_path_iter(interval.resource_id, start_node, Direction::Outgoing)
            .take_while_inclusive(move |&node| node != end_node)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::RangeInclusive;

    use crate::{
        resource::tests::cx_circuit,
        resource::{Interval, Position, ResourceId},
        Circuit,
    };

    use itertools::Itertools;
    use rstest::{fixture, rstest};

    #[test]
    fn test_nodes_in_interval() {
        let circ = cx_circuit(5);
        let subgraph = Circuit::from(&circ).subgraph().unwrap();
        let cx_nodes = subgraph.nodes().to_owned();
        let scope = super::ResourceScope::new(&circ, subgraph);

        assert_eq!(cx_nodes.len(), 5);

        let pos_interval = 1usize..4;

        for resource_id in [0, 1].map(ResourceId::new) {
            let interval = Interval {
                resource_id,
                positions: [
                    Position::new_integer(pos_interval.start as i64),
                    Position::new_integer((pos_interval.end - 1) as i64),
                ],
                nodes: [cx_nodes[pos_interval.start], cx_nodes[pos_interval.end - 1]],
            };

            assert_eq!(
                Interval::new(
                    resource_id,
                    cx_nodes[pos_interval.start],
                    cx_nodes[pos_interval.end - 1],
                    &scope
                ),
                interval
            );

            assert_eq!(
                scope.nodes_in_interval(interval).collect_vec(),
                cx_nodes[pos_interval.clone()]
            );
        }
    }

    #[fixture]
    fn cx_circuit_scope() -> ResourceScope {
        let circ = cx_circuit(5);
        ResourceScope::from_circuit(Circuit::from(circ))
    }

    #[rstest]
    #[case::extend_left(
        1,
        Some(Direction::Incoming),
        1..=3,
    )]
    #[case::extend_right(
        4,
        Some(Direction::Outgoing),
        2..=4,
    )]
    #[case::node_already_in_interval_start(
        2,
        None,
        2..=3,
    )]
    #[case::node_already_in_interval_end(
        3,
        None,
        2..=3,
    )]
    fn test_try_extend_success(
        cx_circuit_scope: ResourceScope,
        #[case] node_to_extend: usize,
        #[case] expected_direction: Option<Direction>,
        #[case] expected_range: RangeInclusive<usize>,
    ) {
        let cx_nodes = cx_circuit_scope.nodes();

        // Create an interval from nodes 2 to 3 (middle of circuit)
        let mut interval = Interval::new(
            ResourceId::new(0),
            cx_nodes[2],
            cx_nodes[3],
            &cx_circuit_scope,
        );

        // Apply the test case
        let result = interval
            .try_extend(cx_nodes[node_to_extend], &cx_circuit_scope)
            .unwrap();

        assert_eq!(result, expected_direction);
        assert_eq!(interval.start_node(), cx_nodes[*expected_range.start()]);
        assert_eq!(interval.end_node(), cx_nodes[*expected_range.end()]);
    }

    #[rstest]
    fn test_try_extend_error(cx_circuit_scope: ResourceScope) {
        let cx_nodes = cx_circuit_scope.nodes();

        // Create an interval from nodes 2 to 3 (middle of circuit)
        let mut interval = Interval::new(
            ResourceId::new(0),
            cx_nodes[2],
            cx_nodes[3],
            &cx_circuit_scope,
        );

        let result = interval
            .try_extend(cx_nodes[0], &cx_circuit_scope)
            .unwrap_err();

        assert_eq!(result, InvalidInterval::NotContiguous(cx_nodes[0]));
        assert_eq!(interval.start_node(), cx_nodes[2]);
        assert_eq!(interval.end_node(), cx_nodes[3]);
    }
}
