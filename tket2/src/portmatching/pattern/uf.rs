use std::collections::{BTreeMap, BTreeSet};

use itertools::Itertools;
use portmatching::pattern::Satisfiable;

type PortConnections<W> = BTreeMap<hugr::Port, W>;
type WireConnections<N> = Vec<(hugr::Port, N)>;

/// Two-in-One Union find data structures to store node and wire identities.
///
/// Tracks which nodes and wires are known to be equal, or known to be unequal.
/// This is more complex than having two traditional UF data structure for two
/// reasons:
///  1. uniting two classes of equivalent nodes potentially required to merge
///     classes of wires too (and vice versa).
///  2. we also track pairs of nodes and wires known to be unequal.
///
/// Wires may be marked as linear, in which case we can make stronger
/// implications from known equalities/inequalities.
///
/// ## Important note on usage:
/// The public facing functions return a [`Satisfiable`] enum that indicates
/// whether the new fact was added, was already known or would have introduced
/// a contradiction. When the latter happens, the data structure is no longer
/// in a valid state. I repeat: do not use the data structure after receiving
/// a [`Satisfiable::No`] response.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Uf<N, W, Op> {
    node_roots: BTreeMap<N, N>,
    wire_roots: BTreeMap<W, W>,

    /// Map nodes to its incident wires and their ports
    ///
    /// Any non-root node is None
    node_connected_wires: BTreeMap<N, Option<PortConnections<W>>>,
    /// Set of nodes known to be unequal the key
    ///
    /// Any non-root node is None
    node_inequality: BTreeMap<N, Option<BTreeSet<N>>>,
    /// Map nodes to their operation
    ///
    /// Any non-root node is None -- roots may be None too.
    node_op: BTreeMap<N, Option<Op>>,

    /// Map wires to its adjacent nodes and their ports, along with whether
    /// the wire is linear
    ///
    /// Any non-root wire is None
    wire_connected_nodes: BTreeMap<W, Option<(bool, WireConnections<N>)>>,
    /// Set of wires known to be unequal the key
    ///
    /// Any non-root wire is None
    wire_inequality: BTreeMap<W, Option<BTreeSet<W>>>,
}

impl<N, W, Op> Default for Uf<N, W, Op> {
    fn default() -> Self {
        Uf {
            node_roots: BTreeMap::new(),
            wire_roots: BTreeMap::new(),
            node_connected_wires: BTreeMap::new(),
            node_inequality: BTreeMap::new(),
            node_op: BTreeMap::new(),
            wire_connected_nodes: BTreeMap::new(),
            wire_inequality: BTreeMap::new(),
        }
    }
}

impl<N: Ord + Copy, W: Ord + Copy, Op: Eq> Uf<N, W, Op> {
    pub(super) fn new() -> Self {
        Uf::default()
    }

    #[must_use]
    pub(super) fn set_op(&mut self, node: N, op: Op) -> Satisfiable {
        self.ensure_node_exists(node);
        let node = self.root_node(node);
        match self.node_op[&node].as_ref() {
            Some(curr_op) if curr_op == &op => return Satisfiable::Tautology,
            Some(_) => return Satisfiable::No,
            None => {
                self.node_op.insert(node, Some(op));
                Satisfiable::Yes(())
            }
        }
    }

    #[must_use]
    pub(super) fn set_link(
        &mut self,
        node: N,
        wire: W,
        port: hugr::Port,
        is_linear: bool,
    ) -> Satisfiable {
        self.ensure_node_exists(node);
        self.ensure_wire_exists(wire, is_linear);
        let node = self.root_node(node);
        let wire = self.root_wire(wire);

        let nodes_conn = self
            .node_connected_wires
            .get_mut(&node)
            .unwrap()
            .as_mut()
            .unwrap();
        let wires_conn = &mut self
            .wire_connected_nodes
            .get_mut(&wire)
            .unwrap()
            .as_mut()
            .unwrap()
            .1;

        if let Some(&curr_wire) = nodes_conn.get(&port) {
            return self.set_wires_equal(wire, curr_wire, is_linear);
        }

        let must_be_unique = port.direction() == hugr::Direction::Outgoing || is_linear;

        if must_be_unique {
            if let Some(&(_, curr_node)) = wires_conn.iter().find(|(p, _)| p == &port) {
                return self.set_nodes_equal(node, curr_node);
            }
            let is_unique = !wires_conn
                .iter()
                .any(|(p, _)| p.direction() == port.direction());
            if !is_unique {
                return Satisfiable::No;
            }
        }

        nodes_conn.insert(port, wire);
        wires_conn.push((port, node));

        // Propagate inequalities
        for w in self.wire_inequalities(wire).clone() {
            let conn_nodes = self
                .connected_nodes(w)
                .iter()
                .filter(|&&(p, _)| p == port)
                .map(|&(_, n)| n)
                .collect_vec();
            for other_node in conn_nodes {
                if self.set_nodes_not_equal(node, other_node) == Satisfiable::No {
                    return Satisfiable::No;
                }
            }
        }
        if must_be_unique {
            for n in self.node_inequalities(node).clone() {
                let Some(&other_wire) = self.connected_wires(n).get(&port) else {
                    continue;
                };
                if self.set_wires_not_equal(wire, other_wire, is_linear) == Satisfiable::No {
                    return Satisfiable::No;
                }
            }
        }

        Satisfiable::Yes(())
    }

    #[must_use]
    fn set_nodes_equal(&mut self, node1: N, node2: N) -> Satisfiable {
        self.ensure_node_exists(node1);
        self.ensure_node_exists(node2);

        let node1 = self.root_node(node1);
        let node2 = self.root_node(node2);

        if node1 == node2 {
            return Satisfiable::Tautology;
        }
        if self.node_inequalities(node1).contains(&node2) {
            return Satisfiable::No;
        }
        match (&self.node_op[&node1], &self.node_op[&node2]) {
            (Some(op1), Some(op2)) if op1 != op2 => return Satisfiable::No,
            _ => {}
        }

        // We (choose to) merge node1 into node2
        self.node_roots.insert(node1, node2);

        // Remove node1 as root
        let (conn1, ineq1, op) = self.remove_node_root(node1);

        // Set node op if known in node1
        if let Some(op) = op {
            self.node_op.insert(node2, Some(op));
        }

        // Insert wires from node1 into node2
        for (port, wire) in conn1 {
            let is_linear = self.is_linear(wire);
            if self.set_link(node2, wire, port, is_linear) == Satisfiable::No {
                return Satisfiable::No;
            }
        }

        // Insert node inequalitites from node1 into node2
        for n in ineq1 {
            if self.set_nodes_not_equal(node2, n) == Satisfiable::No {
                return Satisfiable::No;
            }
        }

        Satisfiable::Yes(())
    }

    #[must_use]
    fn set_nodes_not_equal(&mut self, node1: N, node2: N) -> Satisfiable {
        self.ensure_node_exists(node1);
        self.ensure_node_exists(node2);
        let node1 = self.root_node(node1);
        let node2 = self.root_node(node2);

        if node1 == node2 {
            return Satisfiable::No;
        }
        if self.node_inequalities(node1).contains(&node2) {
            return Satisfiable::Tautology;
        }

        self.node_inequalities_mut(node1).insert(node2);
        self.node_inequalities_mut(node2).insert(node1);

        for (w1, w2, dir) in self.zip_wires(node1, node2).collect_vec() {
            let is_linear = self.is_linear(w1);
            if dir == hugr::Direction::Outgoing || self.is_linear(w1) {
                if self.set_wires_not_equal(w1, w2, is_linear) == Satisfiable::No {
                    return Satisfiable::No;
                }
            }
        }

        Satisfiable::Yes(())
    }

    #[must_use]
    fn set_wires_equal(&mut self, wire1: W, wire2: W, is_linear: bool) -> Satisfiable {
        self.ensure_wire_exists(wire1, is_linear);
        self.ensure_wire_exists(wire2, is_linear);
        let wire1 = self.root_wire(wire1);
        let wire2 = self.root_wire(wire2);

        if wire1 == wire2 {
            return Satisfiable::Tautology;
        }
        if self.wire_inequalities(wire1).contains(&wire2) {
            return Satisfiable::No;
        }

        // We (choose to) merge wire1 into wire2
        self.wire_roots.insert(wire1, wire2);

        // Remove wire1 as root
        let (conn1, ineq1) = self.remove_wire_root(wire1);

        // Insert nodes from wire1 into wire2
        for (port, node) in conn1 {
            if self.set_link(node, wire2, port, is_linear) == Satisfiable::No {
                return Satisfiable::No;
            }
        }

        // Insert wire inequalitites from wire1 into wire2
        for w in ineq1 {
            if self.set_wires_not_equal(wire2, w, is_linear) == Satisfiable::No {
                return Satisfiable::No;
            }
        }

        Satisfiable::Yes(())
    }

    #[must_use]
    pub(super) fn set_wires_not_equal(
        &mut self,
        wire1: W,
        wire2: W,
        is_linear: bool,
    ) -> Satisfiable {
        self.ensure_wire_exists(wire1, is_linear);
        self.ensure_wire_exists(wire2, is_linear);
        let wire1 = self.root_wire(wire1);
        let wire2 = self.root_wire(wire2);

        if wire1 == wire2 {
            return Satisfiable::No;
        }
        if self.wire_inequalities(wire1).contains(&wire2) {
            return Satisfiable::Tautology;
        }

        self.wire_inequalities_mut(wire1).insert(wire2);
        self.wire_inequalities_mut(wire2).insert(wire1);

        for (n1, n2) in self.zip_nodes(wire1, wire2).collect_vec() {
            if self.set_nodes_not_equal(n1, n2) == Satisfiable::No {
                return Satisfiable::No;
            }
        }

        Satisfiable::Yes(())
    }
}

impl<N: Ord + Copy, W: Ord + Copy, Op: Eq> Uf<N, W, Op> {
    fn is_linear(&mut self, w: W) -> bool {
        let root = self.root_wire(w);
        self.wire_connected_nodes[&root].as_ref().unwrap().0
    }

    fn zip_wires(&self, root1: N, root2: N) -> impl Iterator<Item = (W, W, hugr::Direction)> + '_ {
        let conn1 = self.connected_wires(root1);
        let conn2 = self.connected_wires(root2);

        conn1.iter().filter_map(move |(p, &w1)| {
            let &w2 = conn2.get(p)?;
            Some((w1, w2, p.direction()))
        })
    }

    fn zip_nodes(&self, root1: W, root2: W) -> impl Iterator<Item = (N, N)> + '_ {
        let conn1 = self.connected_nodes(root1);
        let conn2 = self.connected_nodes(root2);

        conn1.iter().flat_map(move |&(p, n1)| {
            conn2
                .iter()
                .filter(move |&(p2, _)| *p2 == p)
                .map(move |&(_, n2)| (n1, n2))
        })
    }

    fn remove_node_root(&mut self, root: N) -> (PortConnections<W>, BTreeSet<N>, Option<Op>) {
        let conn_wires = self
            .node_connected_wires
            .get_mut(&root)
            .unwrap()
            .take()
            .unwrap();
        let ineq = self.node_inequality.get_mut(&root).unwrap().take().unwrap();
        let op = self.node_op.get_mut(&root).unwrap().take();

        // Remove connections to `root` at the wires
        for (p, w) in &conn_wires {
            let conn_nodes = self.connected_nodes_mut(*w);
            let pos = conn_nodes
                .iter()
                .position(|(p2, n)| *n == root && *p2 == *p)
                .unwrap();
            conn_nodes.remove(pos);
        }

        // Remove inequalities to `root` at other nodes
        for &n in &ineq {
            self.node_inequalities_mut(n).remove(&root);
        }

        (conn_wires, ineq, op)
    }

    fn remove_wire_root(&mut self, root: W) -> (WireConnections<N>, BTreeSet<W>) {
        let conn_nodes = self
            .wire_connected_nodes
            .get_mut(&root)
            .unwrap()
            .take()
            .unwrap()
            .1;
        let ineq = self.wire_inequality.get_mut(&root).unwrap().take().unwrap();

        // Remove connections to `root` wire at the incident nodes
        for (p, n) in &conn_nodes {
            let conn_wires = self.connected_wires_mut(*n);
            conn_wires.remove(p);
        }

        // Remove inequalities to `root` at other nodes
        for &w in &ineq {
            self.wire_inequalities_mut(w).remove(&root);
        }

        (conn_nodes, ineq)
    }

    fn ensure_node_exists(&mut self, node: N) {
        if !self.node_roots.contains_key(&node) {
            self.node_roots.insert(node, node);
            self.node_connected_wires
                .insert(node, Some(Default::default()));
            self.node_inequality.insert(node, Some(Default::default()));
            self.node_op.insert(node, None);
        }
    }

    fn ensure_wire_exists(&mut self, wire: W, is_linear: bool) {
        if !self.wire_roots.contains_key(&wire) {
            self.wire_roots.insert(wire, wire);
            self.wire_connected_nodes
                .insert(wire, Some((is_linear, Default::default())));
            self.wire_inequality.insert(wire, Some(Default::default()));
        }
    }

    fn connected_wires_mut(&mut self, root: N) -> &mut PortConnections<W> {
        self.node_connected_wires
            .get_mut(&root)
            .unwrap()
            .as_mut()
            .unwrap()
    }

    fn connected_wires(&self, root: N) -> &PortConnections<W> {
        self.node_connected_wires[&root].as_ref().unwrap()
    }

    fn node_inequalities(&self, root: N) -> &BTreeSet<N> {
        self.node_inequality.get(&root).unwrap().as_ref().unwrap()
    }

    fn wire_inequalities(&self, root: W) -> &BTreeSet<W> {
        self.wire_inequality.get(&root).unwrap().as_ref().unwrap()
    }

    fn node_inequalities_mut(&mut self, root: N) -> &mut BTreeSet<N> {
        self.node_inequality
            .get_mut(&root)
            .unwrap()
            .as_mut()
            .unwrap()
    }

    fn wire_inequalities_mut(&mut self, root: W) -> &mut BTreeSet<W> {
        self.wire_inequality
            .get_mut(&root)
            .unwrap()
            .as_mut()
            .unwrap()
    }

    fn connected_nodes_mut(&mut self, wire: W) -> &mut WireConnections<N> {
        &mut self
            .wire_connected_nodes
            .get_mut(&wire)
            .unwrap()
            .as_mut()
            .unwrap()
            .1
    }

    fn connected_nodes(&self, wire: W) -> &WireConnections<N> {
        &self.wire_connected_nodes[&wire].as_ref().unwrap().1
    }

    #[cfg(test)]
    fn linked_wire(&mut self, node: N, port: hugr::Port) -> Option<W> {
        let root = self.root_node(node);
        self.connected_wires(root).get(&port).copied()
    }

    fn root_node(&mut self, node: N) -> N {
        let mut root = self.node_roots[&node];
        if root != node {
            root = self.root_node(root);
            self.node_roots.insert(node, root);
        }
        root
    }

    fn root_wire(&mut self, wire: W) -> W {
        let mut root = self.wire_roots[&wire];
        if root != wire {
            root = self.root_wire(root);
            self.wire_roots.insert(wire, root);
        }
        root
    }
}

#[cfg(test)]
mod tests {
    use hugr::{Direction, Port};
    use rstest::rstest;

    use super::*;

    type TestUf = Uf<isize, isize, ()>;

    #[test]
    fn test_union_nodes() {
        let node1 = 0;
        let node2 = 1;
        let node3 = 2;

        let mut uf = TestUf::new();
        assert_eq!(uf.set_nodes_equal(node1, node2), Satisfiable::Yes(()));
        assert_eq!(uf.set_nodes_equal(node2, node3), Satisfiable::Yes(()));

        assert_eq!(uf.root_node(node1), uf.root_node(node2));
        assert_eq!(uf.root_node(node1), uf.root_node(node3));
        assert_eq!(
            uf.node_roots.into_iter().collect::<Vec<_>>(),
            [(node1, node3), (node2, node3), (node3, node3)]
        );
    }

    #[test]
    fn test_union_nodes_link_prop() {
        let mut uf = TestUf::new();
        assert_eq!(uf.set_nodes_equal(0, 1), Satisfiable::Yes(()));
        let out0: Port = Port::new(Direction::Outgoing, 0);
        let w = 66;

        assert_eq!(uf.set_link(0, w, out0, true), Satisfiable::Yes(()));
        assert_eq!(uf.set_nodes_equal(1, 2), Satisfiable::Yes(()));

        assert_eq!(uf.linked_wire(2, out0), Some(w));
    }

    #[test]
    fn test_union_nodes_op_prop_1() {
        let mut uf = Uf::<isize, isize, bool>::new();
        assert_eq!(uf.set_op(0, false), Satisfiable::Yes(()));
        assert_eq!(uf.set_nodes_equal(0, 1), Satisfiable::Yes(()));
        assert_eq!(uf.set_op(1, true), Satisfiable::No);
    }

    #[test]
    fn test_union_nodes_op_prop_2() {
        let mut uf = Uf::<isize, isize, bool>::new();
        assert_eq!(uf.set_op(0, false), Satisfiable::Yes(()));
        assert_eq!(uf.set_nodes_equal(0, 1), Satisfiable::Yes(()));
        assert_eq!(uf.set_op(2, true), Satisfiable::Yes(()));
        assert_eq!(uf.set_nodes_equal(0, 2), Satisfiable::No);
    }

    #[test]
    fn test_node_creation() {
        let n = 0;
        let mut uf = TestUf::new();

        uf.ensure_node_exists(n);

        assert_eq!(uf.root_node(n), n);

        assert!(uf.node_inequality[&n].is_some());
        assert!(uf.node_connected_wires[&n].is_some());
    }

    #[derive(Clone, Copy, Debug)]
    enum Instruction<N, W> {
        NodeEq(N, N),
        WireEq(W, W, bool),
        Link(N, W, hugr::Port, bool),
        NodeNotEq(N, N),
        WireNotEq(W, W, bool),
    }

    impl<N: Copy + Ord, W: Copy + Ord> Instruction<N, W> {
        #[must_use]
        fn apply(&self, uf: &mut Uf<N, W, ()>) -> Satisfiable {
            match *self {
                Instruction::NodeEq(n1, n2) => uf.set_nodes_equal(n1, n2),
                Instruction::WireEq(w1, w2, is_linear) => uf.set_wires_equal(w1, w2, is_linear),
                Instruction::Link(n, w, port, is_linear) => uf.set_link(n, w, port, is_linear),
                Instruction::NodeNotEq(n1, n2) => uf.set_nodes_not_equal(n1, n2),
                Instruction::WireNotEq(w1, w2, is_linear) => {
                    uf.set_wires_not_equal(w1, w2, is_linear)
                }
            }
        }
    }

    #[rstest]
    #[case(vec![
        Instruction::NodeEq(0, 1),
        Instruction::NodeEq(1, 2),
        Instruction::NodeEq(0, 2),
    ], Satisfiable::Tautology)]
    #[case(vec![
        Instruction::NodeEq(0, 1),
        Instruction::Link(0, 10, Port::new(Direction::Outgoing, 0), true),
        Instruction::Link(2, 10, Port::new(Direction::Outgoing, 0), true),
    ], Satisfiable::Yes(()))]
    #[case(vec![
        Instruction::NodeEq(0, 1),
        Instruction::Link(0, 10, Port::new(Direction::Outgoing, 0), true),
        Instruction::Link(2, 10, Port::new(Direction::Outgoing, 0), true),
        Instruction::NodeEq(1, 2),
    ], Satisfiable::Tautology)]
    #[case(vec![
        Instruction::NodeEq(0, 1),
        Instruction::Link(0, 10, Port::new(Direction::Incoming, 0), false),
        Instruction::Link(2, 10, Port::new(Direction::Incoming, 0), false),
        Instruction::NodeEq(1, 2),
    ], Satisfiable::Yes(()))]
    #[case(vec![
        Instruction::NodeEq(0, 1),
        Instruction::Link(0, 10, Port::new(Direction::Incoming, 0), false),
        Instruction::NodeNotEq(1, 2),
        Instruction::Link(2, 10, Port::new(Direction::Incoming, 0), false),
    ], Satisfiable::Yes(()))]
    #[case(vec![
        Instruction::NodeEq(0, 1),
        Instruction::Link(0, 10, Port::new(Direction::Incoming, 0), true),
        Instruction::NodeNotEq(1, 2),
        Instruction::NodeEq(2, 3),
        Instruction::Link(3, 10, Port::new(Direction::Incoming, 0), true),
    ], Satisfiable::No)]
    #[case(vec![
        Instruction::Link(0, 10, Port::new(Direction::Incoming, 0), true),
        Instruction::Link(1, 10, Port::new(Direction::Outgoing, 0), true),
        Instruction::Link(1, 11, Port::new(Direction::Outgoing, 1), false),
        Instruction::Link(2, 20, Port::new(Direction::Incoming, 0), true),
        Instruction::Link(3, 20, Port::new(Direction::Outgoing, 0), true),
        Instruction::Link(3, 21, Port::new(Direction::Outgoing, 1), false),
        Instruction::WireEq(10, 20, true),
        Instruction::WireEq(11, 21, false)
    ], Satisfiable::Tautology)]
    #[case(vec![
        Instruction::Link(0, 10, Port::new(Direction::Incoming, 0), true),
        Instruction::Link(1, 10, Port::new(Direction::Outgoing, 0), true),
        Instruction::Link(1, 11, Port::new(Direction::Outgoing, 1), false),
        Instruction::Link(2, 20, Port::new(Direction::Incoming, 0), true),
        Instruction::Link(3, 20, Port::new(Direction::Outgoing, 0), true),
        Instruction::Link(3, 21, Port::new(Direction::Outgoing, 1), false),
        Instruction::NodeEq(0, 2),
        Instruction::WireEq(11, 21, false)
    ], Satisfiable::Tautology)]
    #[case(vec![
        Instruction::Link(0, 10, Port::new(Direction::Incoming, 0), true),
        Instruction::Link(1, 10, Port::new(Direction::Outgoing, 0), true),
        Instruction::Link(1, 11, Port::new(Direction::Outgoing, 1), false),
        Instruction::Link(2, 20, Port::new(Direction::Incoming, 0), true),
        Instruction::Link(3, 20, Port::new(Direction::Outgoing, 0), true),
        Instruction::Link(3, 21, Port::new(Direction::Outgoing, 1), false),
        Instruction::WireNotEq(10, 20, true),
        Instruction::WireNotEq(11, 21, false)
    ], Satisfiable::Tautology)]
    #[case(vec![
        Instruction::WireNotEq(10, 20, true),
        Instruction::Link(0, 10, Port::new(Direction::Incoming, 0), true),
        Instruction::Link(1, 10, Port::new(Direction::Outgoing, 0), true),
        Instruction::Link(1, 11, Port::new(Direction::Outgoing, 1), false),
        Instruction::Link(2, 20, Port::new(Direction::Incoming, 0), true),
        Instruction::Link(3, 20, Port::new(Direction::Outgoing, 0), true),
        Instruction::Link(3, 21, Port::new(Direction::Outgoing, 1), false),
        Instruction::WireNotEq(11, 21, false)
    ], Satisfiable::Tautology)]
    #[case(vec![
        Instruction::Link(3, 20, Port::new(Direction::Outgoing, 0), true),
        Instruction::Link(1, 11, Port::new(Direction::Outgoing, 1), false),
        Instruction::WireNotEq(10, 20, true),
        Instruction::Link(2, 20, Port::new(Direction::Incoming, 0), true),
        Instruction::Link(0, 10, Port::new(Direction::Incoming, 0), true),
        Instruction::Link(1, 10, Port::new(Direction::Outgoing, 0), true),
        Instruction::Link(3, 21, Port::new(Direction::Outgoing, 1), false),
        Instruction::WireNotEq(11, 21, false)
    ], Satisfiable::Tautology)]
    fn test_uf_logic(
        #[case] mut instructions: Vec<Instruction<isize, isize>>,
        #[case] expected: Satisfiable,
    ) {
        let last = instructions.pop().unwrap();
        let mut uf = TestUf::new();
        for i in instructions {
            assert_eq!(i.apply(&mut uf), Satisfiable::Yes(()));
        }
        assert_eq!(last.apply(&mut uf), expected);
    }
}
