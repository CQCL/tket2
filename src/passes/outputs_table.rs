use std::collections::{HashMap, HashSet};

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct OutputsId(usize);

#[derive(PartialEq, Debug)]
enum Outputs {
    Single(usize), // an actual output of the graph
    Sequence(OutputsId, OutputsId),
}

/// Represents a sequence of graph-outputs ("output1, output2, output1, output3")
/// efficiently via flyweight pattern
pub(crate) struct OutputsTable {
    table: Vec<Outputs>,
    seq_map: HashMap<(OutputsId, OutputsId), OutputsId>,
}

impl OutputsTable {
    pub fn new(num_outputs: usize) -> Self {
        OutputsTable {
            table: (0..num_outputs).map(Outputs::Single).collect(),
            seq_map: HashMap::new(),
        }
    }

    pub fn for_graph_output(&self, graph_output: usize) -> OutputsId {
        assert!(self.table[graph_output] == Outputs::Single(graph_output));
        OutputsId(graph_output)
    }

    pub fn sequence(&mut self, fst: OutputsId, snd: OutputsId) -> OutputsId {
        match self.seq_map.get(&(fst, snd)) {
            Some(x) => *x,
            None => {
                let idx = OutputsId(self.table.len());
                self.seq_map.insert((fst, snd), idx);
                self.table.push(Outputs::Sequence(fst, snd));
                idx
            }
        }
    }

    pub fn onto_seq_deduped(
        &self,
        id: OutputsId,
        seen: &mut HashSet<OutputsId>,
        out: &mut Vec<usize>,
    ) {
        if !seen.contains(&id) {
            seen.insert(id);
            match self.table[id.0] {
                Outputs::Single(graph_output) => out.push(graph_output),
                Outputs::Sequence(a, b) => {
                    self.onto_seq_deduped(a, seen, out);
                    self.onto_seq_deduped(b, seen, out);
                }
            }
        }
    }

    pub fn to_seq_deduped(&self, id: OutputsId) -> Vec<usize> {
        let mut res = Vec::new();
        self.onto_seq_deduped(id, &mut HashSet::new(), &mut res);
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_single_outputs() {
        let ot = OutputsTable::new(2);
        assert!(ot.for_graph_output(0) == OutputsId(0));
        assert!(ot.for_graph_output(1) == OutputsId(1));
    }

    #[test]
    #[should_panic]
    fn test_out_of_range() {
        let ot = OutputsTable::new(2);
        ot.for_graph_output(2);
    }

    #[test]
    fn test_sequence_flyweight() {
        let mut ot = OutputsTable::new(3);
        let s01 = ot.sequence(ot.for_graph_output(0), ot.for_graph_output(1));
        let a = ot.sequence(s01, ot.for_graph_output(2));

        let s12 = ot.sequence(ot.for_graph_output(1), ot.for_graph_output(2));
        assert_ne!(s01, s12);
        let b = ot.sequence(ot.for_graph_output(0), s12);
        assert_ne!(a, b);

        let s01_ = ot.sequence(ot.for_graph_output(0), ot.for_graph_output(1));
        assert_eq!(s01, s01_);
        let a2 = ot.sequence(s01_, ot.for_graph_output(2));
        assert_eq!(a, a2);

        let s12_ = ot.sequence(ot.for_graph_output(1), ot.for_graph_output(2));
        let b2 = ot.sequence(ot.for_graph_output(0), s12_);
        assert_eq!(b, b2);
    }

    #[test]
    fn test_dedup() {
        let mut ot = OutputsTable::new(3);
        let s12 = ot.sequence(ot.for_graph_output(1), ot.for_graph_output(2));
        let s02 = ot.sequence(ot.for_graph_output(0), ot.for_graph_output(2));
        let s1202 = ot.sequence(s12, s02);
        assert_eq!(ot.to_seq_deduped(s1202), Vec::from([1, 2, 0]));
        let s0212 = ot.sequence(s02, s12);
        let s021202 = ot.sequence(s0212, s02);
        assert_eq!(ot.to_seq_deduped(s0212), Vec::from([0, 2, 1]));
        assert_eq!(ot.to_seq_deduped(s021202), Vec::from([0, 2, 1]));
    }
}
