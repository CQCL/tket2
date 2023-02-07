use petgraph::algo::{floyd_warshall, NegativeCycle};
use petgraph::graph::{NodeIndex, UnGraph};
use std::collections::HashMap;

type Architecture = UnGraph<u32, ()>;
type Distances = HashMap<(NodeIndex, NodeIndex), u32>;
fn distances(arc: &Architecture) -> Result<Distances, NegativeCycle> {
    floyd_warshall(arc, |_| 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn insert_mirror_dists(dists: Distances) -> Distances {
        let mut out = HashMap::new();
        for ((i, j), d) in dists {
            if i <= j {
                out.insert((i, j), d);
                out.insert((j, i), d);
            }
        }
        out
    }
    #[test]
    fn test_distances() -> Result<(), NegativeCycle> {
        /*
        0 - 1 - 2
         \ / \ /
          3 - 4
        */
        let g = Architecture::from_edges(&[(0, 1), (0, 3), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4)]);

        let dists = distances(&g)?;

        let correct = HashMap::from_iter(
            [
                ((0, 1), 1),
                ((0, 2), 2),
                ((0, 3), 1),
                ((0, 4), 2),
                ((1, 2), 1),
                ((1, 3), 1),
                ((1, 4), 1),
                ((2, 4), 1),
                ((2, 3), 2),
                ((3, 4), 1),
                ((0, 0), 0),
                ((1, 1), 0),
                ((2, 2), 0),
                ((3, 3), 0),
                ((4, 4), 0),
            ]
            .map(|((i, j), d)| ((NodeIndex::new(i), NodeIndex::new(j)), d)),
        );

        let correct = insert_mirror_dists(correct);
        assert_eq!(dists, correct);
        Ok(())
    }
}
