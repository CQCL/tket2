//! Update ResourceScope from updates to a Walker

use std::{
    collections::{BTreeSet, HashMap, VecDeque},
    hash::{Hash, Hasher},
    iter,
};

use crate::resource::{CircuitRewriteError, ResourceScope};
use fxhash::FxHasher64;
use hugr::{
    hugr::views::SiblingSubgraph,
    persistent::{CommitId, PersistentHugr, Walker},
    HugrView,
};
use itertools::Itertools;

pub struct ResourceScopeCache {
    cache: HashMap<u64, ResourceScope<PersistentHugr>>,
}

impl ResourceScopeCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    pub fn init(&mut self, walker: &Walker) -> &ResourceScope<PersistentHugr> {
        let h = hash_from_walker(walker);
        self.cache
            .entry(h)
            .or_insert_with(|| to_resource_scope(walker.as_hugr_view()))
    }

    pub fn update(
        &mut self,
        walker: &Walker,
        prev_walker: &Walker,
    ) -> Result<&ResourceScope<PersistentHugr>, CircuitRewriteError> {
        let h = hash_from_walker(walker);
        if self.cache.contains_key(&h) {
            return Ok(self.cache.get(&h).unwrap());
        }

        let prev_h = hash_from_walker(prev_walker);
        let prev_scope = self.cache.get(&prev_h).unwrap();

        let old_commits: BTreeSet<_> = prev_walker.as_hugr_view().all_commit_ids().collect();
        let missing_commits: VecDeque<_> = walker
            .as_hugr_view()
            .all_commit_ids()
            .filter(|id| !old_commits.contains(id))
            .collect();

        let mut new_scope = prev_scope.clone();
        for id in commits_in_topo_order(missing_commits, new_scope.hugr(), walker.as_hugr_view()) {
            let commit = walker.as_hugr_view().get_commit(id);
            new_scope.apply_commit(commit.clone())?;
        }

        self.cache.insert(h, new_scope);
        Ok(self.cache.get(&h).unwrap())
    }
}

fn commits_in_topo_order<'a>(
    mut queue: VecDeque<CommitId>,
    hugr: &PersistentHugr,
    state_space: &'a PersistentHugr,
) -> impl Iterator<Item = CommitId> + 'a {
    let mut known_commits = hugr.all_commit_ids().collect::<BTreeSet<_>>();

    let next_commit = move || {
        if queue.is_empty() {
            return None;
        }
        for _ in 0..queue.len() {
            let id = queue.pop_front().unwrap();
            let commit = state_space.get_commit(id);
            if commit.parents().all(|p| known_commits.contains(&p.id())) {
                known_commits.insert(id);
                return Some(id);
            }

            // not ready yet, back in the queue
            queue.push_back(id);
        }

        panic!("could not apply any commits in queue")
    };

    iter::from_fn(next_commit)
}

fn hash_from_walker(walker: &Walker) -> u64 {
    let mut hasher = FxHasher64::default();
    for id in walker.as_hugr_view().all_commit_ids().sorted_unstable() {
        id.hash(&mut hasher);
    }
    hasher.finish()
}

pub(super) fn to_resource_scope(hugr: &PersistentHugr) -> ResourceScope<PersistentHugr> {
    let mut all_nodes = HugrView::children(hugr, hugr.entrypoint());
    let [input, output] = (&mut all_nodes).take(2).collect_array().unwrap();
    let nodes = all_nodes.collect_vec();
    let incoming_ports = hugr
        .out_value_types(input)
        .map(|(p, _)| hugr.linked_inputs(input, p).collect_vec())
        .collect_vec();
    let outgoing_ports = hugr
        .in_value_types(output)
        .map(|(p, _)| hugr.single_linked_output(output, p).unwrap())
        .collect_vec();
    let subgraph = SiblingSubgraph::new_unchecked(incoming_ports, outgoing_ports, vec![], nodes);
    ResourceScope::new(hugr.clone(), subgraph)
}
