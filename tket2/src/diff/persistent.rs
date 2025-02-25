use std::{
    cmp,
    collections::{BTreeMap, BTreeSet},
};

use bytemuck::TransparentWrapper;
use derive_more::{From, Into};
use derive_where::derive_where;
use hugr::{HugrView, Node};
use relrc::detached::Detached;

use super::{
    sync::{NoSync, SyncProtocol},
    CircuitDiff, CircuitDiffData, CircuitHistory, InvalidNodes,
};

#[repr(transparent)]
#[derive(From, Into)]
#[derive_where(Clone)]
struct PtrEqDiff<H>(CircuitDiff<H>);

unsafe impl<H> TransparentWrapper<CircuitDiff<H>> for PtrEqDiff<H> {}

pub type DetachedDiff<H> = Detached<CircuitDiffData<H>, InvalidNodes>;

impl<H> PartialEq for PtrEqDiff<H> {
    fn eq(&self, other: &Self) -> bool {
        self.0.as_ptr() == other.0.as_ptr()
    }
}
impl<H> Eq for PtrEqDiff<H> {}

impl<H> PartialOrd for PtrEqDiff<H> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(&other))
    }
}

impl<H> Ord for PtrEqDiff<H> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.0.as_ptr().cmp(&other.0.as_ptr())
    }
}

#[derive_where(Clone, Default; S, V)]
pub struct PersistentCircuit<H, S = NoSync, V = DiffUnitValue> {
    diffs: BTreeMap<PtrEqDiff<H>, usize>,
    sync_protocol: S,
    diff_value: V,
}

impl<H, S, V> PersistentCircuit<H, S, V> {
    pub fn new(sync_protocol: S, diff_value: V) -> Self {
        Self {
            diffs: Default::default(),
            sync_protocol,
            diff_value,
        }
    }

    pub fn n_diffs(&self) -> usize
    where
        H: HugrView<Node = Node>,
    {
        let mut all_diffs = BTreeSet::new();
        let mut diff_stack = Vec::from_iter(self.diffs.keys().map(|d| d.0.clone()));
        while let Some(d) = diff_stack.pop() {
            if all_diffs.insert(d.0.hash_id()) {
                for parent in d.all_parents() {
                    diff_stack.push(parent);
                }
            }
        }
        all_diffs.len()
    }

    pub fn diff_iter(&self) -> impl Iterator<Item = CircuitDiff<H>> + '_ {
        self.diffs
            .keys()
            .map(|diff| PtrEqDiff::peel_ref(diff))
            .cloned()
    }
}

impl<H: HugrView, S: SyncProtocol<H>, V: DiffValue<H>> PersistentCircuit<H, S, V> {
    pub fn add_diff(&mut self, diff: CircuitDiff<H>)
    where
        S: SyncProtocol<H>,
        V: DiffValue<H>,
    {
        self.add_diff_exclude(diff, [])
    }

    fn add_diff_exclude(
        &mut self,
        diff: CircuitDiff<H>,
        exclude: impl IntoIterator<Item = S::ProcessID>,
    ) where
        S: SyncProtocol<H>,
        V: DiffValue<H>,
    {
        self.sync_protocol.sync(&diff, exclude);
        let diff_value = self.diff_value.value(&diff);
        self.diffs.insert(diff.into(), diff_value);
    }

    pub fn extract_best(&self) -> CircuitHistory<H> {
        todo!()
    }

    pub fn sync(&mut self) {
        while let Some((diff, origin)) = self.sync_protocol.try_receive(self.diff_iter()) {
            self.add_diff_exclude(diff, [origin]);
        }
    }
}

pub trait DiffValue<H> {
    fn value(&self, diff: &CircuitDiff<H>) -> usize;
}

#[derive(Copy, Clone, Debug, Default)]
pub struct DiffUnitValue;

impl<H> DiffValue<H> for DiffUnitValue {
    fn value(&self, _diff: &CircuitDiff<H>) -> usize {
        1
    }
}

impl<H, F> DiffValue<H> for F
where
    F: Fn(&CircuitDiff<H>) -> usize,
{
    fn value(&self, diff: &CircuitDiff<H>) -> usize {
        self(diff)
    }
}
