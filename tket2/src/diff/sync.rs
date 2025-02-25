use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    iter,
    rc::Rc,
};

use bytemuck::TransparentWrapper;
use hugr::HugrView;
use itertools::Itertools;

use super::{persistent::DetachedDiff, CircuitDiff};

/// A messaging protocol to sync multiple [super::PersistentCircuit] instances.
pub trait SyncProtocol<H> {
    /// Identifiers for instances participating in the syncing
    type ProcessID;

    /// Sync a newly added [CircuitDiff].
    ///
    /// Call this every time a new [CircuitDiff] is added to a [super::PersistentCircuit].
    ///
    /// `detach_from` are the set of circuit diffs that are expected to already
    fn sync(
        &self,
        diff: &CircuitDiff<H>,
        exclude_processes: impl IntoIterator<Item = Self::ProcessID>,
    );

    /// Receive a new diff from the syncing protocol, if any.
    ///
    /// If a [`CircuitDiff`] is incoming, attach
    /// it to the circuit diffs provided and return the new diff, along with
    /// the ID of the message origin. Otherwise returns `None`.
    ///
    /// The expectation is that this should be polled regularly
    /// to process incoming messages. Messages are processed one at a time,
    /// so try_receive should be called repeatedly until `None` is returned.
    fn try_receive(
        &self,
        attach_to: impl IntoIterator<Item = CircuitDiff<H>>,
    ) -> Option<(CircuitDiff<H>, Self::ProcessID)>;
}

/// A trivial syncing protocol that does nothing.
#[derive(Copy, Clone, Debug, Default)]
pub struct NoSync;

impl<H> SyncProtocol<H> for NoSync {
    type ProcessID = ();

    fn sync(
        &self,
        _diff: &CircuitDiff<H>,
        _exclude_processes: impl IntoIterator<Item = Self::ProcessID>,
    ) {
    }

    fn try_receive(
        &self,
        _attach_to: impl IntoIterator<Item = CircuitDiff<H>>,
    ) -> Option<(CircuitDiff<H>, Self::ProcessID)> {
        None
    }
}

/// A queue of detached diffs, along with the ID of the process that sent them.
pub type DetachedQueue<H> = Rc<RefCell<Vec<(DetachedDiff<H>, usize)>>>;

/// A RC queue-based syncing protocol for multiple [super::PersistentCircuit]s
/// that can share memory.
pub struct QueueSync<H> {
    incoming_queue: DetachedQueue<H>,
    outgoing_queues: BTreeMap<usize, DetachedQueue<H>>,
    id: usize,
}

/// Create `N` empty RC queues.
pub fn n_empty_queues<const N: usize, H>() -> [DetachedQueue<H>; N] {
    iter::repeat_with(|| Rc::new(RefCell::new(Vec::new())))
        .take(N)
        .collect_array()
        .unwrap()
}

impl<H> QueueSync<H> {
    /// Construct a new queue sync protocol.
    pub fn new(
        incoming_queue: DetachedQueue<H>,
        outgoing_queues: BTreeMap<usize, DetachedQueue<H>>,
        id: usize,
    ) -> Self {
        Self {
            incoming_queue,
            outgoing_queues,
            id,
        }
    }
}

impl<H: HugrView + Clone> SyncProtocol<H> for QueueSync<H> {
    type ProcessID = usize;

    fn sync(
        &self,
        diff: &CircuitDiff<H>,
        exclude_processes: impl IntoIterator<Item = Self::ProcessID>,
    ) {
        let exclude = BTreeSet::from_iter(exclude_processes);
        let relrc = TransparentWrapper::peel_ref(diff);
        let parents = relrc.all_parents().map(|p| p.hash_id()).collect();
        let detached = relrc.detach(&parents);
        for (id, queue) in self.outgoing_queues.iter() {
            if !exclude.contains(id) {
                queue.borrow_mut().push((detached.clone(), self.id));
            }
        }
    }

    fn try_receive(
        &self,
        attach_to: impl IntoIterator<Item = CircuitDiff<H>>,
    ) -> Option<(CircuitDiff<H>, Self::ProcessID)> {
        let (detached, origin) = self.incoming_queue.borrow_mut().pop()?;
        let relrc = relrc::RelRc::attach(
            detached,
            attach_to
                .into_iter()
                .map(|diff| TransparentWrapper::peel(diff)),
        );
        Some((relrc.into(), origin))
    }
}

#[cfg(feature = "mpi")]
pub use mpi::*;
#[cfg(feature = "mpi")]
mod mpi {
    use std::{
        future::Future,
        task::{Context, Poll},
    };

    use ::mpi::{datatype::DatatypeRef, ffi::MPI_Datatype, traits::Equivalence};
    use derive_more::{From, Into};
    use futures::{future::select_all, task};
    use hugr::Hugr;
    use relrc::mpi::{MPIMode, RelRcCommunicator};

    use crate::diff::{CircuitDiffData, InvalidNodes};

    use super::*;

    /// The rank of an MPI process.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, From, Into)]
    pub struct MPIRank(pub i32);

    /// A syncing protocol that uses MPI to send and receive diffs.
    pub struct MPISync<P> {
        processes: BTreeMap<MPIRank, P>,
    }

    impl<P> MPISync<P> {
        /// Construct a new MPISync protocol.
        pub fn new(processes: BTreeMap<MPIRank, P>) -> Self {
            Self { processes }
        }

        /// A future that returns the first CircuitDiff to be received, along
        /// with the rank of the process that sent it.
        async fn select_recv<H>(
            &self,
            attach_to: impl IntoIterator<Item = CircuitDiff<H>>,
        ) -> (CircuitDiff<H>, MPIRank)
        where
            P: RelRcCommunicator<CircuitDiffData<H>, InvalidNodes>,
        {
            let attach_to = attach_to
                .into_iter()
                .map(|diff| TransparentWrapper::peel(diff))
                .collect_vec();

            let mut futures = Vec::with_capacity(self.processes.len());
            let mut ids = self.processes.keys().copied().collect_vec();
            for (id, proc) in &self.processes {
                ids.push(*id);
                let fut = proc.recv_relrc_async(attach_to.clone(), MPIMode::default());
                futures.push(Box::pin(fut));
            }
            let (value, index, _) = select_all(futures).await;

            (value.into(), ids[index])
        }
    }

    impl<H: HugrView + Clone, P> SyncProtocol<H> for MPISync<P>
    where
        P: RelRcCommunicator<CircuitDiffData<H>, InvalidNodes>,
    {
        type ProcessID = MPIRank;

        fn sync(
            &self,
            diff: &CircuitDiff<H>,
            exclude_processes: impl IntoIterator<Item = Self::ProcessID>,
        ) {
            let exclude = BTreeSet::from_iter(exclude_processes);
            let relrc = TransparentWrapper::peel_ref(diff);
            for (id, proc) in &self.processes {
                if !exclude.contains(id) {
                    proc.send_relrc(&relrc, MPIMode::default());
                }
            }
        }

        fn try_receive(
            &self,
            attach_to: impl IntoIterator<Item = CircuitDiff<H>>,
        ) -> Option<(CircuitDiff<H>, Self::ProcessID)> {
            let mut future = Box::pin(self.select_recv(attach_to));

            match future
                .as_mut()
                .poll(&mut Context::from_waker(task::noop_waker_ref()))
            {
                Poll::Ready(value) => Some(value),
                Poll::Pending => None,
            }
        }
    }

    unsafe impl Equivalence for CircuitDiffData<Hugr> {
        type Out = DatatypeRef<'static>;

        fn equivalent_datatype() -> Self::Out {
            todo!()
        }
    }

    unsafe impl Equivalence for InvalidNodes {
        type Out = DatatypeRef<'static>;

        fn equivalent_datatype() -> Self::Out {
            todo!()
        }
    }
}
