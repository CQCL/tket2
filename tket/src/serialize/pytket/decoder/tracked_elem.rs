//! Pytket qubit and bit elements that we track during decoding.

use std::sync::{Arc, LazyLock};

use hugr::extension::prelude::{bool_t, qb_t};
use hugr::types::Type;
use tket_json_rs::register::ElementId as PytketRegister;

/// An internal lightweight identifier for a [`TrackedQubit`] in the decoder.
#[derive(Clone, Copy, Debug, derive_more::Display, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct TrackedQubitId(#[display(transparent)] pub usize);

/// An internal lightweight identifier for a [`TrackedBit`] in the decoder.
#[derive(Clone, Copy, Debug, derive_more::Display, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct TrackedBitId(#[display(transparent)] pub usize);

/// An identifier for a pytket qubit register in the data carried by a wire.
///
/// After a pytket circuit assigns a new value to the register, older
/// [`TrackedQubit`]s referring to it become _outdated_.
///
/// Outdated values no longer correspond to a pytket circuit register, but they
/// can still be found in the wires of the hugr being extracted.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TrackedQubit {
    outdated: bool,
    reg: Arc<PytketRegister>,
}

/// An identifier for a pytket bit register in the data carried by a wire.
///
/// After a pytket circuit assigns a new value to the register, older
/// [`TrackedBit`]s referring to it become _outdated_.
///
/// Outdated values no longer correspond to a pytket circuit register, but they
/// can still be found in the wires of the hugr being extracted.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TrackedBit {
    outdated: bool,
    reg: Arc<PytketRegister>,
}

impl TrackedQubit {
    /// Return a new tracked qubit.
    pub(super) fn new(reg: Arc<PytketRegister>) -> Self {
        Self {
            outdated: false,
            reg,
        }
    }

    /// Returns the pytket register for this tracked element.
    pub fn pytket_register(&self) -> &PytketRegister {
        &self.reg
    }

    /// Returns the pytket register for this tracked element.
    pub fn pytket_register_arc(&self) -> Arc<PytketRegister> {
        self.reg.clone()
    }

    /// Returns the type of the element.
    pub fn ty(&self) -> Arc<Type> {
        static QUBIT_TYPE: LazyLock<Arc<Type>> = LazyLock::new(|| qb_t().into());
        QUBIT_TYPE.clone()
    }

    /// Returns `true` if the element has been overwritten by a new value.
    pub fn is_outdated(&self) -> bool {
        self.outdated
    }

    /// Mark the element as outdated.
    pub(super) fn mark_outdated(&mut self) {
        self.outdated = true;
    }
}

impl TrackedBit {
    /// Returns a new tracked bit.
    pub(super) fn new(reg: Arc<PytketRegister>) -> Self {
        Self {
            outdated: false,
            reg,
        }
    }

    /// Returns the pytket register for this tracked element.
    pub fn pytket_register(&self) -> &PytketRegister {
        &self.reg
    }

    /// Returns the pytket register for this tracked element.
    pub fn pytket_register_arc(&self) -> Arc<PytketRegister> {
        self.reg.clone()
    }

    /// Returns the type of the element.
    pub fn ty(&self) -> Arc<Type> {
        static BOOL_TYPE: LazyLock<Arc<Type>> = LazyLock::new(|| bool_t().into());
        BOOL_TYPE.clone()
    }

    /// Returns `true` if the element has been overwritten by a new value.
    pub fn is_outdated(&self) -> bool {
        self.outdated
    }

    /// Mark the element as outdated.
    pub(super) fn mark_outdated(&mut self) {
        self.outdated = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hugr::extension::prelude::{bool_t, qb_t};
    use hugr::types::Type;
    use rstest::rstest;
    use std::sync::Arc;
    use tket_json_rs::register::ElementId;

    #[rstest]
    fn tracked_qubit_basic_behaviour() {
        let reg = Arc::new(ElementId("q".to_string(), vec![0]));
        let mut tq = TrackedQubit::new(reg.clone());

        assert!(!tq.is_outdated());
        assert_eq!(tq.pytket_register(), &*reg);
        assert_eq!(tq.pytket_register_arc(), reg);
        assert_eq!(&*tq.ty(), &Type::from(qb_t()));

        tq.mark_outdated();
        assert!(tq.is_outdated());
    }

    #[rstest]
    fn tracked_bit_basic_behaviour() {
        let reg = Arc::new(ElementId("c".to_string(), vec![1]));
        let mut tb = TrackedBit::new(reg.clone());

        assert!(!tb.is_outdated());
        assert_eq!(tb.pytket_register(), &*reg);
        assert_eq!(tb.pytket_register_arc(), reg);
        assert_eq!(&*tb.ty(), &Type::from(bool_t()));

        tb.mark_outdated();
        assert!(tb.is_outdated());
    }
}
