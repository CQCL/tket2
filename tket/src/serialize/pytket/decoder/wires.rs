//! Structures to keep track of pytket [`ElementId`][tket_json_rs::register::ElementId]s and
//! their correspondence to wires in the hugr being defined.

use std::sync::Arc;

use hugr::types::Type;
use hugr::Wire;
use itertools::Itertools;

use crate::serialize::pytket::decoder::Tk1DecoderContext;
use crate::serialize::pytket::Tk1DecodeError;

/// Input wires to a pytket circuit
#[derive(Debug, Clone)]
pub struct InputWires<'a> {
    /// Computed list of wires corresponding to the arguments,
    /// along with their types.
    wires: Vec<WireData<'a>>,
}

impl<'a> InputWires<'a> {
    /// Retrieve the wire data at the given index.
    ///
    /// Panics if the index is out of bounds. See [`InputWires::len`].
    pub fn wire_data(&self, idx: usize) -> &WireData<'a> {
        self.wires.get(idx).unwrap_or_else(|| {
            panic!(
                "Cannot get wire data at index {idx}, only {} wires are tracked",
                self.wires.len()
            )
        })
    }

    /// Return the number of wires tracked.
    ///
    /// To convert the wires into specific types and pack/unpack tuples,
    /// use [`InputWires::into_types`].
    pub fn len(&self) -> usize {
        self.wires.len()
    }

    /// Return whether there are no tracked wires.
    pub fn is_empty(&self) -> bool {
        self.wires.is_empty()
    }

    /// Return an iterator over the wires and their types.
    ///
    /// This returns the wires as-is, without any additional conversions.
    /// If you need to retrieve a specific wire type, use TODO
    pub fn iter(&self) -> impl Iterator<Item = &'_ WireData<'a>> + '_ {
        self.wires.iter()
    }

    /// Transform the current [`InputWires`] into a new `InputWires` with the given
    /// wire types, if possible.
    ///
    /// This transformation packs/unpacks tuples, converts between bool types, etc.
    ///
    /// Any wires not specified by `new_types` will be left unchanged.
    ///
    /// The `operation` parameter is a user-friendly location name used when reporting errors.
    pub fn into_types<'op>(
        self,
        new_types: impl IntoIterator<Item = &'op Type>,
        operation: &str,
        decoder: &mut Tk1DecoderContext<'a>,
    ) -> Result<InputWires<'a>, Tk1DecodeError> {
        let new_wires =
            WireTracker::<'a>::transform_wires(self.wires, new_types, operation, decoder)?;
        Ok(InputWires { wires: new_wires })
    }

    /// Transform the current wires into a new set of wires with the given
    /// types, if possible, and return them as an array.
    ///
    /// This transformation packs/unpacks tuples, converts between bool types, etc.
    ///
    /// Any wires not specified by `new_types` will be left unchanged.
    pub fn into_types_array<'op, const N: usize>(
        self,
        new_types: &[Type; N],
        operation: &str,
        decoder: &mut Tk1DecoderContext<'a>,
    ) -> Result<([WireData<'a>; N], InputWires<'a>), Tk1DecodeError> {
        let new_wires = self.into_types(new_types, operation, decoder)?;
        let wire_arr: [WireData<'a>; N] = new_wires.wires[..N]
            .iter()
            .cloned()
            .collect_array()
            .unwrap_or_else(|| {
                panic!("Expected at least {N} wires, got {}", new_wires.wires.len())
            });
        Ok((wire_arr, new_wires))
    }

    /// Checks that we have the expected number of wires.
    ///
    /// Returns an error otherwise.
    pub fn check_len(&self, expected: usize, operation: &str) -> Result<(), Tk1DecodeError> {
        if self.wires.len() != expected {
            let types = self.wires.iter().map(|wd| wd.ty.to_string()).collect_vec();
            Err(Tk1DecodeError::UnexpectedInputWires {
                expected,
                actual: self.wires.len(),
                types,
                operation: operation.to_string(),
            })
        } else {
            Ok(())
        }
    }
}

impl<'a> IntoIterator for InputWires<'a> {
    type Item = WireData<'a>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.wires.into_iter()
    }
}

/// Tracked data for a wire in [`InputWires`]
#[derive(Debug, Clone, PartialEq)]
pub struct WireData<'a> {
    /// The identifier in the hugr.
    pub wire: Wire,
    /// The type of the wire.
    pub ty: Arc<Type>,
    /// List of pytket arguments corresponding to this wire.
    pub args: Vec<&'a tket_json_rs::register::ElementId>,
}

/// Tracker for wires added to a hugr.
#[derive(Debug, Clone)]
pub struct WireTracker<'a> {
    phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a> WireTracker<'a> {
    /// Transform a list of wires into an equivalent set with the given types.
    ///
    /// This transformation packs/unpacks tuples, converts between bool types, etc.
    ///
    /// The `operation` parameter is a user-friendly location name used when reporting errors.
    pub fn transform_wires<'op>(
        wires: Vec<WireData<'a>>,
        new_types: impl IntoIterator<Item = &'op Type>,
        operation: &str,
        decoder: &mut Tk1DecoderContext<'a>,
    ) -> Result<Vec<WireData<'a>>, Tk1DecodeError> {
        // If we already have the types, we can just return the wires.
        let new_types = new_types.into_iter().collect_vec();
        if wires
            .iter()
            .zip(new_types.iter())
            .all(|(wd, new_type)| wd.ty.as_ref() == *new_type)
        {
            return Ok(wires);
        }

        let new_types = new_types.into_iter();
        let new_wires: Vec<WireData> = Vec::with_capacity(new_types.size_hint().0);
        let _ = (operation, decoder, new_wires);
        // TODO: We need to implement the different mappings here.
        // Check if we can use the memoized unpacking helper from
        // [tket2_hseries::extension::qsystem::barrier].
        todo!()
    }
}
