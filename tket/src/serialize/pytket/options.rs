//! Option structs for the [TKETDecode][super::TKETDecode] trait methods.

use std::sync::Arc;

use hugr::types::Signature;
use hugr::{Hugr, Node};

use crate::serialize::pytket::{PytketDecoderConfig, PytketEncoderConfig};

/// Options used when decoding a pytket
/// [`SerialCircuit`][tket_json_rs::circuit_json::SerialCircuit] into a HUGR.
///
/// See [TKETDecode::decode][super::TKETDecode::decode].
///
/// In contrast to [PytketDecoderConfig] which is normally statically defined by
/// a library, these options may vary between calls.
#[derive(Default, Clone)]
#[non_exhaustive]
pub struct DecodeOptions {
    /// The configuration for the decoder, containing custom
    /// operation decoders.
    ///
    /// When `None`, we will use [`default_decoder_config`][super::default_decoder_config].
    pub config: Option<Arc<PytketDecoderConfig>>,
    /// The name of the function to create.
    ///
    /// If `None`, we will use the name of the circuit, or "main" if the circuit
    /// has no name.
    pub fn_name: Option<String>,
    /// The signature of the function to create. This should match the number of qubits and bits in the circuit.
    ///
    /// If `None`, we will use qubits, bools, and [rotation_type][crate::extension::rotation::rotation_type] parameters.
    pub signature: Option<Signature>,
    /// A list of parameter names to add to the function input.
    ///
    /// If additional parameters are found in the circuit, they will be added
    /// after these using generic names.
    pub input_params: Vec<String>,
}

impl DecodeOptions {
    /// Create a new [`DecodeOptions`] with the default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a decoder configuration.
    pub fn with_config(mut self, config: impl Into<Arc<PytketDecoderConfig>>) -> Self {
        self.config = Some(config.into());
        self
    }

    /// Set the name of the function to create.
    pub fn with_fn_name(mut self, fn_name: impl ToString) -> Self {
        self.fn_name = Some(fn_name.to_string());
        self
    }

    /// Set the signature of the function to create.
    pub fn with_signature(mut self, signature: Signature) -> Self {
        self.signature = Some(signature);
        self
    }

    /// Set the input parameter names.
    pub fn with_input_params(mut self, input_params: impl IntoIterator<Item = String>) -> Self {
        self.input_params = input_params.into_iter().collect();
        self
    }
}

/// Where to insert the decoded circuit when calling
/// [`TKETDecode::decode_inplace`][super::TKETDecode::decode_inplace].
#[derive(Debug, derive_more::Display, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DecodeInsertionTarget {
    /// Insert the decoded circuit as a new function in the HUGR.
    #[default]
    Function,
    /// Insert the decoded circuit as a dataflow region in the HUGR under the given parent.
    Region {
        /// The parent node that will contain the circuit's decoded DFG.
        parent: Node,
    },
}

/// Options used when encoding a HUGR into a pytket
/// [`SerialCircuit`][tket_json_rs::circuit_json::SerialCircuit].
///
/// See [TKETDecode::encode][super::TKETDecode::encode].
///
/// In contrast to [PytketEncoderConfig] which is normally statically defined by
/// a library, these options may vary between calls.
#[derive(Default, Clone)]
#[non_exhaustive]
pub struct EncodeOptions {
    /// The configuration for the decoder, containing custom
    /// operation decoders.
    ///
    /// When `None`, we will use [`default_encoder_config`][super::default_encoder_config].
    pub config: Option<Arc<PytketEncoderConfig<Hugr>>>,
}

impl EncodeOptions {
    /// Create a new [`EncodeOptions`] with the default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a encoder configuration.
    pub fn with_config(mut self, config: impl Into<Arc<PytketEncoderConfig<Hugr>>>) -> Self {
        self.config = Some(config.into());
        self
    }
}
