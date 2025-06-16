//! Configuration for converting [`tket_json_rs::circuit_json::SerialCircuit`]
//! into [`Circuit`]s.
//!
//! A configuration struct contains a list of custom decoders that define
//! translations of legacy tket primitives into HUGR operations.

use hugr::extension::ExtensionId;
use hugr::Node;
use itertools::Itertools;
use std::collections::HashMap;

use crate::serialize::pytket::decoder::{DecodeStatus, Tk1DecoderContext};
use crate::serialize::pytket::extension::PytketDecoder;
use crate::serialize::pytket::Tk1ConvertError;

/// Default pytket decoder configuration.
///
/// Contains a list of custom decoders that define translations of legacy tket
/// primitives into HUGR operations.
pub fn default_decoder_config<'h>() -> Tk1DecoderConfig {
    let config = Tk1DecoderConfig::new();
    // TODO: Add default decoders here.
    config
}

/// Configuration for converting [`tket_json_rs::circuit_json::SerialCircuit`]
/// into [`Circuit`]s.
///
/// Contains custom decoders that define translations for HUGR operations,
/// types, and consts into pytket primitives.
#[derive(derive_more::Debug)]
pub struct Tk1DecoderConfig {
    /// Operation emitters
    #[debug(skip)]
    pub(super) decoders: Vec<Box<dyn PytketDecoder>>,
    /// Pre-computed map from extension ids to corresponding decoders in
    /// `decoders`, identified by their index.
    #[debug("{:?}", extension_decoders.keys().collect_vec())]
    extension_decoders: HashMap<ExtensionId, Vec<usize>>,
    /// Pre-computed map from pytket optypes to corresponding decoders in
    /// `decoders`, identified by their index.
    #[debug("{:?}", extension_decoders.keys().collect_vec())]
    optype_decoders: HashMap<tket_json_rs::OpType, Vec<usize>>,
}

impl<'h> Tk1DecoderConfig {
    /// Create a new [`Tk1DecoderConfig`] with no decoders.
    pub fn new() -> Self {
        Self {
            decoders: vec![],
            extension_decoders: HashMap::new(),
            optype_decoders: HashMap::new(),
        }
    }

    /// Add a decoder to the configuration.
    pub fn add_decoder(&mut self, decoder: impl PytketDecoder + 'static) {
        let idx = self.decoders.len();

        for ext in decoder.extensions() {
            self.extension_decoders.entry(ext).or_default().push(idx);
        }
        for optype in decoder.op_types() {
            self.optype_decoders.entry(optype).or_default().push(idx);
        }

        self.decoders.push(Box::new(decoder));
    }

    /// List the extensions supported by the decoders.
    ///
    /// Use [`Tk1DecoderConfig::add_decoder`] to extend this list.
    pub fn supported_extensions(&self) -> impl Iterator<Item = &ExtensionId> {
        self.extension_decoders.keys()
    }

    /// Encode a HUGR operation using the registered custom encoders.
    ///
    /// Returns `true` if the operation was successfully converted and no further
    /// encoders should be called.
    pub(super) fn op_to_hugr(
        &self,
        op: &tket_json_rs::circuit_json::Operation,
        args: &[tket_json_rs::register::ElementId],
        opgroup: Option<&str>,
        decoder: &mut Tk1DecoderContext<'h>,
    ) -> Result<DecodeStatus, Tk1ConvertError<Node>> {
        let mut result = DecodeStatus::Unsupported;
        for enc in self.decoders_for_optype(&op.op_type) {
            result = enc.op_to_hugr(op, args, opgroup, decoder)?;
            if result == DecodeStatus::Success {
                break;
            }
        }
        Ok(result)
    }

    /// Lists the decoder that can handle a given pytket optype.
    fn decoders_for_optype(
        &self,
        optype: &tket_json_rs::OpType,
    ) -> impl Iterator<Item = &Box<dyn PytketDecoder>> {
        self.optype_decoders
            .get(optype)
            .into_iter()
            .flatten()
            .map(move |idx| &self.decoders[*idx])
    }
}

impl<'h> Default for Tk1DecoderConfig {
    fn default() -> Self {
        Self {
            decoders: Default::default(),
            extension_decoders: Default::default(),
            optype_decoders: Default::default(),
        }
    }
}
