//! Configuration for converting [`Circuit`]s into [`SerialCircuit`]
//!
//! A configuration struct contains a list of custom encoders that define translations
//! of HUGR operations and types into pytket primitives.

use std::collections::HashMap;

use hugr::extension::ExtensionId;
use hugr::ops::ExtensionOp;
use hugr::types::{CustomType, Type, TypeEnum};

use crate::serialize::pytket::op::tk2op::Tk2OpEncoder;
use crate::serialize::pytket::Tk1ConvertError;
use crate::Circuit;

use super::value_tracker::RegisterCount;
use super::Tk1EncoderContext;
use hugr::extension::prelude::bool_t;
use hugr::HugrView;
use itertools::Itertools;

/// Default encoder configuration for [`Circuit`]s.
///
/// Contains encoders for std and tket2 operations.
pub fn default_encoder_config<H: HugrView>() -> Tk1EncoderConfig<H> {
    // TODO: Add std & tket2 encoders
    let mut config = Tk1EncoderConfig::new();
    config.add_encoder(Tk2OpEncoder);
    config
}

/// An encoder of HUGR operations and types that transform them
/// into pytket primitives.
pub trait Tk1Encoder<H: HugrView> {
    /// The name of the extension this encoder/decoder is for.
    ///
    /// [`Tk1Encoder::op_to_pytket`] and [`Tk1Encoder::type_to_pytket`] will
    /// only be called for operations/types of these extensions.
    fn extensions(&self) -> Vec<ExtensionId>;

    /// Given a node in the HUGR circuit and its operation type, try to convert
    /// it to a pytket operation and add it to the pytket encoder.
    ///
    /// Returns `true` if the operation was successfully converted. If that is
    /// the case, no further encoders will be called.
    ///
    /// If the operation is not supported by the encoder, return `false`.
    fn op_to_pytket(
        &self,
        node: H::Node,
        op: &ExtensionOp,
        circ: &Circuit<H>,
        encoder: &mut Tk1EncoderContext<H>,
    ) -> Result<bool, Tk1ConvertError<H::Node>>;

    /// Given a HUGR type, return the number of qubits, bits, and sympy
    /// parameters of its pytket counterpart.
    ///
    /// If the type is not supported by the encoder, return `None`.
    fn type_to_pytket(
        &self,
        #[allow(unused)] op: &CustomType,
    ) -> Result<Option<RegisterCount>, Tk1ConvertError<H::Node>> {
        Ok(None)
    }
}

/// Configuration for converting [`Circuit`] into [`SerialCircuit`].
///
/// Contains custom encoders that define translations for HUGR operations and types
/// into pytket primitives.
#[derive(derive_more::Debug)]
#[debug(bounds(H: HugrView))]
pub struct Tk1EncoderConfig<H: HugrView> {
    /// Operation encoders
    #[debug(skip)]
    pub(super) encoders: Vec<Box<dyn Tk1Encoder<H>>>,
    /// Pre-computed map from extension ids to corresponding encoders in
    /// `encoders`, identified by their index.
    #[debug("{:?}", extension_encoders.keys().collect_vec())]
    extension_encoders: HashMap<ExtensionId, Vec<usize>>,
}

impl<H: HugrView> Tk1EncoderConfig<H> {
    /// Create a new [`Tk1EncoderConfig`] with no encoders.
    pub fn new() -> Self {
        Self {
            encoders: vec![],
            extension_encoders: HashMap::new(),
        }
    }

    /// Add an encoder to the configuration.
    pub fn add_encoder(&mut self, encoder: impl Tk1Encoder<H> + 'static) {
        let idx = self.encoders.len();

        for ext in encoder.extensions() {
            self.extension_encoders.entry(ext).or_default().push(idx);
        }
        self.encoders.push(Box::new(encoder));
    }

    /// List the extensions supported by the encoders.
    ///
    /// Use [`Tk1EncoderConfig::add_encoder`] to extend this list.
    pub fn supported_extensions(&self) -> impl Iterator<Item = &ExtensionId> {
        self.extension_encoders.keys()
    }

    /// Encode a HUGR operation using the registered custom encoders.
    ///
    /// Returns `true` if the operation was successfully converted. If that is
    pub fn op_to_pytket(
        &self,
        node: H::Node,
        op: &ExtensionOp,
        circ: &Circuit<H>,
        encoder: &mut Tk1EncoderContext<H>,
    ) -> Result<bool, Tk1ConvertError<H::Node>> {
        let mut result = false;
        let extension = op.def().extension_id();
        for enc in self.encoders_for_extension(extension) {
            if enc.op_to_pytket(node, op, circ, encoder)? {
                result = true;
                break;
            }
        }
        Ok(result)
    }

    /// Translate a HUGR type into a count of qubits, bits, and parameters,
    /// using the registered custom encodes.
    ///
    /// Only tuple sums, bools, and custom types are supported.
    /// Other types will return `None`.
    pub fn type_to_pytket(
        &self,
        typ: &Type,
    ) -> Result<Option<RegisterCount>, Tk1ConvertError<H::Node>> {
        match typ.as_type_enum() {
            TypeEnum::Sum(sum) => {
                if typ == &bool_t() {
                    return Ok(Some(RegisterCount {
                        qubits: 0,
                        bits: 1,
                        params: 0,
                    }));
                }
                if let Some(tuple) = sum.as_tuple() {
                    let count: Result<Option<RegisterCount>, Tk1ConvertError<H::Node>> = tuple
                        .iter()
                        .map(|ty| {
                            match ty.clone().try_into() {
                                Ok(ty) => Ok(self.type_to_pytket(&ty)?),
                                // Sum types with row variables (variable tuple lengths) are not supported.
                                Err(_) => Ok(None),
                            }
                        })
                        .sum();
                    return count;
                }
            }
            TypeEnum::Extension(custom) => {
                let type_ext = custom.extension();
                for encoder in self.encoders_for_extension(type_ext) {
                    if let Some(count) = encoder.type_to_pytket(custom)? {
                        return Ok(Some(count));
                    }
                }
            }
            _ => {}
        }
        Ok(None)
    }

    /// Lists the encoders that can handle a given extension.
    fn encoders_for_extension(
        &self,
        ext: &ExtensionId,
    ) -> impl Iterator<Item = &Box<dyn Tk1Encoder<H>>> {
        self.extension_encoders
            .get(ext)
            .into_iter()
            .flat_map(move |idxs| idxs.iter().map(move |idx| &self.encoders[*idx]))
    }
}

impl<H: HugrView> Default for Tk1EncoderConfig<H> {
    fn default() -> Self {
        Self {
            encoders: Default::default(),
            extension_encoders: Default::default(),
        }
    }
}
