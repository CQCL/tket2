//! Extension encoder/decoders for the tket2 <-> `pytket` conversion.
//!
//! To add a new extension encoder, implement the [`Tk1Encoder`] trait and add
//! it to the [`Tk1EncoderConfig`](crate::serialize::pytket::Tk1EncoderConfig)
//! used for decoding.
//!
//! This module contains decoders for some common extensions. The
//! [`default_encoder_config`](crate::serialize::pytket::default_encoder_config)
//! creates a configuration with the decoders for the standard library and tket2
//! extension.

mod prelude;
mod tk1;
mod tk2;

pub use prelude::PreludeEncoder;
pub use tk1::Tk1OpEncoder;
pub use tk2::Tk2OpEncoder;

pub(crate) use tk1::OpaqueTk1Op;

use super::encoder::RegisterCount;
use super::Tk1EncoderContext;
use crate::serialize::pytket::Tk1ConvertError;
use crate::Circuit;
use hugr::extension::ExtensionId;
use hugr::ops::ExtensionOp;
use hugr::types::CustomType;
use hugr::HugrView;

/// An encoder of HUGR operations and types that transform them
/// into pytket primitives.
pub trait Tk1Encoder<H: HugrView> {
    /// The name of the extension this encoder/decoder is for.
    ///
    /// [`Tk1Encoder::op_to_pytket`] and [`Tk1Encoder::type_to_pytket`] will
    /// only be called for operations/types of these extensions.
    ///
    /// If the function returns `None`, the encoder will be called for all
    /// operations/types irrespective of their extension.
    fn extensions(&self) -> Option<Vec<ExtensionId>>;

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

    /// Given a HUGR type, return the number of qubits, bits, and parameter
    /// expressions of its pytket counterpart.
    ///
    /// If the type cannot be translated into a list of the aforementioned
    /// values, return `None`. Operations dealing with such types will be
    /// marked as unsupported will be serialized as opaque operations.
    fn type_to_pytket(
        &self,
        typ: &CustomType,
    ) -> Result<Option<RegisterCount>, Tk1ConvertError<H::Node>> {
        let _ = typ;
        Ok(None)
    }
}
