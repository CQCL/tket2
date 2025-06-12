//! Extension encoder/decoders for the tket2 <-> `pytket` conversion.
//!
//! To add a new extension encoder, implement the [`PytketEmitter`] trait and add
//! it to the [`Tk1EncoderConfig`](crate::serialize::pytket::Tk1EncoderConfig)
//! used for decoding.
//!
//! This module contains decoders for some common extensions. The
//! [`default_encoder_config`](crate::serialize::pytket::default_encoder_config)
//! creates a configuration with the decoders for the standard library and tket2
//! extension.

mod bool;
mod float;
mod prelude;
mod rotation;
mod tk1;
mod tk2;

pub use bool::BoolEmitter;
pub use float::FloatEmitter;
pub use prelude::PreludeEmitter;
pub use rotation::RotationEmitter;
pub use tk1::Tk1Emitter;
pub use tk2::Tk2Emitter;

pub(crate) use bool::set_bits_op;
pub(crate) use tk1::OpaqueTk1Op;

use super::encoder::{RegisterCount, TrackedValues};
use super::Tk1EncoderContext;
use crate::serialize::pytket::encoder::EncodeStatus;
use crate::serialize::pytket::Tk1ConvertError;
use crate::Circuit;
use hugr::extension::ExtensionId;
use hugr::ops::constant::OpaqueValue;
use hugr::ops::ExtensionOp;
use hugr::types::CustomType;
use hugr::HugrView;

/// An encoder of HUGR operations and types that transforms them into pytket
/// primitives.
///
/// An [encoder configuration](crate::serialize::pytket::Tk1EncoderConfig)
/// contains a list of such encoders. When encountering a type, operation, or
/// constant in the HUGR being encoded, the configuration will call each of
/// the encoders declaring support for the specific extension sequentially until
/// one of them indicates a successful conversion.
pub trait PytketEmitter<H: HugrView> {
    /// The name of the extension this encoder/decoder is for.
    ///
    /// [`PytketEmitter::op_to_pytket`] and [`PytketEmitter::type_to_pytket`] will
    /// only be called for operations/types of these extensions.
    ///
    /// If the function returns `None`, the encoder will be called for all
    /// operations/types irrespective of their extension.
    fn extensions(&self) -> Option<Vec<ExtensionId>>;

    /// Given a node in the HUGR circuit and its operation type, try to convert
    /// it to a pytket operation and add it to the pytket encoder.
    ///
    /// Returns an [`EncodeStatus`] indicating if the operation was successfully
    /// converted. If the operation is not supported by the encoder, it's
    /// important to **not** modify the `encoder` context as that may invalidate
    /// the context for other encoders that may be called afterwards.
    fn op_to_pytket(
        &self,
        node: H::Node,
        op: &ExtensionOp,
        circ: &Circuit<H>,
        encoder: &mut Tk1EncoderContext<H>,
    ) -> Result<EncodeStatus, Tk1ConvertError<H::Node>> {
        let _ = (node, op, circ, encoder);
        Ok(EncodeStatus::Unsupported)
    }

    /// Given a HUGR type, return the number of qubits, bits, and parameter
    /// expressions of its pytket counterpart.
    ///
    /// If the type cannot be translated into a list of the aforementioned
    /// values, return `None`. Operations dealing with such types will be
    /// marked as unsupported and will be serialized as opaque operations.
    fn type_to_pytket(
        &self,
        typ: &CustomType,
    ) -> Result<Option<RegisterCount>, Tk1ConvertError<H::Node>> {
        let _ = typ;
        Ok(None)
    }

    /// Given an opaque constant value, add it to the pytket encoder and return
    /// the values to associate to the loaded constant.
    fn const_to_pytket(
        &self,
        value: &OpaqueValue,
        encoder: &mut Tk1EncoderContext<H>,
    ) -> Result<Option<TrackedValues>, Tk1ConvertError<H::Node>> {
        let _ = (value, encoder);
        Ok(None)
    }
}
