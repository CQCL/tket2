//! Extension encoder/decoders for the tket <-> `pytket` conversion.
//!
//! To add a new extension encoder, implement the [`PytketEmitter`] trait and add
//! it to the [`Tk1EncoderConfig`](crate::serialize::pytket::Tk1EncoderConfig)
//! used for decoding.
//!
//! This module contains decoders for some common extensions. The
//! [`default_encoder_config`](crate::serialize::pytket::default_encoder_config)
//! creates a configuration with the decoders for the standard library and tket
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

use super::encoder::TrackedValues;
use crate::serialize::pytket::encoder::{EncodeStatus, Tk1EncoderContext};
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
//
// Note: The HugrView type needs to be fixed at the type level rather than on
// the method signatures to ensure the trait is dyn compatible.
pub trait PytketEmitter<H: HugrView> {
    /// The name of the extension this encoder is for.
    ///
    /// [`PytketEmitter::op_to_pytket`] will only be called for operations/types
    /// of these extensions.
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

/// A translator of HUGR types that describes how to encode them as pytket
/// registers (qubits, bits, and parameter expressions).
///
/// This is used both during encoding and decoding of pytket operations and
/// types.
pub trait PytketTypeTranslator {
    /// The name of the extensions this translator is for.
    ///
    /// [`PytketTypeTranslator::type_to_pytket`] will only be called for
    /// operations/types of these extensions.
    fn extensions(&self) -> Vec<ExtensionId>;

    /// Given a HUGR opaque type, return the number of qubits, bits, and
    /// parameter expressions of its pytket counterpart.
    ///
    /// If the type cannot be fully translated into a list of the aforementioned
    /// values, return `None`. Operations dealing with such types will be marked
    /// as unsupported and will be serialized as opaque operations.
    fn type_to_pytket(&self, typ: &CustomType) -> Option<RegisterCount> {
        let _ = typ;
        None
    }
}

/// A count of pytket qubits, bits, and sympy parameters.
///
/// Used as return value for [`TrackedValues::count`].
#[derive(
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Debug,
    Default,
    derive_more::Display,
    derive_more::Add,
    derive_more::Sub,
    derive_more::Sum,
)]
#[display("{qubits} qubits, {bits} bits, {params} parameters")]
#[non_exhaustive]
pub struct RegisterCount {
    /// Amount of qubits.
    pub qubits: usize,
    /// Amount of bits.
    pub bits: usize,
    /// Amount of sympy parameters.
    pub params: usize,
}

impl RegisterCount {
    /// Create a new [`RegisterCount`] from the number of qubits, bits, and parameters.
    pub const fn new(qubits: usize, bits: usize, params: usize) -> Self {
        RegisterCount {
            qubits,
            bits,
            params,
        }
    }

    /// Create a new [`RegisterCount`] containing only qubits.
    pub const fn only_qubits(qubits: usize) -> Self {
        RegisterCount {
            qubits,
            bits: 0,
            params: 0,
        }
    }

    /// Create a new [`RegisterCount`] containing only bits.
    pub const fn only_bits(bits: usize) -> Self {
        RegisterCount {
            qubits: 0,
            bits,
            params: 0,
        }
    }

    /// Create a new [`RegisterCount`] containing only parameters.
    pub const fn only_params(params: usize) -> Self {
        RegisterCount {
            qubits: 0,
            bits: 0,
            params,
        }
    }

    /// Returns the number of qubits, bits, and parameters associated with the wire.
    pub const fn total(&self) -> usize {
        self.qubits + self.bits + self.params
    }
}
