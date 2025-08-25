//! Resource flow logic for tracking how resources move through operations.
//!
//! This module defines the [`ResourceFlow`] trait that specifies how resources
//! flow through operations, along with a default implementation.

use crate::resource::types::ResourceId;
use derive_more::derive::{Display, Error};
use hugr::ops::OpTrait;
use hugr::types::Type;
use hugr::HugrView;
use itertools::{EitherOrBoth, Itertools};

/// Error type for unsupported operations in ResourceFlow implementations.
#[derive(Debug, Display, Clone, PartialEq, Error)]
#[display("Unsupported operation")]
pub struct UnsupportedOp;

/// Trait for specifying how resources flow through operations.
///
/// This trait allows different implementations to define how linear resources
/// are mapped from inputs to outputs through various operation types.
pub trait ResourceFlow<H: HugrView> {
    /// Map resource IDs from operation inputs to outputs.
    ///
    /// Takes an operation type and the resource IDs of the operation's inputs.
    /// The i-th entry is Some(resource_id) if the i-th port is a linear type,
    /// None otherwise. Returns the resource IDs of the operation's outputs
    /// in port order. Output resource IDs should be one of the input resource
    /// IDs for resource-preserving operations, or None for new resources or
    /// non-linear types.
    ///
    /// # Arguments
    /// * `op` - The operation type
    /// * `inputs` - Resource IDs for each input port (None for non-linear
    ///   types)
    ///
    /// # Returns
    /// Resource IDs for each output port, or UnsupportedOp if the operation
    /// cannot be handled by this implementation.
    fn map_resources(
        &self,
        node: H::Node,
        hugr: &H,
        inputs: &[Option<ResourceId>],
    ) -> Result<Vec<Option<ResourceId>>, UnsupportedOp>;

    /// Convert this ResourceFlow into a boxed trait object.
    fn into_boxed<'a>(self) -> Box<dyn 'a + ResourceFlow<H>>
    where
        Self: 'a + Sized,
    {
        Box::new(self)
    }
}

impl<H: HugrView> ResourceFlow<H> for Box<dyn '_ + ResourceFlow<H>> {
    fn map_resources(
        &self,
        node: H::Node,
        hugr: &H,
        inputs: &[Option<ResourceId>],
    ) -> Result<Vec<Option<ResourceId>>, UnsupportedOp> {
        self.as_ref().map_resources(node, hugr, inputs)
    }
}

/// Default implementation of ResourceFlow.
///
/// This implementation considers that an operation is resource-preserving if
/// whenever the i-th input or output is linear, then the i-th input type
/// matches the i-th output. The i-th input is then mapped to the i-th output.
///
/// Otherwise, all input resources are discarded and all outputs will be given
/// fresh resource IDs.
#[derive(Debug, Clone, Default)]
pub struct DefaultResourceFlow;

impl DefaultResourceFlow {
    /// Create a new DefaultResourceFlow instance.
    pub fn new() -> Self {
        Self
    }

    /// Check if a type is linear (non-copyable).
    fn is_linear_type(ty: &Type) -> bool {
        !ty.copyable()
    }

    /// Determine if an operation is resource-preserving based on input/output
    /// types.
    fn is_resource_preserving(input_types: &[Type], output_types: &[Type]) -> bool {
        // An operation is resource-preserving if for each i, if input[i] or
        // output[i] is linear, then type(input[i]) == type(output[i])

        for io_ty in input_types.iter().zip_longest(output_types.iter()) {
            let (input_ty, output_ty) = match io_ty {
                EitherOrBoth::Both(input_ty, output_ty) => (input_ty, output_ty),
                EitherOrBoth::Left(ty) | EitherOrBoth::Right(ty) => {
                    if Self::is_linear_type(ty) {
                        // linear type on one side, nothing on the other
                        return false;
                    }
                    continue;
                }
            };

            if Self::is_linear_type(input_ty) || Self::is_linear_type(output_ty) {
                // If input/output is linear, both must be the same type
                if input_ty != output_ty {
                    return false;
                }
            }
        }

        true
    }
}

impl<H: HugrView> ResourceFlow<H> for DefaultResourceFlow {
    fn map_resources(
        &self,
        node: H::Node,
        hugr: &H,
        inputs: &[Option<ResourceId>],
    ) -> Result<Vec<Option<ResourceId>>, UnsupportedOp> {
        let op = hugr.get_optype(node);
        let signature = op.dataflow_signature().expect("dataflow op");
        let input_types = signature.input_types();
        let output_types = signature.output_types();

        debug_assert_eq!(
            inputs.len(),
            input_types.len(),
            "Input resource array length must match operation input count"
        );

        if Self::is_resource_preserving(input_types, output_types) {
            Ok(retain_linear_types(inputs.to_vec(), output_types))
        } else {
            // Resource-producing/consuming: all linear outputs are new resources (None)
            Ok(vec![None; output_types.len()])
        }
    }
}

fn retain_linear_types(
    mut resources: Vec<Option<ResourceId>>,
    types: &[Type],
) -> Vec<Option<ResourceId>> {
    for (ty, resource) in types.iter().zip(resources.iter_mut()) {
        if ty.copyable() {
            *resource = None;
        }
    }
    resources
}
