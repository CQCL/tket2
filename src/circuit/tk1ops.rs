//! Implementation of Hugr's CustomOp trait for the TKET1 bindings in `tket-rs`.
//! 
//! TODO: This cannot be defined here. Should we do it in `hugr` or in `tket-rs`?

use hugr::ops::CustomOp;
use hugr::types::Signature;
use tket_json_rs::circuit_json::Operation;

#[typetag::serde]
impl CustomOp for Operation {
    fn resources(&self) ->  &ResourceSet {
        todo!()
    }

fn name(&self) -> SmolStr {
        todo!()
    }

fn signature(&self) -> Signature {
        todo!()
    }
}
