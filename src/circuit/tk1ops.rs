use super::operation::CustomOp;
use super::operation::Signature;
use super::operation::ToCircuitFail;
use tket_json_rs::circuit_json::Operation;

impl CustomOp for Operation {
    fn signature(&self) -> Option<Signature> {
        None
    }

    fn to_circuit(&self) -> Result<super::circuit::Circuit, super::operation::ToCircuitFail> {
        Err(ToCircuitFail)
    }
}
