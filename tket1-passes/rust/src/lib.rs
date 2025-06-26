//! Rust interface for tket1-passes library
//!
//! This crate provides safe Rust bindings to the `libtket1-passes.dylib` C library
//! that exposes some of TKET1's passes as Rust functions.

use std::ffi::{c_char, CStr, CString};
use std::ptr;
use thiserror::Error;
use tket_json_rs::{OpType, SerialCircuit};

// Include the auto-generated bindings
mod ffi {
    #![allow(non_camel_case_types)]
    #![allow(non_upper_case_globals)]
    #![allow(dead_code)]
    #![allow(non_snake_case)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

/// Error types that can occur when using the tket library
#[derive(Error, Debug)]
pub enum PassError {
    /// Null pointer provided
    #[error("Null pointer provided")]
    NullPointer,
    /// Invalid argument provided
    #[error("Invalid argument provided")]
    InvalidArgument,
    /// Circuit is invalid
    #[error("Circuit is invalid")]
    CircuitInvalid,
    /// Memory allocation failed
    #[error("Memory allocation failed")]
    Memory,
    /// Failed to process JSON
    #[error("Failed to process JSON: {0}")]
    JsonError(String),
    /// Unknown error occurred
    #[error("Unknown error occurred")]
    Unknown,
    /// FFI error
    #[error("Invalid target gate: {0:?}")]
    InvalidTargetGate(OpType),
}

impl From<std::ffi::NulError> for PassError {
    fn from(_: std::ffi::NulError) -> Self {
        PassError::NullPointer
    }
}

impl From<serde_json::Error> for PassError {
    fn from(err: serde_json::Error) -> Self {
        PassError::JsonError(err.to_string())
    }
}

impl From<ffi::TketError> for PassError {
    fn from(err: ffi::TketError) -> Self {
        match err {
            ffi::TketError_TKET_ERROR_NULL_POINTER => PassError::NullPointer,
            ffi::TketError_TKET_ERROR_INVALID_ARGUMENT => PassError::InvalidArgument,
            ffi::TketError_TKET_ERROR_CIRCUIT_INVALID => PassError::CircuitInvalid,
            ffi::TketError_TKET_ERROR_MEMORY => PassError::Memory,
            ffi::TketError_TKET_ERROR_PARSE_JSON => {
                PassError::JsonError("Failed to parse JSON".to_string())
            }
            ffi::TketError_TKET_ERROR_UNKNOWN => PassError::Unknown,
            _ => PassError::Unknown,
        }
    }
}

fn try_from_optype(optype: OpType) -> Result<ffi::TketTargetGate, PassError> {
    match optype {
        OpType::CX => Ok(ffi::TketTargetGate_TKET_TARGET_CX),
        OpType::TK2 => Ok(ffi::TketTargetGate_TKET_TARGET_TK2),
        _ => Err(PassError::InvalidTargetGate(optype)),
    }
}

/// A quantum circuit in TKET1's in-memory format
pub struct Tket1Circuit {
    inner: *mut ffi::TketCircuit,
}

impl From<&SerialCircuit> for Tket1Circuit {
    fn from(serial_circuit: &SerialCircuit) -> Self {
        Tket1Circuit::from_serial_circuit(serial_circuit).unwrap()
    }
}

impl From<SerialCircuit> for Tket1Circuit {
    fn from(serial_circuit: SerialCircuit) -> Self {
        Tket1Circuit::from_serial_circuit(&serial_circuit).unwrap()
    }
}

impl From<Tket1Circuit> for SerialCircuit {
    fn from(circuit: Tket1Circuit) -> Self {
        circuit.to_serial_circuit().unwrap()
    }
}

impl Tket1Circuit {
    /// Create a new circuit from a JSON string
    pub fn from_serial_circuit(serial_circuit: &SerialCircuit) -> Result<Self, PassError> {
        let json = serde_json::to_vec(&serial_circuit)?;
        let c_json = CString::new(json)?;
        let circuit = unsafe { ffi::tket_circuit_from_json(c_json.as_ptr()) };

        if circuit.is_null() {
            return Err(PassError::JsonError(
                "Failed to create Circuit from JSON".to_string(),
            ));
        }

        Ok(Tket1Circuit { inner: circuit })
    }

    /// Convert the circuit to a JSON string
    pub fn to_serial_circuit(&self) -> Result<SerialCircuit, PassError> {
        let mut json_ptr: *mut c_char = ptr::null_mut();

        let result = unsafe { ffi::tket_circuit_to_json(self.inner, &mut json_ptr) };

        if result != ffi::TketError_TKET_SUCCESS {
            return Err(result.into());
        }

        if json_ptr.is_null() {
            return Err(PassError::NullPointer);
        }

        let c_str = unsafe { CStr::from_ptr(json_ptr) };
        let serial_circuit = serde_json::from_str(c_str.to_string_lossy().as_ref())?;

        unsafe { ffi::tket_free_string(json_ptr) };

        Ok(serial_circuit)
    }

    /// Apply two-qubit squash transform to the circuit
    ///
    /// Squash sequences of two-qubit operations into minimal form using KAK
    /// decomposition. Can decompose to TK2 or CX gates.
    pub fn two_qubit_squash(
        &mut self,
        target_gate: impl Into<OpType>,
        cx_fidelity: f64,
        allow_swaps: bool,
    ) -> Result<(), PassError> {
        let target_gate = try_from_optype(target_gate.into())?;
        let result = unsafe {
            ffi::tket_two_qubit_squash(self.inner, target_gate, cx_fidelity, allow_swaps)
        };

        if result != ffi::TketError_TKET_SUCCESS {
            return Err(result.into());
        }

        Ok(())
    }

    /// Apply clifford resynthesis transform to the circuit
    ///
    /// Resynthesise all Clifford subcircuits and simplify using Clifford rules.
    /// This can significantly reduce the two-qubit gate count for Clifford-heavy
    /// circuits.
    pub fn clifford_resynthesis(&mut self, allow_swaps: bool) -> Result<(), PassError> {
        let result = unsafe { ffi::tket_clifford_resynthesis(self.inner, allow_swaps) };

        if result != ffi::TketError_TKET_SUCCESS {
            return Err(result.into());
        }

        Ok(())
    }
}

impl Drop for Tket1Circuit {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe { ffi::tket_circuit_destroy(self.inner) };
        }
    }
}

impl std::fmt::Debug for Tket1Circuit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Circuit")
            .field("inner", &self.inner)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    const CIRC_STR: &'static str = include_str!("../../../test_files/2cx.json");

    #[fixture]
    fn circuit() -> SerialCircuit {
        serde_json::from_str(CIRC_STR).unwrap()
    }

    #[rstest]
    fn test_circuit_creation(circuit: SerialCircuit) {
        let circuit_ptr = Tket1Circuit::from_serial_circuit(&circuit).unwrap();
        let serial_circuit = circuit_ptr.to_serial_circuit().unwrap();
        assert_eq!(circuit, serial_circuit);
    }

    #[rstest]
    fn test_two_qubit_squash(circuit: SerialCircuit) {
        assert_eq!(circuit.commands.len(), 2);
        let mut circuit_ptr = Tket1Circuit::from_serial_circuit(&circuit).unwrap();
        circuit_ptr.two_qubit_squash(OpType::CX, 1., true).unwrap();
        let serial_circuit = circuit_ptr.to_serial_circuit().unwrap();
        assert_eq!(serial_circuit.commands.len(), 0);
    }

    #[rstest]
    fn test_clifford_resynthesis(circuit: SerialCircuit) {
        assert_eq!(circuit.commands.len(), 2);
        let mut circuit_ptr = Tket1Circuit::from_serial_circuit(&circuit).unwrap();
        circuit_ptr.clifford_resynthesis(true).unwrap();
        let serial_circuit = circuit_ptr.to_serial_circuit().unwrap();
        assert_eq!(serial_circuit.commands.len(), 0);
    }

    #[test]
    fn test_target_gate_conversion() {
        assert_eq!(
            ffi::TketTargetGate_TKET_TARGET_CX,
            try_from_optype(OpType::CX).unwrap()
        );
        assert_eq!(
            ffi::TketTargetGate_TKET_TARGET_TK2,
            try_from_optype(OpType::TK2).unwrap()
        );
    }
}
