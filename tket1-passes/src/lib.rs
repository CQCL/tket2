//! Rust interface for tket1-passes library
//!
//! This crate provides safe Rust bindings to the `libtket1-passes` C library
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

/// Error types that can occur when using the tket1-passes library
#[derive(Error, Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PassError {
    /// An error happened in TKET1
    #[error("TKET1 error: {0}")]
    Tket1Error(String),
    /// Expected TKET1-allocated data, got NULL pointer
    #[error("Expected TKET1-allocated data, got NULL pointer")]
    NullPointer,
    /// Failed to process JSON
    #[error("Failed to process JSON: {0}")]
    JsonError(String),
    /// Target gate type is not supported
    #[error("Invalid target gate: {0:?}. Supported gates are CX and TK2.")]
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

impl TryFrom<ffi::TketError> for PassError {
    type Error = &'static str;

    fn try_from(value: ffi::TketError) -> Result<Self, Self::Error> {
        if value == ffi::TketError_TKET_SUCCESS {
            return Err("No error occurred");
        }
        let c_str = unsafe { CStr::from_ptr(ffi::tket_error_string(value)) };
        let s = c_str.to_string_lossy().to_string();
        Ok(PassError::Tket1Error(s))
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
#[derive(Debug)]
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

        let error_code = unsafe { ffi::tket_circuit_to_json(self.inner, &mut json_ptr) };

        if let Ok(pass_error) = error_code.try_into() {
            return Err(pass_error);
        }

        if json_ptr.is_null() {
            return Err(PassError::NullPointer);
        }

        let c_str = unsafe { CStr::from_ptr(json_ptr) };
        let serial_circuit = serde_json::from_str(c_str.to_string_lossy().as_ref())?;

        unsafe { ffi::tket_free_string(json_ptr) };

        Ok(serial_circuit)
    }

    /// Apply TKET1's two_qubit_squash transform to the circuit
    pub fn two_qubit_squash(
        &mut self,
        target_gate: impl Into<OpType>,
        cx_fidelity: f64,
        allow_swaps: bool,
    ) -> Result<(), PassError> {
        let target_gate = try_from_optype(target_gate.into())?;
        let error_code = unsafe {
            ffi::tket_two_qubit_squash(self.inner, target_gate, cx_fidelity, allow_swaps)
        };

        if let Ok(pass_error) = error_code.try_into() {
            return Err(pass_error);
        }

        Ok(())
    }

    /// Apply TKET1's clifford_simp transform to the circuit
    pub fn clifford_simp(
        &mut self,
        target_gate: impl Into<OpType>,
        allow_swaps: bool,
    ) -> Result<(), PassError> {
        let target_gate = try_from_optype(target_gate.into())?;
        let error_code = unsafe { ffi::tket_clifford_simp(self.inner, target_gate, allow_swaps) };

        if let Ok(pass_error) = error_code.try_into() {
            return Err(pass_error);
        }

        Ok(())
    }

    /// Apply TKET1's squash_phasedx_rz transform to the circuit
    pub fn squash_phasedx_rz(&mut self) -> Result<(), PassError> {
        let error_code = unsafe { ffi::tket_squash_phasedx_rz(self.inner) };

        if let Ok(pass_error) = error_code.try_into() {
            return Err(pass_error);
        }

        Ok(())
    }
}

impl Drop for Tket1Circuit {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe { ffi::tket_free_circuit(self.inner) };
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;
    use std::sync::Mutex;

    use super::*;
    use rstest::*;

    const CIRC_STR: &str = include_str!("../../test_files/2cx.json");

    // Mutex to ensure tests don't run in parallel, due to TKET1 bug, see
    // https://github.com/CQCL/tket/issues/2009
    static TEST_MUTEX: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

    #[fixture]
    fn circuit() -> SerialCircuit {
        serde_json::from_str(CIRC_STR).unwrap()
    }

    #[rstest]
    fn test_circuit_creation(circuit: SerialCircuit) {
        let _lock = TEST_MUTEX.lock().unwrap();
        let circuit_ptr = Tket1Circuit::from_serial_circuit(&circuit).unwrap();
        let serial_circuit = circuit_ptr.to_serial_circuit().unwrap();
        assert_eq!(circuit, serial_circuit);
    }

    #[rstest]
    fn test_two_qubit_squash(circuit: SerialCircuit) {
        let _lock = TEST_MUTEX.lock().unwrap();
        assert_eq!(circuit.commands.len(), 2);
        let mut circuit_ptr = Tket1Circuit::from_serial_circuit(&circuit).unwrap();
        circuit_ptr.two_qubit_squash(OpType::CX, 1., true).unwrap();
        let serial_circuit = circuit_ptr.to_serial_circuit().unwrap();
        assert_eq!(serial_circuit.commands.len(), 0);
    }

    #[rstest]
    fn test_clifford_simp(circuit: SerialCircuit) {
        let _lock = TEST_MUTEX.lock().unwrap();
        assert_eq!(circuit.commands.len(), 2);
        let mut circuit_ptr = Tket1Circuit::from_serial_circuit(&circuit).unwrap();
        circuit_ptr.clifford_simp(OpType::CX, true).unwrap();
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

    #[rstest]
    fn test_error_handling(circuit: SerialCircuit) {
        let mut circuit_ptr = Tket1Circuit::from_serial_circuit(&circuit).unwrap();
        assert_eq!(
            circuit_ptr.clifford_simp(OpType::CZ, true).unwrap_err(),
            PassError::InvalidTargetGate(OpType::CZ)
        );
        assert_eq!(
            circuit_ptr
                .two_qubit_squash(OpType::CZ, 1., true)
                .unwrap_err(),
            PassError::InvalidTargetGate(OpType::CZ)
        );
    }

    #[test]
    fn test_error_null_circ() {
        let mut null_circ = Tket1Circuit {
            inner: ptr::null_mut(),
        };
        assert_eq!(
            null_circ.clifford_simp(OpType::CX, true).unwrap_err(),
            PassError::Tket1Error("Invalid NULL pointer in arguments".to_string())
        );
        assert_eq!(
            null_circ
                .two_qubit_squash(OpType::CX, 1., true)
                .unwrap_err(),
            PassError::Tket1Error("Invalid NULL pointer in arguments".to_string())
        );
    }

    #[test]
    fn test_error_handling_from_json() {
        // badly formatted JSON
        let circuit_json = r#"{"bits": [], "commands": [{"args": [["q", [0]], ["q", [1]]], "op": {"type": "CX"}}, {"args": [["q", [0]], ["q", [1]]], "op": {"type": "CX"}}], "created_qubits": [], "discarded_qubits": [], "implicit_permutation": [[["q", [0]], ["q", [0]]], [["q", [1]], ["q", [1]]]], "phase": "0.0", "qubits": [["q", [0]] ["q", [1]]]}"#;
        let c_str = CString::new(circuit_json).unwrap();
        let circ_ptr = unsafe { ffi::tket_circuit_from_json(c_str.as_ptr()) };
        assert!(circ_ptr.is_null());
    }
}
