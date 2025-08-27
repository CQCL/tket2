//! Module for richer circuit representation and operations.
//! This module provides three extensions: modifiers, global phase, and safe drop.
//!
//! ## Modifiers
//! Modifiers are functions that takes circuits and return modified circuits
//! by applying modifiers: control, dagger, or power.
//!
//! ## Global Phase
//! Global phase is an operation that represents the global phase of a circuit.
//! It is implemented as a side-effect that takes a rotation angle as an input.
//!
//! ## Safe Drop
//! A vefiried safe_drop function that deallocates
//! a qubit from a circuit, which must be at the initial state |0>.
pub mod control;
pub mod dagger;
pub mod modifier_resolver;
pub mod power;
use std::{
    str::FromStr,
    sync::{Arc, Weak},
};

use crate::{
    extension::rotation::rotation_type,
    rich_circuit::{control::ModifierControl, dagger::ModifierDagger, power::ModifierPower},
};
use hugr::{
    extension::{
        prelude::qb_t,
        simple_op::{MakeOpDef, MakeRegisteredOp, OpLoadError},
        ExtensionId, OpDef, SignatureFunc, Version,
    },
    ops::{ExtensionOp, OpName},
    type_row,
    types::Signature,
    Extension,
};
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use strum::{EnumIter, EnumString, IntoStaticStr};

/// Types of modifers.
#[derive(
    Clone,
    Copy,
    Debug,
    Serialize,
    Deserialize,
    Hash,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    EnumIter,
    IntoStaticStr,
    EnumString,
)]
pub enum Modifier {
    /// Control modifier.
    ControlModifier,
    /// Dagger modifier.
    DaggerModifier,
    /// Power modifier.
    PowerModifier,
}

/// The extension ID for the modifier extension.
pub const MODIFIER_EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("modifier");
/// The version of the modifier extension.
pub const MODIFIER_VERSION: Version = Version::new(0, 1, 0);

lazy_static! {
    /// The extension definition for modifier operations.
    pub static ref MODIFIER_EXTENSION: Arc<Extension> =  {
            Extension::new_arc(MODIFIER_EXTENSION_ID, MODIFIER_VERSION, |modifier, extension_ref| {
                modifier.add_op(
                    CONTROL_OP_ID,
                    "Quantum control operation".to_string(),
                    ModifierControl::signature(),
                    extension_ref,
                ).unwrap();

                modifier.add_op(
                    DAGGER_OP_ID,
                    "Dagger Operator".to_string(),
                    ModifierDagger::signature(),
                    extension_ref,
                ).unwrap();

                modifier.add_op(
                    POWER_OP_ID,
                    "Power Operator".to_string(),
                    ModifierPower::signature(),
                    extension_ref,
                ).unwrap();
            }
    )};
}

#[allow(missing_docs)]
pub const CONTROL_OP_ID: OpName = OpName::new_inline("ControlModifier");
#[allow(missing_docs)]
pub const DAGGER_OP_ID: OpName = OpName::new_inline("DaggerModifier");
#[allow(missing_docs)]
pub const POWER_OP_ID: OpName = OpName::new_inline("PowerModifier");

impl MakeOpDef for Modifier {
    fn opdef_id(&self) -> OpName {
        match self {
            Modifier::ControlModifier => CONTROL_OP_ID.clone(),
            Modifier::DaggerModifier => DAGGER_OP_ID.clone(),
            Modifier::PowerModifier => POWER_OP_ID.clone(),
        }
    }

    fn from_def(op_def: &OpDef) -> Result<Self, OpLoadError>
    where
        Self: Sized,
    {
        // [WIP] I still don't understand this.
        hugr::extension::simple_op::try_from_name(op_def.name(), op_def.extension_id())
    }

    fn init_signature(&self, _extension_ref: &std::sync::Weak<hugr::Extension>) -> SignatureFunc {
        match self {
            Modifier::ControlModifier => ModifierControl::signature(),
            Modifier::DaggerModifier => ModifierDagger::signature(),
            Modifier::PowerModifier => ModifierPower::signature(),
        }
    }

    fn extension_ref(&self) -> Weak<hugr::Extension> {
        Arc::downgrade(&MODIFIER_EXTENSION)
    }

    fn extension(&self) -> ExtensionId {
        MODIFIER_EXTENSION_ID.to_owned()
    }

    fn description(&self) -> String {
        match self {
            Modifier::ControlModifier => {
                "Generates a quantum-controlled circuit from a circuit.".into()
            }
            Modifier::DaggerModifier => "Dagger operation on a circuit.".into(),
            Modifier::PowerModifier => {
                "Generates a circuit that applies a circuit many times.".into()
            }
        }
    }

    // [TODO]: Do we need this?
    // fn post_opdef(&self, _def: &mut OpDef);
}

#[allow(missing_docs)]
pub const GLOBAL_PHASE_EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("global_phase_ext");
#[allow(missing_docs)]
pub const GLOBAL_PHASE_VERSION: Version = Version::new(0, 1, 0);

lazy_static! {
    /// The extension definition for
    pub static ref GLOBAL_PHASE_EXTENSION: Arc<Extension> =  {
            Extension::new_arc(GLOBAL_PHASE_EXTENSION_ID, GLOBAL_PHASE_VERSION, |op, extension_ref| {
                op.add_op(
                    GLOBAL_PHASE_OP_ID.clone(),
                    "Global phase of a circuit".to_string(),
                    GlobalPhase::signature(),
                    extension_ref,
                ).unwrap();
            }
    )};
}

#[allow(missing_docs)]
pub static GLOBAL_PHASE_OP_ID: OpName = OpName::new_inline("global_phase");

/// Global phase of a circuit.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct GlobalPhase;

impl GlobalPhase {
    /// Wraps the operation in an [`ExtensionOp`]
    pub fn into_extension_op(self) -> ExtensionOp {
        <Self as MakeRegisteredOp>::to_extension_op(self)
            .expect("Failed to convert to extension op.")
    }

    fn signature() -> Signature {
        Signature::new(vec![rotation_type()], type_row![])
    }
}

impl FromStr for GlobalPhase {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == Self.opdef_id() {
            Ok(Self)
        } else {
            Err(())
        }
    }
}

impl MakeOpDef for GlobalPhase {
    fn opdef_id(&self) -> OpName {
        GLOBAL_PHASE_OP_ID.clone()
    }

    fn from_def(op_def: &OpDef) -> Result<Self, OpLoadError>
    where
        Self: Sized,
    {
        hugr::extension::simple_op::try_from_name(op_def.name(), op_def.extension_id())
    }

    fn init_signature(&self, _extension_ref: &Weak<Extension>) -> SignatureFunc {
        SignatureFunc::from(GlobalPhase::signature())
    }

    fn extension_ref(&self) -> Weak<Extension> {
        Arc::downgrade(&GLOBAL_PHASE_EXTENSION)
    }

    fn extension(&self) -> ExtensionId {
        GLOBAL_PHASE_EXTENSION_ID.to_owned()
    }

    fn description(&self) -> String {
        "Global phase of a circuit.".into()
    }

    // TODO: Do we need this?
    // fn post_opdef(&self, def: &mut OpDef) {
}

impl MakeRegisteredOp for GlobalPhase {
    fn extension_id(&self) -> ExtensionId {
        GLOBAL_PHASE_EXTENSION_ID.to_owned()
    }

    fn extension_ref(&self) -> Weak<hugr::Extension> {
        Arc::<hugr::Extension>::downgrade(&GLOBAL_PHASE_EXTENSION)
    }
}

/// Safe drop function for qubits.
/// This function deallocates a qubit from a circuit, which needs to be verified to be at the initial state.
#[derive(
    Clone,
    Copy,
    Debug,
    Serialize,
    Deserialize,
    Hash,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    EnumIter,
    IntoStaticStr,
    EnumString,
)]
pub enum SafeDrop {
    /// Verified drop operation.
    VerifiedZero,
    // FUTURE: automatic uncomputation.
}

/// The extension ID for the verified extension.
pub const SAFEDROP_EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("safe_drop");
/// The version of the modifier extension.
pub const SAFEDROP_VERSION: Version = Version::new(0, 1, 0);

lazy_static! {
    /// The extension definition for TKET2 rotation type and ops.
    pub static ref SAFEDROP_EXTENSION: Arc<Extension> =  {
        Extension::new_arc(SAFEDROP_EXTENSION_ID, SAFEDROP_VERSION, |safe_drop, extension_ref| {
            safe_drop.add_op(
                VERIFIED_ZERO_OP_ID,
                "Reset qubit which is guaranteed to be |0>".to_string(),
                SafeDrop::VerifiedZero.signature(),
                extension_ref,
            ).unwrap();
        }
    )};
}

#[allow(missing_docs)]
pub const VERIFIED_ZERO_OP_ID: OpName = OpName::new_inline("VerifiedZero");

impl MakeOpDef for SafeDrop {
    fn opdef_id(&self) -> OpName {
        VERIFIED_ZERO_OP_ID.clone()
    }

    fn from_def(op_def: &OpDef) -> Result<Self, OpLoadError>
    where
        Self: Sized,
    {
        hugr::extension::simple_op::try_from_name(op_def.name(), op_def.extension_id())
    }

    fn init_signature(&self, _extension_ref: &std::sync::Weak<hugr::Extension>) -> SignatureFunc {
        Self::signature(self)
    }

    fn extension_ref(&self) -> Weak<hugr::Extension> {
        Arc::downgrade(&SAFEDROP_EXTENSION)
    }

    fn extension(&self) -> ExtensionId {
        SAFEDROP_EXTENSION_ID.to_owned()
    }

    fn description(&self) -> String {
        "Safe drop operation for qubits.".into()
    }
}

impl SafeDrop {
    /// Signature for the safe drop operation.
    pub fn signature(&self) -> SignatureFunc {
        Signature::new(vec![qb_t()], type_row![]).into()
    }
}

#[cfg(test)]
mod test {
    use hugr::{
        builder::{Dataflow, DataflowSubContainer, HugrBuilder, ModuleBuilder},
        extension::prelude::{bool_t, qb_t},
        ops::CallIndirect,
        type_row,
        types::{Signature, Term},
        HugrView,
    };
    use serde::Serialize;

    use crate::rich_circuit::{CONTROL_OP_ID, MODIFIER_EXTENSION};

    #[test]
    fn test_control_op() {
        let mut module = ModuleBuilder::new();
        let decl = module
            .declare("dummy_decl", Signature::new(bool_t(), type_row![]).into())
            .unwrap();
        let mut func = module
            .define_function(
                "dummy_function",
                Signature::new(vec![qb_t(), bool_t()], vec![qb_t()]),
            )
            .unwrap();
        let [fin1, fin2] = func.input_wires_arr();

        let control_op = MODIFIER_EXTENSION
            .instantiate_extension_op(
                &CONTROL_OP_ID,
                [Term::new_list([bool_t().into()]), Term::new_list([])],
            )
            .unwrap();

        let loaded_func = func.load_func(&decl, &[]).unwrap();
        let controlled = func
            .add_dataflow_op(control_op.clone(), vec![loaded_func])
            .unwrap()
            .out_wire(0);
        let ret = func
            .add_dataflow_op(
                CallIndirect {
                    signature: Signature::new(vec![qb_t(), bool_t()], vec![qb_t()]),
                },
                vec![controlled, fin1, fin2],
            )
            .unwrap()
            .out_wire(0);

        func.finish_with_outputs(vec![ret]).unwrap();

        let h = module.finish_hugr().unwrap();
        println!("{}", h.mermaid_string());
    }

    #[test]
    fn test_gen_json_modifier() {
        let ext = &MODIFIER_EXTENSION;
        // open file "modifier.json"
        let writer = std::fs::File::create("modifier.json").unwrap();
        ext.to_owned()
            .serialize(&mut serde_json::Serializer::pretty(writer))
            .unwrap();
    }
}
