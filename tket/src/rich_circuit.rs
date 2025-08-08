pub mod control;
pub mod dagger;
pub mod flatten_modifier;
pub mod power;
use std::sync::{Arc, Weak};

use crate::{
    extension::{TKET2_EXTENSION, TKET2_EXTENSION_ID as EXTENSION_ID},
    rich_circuit::{control::ModifierControl, dagger::ModifierDagger, power::ModifierPower},
};
use hugr::{
    extension::{
        prelude::qb_t,
        simple_op::{MakeExtensionOp, MakeOpDef, MakeRegisteredOp, OpLoadError},
        ExtensionId, OpDef, SignatureError, SignatureFromArgs, SignatureFunc, TypeDef, Version,
    },
    ops::{ExtensionOp, OpName, OpType},
    types::{
        type_param::TypeParam, FuncValueType, PolyFuncTypeRV, Signature, Type, TypeArg, TypeBound,
        TypeRV,
    },
    Extension, Hugr, HugrView,
};
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use smol_str::ToSmolStr;
use strum::{EnumIter, EnumString, IntoStaticStr};

/// A circuit that possibly controlled by modifiers.
///
use crate::circuit::Circuit;

pub const MODIFIER_EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("modifier");
pub const MODIFIER_VERSION: Version = Version::new(0, 1, 0);

lazy_static! {
    /// The extension definition for TKET2 rotation type and ops.
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
                    DAGGER_OP_ID,
                    "Power Operator".to_string(),
                    ModifierPower::signature(),
                    extension_ref,
                ).unwrap();
            }
    )};
}

pub const CONTROL_OP_ID: OpName = OpName::new_inline("ControlOperator");
pub const DAGGER_OP_ID: OpName = OpName::new_inline("DaggerOperator");
pub const POWER_OP_ID: OpName = OpName::new_inline("PowerOperator");

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
    Control(u32),
    Dagger,
    Power,
}

impl Modifier {}

impl MakeOpDef for Modifier {
    fn opdef_id(&self) -> OpName {
        match self {
            Modifier::Control(_) => CONTROL_OP_ID.clone(),
            Modifier::Dagger => DAGGER_OP_ID.clone(),
            Modifier::Power => POWER_OP_ID.clone(),
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
            Modifier::Control(_) => ModifierControl::signature(),
            Modifier::Dagger => ModifierDagger::signature(),
            Modifier::Power => ModifierPower::signature(),
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
            Modifier::Control(_) => "Generates a quantum-controlled circuit from a circuit.".into(),
            Modifier::Dagger => "Dagger operation on a circuit.".into(),
            Modifier::Power => "Generates a circuit that applies a circuit many times.".into(),
        }
    }

    // [TODO]: do we need this?
    // fn post_opdef(&self, _def: &mut OpDef);
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
}
