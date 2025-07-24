pub mod control;
pub mod dagger;
pub mod modifier_call_graph;
pub mod modifier_resolver;
pub mod power;
use std::collections::HashMap;
use std::sync::{Arc, Weak};

use crate::circuit::Circuit;
use crate::rich_circuit::{control::ModifierControl, dagger::ModifierDagger, power::ModifierPower};
use hugr::Node;
use hugr::{
    extension::{
        simple_op::{MakeOpDef, OpLoadError},
        ExtensionId, OpDef, SignatureFunc, Version,
    },
    ops::OpName,
    Extension,
};
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use strum::{EnumIter, EnumString, IntoStaticStr};

/// Circuit with hash map of modifier informationj
#[derive(Debug, Clone)]
struct RichCircuit {
    circ: Circuit,
    modifier_map: HashMap<Node, Vec<Modifier>>,
}

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
    Control(usize),
    Dagger,
    Power(usize),
}

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

impl MakeOpDef for Modifier {
    fn opdef_id(&self) -> OpName {
        match self {
            Modifier::Control(_) => CONTROL_OP_ID.clone(),
            Modifier::Dagger => DAGGER_OP_ID.clone(),
            Modifier::Power(_) => POWER_OP_ID.clone(),
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
            Modifier::Power(_) => ModifierPower::signature(),
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
            Modifier::Power(_) => "Generates a circuit that applies a circuit many times.".into(),
        }
    }

    // [TODO]: do we need this?
    // fn post_opdef(&self, _def: &mut OpDef);
}

pub struct CombinedModifier {
    control: usize,
    dagger: bool,
    power: usize,
}

impl From<Vec<Modifier>> for CombinedModifier {
    fn from(modifiers: Vec<Modifier>) -> Self {
        let mut control = 0;
        let mut dagger = false;
        let mut power = 0;

        for modifier in modifiers {
            match modifier {
                Modifier::Control(c) => control += c as usize,
                Modifier::Dagger => dagger = true,
                Modifier::Power(p) => power += p as usize,
            }
        }

        CombinedModifier {
            control,
            dagger,
            power,
        }
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
