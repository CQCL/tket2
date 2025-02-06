//! Tracking of encoded qubit wires during pytket circuit encoding.

use std::collections::HashMap;

use hugr::ops::{OpTrait, OpType};
use hugr::std_extensions::arithmetic::float_types::float64_type;
use hugr::types::Type;
use hugr::{CircuitUnit, HugrView, Wire};

use crate::circuit::{Circuit, Command};
use crate::extension::rotation::rotation_type;
use crate::serialize::pytket::{OpConvertError, METADATA_INPUT_PARAMETERS};

use super::super::param::encode::fold_param_op;

/// A structure for tracking the parameters of a circuit being encoded.
#[derive(Debug, Clone, Default)]
pub struct ParameterTracker {
    /// The parameters associated with each wire.
    parameters: HashMap<Wire, String>,
}

impl ParameterTracker {
    /// Create a new [`ParameterTracker`] from the input parameters of a [`Circuit`].
    pub fn new(circ: &Circuit<impl HugrView>) -> Self {
        let mut tracker = ParameterTracker::default();

        let angle_input_wires = circ.units().filter_map(|u| match u {
            (CircuitUnit::Wire(w), _, ty) if [rotation_type(), float64_type()].contains(&ty) => {
                Some(w)
            }
            _ => None,
        });

        // The input parameter names may be specified in the metadata.
        let fixed_input_names: Vec<String> = circ
            .hugr()
            .get_metadata(circ.parent(), METADATA_INPUT_PARAMETERS)
            .and_then(|params| serde_json::from_value(params.clone()).ok())
            .unwrap_or_default();
        let extra_names = (fixed_input_names.len()..).map(|i| format!("f{i}"));
        let mut param_name = fixed_input_names.into_iter().chain(extra_names);

        for wire in angle_input_wires {
            tracker.add_parameter(wire, param_name.next().unwrap());
        }

        tracker
    }

    /// Record any output of the command that can be used as a TKET1 parameter.
    /// Returns whether parameters were recorded.
    /// Associates the output wires with the parameter expression.
    pub fn record_parameters<T: HugrView>(
        &mut self,
        command: &Command<'_, T>,
        optype: &OpType,
    ) -> Result<bool, OpConvertError> {
        let input_count = if let Some(signature) = optype.dataflow_signature() {
            // Only consider commands where all inputs and some outputs are
            // parameters that we can track.
            let tracked_params: [Type; 2] = [rotation_type(), float64_type()];
            let all_inputs = signature
                .input()
                .iter()
                .all(|ty| tracked_params.contains(ty));
            let some_output = signature
                .output()
                .iter()
                .any(|ty| tracked_params.contains(ty));
            if !all_inputs || !some_output {
                return Ok(false);
            }
            signature.input_count()
        } else if let OpType::Const(_) = optype {
            // `Const` is a special non-dataflow command we can handle.
            // It has zero inputs.
            0
        } else {
            // Not a parameter-generating command.
            return Ok(false);
        };

        // Collect the input parameters.
        let mut inputs = Vec::with_capacity(input_count);
        for (unit, _, _) in command.inputs() {
            let CircuitUnit::Wire(wire) = unit else {
                panic!("Angle types are not linear")
            };
            let Some(param) = self.parameters.get(&wire) else {
                let typ = rotation_type();
                return Err(OpConvertError::UnresolvedParamInput {
                    typ,
                    optype: optype.clone(),
                    node: command.node(),
                });
            };
            inputs.push(param.as_str());
        }

        let Some(param) = fold_param_op(optype, &inputs) else {
            return Ok(false);
        };

        for (unit, _, _) in command.outputs() {
            if let CircuitUnit::Wire(wire) = unit {
                self.add_parameter(wire, param.clone())
            }
        }
        Ok(true)
    }

    /// Associate a parameter expression with a wire.
    pub fn add_parameter(&mut self, wire: Wire, param: String) {
        self.parameters.insert(wire, param);
    }

    /// Returns the parameter expression for a wire, if it exists.
    pub fn get(&self, wire: &Wire) -> Option<&String> {
        self.parameters.get(wire)
    }
}
