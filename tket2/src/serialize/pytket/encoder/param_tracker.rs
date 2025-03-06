//! Tracking of encoded qubit wires during pytket circuit encoding.

use std::collections::HashMap;

use hugr::core::HugrNode;
use hugr::ops::{OpTrait, OpType};
use hugr::std_extensions::arithmetic::float_types::float64_type;
use hugr::types::Type;
use hugr::{CircuitUnit, HugrView, Wire};

use crate::circuit::Circuit;
use crate::extension::rotation::rotation_type;
use crate::serialize::pytket::{OpConvertError, METADATA_INPUT_PARAMETERS};

use super::super::param::encode::fold_param_op;

/// A structure for tracking the parameters of a circuit being encoded.
#[derive(derive_more::Debug, Clone)]
pub struct ParameterTracker<N> {
    /// The parameters associated with each wire.
    parameters: HashMap<Wire<N>, String>,
}

impl<N: HugrNode> ParameterTracker<N> {
    /// Create a new [`ParameterTracker`] from the input parameters of a [`Circuit`].
    pub fn new(circ: &Circuit<impl HugrView<Node = N>>) -> Self {
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
    pub fn record_parameters<T: HugrView<Node = N>>(
        &mut self,
        node: N,
        optype: &OpType,
        circ: &Circuit<impl HugrView<Node = N>>,
    ) -> Result<bool, OpConvertError<N>> {
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
        //
        // Due to the previous checks, we know that all inputs are parameters
        // so they should be already tracked.
        let mut inputs = Vec::with_capacity(input_count);
        let missing_param_err = || {
            Err(OpConvertError::UnresolvedParamInput {
                typ: rotation_type(),
                optype: optype.clone(),
                node,
            })
        };
        for inp in circ.hugr().node_inputs(node) {
            let Some((neigh, neigh_out)) = circ.hugr().single_linked_output(node, inp) else {
                return missing_param_err();
            };
            let wire = Wire::new(neigh, neigh_out);
            let Some(param) = self.parameters.get(&wire) else {
                return missing_param_err();
            };
            inputs.push(param.as_str());
        }

        let Some(param) = fold_param_op(optype, &inputs) else {
            return Ok(false);
        };

        for out in circ.hugr().node_outputs(node) {
            let wire = Wire::new(node, out);
            self.add_parameter(wire, param.clone())
        }
        Ok(true)
    }

    /// Associate a parameter expression with a wire.
    pub fn add_parameter(&mut self, wire: Wire<N>, param: String) {
        self.parameters.insert(wire, param);
    }

    /// Returns the parameter expression for a wire, if it exists.
    pub fn get(&self, wire: Wire<N>) -> Option<&str> {
        self.parameters.get(&wire).map(|s| s.as_str())
    }
}

impl<N> Default for ParameterTracker<N> {
    fn default() -> Self {
        Self {
            parameters: Default::default(),
        }
    }
}
