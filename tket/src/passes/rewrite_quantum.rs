#![allow(missing_docs)]

use itertools::Itertools as _;
use std::collections::BTreeMap;

use hugr::{
    algorithms::{
        replace_types::{NodeTemplate, ReplaceTypesError},
        ComposablePass, ReplaceTypes,
    },
    builder::{
        BuildError, Container, Dataflow,
        DataflowSubContainer, HugrBuilder, ModuleBuilder,
    },
    extension::{
        prelude::{bool_t, qb_t},
        SignatureError,
    },
    hugr::{hugrmut::HugrMut, ValidationError},
    ops::{handle::NodeHandle as _, Call, DataflowOpTrait, OpType},
    std_extensions::arithmetic::int_types::INT_TYPES,
    types::{PolyFuncType, Type, TypeRV},
    HugrView, Node,
};
use strum::IntoEnumIterator as _;

use crate::{
    extension::bool::{bool_type, BoolOpBuilder},
    TketOp,
};

#[derive(derive_more::Error, Debug, derive_more::Display, derive_more::From)]
#[non_exhaustive]
pub enum RewriteQuantumPassError {
    #[from]
    ReplaceTypesError(ReplaceTypesError),
    #[from]
    ValidationError(ValidationError<Node>),
    #[from(SignatureError, BuildError)]
    BuildError(BuildError),
    #[display("Existing function '{name}' node {node} does not have the expected signature for op '{}'. Expected: {expected}. Found {found}", <&'static str>::from(op))]
    ExistingFunctionSignatureMismatch {
        op: TketOp,
        node: Node,
        name: String,
        expected: PolyFuncType,
        found: PolyFuncType,
    },
}

#[derive(Debug, Clone)]
pub struct RewriteQuantumPass {
    qubit_to_ty: Type,
    funcs: BTreeMap<TketOp, String>,
}

impl Default for RewriteQuantumPass {
    fn default() -> Self {
        let int: TypeRV = INT_TYPES[6].clone().into();
        Self {
            qubit_to_ty: Type::new_tuple(vec![int.clone(), int]),
            funcs: TketOp::iter()
                .map(|op| {
                    let name: &'static str = op.into();
                    (op, format!("_{name}"))
                })
                .collect(),
        }
    }
}

impl<H: HugrMut<Node = Node>> ComposablePass<H> for RewriteQuantumPass {
    type Error = RewriteQuantumPassError;
    type Result = bool;

    fn run(&self, hugr: &mut H) -> Result<Self::Result, Self::Error> {
        let r = self.lowerer(self.find_or_create_funcs(hugr)?)?.run(hugr)?;
        eprintln!("{}", hugr.mermaid_string());
        hugr.validate()?;
        Ok(r)
    }
}

fn all_funcs_by_name<N: Clone>(
    hugr: &impl HugrView<Node = N>,
) -> BTreeMap<String, (N, PolyFuncType)> {
    let mod_node = hugr.module_root();
    hugr.children(mod_node)
        .filter_map(|n| {
            Some(match hugr.get_optype(n.clone()) {
                OpType::FuncDecl(decl) => (decl.func_name(), decl.signature()),
                OpType::FuncDefn(defn) => (defn.func_name(), defn.signature()),
                _ => None?,
            })
            .map(|(name, sig)| (name.clone(), (n.clone(), sig.clone())))
        })
        .collect()
}

impl RewriteQuantumPass {
    fn find_or_create_funcs(
        &self,
        hugr: &mut impl HugrMut<Node = Node>,
    ) -> Result<BTreeMap<TketOp, Node>, RewriteQuantumPassError> {
        let mod_node = hugr.module_root();
        let mut func_nodes = BTreeMap::default();
        let funcs_by_name = all_funcs_by_name(hugr);
        for (&op, name) in &self.funcs {
            let ext_op = op.into_extension_op();
            let expected_sig = {
                // this is kind of gross, ReplaceTypes already does this we
                // should be able to call into it here.
                let mut sig = ext_op.signature().into_owned();
                sig.input
                    .iter_mut()
                    .chain(sig.output.iter_mut())
                    .for_each(|ty| {
                        if ty == &qb_t() {
                            *ty = self.qubit_to_ty.clone();
                        } else if ty == &bool_t() {
                            *ty = bool_type()
                        }
                    });
                sig
            };
            let node = {
                let func_node = {
                    let expected_sig = expected_sig.clone().into();
                    if let Some((n, sig)) = funcs_by_name.get(name) {
                        let node = *n;
                        if sig != &expected_sig {
                            Err(RewriteQuantumPassError::ExistingFunctionSignatureMismatch {
                                op,
                                node,
                                name: name.clone(),
                                expected: expected_sig,
                                found: sig.clone(),
                            })?
                        }
                        *n
                    } else {
                        let decl_hugr = {
                            let mut builder = ModuleBuilder::new();
                            let id = builder.declare(name, expected_sig)?;
                            let mut hugr = builder.finish_hugr()?;
                            hugr.set_entrypoint(id.node());
                            hugr
                        };
                        let ir = hugr.insert_hugr(mod_node, decl_hugr);
                        ir.inserted_entrypoint
                    }
                };
                match op {
                    // TketOps return hugr bools, but guppy functions return
                    // opaque tket2 bools. Gross.  We insert wrapper functions
                    // for these two ops that read the result.
                    TketOp::Measure => {
                        let wrapper_sig = {
                            let mut sig = expected_sig.clone();
                            sig.output.iter_mut().for_each(|ty| {
                                if ty == &bool_type() {
                                    *ty = bool_t()
                                }
                            });
                            sig
                        };
                        let (decl_hugr, (call_node, call_port)) = {
                            let mut builder = ModuleBuilder::new();
                            let dummy = builder.declare(name, expected_sig.clone().into())?;
                            let (wrapper, call) = {
                                let mut wrapper = builder.define_function(
                                    format!("_Wrapped_{name}"),
                                    wrapper_sig.clone(),
                                )?;
                                let call = Call::try_new(expected_sig.clone().into(), [])?;
                                let port = call.called_function_port();
                                let call = wrapper.add_dataflow_op(call, wrapper.input_wires())?;
                                wrapper
                                    .hugr_mut()
                                    .connect(dummy.node(), 0, call.node(), port);
                                let call_outs: Vec<_> =
                                    itertools::zip_eq(wrapper_sig.output().iter(), call.outputs())
                                        .map(|(ty, wire)| {
                                            if ty == &bool_t() {
                                                Ok::<_, BuildError>(wrapper.add_bool_read(wire)?[0])
                                            } else {
                                                Ok(wire)
                                            }
                                        })
                                        .try_collect()?;
                                (
                                    wrapper.finish_with_outputs(call_outs)?.node(),
                                    (call.node(), port),
                                )
                            };

                            let mut hugr = builder.finish_hugr()?;
                            hugr.set_entrypoint(wrapper);
                            (hugr, call)
                        };
                        let ir = hugr.insert_hugr(mod_node, decl_hugr);
                        hugr.disconnect(ir.node_map[&call_node], call_port);
                        hugr.connect(func_node, 0, ir.node_map[&call_node], call_port);
                        ir.inserted_entrypoint
                    }
                    _ => func_node,
                }
            };
            let _ = func_nodes.insert(op, node);
        }
        Ok(func_nodes)
    }

    pub fn lowerer(
        &self,
        funcs: BTreeMap<TketOp, Node>,
    ) -> Result<ReplaceTypes, RewriteQuantumPassError> {
        let mut lw = ReplaceTypes::default();
        lw.replace_type(
            qb_t().as_extension().cloned().unwrap(),
            self.qubit_to_ty.clone(),
        );
        for (op, node) in funcs {
            lw.replace_op(&op.into_extension_op(), NodeTemplate::Call(node, vec![]));
        }
        Ok(lw)
    }
}

#[cfg(test)]
mod test {
    use hugr::{
        algorithms::call_graph::CallGraph,
        builder::{Dataflow, DataflowSubContainer as _},
        type_row,
        types::Signature,
    };
    use petgraph::Direction;

    use crate::extension::{bool::ConstBool, TKET_EXTENSION_ID};

    use super::*;

    #[test]
    fn example() {
        let pass = RewriteQuantumPass::default();
        let mut module = ModuleBuilder::new();
        let h_node = {
            let t = &pass.qubit_to_ty;
            module
                .declare("_H", Signature::new_endo(t.clone()).into())
                .unwrap()
                .node()
        };
        let measure_node = {
            let t = &pass.qubit_to_ty;
            let mut measure = module
                .define_function(
                    "_Measure",
                    Signature::new(t.clone(), vec![t.clone(), bool_type()]),
                )
                .unwrap();
            let [q] = measure.input_wires_arr();
            let r = measure.add_load_value(ConstBool::new(true));
            measure.finish_with_outputs([q, r]).unwrap().node()
        };

        let main = {
            let mut main = module
                .define_function("main", Signature::new(type_row![], bool_t()))
                .unwrap();

            let qb1 = main
                .add_dataflow_op(TketOp::QAlloc, [])
                .unwrap()
                .out_wire(0);
            let qb2 = main
                .add_dataflow_op(TketOp::QAlloc, [])
                .unwrap()
                .out_wire(0);

            let qb1 = main.add_dataflow_op(TketOp::H, [qb1]).unwrap().out_wire(0);

            let [qb1, qb2] = main
                .add_dataflow_op(TketOp::CX, [qb1, qb2])
                .unwrap()
                .outputs_arr();

            let [qb1, r] = main
                .add_dataflow_op(TketOp::Measure, [qb1])
                .unwrap()
                .outputs_arr();

            let [qb1] = main
                .add_dataflow_op(TketOp::Reset, [qb1])
                .unwrap()
                .outputs_arr();

            main.add_dataflow_op(TketOp::QFree, [qb1]).unwrap();

            main.add_dataflow_op(TketOp::QFree, [qb2]).unwrap();

            main.finish_with_outputs([r]).unwrap().node()
        };

        let mut hugr = {
            let mut hugr = module.finish_hugr().unwrap();
            hugr.set_entrypoint(main);
            hugr
        };

        pass.run(&mut hugr).unwrap();

        for n in hugr.nodes() {
            if let Some(ext) = hugr.get_optype(n).as_extension_op() {
                assert_ne!(ext.extension_id(), &TKET_EXTENSION_ID);
            }
        }

        let call_graph = CallGraph::new(&hugr);
        let g = call_graph.graph();
        let funcs_by_name = all_funcs_by_name(&hugr);

        assert_eq!(h_node, funcs_by_name["_H"].0);
        assert!(hugr.get_optype(h_node).is_func_decl());
        assert!(
            g.edges_directed(call_graph.node_index(h_node).unwrap(), Direction::Incoming)
                .count()
                > 0
        );

        assert_eq!(measure_node, funcs_by_name["_Measure"].0);
        assert!(hugr.get_optype(measure_node).is_func_defn());
        assert!(
            g.edges_directed(
                call_graph.node_index(measure_node).unwrap(),
                Direction::Incoming
            )
            .count()
                > 0
        );
    }
}
