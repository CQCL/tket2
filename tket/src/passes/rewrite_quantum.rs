#![allow(missing_docs)]

use std::collections::BTreeMap;

use hugr::{
    algorithms::{
        replace_types::{NodeTemplate, ReplaceTypesError},
        ComposablePass, ReplaceTypes,
    },
    builder::{BuildError, HugrBuilder, ModuleBuilder},
    extension::prelude::qb_t,
    hugr::{hugrmut::HugrMut, ValidationError},
    ops::{handle::NodeHandle as _, DataflowOpTrait, OpType},
    std_extensions::arithmetic::int_types::INT_TYPES,
    types::{PolyFuncType, Type, TypeRV},
    HugrView, Node,
};
use strum::IntoEnumIterator as _;

use crate::TketOp;

#[derive(derive_more::Error, Debug, derive_more::Display, derive_more::From)]
#[non_exhaustive]
pub enum RewriteQuantumPassError {
    ReplaceTypesError(ReplaceTypesError),
    ValidationError(ValidationError<Node>),
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
        let r = self.lowerer(self.find_or_create_funcs(hugr)?).run(hugr)?;
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
            let expected_sig: PolyFuncType = {
                // this is kind of gross, ReplaceTypes already does this we
                // should be able to call into it here.
                let mut sig = ext_op.signature().into_owned();
                sig.input
                    .iter_mut()
                    .chain(sig.output.iter_mut())
                    .for_each(|ty| {
                        if ty == &qb_t() {
                            *ty = self.qubit_to_ty.clone();
                        }
                    });
                sig.into()
            };
            let node = if let Some((n, sig)) = funcs_by_name.get(name) {
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
                    let id = builder.declare(name, expected_sig.clone())?;
                    let mut hugr = builder.finish_hugr()?;
                    hugr.set_entrypoint(id.node());
                    hugr
                };
                let ir = hugr.insert_hugr(mod_node, decl_hugr);
                ir.inserted_entrypoint
            };
            let _ = func_nodes.insert(op, node);
        }
        Ok(func_nodes)
    }

    pub fn lowerer(&self, funcs: BTreeMap<TketOp, Node>) -> ReplaceTypes {
        let mut lw = ReplaceTypes::default();
        lw.replace_type(
            qb_t().as_extension().cloned().unwrap(),
            self.qubit_to_ty.clone(),
        );
        for (op, node) in funcs {
            lw.replace_op(&op.into_extension_op(), NodeTemplate::Call(node, vec![]));
        }
        lw
    }
}

#[cfg(test)]
mod test {
    use hugr::{
        algorithms::call_graph::CallGraph,
        builder::{Dataflow, DataflowSubContainer as _, SubContainer},
        type_row,
        types::Signature,
    };
    use petgraph::Direction;

    use crate::extension::TKET_EXTENSION_ID;

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
        let reset_node = {
            let t = &pass.qubit_to_ty;
            let reset = module
                .define_function("_Reset", Signature::new_endo(t.clone()))
                .unwrap();
            let [q] = reset.input_wires_arr();
            reset.finish_with_outputs([q]).unwrap().node()
        };

        let main = {
            let mut main = module
                .define_function("main", Signature::new_endo(type_row![]))
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

            let [qb1, _r] = main
                .add_dataflow_op(TketOp::Measure, [qb1])
                .unwrap()
                .outputs_arr();

            let [qb1] = main
                .add_dataflow_op(TketOp::Reset, [qb1])
                .unwrap()
                .outputs_arr();

            main.add_dataflow_op(TketOp::QFree, [qb1]).unwrap();

            main.add_dataflow_op(TketOp::QFree, [qb2]).unwrap();

            main.finish_sub_container().unwrap().node()
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

        assert_eq!(reset_node, funcs_by_name["_Reset"].0);
        assert!(hugr.get_optype(reset_node).is_func_defn());
        assert!(
            g.edges_directed(
                call_graph.node_index(reset_node).unwrap(),
                Direction::Incoming
            )
            .count()
                > 0
        );
    }
}
