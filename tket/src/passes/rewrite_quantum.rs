#![allow(missing_docs)]

use hugr_core::hugr::linking::{
    self, HugrLinking as _, NodeLinkingDirective, NodeLinkingDirectives, NodeLinkingError,
};
use itertools::Itertools as _;
use petgraph::visit::{IntoEdgeReferences, Walker};
use std::collections::{BTreeMap, BTreeSet, HashMap};

use hugr::{
    algorithms::{
        call_graph::{CallGraph, CallGraphEdge, CallGraphNode},
        composable::ValidatePassError,
        remove_dead_funcs,
        replace_types::{NodeTemplate, ReplaceTypesError},
        ComposablePass, RemoveDeadFuncsError, ReplaceTypes,
    },
    builder::{BuildError, Container, Dataflow, DataflowSubContainer, HugrBuilder, ModuleBuilder},
    extension::{
        prelude::{bool_t, qb_t},
        SignatureError,
    },
    hugr::{hugrmut::HugrMut, views::ExtractionResult, ValidationError},
    ops::{handle::NodeHandle as _, Call, DataflowOpTrait, OpType},
    std_extensions::arithmetic::int_types::INT_TYPES,
    types::{CustomType, PolyFuncType, Signature, Transformable, Type, TypeRV, TypeTransformer},
    Hugr, HugrView, Node,
};
use strum::IntoEnumIterator as _;

use crate::{
    extension::bool::{bool_custom_type, bool_type, BoolOpBuilder},
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
    #[from]
    ValidatePassError(ValidatePassError<Node, RemoveDeadFuncsError>),
    #[display("Missing function '{name}'")]
    MissingFunction { name: String },
    #[from]
    NodeLinkingError(NodeLinkingError<Node, Node>),
}

#[derive(Debug, Clone)]
pub struct RewriteQuantumPass {
    qubit_to_ty: Type,
    pub entrypoint: Option<String>,
    funcs: BTreeMap<TketOp, String>,
}

impl Default for RewriteQuantumPass {
    fn default() -> Self {
        let int: TypeRV = INT_TYPES[6].clone().into();
        Self {
            qubit_to_ty: Type::new_tuple(vec![int.clone(), int]),
            entrypoint: None,
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
    type Result = ();

    fn run(&self, hugr: &mut H) -> Result<Self::Result, Self::Error> {
        let hugr_all_funcs_by_name = all_funcs_by_name(hugr);

        let mut state = if let Some(entrypoint) = self.entrypoint.as_ref() {
            let node = hugr_all_funcs_by_name
                .get(entrypoint)
                .ok_or(RewriteQuantumPassError::MissingFunction {
                    name: entrypoint.clone(),
                })?
                .0;
            RewriteQuantumState::new(&hugr_all_funcs_by_name, self, &hugr.with_entrypoint(node))?
        } else {
            RewriteQuantumState::new(&hugr_all_funcs_by_name, self, &hugr)?
        };
        // let new_entrypoint_node = hugr_map.extracted_node(old_entrypoint_node);
        // hugr2.set_entrypoint(new_entrypoint_node);

        // remove_dead_funcs(&mut hugr2, [])?;
        for op in TketOp::iter() {
            state.op(op)?;
        }

        let (rw_hugr, nlds, e) = state.finish()?;

        // for (op, new_node) in op_funcs {
        //     let name = &self.funcs[&op];
        //     if let Some((n, _)) = hugr_all_funcs_by_name.get(name) {
        //         nlds.insert(new_node, NodeLinkingDirective::UseExisting(*n));
        //     } else {
        //         nlds.insert(new_node, NodeLinkingDirective::add());
        //     }
        let orig_entrypoint = hugr.entrypoint();
        hugr.set_entrypoint(hugr.module_root());
        eprintln!("rw_hugr: {}", rw_hugr.mermaid_string());
        eprintln!("nlds: {nlds:?}");
        eprintln!("hugr: {}", hugr.mermaid_string());

        let link_r = hugr.insert_link_hugr_by_node(None, rw_hugr, nlds).unwrap();
        hugr.set_entrypoint(link_r.node_map[&e]);
        eprintln!("{}", hugr.mermaid_string());
        hugr.validate()?;
        Ok(())
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

#[derive(Debug)]
struct RewriteQuantumState<'a> {
    orig_funcs_by_name: &'a BTreeMap<String, (Node, PolyFuncType)>,
    pass: &'a RewriteQuantumPass,
    op_funcs: BTreeMap<TketOp, Node>,
    nlds: NodeLinkingDirectives<Node, Node>,
    hugr: Hugr,
}

impl<'a> RewriteQuantumState<'a> {
    pub fn new(
        orig_funcs_by_name: &'a BTreeMap<String, (Node, PolyFuncType)>,
        pass: &'a RewriteQuantumPass,
        hugr: &impl HugrView<Node = Node>,
    ) -> Result<Self, RewriteQuantumPassError> {
        let (mut new_hugr, hugr_map) = hugr.extract_hugr(hugr.module_root());
        let new_entrypoint_node = hugr_map.extracted_node(hugr.entrypoint());
        let mut nlds = HashMap::default();
        if new_entrypoint_node != new_hugr.module_root() {
            new_hugr.set_entrypoint(new_entrypoint_node);
            nlds.insert(
                new_entrypoint_node,
                NodeLinkingDirective::replace([hugr.entrypoint()]),
            );
            remove_dead_funcs(&mut new_hugr, [])?;
        }
        Ok(Self {
            orig_funcs_by_name,
            pass,
            hugr: new_hugr,
            nlds,
            op_funcs: Default::default(),
        })
    }

    pub fn op(&mut self, op: TketOp) -> Result<(), RewriteQuantumPassError> {
        let new_qb_ty = &self.pass.qubit_to_ty;
        let func_name = &self.pass.funcs[&op];
        let ext_op = op.into_extension_op();
        let func_sig: PolyFuncType = {
            // this is kind of gross, ReplaceTypes already does this we
            // should be able to call into it here.
            let mut sig = ext_op.signature().into_owned();
            sig.input
                .iter_mut()
                .chain(sig.output.iter_mut())
                .for_each(|ty| {
                    if ty == &qb_t() {
                        *ty = new_qb_ty.clone();
                    } else if ty == &bool_t() {
                        *ty = bool_type();
                    }
                });
            sig.into()
        };
        let existing_node = if let Some((n, sig)) = self.orig_funcs_by_name.get(func_name) {
            if sig != &func_sig {
                return Err(RewriteQuantumPassError::ExistingFunctionSignatureMismatch {
                    op,
                    node: *n,
                    name: func_name.clone(),
                    expected: func_sig,
                    found: sig.clone(),
                });
            }
            Some(n)
        } else {
            None
        };
        // let op_sig = ext_op.signature().into_owned();

        let mut mod_builder = ModuleBuilder::new();
        let func_id = mod_builder.declare(func_name, func_sig.clone())?;
        let mut funcs_to_link = vec![(func_id.node(), existing_node)];

        let func_node = if matches!(op, TketOp::Measure) {
            let mut wrapper_func = mod_builder.define_function(
                format!("func_name.Wrapped"),
                Signature::new(
                    self.pass.qubit_to_ty.clone(),
                    vec![new_qb_ty.clone(), bool_t()],
                ),
            )?;
            let [q] = wrapper_func.input_wires_arr();
            let [q, r] = wrapper_func.call(&func_id, &[], [q])?.outputs_arr();
            let [r] = wrapper_func.add_bool_read(r)?;
            let n = wrapper_func.finish_with_outputs([q, r])?.node();
            funcs_to_link.push((n.node(), None));
            n
        } else {
            func_id.node()
        };
        let func_hugr = mod_builder.finish_hugr()?;

        let link_r = self
            .hugr
            .insert_link_hugr_by_node(
                None,
                func_hugr,
                funcs_to_link
                    .iter()
                    .map(|(n, _)| (*n, NodeLinkingDirective::add()))
                    .collect(),
            )
            .unwrap();
        self.op_funcs.insert(op, link_r.node_map[&func_node]);
        for (n, mb_existing) in funcs_to_link {
            let nld = mb_existing.map_or(NodeLinkingDirective::add(), |existing| {
                NodeLinkingDirective::UseExisting(*existing)
            });
            self.nlds.insert(link_r.node_map[&n], nld);
        }
        Ok(())
    }

    pub fn finish(
        mut self,
    ) -> Result<(Hugr, NodeLinkingDirectives<Node, Node>, Node), RewriteQuantumPassError> {
        let mut lw = ReplaceTypes::default();
        lw.replace_type(
            qb_t().as_extension().cloned().unwrap(),
            self.pass.qubit_to_ty.clone(),
        );
        for (op, node) in self.op_funcs {
            lw.replace_op(&op.into_extension_op(), NodeTemplate::Call(node, vec![]));
        }
        lw.run(&mut self.hugr)?;
        let e = self.hugr.entrypoint();
        self.hugr.set_entrypoint(self.hugr.module_root());
        Ok((self.hugr, self.nlds, e))
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

    struct Case {
        hugr: Hugr,
        h_node: Node,
        measure_node: Node,
        main_node: Node,
    }

    #[rstest::fixture]
    fn example() -> Case {
        let qubit_to_ty = RewriteQuantumPass::default().qubit_to_ty;
        let mut module = ModuleBuilder::new();
        let h_node = {
            let t = &qubit_to_ty;
            module
                .declare("_H", Signature::new_endo(t.clone()).into())
                .unwrap()
                .node()
        };
        let measure_node = {
            let t = &qubit_to_ty;
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

        let main_node = {
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
        let hugr = module.finish_hugr().unwrap();
        Case {
            hugr,
            h_node,
            measure_node,
            main_node,
        }
    }

    #[rstest::rstest]
    fn main_entrypoint(example: Case) {
        let Case {
            mut hugr,
            h_node,
            measure_node,
            main_node,
        } = example;
        let pass = RewriteQuantumPass::default();
        hugr.set_entrypoint(main_node);
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

    #[rstest::rstest]
    fn module_entrypoint(example: Case) {
        let Case {
            mut hugr,
            h_node,
            measure_node,
            main_node,
        } = example;
        let pass = RewriteQuantumPass::default();
        hugr.set_entrypoint(hugr.module_root());
        let pass = RewriteQuantumPass::default();
        hugr.set_entrypoint(main_node);
        pass.run(&mut hugr).unwrap();
    }
}
