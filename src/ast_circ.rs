use std::collections::HashMap;

use portgraph::graph::Direction;
use quantraption::ast::{
    Arg, BasicBlock, BitWidth, Const, FuncKind, Instr, Op, Phi, QuantumGateFunc, QubitId, RTFunc,
    Reg, ResultId, Term,
};

use crate::circuit::{
    circuit::{Circuit, UnitID},
    dag::Edge,
    operation::{AngleValue, ConstValue, CustomOp, Op as CircOp, Signature, WireType},
};

impl CustomOp for FuncKind {
    fn signature(&self) -> Option<Signature> {
        use FuncKind::CFunc as C;
        use FuncKind::QFunc as Q;
        use QuantumGateFunc::*;
        use RTFunc::*;

        Some(match self {
            Q(ReadResult) => {
                Signature::new(vec![WireType::LinearBit], [vec![], vec![WireType::Bool]])
            }
            C(TupleStartRecord) => Signature::new_linear(vec![WireType::Control]),
            C(TupleEndRecord) => Signature::new_linear(vec![WireType::Control]),
            C(RecordResult) => Signature::new(
                vec![WireType::Control, WireType::LinearBit],
                [vec![WireType::I64], vec![]],
            ),
            Q(_) => panic!("should be converted to native op"),
            C(_) => todo!(),
        })
    }
}

// Encapsulate a basic block as a circuit with some extra data
pub struct CircBlock {
    circ: Circuit,
    invars: HashMap<Var, Edge>,
    phis: Vec<Phi>,
    term: Term,
    name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Var {
    Qubit(QubitId),
    Result(ResultId),
    Reg(Reg),
    Control,
}

impl CircBlock {
    fn get_var_edge(&mut self, var: Var) -> Edge {
        let isreg = matches!(&var, Var::Reg(_));
        // if let Var::Reg(_) = &var {
        //     if let Some(exist_e) = self.invars.get(&var) {
        //         // TODO a "merge copies" rewriter would be nice
        //         let coped = self.circ.copy_edge(*exist_e, 2).expect("copy failed");
        //         return coped[0];
        //     }
        // }
        let e = self.invars.get(&var).copied().unwrap_or_else(|| {
            // input to basic block
            let uid = match &var {
                Var::Reg(reg) => match reg.reg_type {
                    BitWidth::I64 => UnitID::I64(reg.reg_name.clone()),
                    BitWidth::I1 => UnitID::Bool(reg.reg_name.clone()),
                },
                Var::Qubit(qid) => UnitID::new_q(qid.index as u32),
                Var::Result(rid) => UnitID::new_b(rid.index as u32),
                Var::Control => panic!("there should always be a control edge."),
            };

            let e = self.circ.add_unitid(uid);
            if let Var::Reg(_) = &var {
                self.connect_output(e);
            }
            self.invars.insert(var, e);
            e
        });

        if isreg {
            let coped = self.circ.copy_edge(e, 2).expect("copy failed");
            coped[0]
        } else {
            e
        }
    }

    fn connect_output(&mut self, e: Edge) {
        self.circ
            .dag
            .connect(self.circ.boundary()[1], e, Direction::Incoming, None)
            .expect("connecting reg to output failed.");
    }

    fn get_or_insert_input(&mut self, arg: &FuncInput) -> Edge {
        let var = match arg {
            // Arg::Register(r) => Var::Reg(r.clone()),
            // Arg::QId(qid) => Var::Qubit(qid.clone()),
            // Arg::Result(rid) => Var::Result(rid.clone()),
            FuncInput::Const(c) => {
                let cv = match c {
                    quantraption::ast::Const::Int {
                        value,
                        bits: BitWidth::I64,
                        // TODO watch out for problems caused by this cast
                    } => ConstValue::I64(*value as i64),
                    quantraption::ast::Const::Int {
                        value,
                        bits: BitWidth::I1,
                    } => ConstValue::Bool(*value > 0),
                    quantraption::ast::Const::F64(f) => ConstValue::Angle(AngleValue::F64(*f)),
                };
                let e = self.circ.add_edge(cv.get_type());
                self.circ
                    .add_vertex_with_edges(CircOp::Const(cv), vec![], vec![e]);
                return e;
            }
            FuncInput::Var(v) => v.clone(),
        };

        self.get_var_edge(var)
    }

    fn update_var(&mut self, var: Var, e: Edge) {
        // let e = match var {
        //     Var::Qubit(_) | Var::Result(_) => e,
        //     Var::Reg(_) => {
        //         if let Some(exist_e) = self.invars.get(&var) {
        //             // TODO a "merge copies" rewriter would be nice
        //             let coped = self.circ.copy_edge(*exist_e, 2).expect("copy failed");
        //             coped[0]
        //         } else {
        //             e
        //         }
        //     }
        // };
        if let Var::Reg(_) = &var {
            self.connect_output(e);
        }
        self.invars.insert(var, e);
    }

    fn get_args<FI: Into<FuncInput> + Clone>(&mut self, args: &[FI]) -> Vec<Edge> {
        args.iter()
            .map(|a| self.get_or_insert_input(&a.clone().into()))
            .collect()
    }

    fn load_instr(&mut self, inst: Instr) -> Result<(), ()> {
        let (dest, op) = match inst {
            Instr::Assign(d, op) => (Some(d), op),
            Instr::NoAssign(op) => (None, op),
        };

        let o = match op {
            Op::Call(fk, args) => self.add_func(fk, args),
            Op::Add(_, _) => todo!(),
            Op::Sub(_, _) => todo!(),
            Op::Mul(_, _) => todo!(),
            Op::Div(_, _) => todo!(),
            Op::Shl(_, _) => todo!(),
            Op::LShr(_, _) => todo!(),
            Op::Select(c, a, b) => {
                let args = self.get_args(&[c, a, b]);
                // let wt = self.circ.edge_type(args[1]).expect("edge type
                // missing");
                let wt = dest.as_ref().unwrap().reg_type.into();
                let e = self.circ.add_edge(wt);

                self.circ
                    .add_vertex_with_edges(CircOp::Select(wt), args, vec![e]);
                Some(e)
            }
            Op::Xor(a, b) => {
                let e = self.circ.add_edge(WireType::Bool);
                let args = self.get_args(&[a, b]);

                self.circ.add_vertex_with_edges(CircOp::Xor, args, vec![e]);
                Some(e)
            }
            Op::Or(_, _) => todo!(),
            Op::And(_, _) => todo!(),
            Op::Phi(phi) => {
                self.phis.push(phi);
                return Ok(());
            }
        };

        if let Some(reg) = dest {
            let e = o.expect("missing output edge.");
            let wt = self.circ.edge_type(e).expect("edge type missing.");
            self.update_var(Var::Reg(reg), o.expect("missing output edge"));
        }
        Ok(())
    }

    fn add_func(&mut self, fk: FuncKind, args: Vec<Arg>) -> Option<Edge> {
        let (cop, args) = map_op(fk, args);
        let sig = cop.signature().expect("signature missing.");
        let in_edges = self.get_args(&args);
        let outs = add_out_edges(&mut self.circ, &sig);

        self.circ.add_vertex_with_edges(cop, in_edges, outs.clone());

        let mut outiter = outs.iter();
        for (arg, e) in args
            .into_iter()
            .map(|fi| fi.expect_var("const not valid."))
            .zip(&mut outiter)
            .take(sig.linear.len())
        {
            self.update_var(arg, *e);
        }
        let remaining_outs: Vec<&Edge> = outiter.collect();
        assert!(sig.nonlinear[1].len() < 2, "not multi output function.");
        match &remaining_outs[..] {
            &[e] => Some(*e),
            &[] => None,
            _ => panic!("unexpected num of outs remaining."),
        }
    }
}

// fn arg_to_finput( arg: Arg) -> FuncInput {
//     match arg {
//         Arg::ConstVal(c) => FuncInput::Left(c),
//         _ => FuncInput::Right(self.arg_to_var(arg).unwrap()),
//     }
// }

fn arg_to_var(arg: Arg) -> Option<Var> {
    Some(match arg {
        Arg::Register(r) => Var::Reg(r),
        Arg::QId(qid) => Var::Qubit(qid),
        Arg::Result(rid) => Var::Result(rid),
        Arg::ConstVal(_) => return None,
    })
}
fn add_out_edges(c: &mut Circuit, sig: &Signature) -> Vec<Edge> {
    sig.linear
        .iter()
        .chain(sig.nonlinear[1].iter())
        .map(|wt| c.add_edge(*wt))
        .collect()
}

#[derive(Clone)]
enum FuncInput {
    Const(Const),
    Var(Var),
}

impl FuncInput {
    fn expect_var(self, msg: &str) -> Var {
        if let Self::Var(v) = self {
            v
        } else {
            panic!("{}", msg);
        }
    }
}
impl From<Arg> for FuncInput {
    fn from(value: Arg) -> Self {
        match value {
            Arg::ConstVal(c) => FuncInput::Const(c),
            _ => FuncInput::Var(arg_to_var(value).unwrap()),
        }
    }
}

impl From<BitWidth> for WireType {
    fn from(value: BitWidth) -> Self {
        match value {
            BitWidth::I64 => WireType::I64,
            BitWidth::I1 => WireType::Bool,
        }
    }
}
fn map_op(fk: FuncKind, mut args: Vec<Arg>) -> (CircOp, Vec<FuncInput>) {
    // map ast quantum op, and permute arguments as necessary
    use CircOp::*;
    use QuantumGateFunc as Q;
    let cop = match &fk {
        FuncKind::QFunc(gate) => match gate {
            Q::H => H,
            Q::T => T,
            Q::S => S,
            Q::Cnot => CX,
            Q::Tadj => Tadj,
            Q::Sadj => Sadj,
            Q::Cz => todo!(),
            Q::X => X,
            Q::Y => todo!(),
            Q::Z => Z,
            Q::Rx => RxF64,
            Q::Ry => todo!(),
            Q::Rz => RzF64,
            Q::ZZ => ZZMax,
            Q::Rzz => todo!(),
            Q::Rxxyyzz => todo!(),
            Q::Mz => Measure,
            Q::Reset => Reset,
            Q::ReadResult => Custom(Box::new(fk)),
        },
        FuncKind::CFunc(_) => Custom(Box::new(fk)),
    };

    let args = match cop {
        RxF64 | RzF64 => vec![args.pop().unwrap(), args.pop().unwrap()],

        _ => args,
    };

    let sig = cop.signature().expect("signature missing.");
    let mut args: Vec<FuncInput> = args.into_iter().map(Into::into).collect();
    if let Some(WireType::Control) = sig.linear.get(0).copied() {
        args.insert(0, FuncInput::Var(Var::Control));
    }
    (cop, args)
}

impl TryFrom<BasicBlock> for CircBlock {
    type Error = ();

    fn try_from(block: BasicBlock) -> Result<Self, Self::Error> {
        let BasicBlock { name, instrs, term } = block;

        let mut cblock = Self {
            circ: Circuit::new(),
            phis: vec![],
            invars: HashMap::new(),
            term,
            name,
        };
        let c_e = cblock.circ.new_input(WireType::Control);
        cblock.invars.insert(Var::Control, c_e);

        for inst in instrs {
            cblock.load_instr(inst)?;
        }
        let out_edges: Vec<_> = cblock
            .invars
            .values()
            .copied()
            // skip already connected reg edges
            .filter(|e| cblock.circ.edge_endpoints(*e).is_none())
            .collect();

        cblock
            .circ
            .dag
            .connect_many(
                cblock.circ.boundary()[1],
                out_edges,
                Direction::Incoming,
                None,
            )
            .unwrap();
        // .map_err(|_| ())?;

        Ok(cblock)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validate::check_soundness;
    use quantraption::ast::AST;
    #[test]
    fn test_roundtrip() {
        let ast = AST::read_path("qir_ex.bc").unwrap();
        let AST::Func(f) = &ast;
        for b in f {
            let cb: CircBlock = b.clone().try_into().unwrap();
            println!("{}", cb.circ.dot_string());
            check_soundness(&cb.circ).unwrap();
        }
        ast.write_bitcode_to_path("dump.bc", "test_roundtrip");
    }
}
