use std::collections::HashMap;

use portgraph::graph::Direction;
use quantraption::ast::{
    Arg, BasicBlock, BitWidth, FuncKind, Instr, Op, Phi, QuantumGateFunc, QubitId, RTFunc, Reg,
    ResultId, Term,
};

use crate::circuit::{
    circuit::{Circuit, UnitID},
    dag::Edge,
    operation::{ConstValue, CustomOp, Op as CircOp, Signature, WireType},
};

impl CustomOp for FuncKind {
    fn signature(&self) -> Option<Signature> {
        use FuncKind::CFunc as C;
        use FuncKind::QFunc as Q;
        use QuantumGateFunc::*;
        use RTFunc::*;

        Some(match self {
            Q(ReadResult) => {
                Signature::new(vec![WireType::LinearBit], [vec![], vec![WireType::I64]])
            }
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
}

impl CircBlock {
    fn get_var_edge(&mut self, var: Var) -> Edge {
        self.invars.get(&var).copied().unwrap_or_else(|| {
            let uid = match &var {
                Var::Reg(reg) => UnitID::I64(reg.reg_name.clone()),
                Var::Qubit(qid) => UnitID::new_q(qid.index as u32),
                Var::Result(rid) => UnitID::new_b(rid.index as u32),
            };

            let e = self.circ.add_unitid(uid);
            self.invars.insert(var, e);
            e
        })
    }

    fn get_or_insert_arg(&mut self, arg: &Arg) -> Edge {
        let var = match arg {
            // Arg::Register(r) => Var::Reg(r.clone()),
            // Arg::QId(qid) => Var::Qubit(qid.clone()),
            // Arg::Result(rid) => Var::Result(rid.clone()),
            Arg::ConstVal(c) => {
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
                    quantraption::ast::Const::F64(f) => ConstValue::F64(*f),
                };
                let e = self.circ.add_edge(cv.get_type());
                self.circ
                    .add_vertex_with_edges(CircOp::Const(cv), vec![], vec![e]);
                return e;
            }
            _ => arg_to_var(arg.clone()).unwrap(),
        };

        self.get_var_edge(var)
    }

    fn update_var(&mut self, var: Var, e: Edge) {
        let e = match var {
            Var::Qubit(_) | Var::Result(_) => e,
            Var::Reg(_) => {
                if let Some(exist_e) = self.invars.get(&var) {
                    // TODO a "merge copies" rewriter would be nice
                    let coped = self.circ.copy_edge(*exist_e, 2).expect("copy failed");
                    coped[0]
                } else {
                    e
                }
            }
        };
        self.invars.insert(var, e);
    }

    fn get_args(&mut self, args: &[Arg]) -> Vec<Edge> {
        args.iter().map(|a| self.get_or_insert_arg(a)).collect()
    }

    fn load_instr(&mut self, inst: Instr) -> Result<(), ()> {
        let (dest, op) = match inst {
            Instr::Assign(d, op) => (Some(d), op),
            Instr::NoAssign(op) => (None, op),
        };

        dbg!(&op);
        let o = match op {
            Op::Call(fk, args) => {
                let (cop, args) = map_op(fk, args);
                let in_edges = self.get_args(&args);
                let sig = cop.signature().expect("signature missing.");
                let outs = add_out_edges(&mut self.circ, &sig);
                self.circ.add_vertex_with_edges(cop, in_edges, outs.clone());
                let mut outiter = outs.iter();
                for (arg, e) in args
                    .into_iter()
                    .map(arg_to_var)
                    .flatten()
                    .zip(&mut outiter)
                    .take(sig.linear.len())
                {
                    self.update_var(arg, *e);
                }
                let remaining_outs: Vec<_> = outiter.collect();
                assert!(sig.nonlinear[1].len() < 2, "not multi output function.");
                match &remaining_outs[..] {
                    &[e] => Some(*e),
                    &[] => None,
                    _ => panic!("unexpected num of outs remaining."),
                }
            }
            Op::Add(_, _) => todo!(),
            Op::Sub(_, _) => todo!(),
            Op::Mul(_, _) => todo!(),
            Op::Div(_, _) => todo!(),
            Op::Shl(_, _) => todo!(),
            Op::LShr(_, _) => todo!(),
            Op::Select(_, _, _) => todo!(),
            Op::Xor(a, b) => {
                let e = self.circ.add_edge(WireType::I64);
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
            _ => todo!(),
        };

        if let Some(reg) = dest {
            self.update_var(Var::Reg(reg), o.expect("missing output edge"));
        }
        Ok(())
    }
}
fn add_out_edges(c: &mut Circuit, sig: &Signature) -> Vec<Edge> {
    sig.linear
        .iter()
        .chain(sig.nonlinear[1].iter())
        .map(|wt| c.add_edge(*wt))
        .collect()
}

fn map_op(fk: FuncKind, mut args: Vec<Arg>) -> (CircOp, Vec<Arg>) {
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
            Q::Z => todo!(),
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
        FuncKind::CFunc(_) => todo!(),
    };

    let args = match cop {
        RxF64 | RzF64 => vec![args.pop().unwrap(), args.pop().unwrap()],

        _ => args,
    };

    (cop, args)
}

fn arg_to_var(arg: Arg) -> Option<Var> {
    Some(match arg {
        Arg::Register(r) => Var::Reg(r),
        Arg::QId(qid) => Var::Qubit(qid),
        Arg::Result(rid) => Var::Result(rid),
        Arg::ConstVal(_) => return None,
    })
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

        for inst in instrs {
            cblock.load_instr(inst)?;
        }

        let out_edges: Vec<_> = cblock.invars.values().copied().collect();

        cblock
            .circ
            .dag
            .connect_many(
                cblock.circ.boundary()[1],
                out_edges,
                Direction::Incoming,
                None,
            )
            .map_err(|_| ())?;

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
        let cb: CircBlock = f[0].clone().try_into().unwrap();
        check_soundness(&cb.circ).unwrap();
        println!("{}", cb.circ.dot_string());
        ast.write_bitcode_to_path("dump.bc", "test_roundtrip");
    }
}
