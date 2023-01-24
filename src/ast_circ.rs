use std::collections::HashMap;

use portgraph::graph::Direction;
use quantraption::ast::{
    Arg, BasicBlock, BitWidth, Const, FuncKind, Instr, Op, Phi, QuantumGateFunc, QubitId, RTFunc,
    Reg, ResultId, Term,
};

use crate::circuit::{
    circuit::{Circuit, UnitID},
    dag::{Edge, TopSorter},
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
    phis: Vec<(Reg, Phi)>,
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
                let cv: ConstValue = c.into();
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
                self.phis.push((dest.expect("phi must be assigned"), phi));
                return Ok(());
            }
        };

        if let Some(reg) = dest {
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
        match remaining_outs[..] {
            [e] => Some(*e),
            [] => None,
            _ => panic!("unexpected num of outs remaining."),
        }
    }
}

impl From<&Const> for ConstValue {
    fn from(c: &Const) -> Self {
        match c {
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
        }
    }
}

impl From<&ConstValue> for Const {
    fn from(cv: &ConstValue) -> Self {
        match cv {
            ConstValue::Bool(b) => Const::Int {
                value: *b as u64,
                bits: BitWidth::I1,
            },
            ConstValue::I64(i) => Const::Int {
                value: *i as u64,
                bits: BitWidth::I64,
            },
            ConstValue::F64(f) => Const::F64(*f),
            ConstValue::Angle(a) => Const::F64(a.to_f64()),
            ConstValue::Quat64(_) => panic!("Can't convert quaternion."),
        }
    }
}

// fn arg_to_finput( arg: Arg) -> FuncInput {
//     match arg {
//         Arg::ConstVal(c) => FuncInput::Left(c),
//         _ => FuncInput::Right(self.arg_to_var(arg).unwrap()),
//     }
// }

impl TryFrom<Arg> for Var {
    type Error = ();

    fn try_from(arg: Arg) -> Result<Self, Self::Error> {
        Ok(match arg {
            Arg::Register(r) => Var::Reg(r),
            Arg::QId(qid) => Var::Qubit(qid),
            Arg::Result(rid) => Var::Result(rid),
            Arg::ConstVal(_) => return Err(()),
        })
    }
}

impl TryInto<Arg> for Var {
    type Error = ();

    fn try_into(self) -> Result<Arg, Self::Error> {
        Ok(match self {
            Var::Qubit(qid) => Arg::QId(qid),
            Var::Result(rid) => Arg::Result(rid),
            Var::Reg(r) => Arg::Register(r),
            Var::Control => return Err(()),
        })
    }
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
            _ => FuncInput::Var(value.try_into().unwrap()),
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
        RxF64 | RzF64 => {
            args.reverse();
            args
        }

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
        cblock.circ.uids.push(UnitID::Bool("__CONTROL".into()));
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
            .map_err(|_| ())?;

        Ok(cblock)
    }
}

impl TryFrom<CircBlock> for BasicBlock {
    type Error = ();

    fn try_from(cblock: CircBlock) -> Result<Self, Self::Error> {
        let CircBlock {
            circ,
            invars,
            phis,
            term,
            name,
        } = cblock;
        let mut instrs: Vec<_> = phis
            .into_iter()
            .map(|(r, p)| Instr::Assign(r, Op::Phi(p)))
            .collect();

        let mut argmap: HashMap<Edge, Arg> = HashMap::new();
        for nid in TopSorter::new_zero_input(circ.dag_ref()) {
            let op = circ.node_op(nid).expect("missing op.");

            if matches!(
                op,
                CircOp::Input
                    | CircOp::Output
                    | CircOp::Noop(_)
                    | CircOp::Const(_)
                    | CircOp::Copy { .. }
            ) {
                continue;
            }
            let sig = op.signature().expect("signature missing.");

            let mut inedges = circ.node_edges(nid, Direction::Incoming);
            let mut linlength = sig.linear.len();
            if linlength > 0 && sig.linear.get(0) == Some(&WireType::Control) {
                inedges.remove(0);
                // TODO nasty, make nicer
                linlength -= 1;
            }
            let inargs: Vec<_> = inedges
                .iter()
                .map(|e| get_arg(*e, &mut argmap, &invars, &circ))
                .collect();
            let astop = get_op(op, inargs);
            let mut outedges = circ.node_edges(nid, Direction::Outgoing);
            if linlength > 0 && sig.linear[0] == WireType::Control {
                outedges.remove(0);
            }
            let nonlin_out = sig.nonlinear[1].len();

            for (ine, oute) in inedges.iter().take(linlength).zip(outedges.iter()) {
                if Some(WireType::Control) == circ.edge_type(*ine) {
                    continue;
                }
                let arg = argmap.remove(ine).expect("incoming linear edge not found.");
                argmap.insert(*oute, arg);
            }

            let instr = match nonlin_out {
                0 => Instr::NoAssign(astop),
                1 => {
                    let e = outedges[outedges.len() - 1];
                    let dest = get_edge_arg(&invars, e);
                    if let Some(Arg::Register(r)) = dest {
                        Instr::Assign(r, astop)
                    } else {
                        panic!("destination arg not register.");
                    }
                }
                _ => panic!("More than 1 nonlinear output invalid."),
            };

            instrs.push(instr);
        }

        Ok(Self { term, name, instrs })
    }
}

fn vec_arr<const N: usize, T>(mut v: Vec<T>) -> [T; N] {
    [0; N].map(|i| v.remove(i))
}

fn get_op(op: &CircOp, inargs: Vec<Arg>) -> Op {
    match op {
        CircOp::Xor => {
            let [a, b] = vec_arr(inargs);
            Op::Xor(a, b)
        }
        CircOp::Select(..) => {
            let [a, b, c] = vec_arr(inargs);
            Op::Select(a, b, c)
        }

        _ => {
            let (fk, args) = map_op_rev(op.clone(), inargs);
            Op::Call(fk, args)
        }
    }
}

fn get_arg(
    e: Edge,
    argmap: &mut HashMap<Edge, Arg>,
    invars: &HashMap<Var, Edge>,
    circ: &Circuit,
) -> Arg {
    if let Some(a) = argmap.get(&e) {
        return a.clone();
    }
    if let Some(value) = get_edge_arg(invars, e) {
        return value;
    }
    let (src, _) = circ.edge_endpoints(e).expect("missing edge.");
    let srcop = circ.node_op(src).expect("src node missing.");
    // dbg!(e, srcop);
    match srcop {
        CircOp::Const(c) => Arg::ConstVal(c.into()),
        CircOp::Copy { .. } => get_arg(
            circ.node_edges(src, Direction::Incoming)[0],
            argmap,
            invars,
            circ,
        ),
        CircOp::Input => {
            // input to basic block, convert arg from unitID
            let port = circ.port_of_edge(src, e, Direction::Outgoing).unwrap();
            let arg = match &circ.uids[port] {
                UnitID::Qubit { index, .. } => Arg::QId(QubitId {
                    index: index[0] as u64,
                }),
                UnitID::Bit { index, .. } => Arg::Result(ResultId {
                    index: index[0] as u64,
                }),
                UnitID::I64(r) => Arg::Register(Reg {
                    reg_name: r.clone(),
                    reg_type: BitWidth::I64,
                }),
                UnitID::Bool(r) => Arg::Register(Reg {
                    reg_name: r.clone(),
                    reg_type: BitWidth::I1,
                }),
                _ => panic!("invalid input type."),
            };

            argmap.insert(e, arg.clone());
            arg
        }
        _ => panic!("what do I do with this edge."),
    }
}

fn get_edge_arg(invars: &HashMap<Var, Edge>, e: Edge) -> Option<Arg> {
    invars
        .iter()
        .find_map(|(v, ve)| (*ve == e).then(|| v.clone().try_into().ok()).flatten())
}

fn map_op_rev(op: CircOp, mut args: Vec<Arg>) -> (FuncKind, Vec<Arg>) {
    use CircOp::*;
    use QuantumGateFunc as Q;

    match &op {
        CircOp::RzF64 | CircOp::RxF64 => args.reverse(),
        _ => (),
    };
    let fk = if let Custom(custom) = op {
        *(custom).downcast::<FuncKind>().expect("Unknown custom op.")
    } else {
        FuncKind::QFunc(match op {
            H => Q::H,
            T => Q::T,
            S => Q::S,
            X => Q::X,
            Y => Q::Y,
            Z => Q::Z,
            Tadj => Q::Tadj,
            Sadj => Q::Sadj,
            CX => Q::Cnot,
            ZZMax => Q::ZZ,
            Reset => Q::Reset,
            Measure => Q::Mz,
            RxF64 => Q::Rx,
            RzF64 => Q::Rz,
            Barrier => todo!(),
            AngleAdd => todo!(),
            AngleMul => todo!(),
            AngleNeg => todo!(),
            QuatMul => todo!(),
            TK1 => todo!(),
            Rotation => todo!(),
            ToRotation => todo!(),
            Custom(_) | Select(_) | Xor | Copy { .. } | Const(_) | Output | Input | Noop(_) => {
                unreachable!("Should be skipped over at op level.")
            }
        })
    };

    (fk, args)
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
        // use std::fs;

        // fs::write("ast_before.txt", format!("{:#?}", &ast)).unwrap();
        // dbg!(&ast);
        let ast_new = AST::Func(
            f.iter()
                .map(|b| {
                    let cb: CircBlock = b.clone().try_into().unwrap();
                    // println!("{}", cb.circ.dot_string());
                    check_soundness(&cb.circ).unwrap();
                    cb.try_into().unwrap()
                })
                .collect(),
        );
        let AST::Func(f_new) = &ast_new;
        for (b1, b2) in f.iter().zip(f_new.iter()) {
            assert_eq!(b1.name, b2.name);
            assert_eq!(b1.term, b2.term);
            assert_eq!(b1.instrs.len(), b2.instrs.len());
        }
        // fs::write("ast_after.txt", format!("{:#?}", &ast)).unwrap();

        ast.print_to_path("dump_rt.ll", "mod");
    }
}
