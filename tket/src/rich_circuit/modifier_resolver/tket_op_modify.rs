//! This module provides functionality to modify TketOp operations in a quantum circuit.
use hugr::{
    ops::handle::NodeHandle,
    std_extensions::arithmetic::{float_ops::FloatOps, float_types::ConstF64},
};

use crate::{
    extension::rotation::{ConstRotation, RotationOp},
    rich_circuit::modifier_resolver::*,
};
use TketOp::*;

impl<N: HugrNode> ModifierResolver<N> {
    /// Modify a TketOp operation. The returned `PortVector` contains the incoming and outgoing ports of the modified operation.
    /// Ancilla qubits are dirty qubits that are used to store intermediate results.
    pub fn modify_tket_op(
        &mut self,
        n: N,
        op: TketOp,
        new_fn: &mut impl Dataflow,
        ancilla: &mut Vec<Wire<Node>>,
    ) -> Result<PortVector, ModifierResolverErrors<N>> {
        use TketOp::*;

        let control = self.control_num();
        let dagger = self.modifiers.dagger;

        if control != 0 || dagger {
            if !op.is_quantum() {
                return Err(ModifierResolverErrors::UnResolvable(
                    n,
                    "None quantum operation cannot be modified".to_string(),
                    op.into(),
                )
                .into());
            }
        }
        match op {
            X | CX | Toffoli | Y | CY | Z | CZ | S | Sdg | T | Tdg | V | Vdg | H
                if (control == 0)
                    || (control < 3 && op == X)
                    || (control == 1 && matches!(op, CX | Y | Z)) =>
            {
                let gate = self
                    .modifiers
                    .modified(op)
                    .unwrap_or_else(|| unreachable!());

                let qubits = match op {
                    X | Y | Z | S | T | V | Sdg | Tdg | Vdg | H => 1,
                    CX | CY | CZ => 2,
                    Toffoli => 3,
                    _ => unreachable!(),
                };

                let new = self.add_node_control(new_fn, gate);
                let incoming = control..new_fn.hugr().num_inputs(new);
                let outgoing = control..new_fn.hugr().num_outputs(new);
                let if_rev = control..(control + qubits);
                Ok(self.port_vector_dagger(new, incoming, outgoing, if_rev))
            }
            Rz | CRz | Rx | Ry if control == 0 || (control == 1 && op == Rz) => {
                let qubits = if CRz == op { 2 } else { 1 };

                let new_op = self
                    .modifiers
                    .modified(op)
                    .unwrap_or_else(|| unreachable!());
                let new = self.add_node_control(new_fn, new_op);

                if !dagger {
                    let incoming = control..new_fn.hugr().num_inputs(new);
                    let outgoing = control..new_fn.hugr().num_outputs(new);
                    Ok(PortVector::port_vector(new, incoming, outgoing))
                } else {
                    // If dagered
                    let halfturn = new_fn.add_child_node(RotationOp::to_halfturns);
                    let reversed_float = new_fn
                        .add_dataflow_op(FloatOps::fneg, vec![Wire::new(halfturn, 0)])
                        .map(|out| out.out_wire(0))?;
                    let reversed = new_fn
                        .add_dataflow_op(RotationOp::from_halfturns_unchecked, vec![reversed_float])
                        .map(|out| out.out_wire(0))?;

                    new_fn.hugr_mut().connect(
                        reversed.node(),
                        reversed.source(),
                        new,
                        qubits + control,
                    );

                    let incoming = (control..new_fn.hugr().num_inputs(new))
                        .filter_map(|i| {
                            if i < qubits + control {
                                Some((new, OutgoingPort::from(i)).into())
                            } else if i == qubits + control {
                                Some((halfturn, IncomingPort::from(0)).into())
                            } else {
                                // FIXME: forget state order
                                None
                            }
                        })
                        .collect();
                    let outgoing = (control..new_fn.hugr().num_outputs(new))
                        .filter_map(|i| {
                            let dw: DirWire = (new, IncomingPort::from(i)).into();
                            if i >= qubits + control {
                                Some(dw.shift(1))
                            } else {
                                Some(dw)
                            }
                        })
                        .collect();
                    Ok(PortVector { incoming, outgoing })
                }
            }
            H => {
                // H = X * Ry(pi/2).
                let (mut pv_ry, pv_x) = if !dagger {
                    (
                        self.modify_tket_op(n, Ry, new_fn, ancilla)?,
                        self.modify_tket_op(n, X, new_fn, ancilla)?,
                    )
                } else {
                    let (pv_x, pv_ry) = (
                        self.modify_tket_op(n, X, new_fn, ancilla)?,
                        self.modify_tket_op(n, Ry, new_fn, ancilla)?,
                    );
                    (pv_ry, pv_x)
                };
                let angle = new_fn.add_load_value(ConstRotation::new(0.5).unwrap());
                // let angle = new_fn
                //     .add_dataflow_op(RotationOp::from_halfturns_unchecked, vec![angle])
                //     .unwrap()
                //     .out_wire(0);
                let rot_in = pv_ry.incoming.remove(1);
                connect(new_fn, &rot_in, &angle.into())?;
                connect(new_fn, &pv_ry.outgoing[0], &pv_x.incoming[0])?;

                Ok(PortVector {
                    incoming: pv_ry.incoming,
                    outgoing: pv_x.outgoing,
                })
            }
            Rx => {
                let h1 = new_fn.add_child_node(H);
                let h2 = new_fn.add_child_node(H);
                let mut pv = self.modify_tket_op(n, Rz, new_fn, ancilla)?;
                pv.incoming[0] = connect_by_num(new_fn, &pv.incoming[0], h1, 0);
                pv.outgoing[0] = connect_by_num(new_fn, &pv.outgoing[0], h2, 0);
                Ok(pv)
            }
            Ry | CY => {
                let (gate, targ) = match op {
                    Ry => (Rx, 0),
                    CY => (CX, 1),
                    _ => unreachable!(),
                };
                let s = new_fn.add_child_node(S);
                let sdg = new_fn.add_child_node(Sdg);
                let mut pv = self.modify_tket_op(n, gate, new_fn, ancilla)?;
                if !dagger {
                    pv.incoming[0] = connect_by_num(new_fn, &pv.incoming[targ], sdg, 0);
                    pv.outgoing[0] = connect_by_num(new_fn, &pv.outgoing[targ], s, 0);
                } else {
                    pv.outgoing[0] = connect_by_num(new_fn, &pv.outgoing[targ], sdg, 0);
                    pv.incoming[0] = connect_by_num(new_fn, &pv.incoming[targ], s, 0);
                }
                Ok(pv)
            }
            T | Tdg | S | Sdg | V | Vdg => {
                // op(t) = Phase(θ) * U(t, 2θ)
                self.modifiers.dagger = false;

                let Some((gate, angle)) = self.modifiers.rot_angle(op) else {
                    unreachable!()
                };
                let rot = new_fn.add_load_value(ConstRotation::new(angle).unwrap());
                let rot_2 = new_fn.add_load_value(ConstRotation::new(angle * 2.0).unwrap());

                // CU(cs,t,2θ);
                let mut pv_u = self.modify_tket_op(n, gate, new_fn, ancilla)?;
                connect(new_fn, &rot_2.into(), &pv_u.incoming[1])?;
                let mut t = pv_u.outgoing[0].clone().try_into().unwrap();

                // CPhase(cs,θ);
                let theta_inputs = self.with_ancilla(&mut t, ancilla, |this, ancilla| {
                    this.modify_global_phase(n, new_fn, ancilla)
                })?;
                pv_u.outgoing[0] = t.into();
                for theta_in in theta_inputs {
                    new_fn
                        .hugr_mut()
                        .connect(rot.node(), rot.source(), theta_in.0, theta_in.1);
                }

                if dagger {
                    mem::swap(&mut pv_u.incoming, &mut pv_u.outgoing)
                }
                self.modifiers.dagger = dagger;

                Ok(pv_u)
            }
            // If more control qubits
            Toffoli if !ancilla.is_empty() => {
                // Cn+m+2X(cs1,cs2,x,y,t) = Cn+2X(cs1,x,y,a); Cm+1X(cs2,a,t); Cn+2X(cs1,x,y,a); Cm+1X(cs2,a,t);
                let nd = n;
                self.modifiers.dagger = false;
                let mut a = ancilla.pop().unwrap().into();

                let n = control / 2;
                let m = control - n;
                // We know 1, n <= m.
                let mut cs2 = self.controls().split_off(n);

                println!("P1");
                // 1. Cn+2X(cs1,x,y,a)
                self.modifiers.control = n;
                let cs2_last = cs2.last_mut().unwrap();
                let pv1 = self.with_ancilla(cs2_last, ancilla, |this, ancilla| {
                    this.modify_tket_op(nd, Toffoli, new_fn, ancilla)
                })?;
                connect(new_fn, &a, &pv1.incoming[2])?;
                let x_in = pv1.incoming[0].clone();
                let y_in = pv1.incoming[1].clone();
                let mut x = pv1.outgoing[0].clone().try_into().unwrap();
                let mut y = pv1.outgoing[1].clone();
                a = pv1.outgoing[2].clone();

                println!("P2");
                // 2. Cm+1X(cs2,a,t)
                self.modifiers.control = m;
                let cs1 = mem::replace(self.controls(), cs2);
                let pv2 = self.with_ancilla(&mut x, ancilla, |this, ancilla| {
                    this.modify_tket_op(nd, CX, new_fn, ancilla)
                })?;
                connect(new_fn, &a, &pv2.incoming[0])?;
                a = pv2.outgoing[0].clone();
                let t_in = pv2.incoming[1].clone();
                let mut t = pv2.outgoing[1].clone();

                println!("P3");
                // 3. Cn+2X(cs1,x,y,a)
                self.modifiers.control = n;
                let mut cs2 = mem::replace(self.controls(), cs1);
                let cs2_last = cs2.last_mut().unwrap();
                let pv3 = self.with_ancilla(cs2_last, ancilla, |this, ancilla| {
                    this.modify_tket_op(nd, Toffoli, new_fn, ancilla)
                })?;
                connect(new_fn, &x.into(), &pv3.incoming[0])?;
                connect(new_fn, &y, &pv3.incoming[1])?;
                connect(new_fn, &a, &pv3.incoming[2])?;
                x = pv3.outgoing[0].clone().try_into().unwrap();
                y = pv3.outgoing[1].clone();
                a = pv3.outgoing[2].clone();

                println!("P4");
                // 4. Cm+1X(cs2,a,t)
                self.modifiers.control = m;
                let cs1 = mem::replace(self.controls(), cs2);
                let pv4 = self.with_ancilla(&mut x, ancilla, |this, ancilla| {
                    this.modify_tket_op(nd, CX, new_fn, ancilla)
                })?;
                connect(new_fn, &a, &pv4.incoming[0])?;
                connect(new_fn, &t, &pv4.incoming[1])?;
                a = pv4.outgoing[0].clone();
                t = pv4.outgoing[1].clone();

                self.modifiers.control = control;
                self.modifiers.dagger = dagger;
                let cs2 = mem::replace(self.controls(), cs1);
                self.controls().extend(cs2);
                ancilla.push(a.try_into().unwrap());
                let mut incoming = vec![x_in, y_in, t_in];
                let mut outgoing = vec![x.into(), y, t];
                if dagger {
                    mem::swap(&mut incoming, &mut outgoing);
                }
                Ok(PortVector { incoming, outgoing })
            }
            CX | X if !ancilla.is_empty() => {
                let c_num = if op == X { 2 } else { 1 };
                let mut ctrls = vec![];
                for _ in 0..c_num {
                    ctrls.push(self.pop_control().unwrap());
                }

                let mut pv = self.modify_tket_op(n, Toffoli, new_fn, ancilla)?;

                if dagger {
                    mem::swap(&mut pv.incoming, &mut pv.outgoing)
                }
                for _ in 0..c_num {
                    let c = ctrls.pop().unwrap();
                    let c_in = pv.incoming.remove(0);
                    connect(new_fn, &c.into(), &c_in)?;
                    let c_out = pv.outgoing.remove(0).try_into().unwrap();
                    self.push_control(c_out);
                }
                if dagger {
                    mem::swap(&mut pv.incoming, &mut pv.outgoing)
                }
                Ok(pv)
            }
            CRz => {
                // Cn+1Rz(cs,c,t,theta) = Rz(t,theta/2); Cn+1X(cs,c,t); Rz(t,-theta/2); Cn+1X(cs,c,t);
                self.modifiers.dagger = false;

                // rotations
                let halfturns = new_fn.add_child_node(RotationOp::to_halfturns);
                let half_const = new_fn.add_load_value(ConstF64::new(0.5));
                let half_f64 = new_fn
                    .add_dataflow_op(FloatOps::fmul, vec![Wire::new(halfturns, 0), half_const])
                    .map(|out| out.out_wire(0))?;
                let half_f64_neg = new_fn
                    .add_dataflow_op(FloatOps::fneg, vec![half_f64])
                    .map(|out| out.out_wire(0))?;
                let mut angle_pos = new_fn
                    .add_dataflow_op(RotationOp::from_halfturns_unchecked, vec![half_f64])
                    .map(|out| out.node())?;
                let mut angle_neg = new_fn
                    .add_dataflow_op(RotationOp::from_halfturns_unchecked, vec![half_f64_neg])
                    .map(|out| out.node())?;
                if dagger {
                    mem::swap(&mut angle_pos, &mut angle_neg);
                }

                // Rz(t,theta/2)
                let crz_pos = new_fn.add_child_node(Rz);
                new_fn.hugr_mut().connect(angle_pos, 0, crz_pos, 1);
                let mut t = Wire::new(crz_pos, 0).into();

                // CnCX(cs,c,t)
                let pv1 = self.modify_tket_op(n, CX, new_fn, ancilla)?;
                let mut incoming = vec![
                    pv1.incoming[0].clone(),
                    (crz_pos, IncomingPort::from(0)).into(),
                ];
                connect(new_fn, &t, &pv1.incoming[1])?;
                let mut c = pv1.outgoing[0].clone();
                t = pv1.outgoing[1].clone();

                // Rz(t,-theta/2)
                let crz_neg = new_fn.add_child_node(Rz);
                t = connect_by_num(new_fn, &t, crz_neg, 0);
                new_fn.hugr_mut().connect(angle_neg, 0, crz_neg, 1);

                // CnCX(cs,c,t)
                let pv2 = self.modify_tket_op(n, CX, new_fn, ancilla)?;
                connect(new_fn, &c, &pv2.incoming[0])?;
                connect(new_fn, &t, &pv2.incoming[1])?;
                c = pv2.outgoing[0].clone();
                t = pv2.outgoing[1].clone();
                let mut outgoing = vec![c, t];

                self.modifiers.dagger = dagger;
                if dagger {
                    mem::swap(&mut incoming, &mut outgoing)
                }
                incoming.push((halfturns, IncomingPort::from(0)).into());
                // FIXME: Ignoring StateOrder
                Ok(PortVector { incoming, outgoing })
            }
            Rz | Y | Z => {
                // reduce Rz, Y, Z to CRz, CY, CZ
                let c_op = if op == Rz {
                    CRz
                } else if op == Y {
                    CY
                } else {
                    CZ
                };
                let mut last_control = self.pop_control().unwrap();

                let mut pv = self.modify_tket_op(n, c_op, new_fn, ancilla)?;
                let last_dw = if !dagger {
                    let c_in = pv.incoming.remove(0);
                    connect(new_fn, &c_in, &last_control.into())?;
                    pv.outgoing.remove(0)
                } else {
                    let c_out = pv.outgoing.remove(0);
                    connect(new_fn, &c_out, &last_control.into())?;
                    pv.incoming.remove(0)
                };

                last_control = last_dw.try_into().map_err(|_| {
                    ModifierResolverErrors::Unreachable(format!(
                        "Expected outgoing wire, found incoming wire while modifying Rz",
                    ))
                })?;

                self.push_control(last_control);

                Ok(pv)
            }
            CZ => {
                // reduce CZ to CRz(pi)
                let mut pv = self.modify_tket_op(n, CRz, new_fn, ancilla)?;
                let halfturn = new_fn.add_load_value(ConstRotation::new(1.0).unwrap());
                let dw = pv.incoming.remove(2);
                connect(new_fn, &dw, &halfturn.into())?;
                Ok(pv)
            }
            X | CX | Toffoli => {
                // Cn+1X(cs,c,t) = CV(c,t); CnX(cs,c); CVdg(c,t); CnX(cs,c); CnV(cs,t);
                let gate_control = match op {
                    X => 0,
                    CX => 1,
                    Toffoli => 2,
                    _ => unreachable!(),
                };
                self.modifiers.dagger = false;
                let mut incoming = Vec::new();
                let mut outgoing = Vec::new();

                // CV(c,t)
                self.modifiers.control = 1;
                let c = self.controls().pop().unwrap();
                let cs = mem::replace(self.controls(), vec![c]);
                let pv_crx1 = self.modify_tket_op(n, V, new_fn, ancilla)?;
                incoming.push(pv_crx1.incoming[0].clone());
                let mut targ = pv_crx1.outgoing[0].clone().try_into().unwrap();

                // CnX(cs,c)
                self.modifiers.control = control - 1;
                let c = mem::replace(self.controls(), cs)[0];
                let pv_x1 = self.with_ancilla(&mut targ, ancilla, |this, ancilla| {
                    this.modify_tket_op(n, op, new_fn, ancilla)
                })?;
                connect(new_fn, &c.into(), &pv_x1.incoming[gate_control])?;
                let c = pv_x1.outgoing[gate_control].clone().try_into().unwrap();
                for i in 0..gate_control {
                    incoming.insert(i, pv_x1.incoming[i].clone());
                }

                // CVdg(c,t)
                self.modifiers.control = 1;
                let cs = mem::replace(self.controls(), vec![c]);
                println!("Control of Vdg: {}", self.controls()[0]);
                let pv_crx2 = self.modify_tket_op(n, Vdg, new_fn, ancilla)?;
                connect(new_fn, &targ.into(), &pv_crx2.incoming[0])?;
                targ = pv_crx2.outgoing[0].clone().try_into().unwrap();

                // CnX(cs,c)
                self.modifiers.control = control - 1;
                let mut c = mem::replace(self.controls(), cs)[0];
                assert_eq!(self.controls().len(), self.control_num());
                let pv_x2 = self.with_ancilla(&mut targ, ancilla, |this, ancilla| {
                    this.modify_tket_op(n, op, new_fn, ancilla)
                })?;
                connect(new_fn, &c.into(), &pv_x2.incoming[gate_control])?;
                c = pv_x2.outgoing[gate_control].clone().try_into().unwrap();
                for i in 0..gate_control {
                    connect(new_fn, &pv_x1.outgoing[i], &pv_x2.incoming[i])?;
                }

                // CnV(cs,t)
                // self.control_num() = control + gate_control - 1;
                for i in 0..gate_control {
                    self.push_control(pv_x2.outgoing[i].clone().try_into().unwrap());
                }
                let pv_cnrx = self.with_ancilla(&mut c, ancilla, |this, ancilla| {
                    this.modify_tket_op(n, V, new_fn, ancilla)
                })?;
                for _ in 0..gate_control {
                    outgoing.push(self.pop_control().unwrap().into());
                }
                connect(new_fn, &targ.into(), &pv_cnrx.incoming[0])?;
                // connect(new_fn, &half_pos.into(), &pv_cnrx.incoming[1])?;
                outgoing.push(pv_cnrx.outgoing[0].clone());

                self.push_control(c);
                assert_eq!(control, self.control_num());
                self.modifiers.dagger = dagger;

                // TODO: This does not handle invisible wires
                if !dagger {
                    Ok(PortVector { incoming, outgoing })
                } else {
                    Ok(PortVector {
                        incoming: outgoing,
                        outgoing: incoming,
                    })
                }
            }
            Measure | MeasureFree | QAlloc | TryQAlloc | QFree | Reset => {
                let new = new_fn.add_child_node(op);
                let incoming = 0..new_fn.hugr().num_inputs(new);
                let outgoing = 0..new_fn.hugr().num_outputs(new);
                Ok(PortVector::port_vector(new, incoming, outgoing))
            }
        }
    }
}

impl CombinedModifier {
    /// If the modified operation can be represented as a TketOp,
    /// returns the modified operation, otherwise returns `None`.
    fn modified(&self, op: TketOp) -> Option<TketOp> {
        match op {
            X if self.control == 0 => Some(X),
            X if self.control == 1 => Some(CX),
            X if self.control == 2 => Some(Toffoli),
            Y if self.control == 0 => Some(Y),
            Y if self.control == 1 => Some(CY),
            Z if self.control == 0 => Some(Z),
            Z if self.control == 1 => Some(CZ),
            CX if self.control == 0 => Some(CX),
            CX if self.control == 1 => Some(Toffoli),
            CY if self.control == 0 => Some(CY),
            CZ if self.control == 0 => Some(CZ),
            Toffoli if self.control == 0 => Some(Toffoli),
            H if self.control == 0 => Some(H),
            Rz if self.control == 0 => Some(Rz),
            Rz if self.control == 1 => Some(CRz),
            CRz if self.control == 0 => Some(CRz),
            Rx if self.control == 0 => Some(Rx),
            Ry if self.control == 0 => Some(Ry),
            T if self.control == 0 => match self.dagger {
                false => Some(T),
                true => Some(Tdg),
            },
            Tdg if self.control == 0 => match self.dagger {
                false => Some(Tdg),
                true => Some(T),
            },
            S if self.control == 0 => match self.dagger {
                false => Some(S),
                true => Some(Sdg),
            },
            Sdg if self.control == 0 => match self.dagger {
                false => Some(Sdg),
                true => Some(S),
            },
            V if self.control == 0 => match self.dagger {
                false => Some(V),
                true => Some(Vdg),
            },
            Vdg if self.control == 0 => match self.dagger {
                false => Some(Vdg),
                true => Some(V),
            },
            Measure | MeasureFree | QAlloc | TryQAlloc | QFree | Reset
                if self.control == 0 && !self.dagger =>
            {
                Some(op)
            }
            _ => None,
        }
    }

    // op = exp(θ) * U(2θ)
    fn rot_angle(&self, op: TketOp) -> Option<(TketOp, f64)> {
        let (op, mut angle) = match op {
            S => (Rz, 0.25),
            T => (Rz, 0.125),
            Tdg => (Rz, -0.125),
            Sdg => (Rz, -0.25),
            V => (Rx, 0.25),
            Vdg => (Rx, -0.25),
            _ => return None,
        };
        if self.dagger {
            angle = -angle;
        }
        return Some((op, angle));
    }
}

#[cfg(test)]
mod test {
    use std::{fs::File, io::Write, path::Path};

    use hugr::{
        algorithms::{dead_code, ComposablePass},
        builder::{Container, Dataflow, DataflowSubContainer, HugrBuilder, ModuleBuilder},
        envelope::{EnvelopeConfig, EnvelopeFormat},
        extension::{prelude::qb_t, ExtensionRegistry},
        ops::{handle::NodeHandle, CallIndirect, ExtensionOp},
        std_extensions::collections::array::{array_type, ArrayOpBuilder},
        types::{Signature, Term},
    };

    use crate::{
        extension::{
            bool::BOOL_EXTENSION,
            rotation::{rotation_type, ROTATION_EXTENSION},
            TKET_EXTENSION,
        },
        rich_circuit::*,
    };
    use crate::{
        extension::{debug::StateResult, rotation::ConstRotation},
        rich_circuit::modifier_resolver::*,
    };

    #[test]
    fn test_control_simple() {
        let mut module = ModuleBuilder::new();
        let foo_sig = Signature::new(vec![qb_t(), qb_t()], vec![qb_t(), qb_t()]);
        let main_sig = Signature::new(vec![qb_t(), qb_t(), qb_t()], vec![qb_t(), qb_t(), qb_t()]);
        let call_sig = Signature::new(
            vec![array_type(1, qb_t()), qb_t(), qb_t()],
            vec![array_type(1, qb_t()), qb_t(), qb_t()],
        );

        let control_op = MODIFIER_EXTENSION
            .instantiate_extension_op(
                &CONTROL_OP_ID,
                [
                    Term::BoundedNat(1),
                    Term::new_list([qb_t().into(), qb_t().into()]),
                    Term::new_list([]),
                ],
            )
            .unwrap_or_else(|e| panic!("Failed to instantiate control op: {}", e));

        // fn foo {
        //     -- • -- Y ---
        //        |
        //     -- X -- Z --
        // }
        let foo = {
            let mut func = module.define_function("foo", foo_sig.clone()).unwrap();
            let [in1, in2] = func.input_wires_arr();
            let [w1, w2] = func
                .add_dataflow_op(TketOp::CX, vec![in1, in2])
                .unwrap()
                .outputs_arr();
            let o1 = func
                .add_dataflow_op(TketOp::Y, vec![w1])
                .unwrap()
                .out_wire(0);
            let o2 = func
                .add_dataflow_op(TketOp::Z, vec![w2])
                .unwrap()
                .out_wire(0);
            func.finish_with_outputs(vec![o1, o2]).unwrap()
        };

        let _main = {
            let mut func = module.define_function("main", main_sig.clone()).unwrap();
            let [in1, in2, in3] = func.input_wires_arr();
            let loaded = func.load_func(foo.handle(), &[]).unwrap();
            let in1 = func.add_new_array(qb_t(), vec![in1]).unwrap();
            let controlled = func
                .add_dataflow_op(control_op, vec![loaded])
                .unwrap()
                .out_wire(0);
            let [out1, out2, out3] = func
                .add_dataflow_op(
                    CallIndirect {
                        signature: call_sig,
                    },
                    vec![controlled, in1, in2, in3],
                )
                .unwrap()
                .outputs_arr();
            let out1 = func.add_array_unpack(qb_t(), 1, out1).unwrap()[0];
            func.finish_with_outputs(vec![out1, out2, out3]).unwrap()
        };

        println!("Before modification:\n{}", module.hugr().mermaid_string());
        let mut h = module.finish_hugr().unwrap();
        println!("Before modification:\n{}", h.mermaid_string());

        let entrypoint = h.entrypoint().clone();
        resolve_modifier_with_entrypoints(&mut h, vec![entrypoint].into_iter()).unwrap();
        println!("After modification\n{}", h.mermaid_string());
    }

    #[test]
    fn test_control_ry_s() {
        let mut module = ModuleBuilder::new();
        let foo_sig = Signature::new(vec![qb_t(), qb_t()], vec![qb_t(), qb_t()]);
        let main_sig = Signature::new(vec![qb_t(), qb_t(), qb_t()], vec![qb_t(), qb_t(), qb_t()]);
        let call_sig = Signature::new(
            vec![array_type(1, qb_t()), qb_t(), qb_t()],
            vec![array_type(1, qb_t()), qb_t(), qb_t()],
        );

        let control_op = MODIFIER_EXTENSION
            .instantiate_extension_op(
                &CONTROL_OP_ID,
                [
                    Term::BoundedNat(1),
                    Term::new_list([qb_t().into(), qb_t().into()]),
                    Term::new_list([]),
                ],
            )
            .unwrap();

        // fn foo {
        //     -- • -- S ---
        //        |
        //     -- X -- Ry --
        //             ||
        //      0.5 ====
        // }
        let foo = {
            let mut func = module.define_function("foo", foo_sig.clone()).unwrap();
            let [mut i1, mut i2] = func.input_wires_arr();
            // [i1, i2] = func
            //     .add_dataflow_op(TketOp::CX, vec![i1, i2])
            //     .unwrap()
            //     .outputs_arr();
            // i1 = func
            //     .add_dataflow_op(TketOp::S, vec![i1])
            //     .unwrap()
            //     .out_wire(0);
            let theta = {
                let angle = ConstRotation::new(0.5).unwrap();
                func.add_load_value(angle)
            };
            i2 = func
                .add_dataflow_op(TketOp::Rz, vec![i2, theta])
                .unwrap()
                .out_wire(0);
            func.finish_with_outputs(vec![i1, i2]).unwrap()
        };

        let _main = {
            let mut func = module.define_function("main", main_sig.clone()).unwrap();
            let loaded = func.load_func(foo.handle(), &[]).unwrap();
            let controlled = func
                .add_dataflow_op(control_op, vec![loaded])
                .unwrap()
                .out_wire(0);
            // let theta = func.add_load_value(ConstRotation::new(0.75).unwrap());
            let mut fn_inputs = vec![controlled];
            let mut inputs = func.input_wires().collect::<Vec<_>>();
            inputs[0] = func.add_new_array(qb_t(), vec![inputs[0]]).unwrap();
            fn_inputs.extend(inputs);
            // inputs.push(theta);
            let mut outs = func
                .add_dataflow_op(
                    CallIndirect {
                        signature: call_sig,
                    },
                    fn_inputs,
                )
                .unwrap()
                .outputs()
                .collect::<Vec<_>>();
            outs[0] = func.add_array_unpack(qb_t(), 1, outs[0]).unwrap()[0];
            func.finish_with_outputs(outs).unwrap()
        };

        let mut h = module.finish_hugr().unwrap();
        println!("Before modification:\n{}", h.mermaid_string());

        let entrypoint = h.entrypoint().clone();
        resolve_modifier_with_entrypoints(&mut h, vec![entrypoint].into_iter()).unwrap();
        println!("After modification\n{}", h.mermaid_string());
        let env_format = EnvelopeFormat::PackageJson;
        let env_conf: EnvelopeConfig = EnvelopeConfig::new(env_format);
        let iter: Vec<Arc<Extension>> = vec![
            ROTATION_EXTENSION.to_owned(),
            TKET_EXTENSION.to_owned(),
            BOOL_EXTENSION.to_owned(),
        ];
        let regist: ExtensionRegistry = ExtensionRegistry::new(iter);
        println!(
            "hugr\n{}",
            h.store_str_with_exts(env_conf, &regist).unwrap()
        );
    }

    #[test]
    fn test_simple_dagger() {
        // Reversed Rz(θ) gate

        let mut module = ModuleBuilder::new();
        let foo_sig = Signature::new(vec![qb_t(), rotation_type()], vec![qb_t()]);
        let fn_sig = Signature::new(vec![qb_t()], vec![qb_t()]);

        let dagger_op = MODIFIER_EXTENSION
            .instantiate_extension_op(
                &DAGGER_OP_ID,
                [
                    Term::new_list([qb_t().into()]),
                    Term::new_list([rotation_type().into()]),
                ],
            )
            .unwrap();

        // fn foo {
        //     -- Rz -- S --
        //        ||
        //     ====
        // }
        let foo = {
            let mut func = module.define_function("foo", foo_sig.clone()).unwrap();
            let [in1, in2] = func.input_wires_arr();
            let rxgate = func
                .add_dataflow_op(TketOp::Rz, vec![in1, in2])
                .unwrap()
                .out_wire(0);
            let sgate = func.add_dataflow_op(TketOp::S, vec![rxgate]).unwrap();
            func.finish_with_outputs(sgate.outputs()).unwrap()
        };

        let _main = {
            let mut func = module.define_function("main", fn_sig.clone()).unwrap();
            let [in1] = func.input_wires_arr();
            let loaded = func.load_func(foo.handle(), &[]).unwrap();
            let theta = {
                let angle = ConstRotation::new(0.25).unwrap();
                func.add_load_value(angle)
            };
            let daggered = func
                .add_dataflow_op(dagger_op, vec![loaded])
                .unwrap()
                .out_wire(0);
            let [out1] = func
                .add_dataflow_op(
                    CallIndirect { signature: foo_sig },
                    vec![daggered, in1, theta],
                )
                .unwrap()
                .outputs_arr();
            func.finish_with_outputs(vec![out1]).unwrap()
        };

        let mut h = module.finish_hugr().unwrap();
        println!("Before modification:\n{}", h.mermaid_string());
        let entrypoint = h.entrypoint().clone();
        resolve_modifier_with_entrypoints(&mut h, vec![entrypoint].into_iter()).unwrap();
        println!("After modification\n{}", h.mermaid_string());
    }

    #[test]
    fn test_combined() {
        let mut module = ModuleBuilder::new();
        let foo_sig = Signature::new(
            vec![qb_t(), qb_t(), qb_t(), rotation_type()],
            vec![qb_t(), qb_t(), qb_t()],
        );
        let call_sig = Signature::new(
            vec![
                array_type(1, qb_t()),
                qb_t(),
                qb_t(),
                qb_t(),
                rotation_type(),
            ],
            vec![array_type(1, qb_t()), qb_t(), qb_t(), qb_t()],
        );
        let main_sig = Signature::new(
            vec![array_type(1, qb_t()), qb_t(), qb_t(), qb_t()],
            vec![array_type(1, qb_t()), qb_t(), qb_t(), qb_t()],
        );

        let control_op = MODIFIER_EXTENSION
            .instantiate_extension_op(
                &CONTROL_OP_ID,
                [
                    Term::BoundedNat(1),
                    Term::new_list([qb_t().into(), qb_t().into(), qb_t().into()]),
                    Term::new_list([rotation_type().into()]),
                ],
            )
            .unwrap();
        let dagger_op = MODIFIER_EXTENSION
            .instantiate_extension_op(
                &DAGGER_OP_ID,
                [
                    Term::new_list([
                        array_type(1, qb_t()).into(),
                        qb_t().into(),
                        qb_t().into(),
                        qb_t().into(),
                    ]),
                    Term::new_list([rotation_type().into()]),
                ],
            )
            .unwrap();

        // fn foo {
        //     -------- Ry -- V --
        //              ||
        //        0.5 ===
        //
        //     --- H --- S -------
        //
        //     --- Z ---- Rx -----
        //                ||
        //     ============
        // }
        let foo = {
            let mut func = module.define_function("foo", foo_sig.clone()).unwrap();
            let [mut in1, mut in2, mut in3, in4] = func.input_wires_arr();
            let theta = func.add_load_value(ConstRotation::new(0.46).unwrap());
            in1 = func
                .add_dataflow_op(TketOp::Ry, vec![in1, theta])
                .unwrap()
                .out_wire(0);
            in1 = func
                .add_dataflow_op(TketOp::V, vec![in1])
                .unwrap()
                .out_wire(0);
            in2 = func
                .add_dataflow_op(TketOp::H, vec![in2])
                .unwrap()
                .out_wire(0);
            in2 = func
                .add_dataflow_op(TketOp::S, vec![in2])
                .unwrap()
                .out_wire(0);
            in3 = func
                .add_dataflow_op(TketOp::Z, vec![in3])
                .unwrap()
                .out_wire(0);
            in3 = func
                .add_dataflow_op(TketOp::Rx, vec![in3, in4])
                .unwrap()
                .out_wire(0);
            func.finish_with_outputs(vec![in1, in2, in3]).unwrap()
        };

        let _main = {
            let mut func = module.define_function("main", main_sig.clone()).unwrap();
            let loaded = func.load_func(foo.handle(), &[]).unwrap();
            let controlled = func
                .add_dataflow_op(control_op, vec![loaded])
                .unwrap()
                .out_wire(0);
            let daggered = func
                .add_dataflow_op(dagger_op, vec![controlled])
                .unwrap()
                .out_wire(0);
            let theta = func.add_load_value(ConstRotation::new(0.75).unwrap());
            let mut inputs = vec![daggered];
            inputs.extend(func.input_wires());
            inputs.push(theta);
            let outs = func
                .add_dataflow_op(
                    CallIndirect {
                        signature: call_sig,
                    },
                    inputs,
                )
                .unwrap()
                .outputs();
            func.finish_with_outputs(outs).unwrap()
        };

        let mut h = module.finish_hugr().unwrap();
        println!("Before modification:\n{}", h.mermaid_string());

        let entrypoint = h.entrypoint().clone();
        resolve_modifier_with_entrypoints(&mut h, vec![entrypoint].into_iter()).unwrap();
        println!("After modification\n{}", h.mermaid_string());
    }

    #[test]
    fn test_cccx() {
        let mut module = ModuleBuilder::new();
        let t_num = 4;
        let c_num = 1;
        let targs = iter::repeat(qb_t()).take(t_num).collect::<Vec<_>>();
        let foo_sig = Signature::new_endo(targs.clone());
        let qubits = iter::repeat(qb_t()).take(c_num + t_num).collect::<Vec<_>>();
        let mut call_arg_ty = vec![array_type(c_num as u64, qb_t())];
        call_arg_ty.extend(iter::repeat(qb_t()).take(t_num));
        let call_sig = Signature::new_endo(call_arg_ty);
        let main_sig = Signature::new_endo(qubits);

        let control_op: ExtensionOp = {
            let term_list: Vec<Term> = targs.into_iter().map_into().collect();
            MODIFIER_EXTENSION
                .instantiate_extension_op(
                    &CONTROL_OP_ID,
                    [
                        Term::BoundedNat(c_num as u64),
                        Term::new_list(term_list),
                        Term::new_list([]),
                    ],
                )
                .unwrap()
        };

        // fn foo {
        //     ----•--------
        //         |
        //     ----•--------
        //         |
        //     ----X--------
        // }
        let foo = {
            let mut func = module.define_function("foo", foo_sig.clone()).unwrap();
            let mut inputs: Vec<Wire> = func.input_wires().collect();
            let (i1, i2, i3) = inputs.iter_mut().take(3).collect_tuple().unwrap();
            // let theta = func.add_load_value(ConstRotation::new(0.46).unwrap());
            [*i1, *i2, *i3] = func
                .add_dataflow_op(TketOp::Toffoli, vec![*i1, *i2, *i3])
                .unwrap()
                .outputs_arr();
            // [*i1] = func.add_dataflow_op(TketOp::V, input_wires)
            func.finish_with_outputs(inputs).unwrap()
        };

        let _main = {
            let mut func = module.define_function("main", main_sig).unwrap();
            let mut call = func.load_func(foo.handle(), &[]).unwrap();
            call = func
                .add_dataflow_op(control_op, vec![call])
                .unwrap()
                .out_wire(0);
            let mut inputs: Vec<_> = func.input_wires().collect();
            let targets = inputs.split_off(c_num);
            let control_arr = func.add_new_array(qb_t(), inputs).unwrap();
            let mut fn_inputs = vec![call, control_arr];
            fn_inputs.extend(targets);

            let mut outs = func
                .add_dataflow_op(
                    CallIndirect {
                        signature: call_sig,
                    },
                    fn_inputs,
                )
                .unwrap()
                .outputs();
            let control_arr = outs.next().unwrap();
            let mut controls = func
                .add_array_unpack(qb_t(), c_num as u64, control_arr)
                .unwrap();
            controls.extend(outs);

            func.finish_with_outputs(controls).unwrap()
        };

        let mut h = module.finish_hugr().unwrap();
        println!("Before modification:\n{}", h.mermaid_string());

        let entrypoint = h.entrypoint().clone();
        resolve_modifier_with_entrypoints(&mut h, vec![entrypoint].into_iter()).unwrap();
        println!("After modification\n{}", h.mermaid_string());
        {
            let f = File::create(Path::new("test_cccx.mermaid")).unwrap();
            let mut writer = std::io::BufWriter::new(f);
            write!(writer, "{}", h.mermaid_string()).unwrap();
        }
    }

    #[test]
    fn test_multi_control_ancilla() {
        let mut module = ModuleBuilder::new();
        let t_num = 1;
        let c_num = 7;
        let num = (t_num + c_num).try_into().unwrap();

        let targs = iter::repeat(qb_t()).take(t_num).collect::<Vec<_>>();
        let foo_sig = Signature::new_endo(targs.clone());
        // let qubits = iter::repeat(qb_t()).take(c_num + t_num).collect::<Vec<_>>();
        let mut call_arg_ty = vec![array_type(c_num as u64, qb_t())];
        call_arg_ty.extend(iter::repeat(qb_t()).take(t_num));
        let call_sig = Signature::new_endo(call_arg_ty);
        let main_sig = Signature::new(type_row![], array_type(num, qb_t()));

        let control_op: ExtensionOp = {
            let term_list: Vec<Term> = targs.into_iter().map_into().collect();
            MODIFIER_EXTENSION
                .instantiate_extension_op(
                    &CONTROL_OP_ID,
                    [
                        Term::BoundedNat(c_num as u64),
                        Term::new_list(term_list),
                        Term::new_list([]),
                    ],
                )
                .unwrap()
        };

        let foo = {
            let mut func = module.define_function("foo", foo_sig.clone()).unwrap();
            let mut inputs: Vec<Wire> = func.input_wires().collect();
            // let (i1, i2, i3) = inputs.iter_mut().take(t_num).collect_tuple().unwrap();
            let (i1,) = inputs.iter_mut().take(t_num).collect_tuple().unwrap();
            // let theta = func.add_load_value(ConstRotation::new(1.0).unwrap());
            // [*i1, *i2, *i3] = func
            //     .add_dataflow_op(TketOp::Toffoli, vec![*i1, *i2, *i3])
            //     .unwrap()
            //     .outputs_arr();
            [*i1] = func
                .add_dataflow_op(TketOp::X, vec![*i1])
                .unwrap()
                .outputs_arr();
            func.finish_with_outputs(inputs).unwrap()
        };

        let _main = {
            let mut func = module.define_function("main", main_sig).unwrap();
            let mut call = func.load_func(foo.handle(), &[]).unwrap();
            call = func
                .add_dataflow_op(control_op, vec![call])
                .unwrap()
                .out_wire(0);

            let mut controls = Vec::new();
            for _ in 0..c_num {
                controls.push({
                    let mut q = func
                        .add_dataflow_op(TketOp::QAlloc, vec![])
                        .unwrap()
                        .out_wire(0);
                    q = func
                        .add_dataflow_op(TketOp::H, vec![q])
                        .unwrap()
                        .out_wire(0);
                    q = func
                        .add_dataflow_op(TketOp::X, vec![q])
                        .unwrap()
                        .out_wire(0);
                    q
                });
            }

            let mut targets = Vec::new();
            for _ in 0..t_num {
                targets.push({
                    let mut q = func
                        .add_dataflow_op(TketOp::QAlloc, vec![])
                        .unwrap()
                        .out_wire(0);
                    let theta = func.add_load_value(ConstRotation::new(0.29).unwrap());
                    q = func
                        .add_dataflow_op(TketOp::Ry, vec![q, theta])
                        .unwrap()
                        .out_wire(0);
                    q
                })
            }
            for i in 0..c_num {
                [controls[i], targets[t_num - 1]] = func
                    .add_dataflow_op(TketOp::CX, vec![controls[i], targets[t_num - 1]])
                    .unwrap()
                    .outputs_arr();
            }

            let mut init_state = controls;
            init_state.extend(targets);
            let init_state_arr = func.add_new_array(qb_t(), init_state).unwrap();
            let state_result = StateResult::new("input_state".to_string(), num);
            let init_state_arr = func
                .add_dataflow_op(state_result, vec![init_state_arr])
                .unwrap()
                .out_wire(0);
            let mut controls = func.add_array_unpack(qb_t(), num, init_state_arr).unwrap();
            let mut fn_inputs = controls.split_off(c_num);
            let control_arr = func.add_new_array(qb_t(), controls).unwrap();
            fn_inputs.insert(0, control_arr);
            fn_inputs.insert(0, call);

            let mut fn_outs = func
                .add_dataflow_op(
                    CallIndirect {
                        signature: call_sig,
                    },
                    fn_inputs,
                )
                .unwrap()
                .outputs();

            let control_arr = fn_outs.next().unwrap();
            let mut outputs = func
                .add_array_unpack(qb_t(), c_num as u64, control_arr)
                .unwrap();
            outputs.extend(fn_outs);
            let out_array = func.add_new_array(qb_t(), outputs).unwrap();
            let state_result = StateResult::new("output_state".to_string(), num);
            let out_array = func
                .add_dataflow_op(state_result, vec![out_array])
                .unwrap()
                .outputs();

            func.finish_with_outputs(out_array).unwrap()
        };

        let mut h = module.finish_hugr().unwrap();
        h.validate().unwrap();
        println!("Before modification:\n{}", h.mermaid_string());

        let entrypoint = h.entrypoint().clone();
        resolve_modifier_with_entrypoints(&mut h, vec![entrypoint].into_iter()).unwrap();
        dead_code::DeadCodeElimPass::default()
            .with_entry_points(vec![_main.node()])
            .run(&mut h)
            .unwrap();
        h.validate().unwrap();
        println!("After modification\n{}", h.mermaid_string());
        {
            let f = File::create(Path::new("test_multi_control_ancilla.mermaid")).unwrap();
            let mut writer = std::io::BufWriter::new(f);
            write!(writer, "{}", h.mermaid_string()).unwrap();
        }
        let env_format = EnvelopeFormat::PackageJson;
        let env_conf: EnvelopeConfig = EnvelopeConfig::new(env_format);
        let iter: Vec<Arc<Extension>> = vec![
            ROTATION_EXTENSION.to_owned(),
            TKET_EXTENSION.to_owned(),
            BOOL_EXTENSION.to_owned(),
        ];
        let regist: ExtensionRegistry = ExtensionRegistry::new(iter);
        let f = File::create(Path::new("test_multi_control_ancilla.json")).unwrap();
        let writer = std::io::BufWriter::new(f);
        h.store_with_exts(writer, env_conf, &regist).unwrap();
        // println!(
        //     "hugr\n{}",
        //     h.store_str_with_exts(env_conf, &regist).unwrap()
        // );
    }
}
