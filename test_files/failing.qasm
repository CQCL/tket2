OPENQASM 2.0;
include "qelib1.inc";

qreg q[5];
cx q[4],q[1];
cx q[3],q[4];
cx q[4],q[0];
cx q[0],q[2];
cx q[1],q[2];
u3(0.5*pi,0.0*pi,0.5*pi) q[1];
cx q[3],q[1];
cx q[0],q[2];