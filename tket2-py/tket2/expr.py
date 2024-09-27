from __future__ import annotations
from typing import cast, List, Dict
from hugr.hugr import Hugr, Wire, OutPort
from hugr import ops
from hugr.std import _load_extension
from hugr.build import DfBase, Dfg
from hugr.std.float import FloatVal, FLOAT_T
from sympy.printing import StrPrinter
from sympy.core import Add, Mul, Symbol
from sympy.core.numbers import Integer, Number
import sympy


FLOAT_EXT = _load_extension("arithmetic.float")

def float_op(name: str) -> ops.ExtOp:
    return ops.ExtOp(FLOAT_EXT.get_op(name))

class SympyBuilder(Dfg):
    vars_to_wire: Dict[str, OutPort]
    def __init__(self, vars: List[str], **kwargs):
        input_tys = [FLOAT_T for _ in vars]
        super().__init__(*input_tys, **kwargs)
        self.vars_to_wire = { var:  OutPort(node=self.input_node, offset=i) for (i, var) in enumerate(vars) }

    def expr_wire(self, expr: sympy.Expr) -> Wire:
        if isinstance(expr, Symbol):
            return self.vars_to_wire[str(expr)]
        if isinstance(expr, Add):
            w1 = self.expr_wire(expr.args[0])
            w2 = self.expr_wire(expr.args[1])
            return self.add_op(float_op("fadd"), w1, w2)
        elif isinstance(expr, Mul):
            w1 = self.expr_wire(expr.args[0])
            w2 = self.expr_wire(expr.args[1])
            return self.add_op(float_op("fmul"), w1, w2)
        elif isinstance(expr, Number):
            return self.load(FloatVal(float(expr)))
        else:
            raise NotImplementedError(f"Unsupported sympy func: {expr}")





def parse_expr(expr: str, vars: List[str]) -> str:
    sym_expr = sympy.parse_expr(expr, evaluate=False)
    builder = SympyBuilder(vars)
    out = builder.expr_wire(sym_expr)
    builder.set_outputs(out)
    return builder.hugr.to_json()
