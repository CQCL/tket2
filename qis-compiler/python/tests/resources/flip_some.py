# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "guppylang ==0.21.6",
# ]
# ///

from pathlib import Path

from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import measure, qubit, x


@guppy
def main() -> None:
    q0: qubit = qubit()
    q1: qubit = qubit()
    q2: qubit = qubit()
    q3: qubit = qubit()
    x(q0)
    x(q2)
    x(q3)
    result("c0", measure(q0))
    result("c1", measure(q1))
    result("c2", measure(q2))
    result("c3", measure(q3))

program = main.compile()
Path(__file__).with_suffix(".hugr").write_bytes(program.to_bytes())