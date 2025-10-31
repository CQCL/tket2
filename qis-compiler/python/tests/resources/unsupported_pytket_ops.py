# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "guppylang ==0.21.6",
#     "tket",
# ]
# ///

from pathlib import Path

from guppylang import guppy
from guppylang.std.quantum import discard, qubit
from pytket._tket.circuit import Circuit

circ = Circuit(2).CSXdg(0, 1)
guppy_circ = guppy.load_pytket("guppy_circ", circ, use_arrays=False)


@guppy
def main() -> None:
    a, b = qubit(), qubit()
    guppy_circ(a, b)
    discard(a)
    discard(b)


program = main.compile()
Path(__file__).with_suffix(".hugr").write_bytes(program.to_bytes())
