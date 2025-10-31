# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "guppylang ==0.21.6",
# ]
# ///

from pathlib import Path

from guppylang import guppy
from guppylang.std.quantum import h, measure, qubit


@guppy
def bar() -> None:
    q0 = qubit()
    h(q0)
    measure(q0)

program = bar.compile()
Path(__file__).with_suffix(".hugr").write_bytes(program.to_bytes())