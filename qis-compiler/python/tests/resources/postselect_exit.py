# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "guppylang ==0.21.6",
# ]
# ///

from pathlib import Path

from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import h, measure, qubit


@guppy
def main() -> None:
    q = qubit()
    h(q)
    outcome = measure(q)
    if outcome:
        exit("Postselection failed", 42)
    result("c", outcome)

program = main.compile()
Path(__file__).with_suffix(".hugr").write_bytes(program.to_bytes())