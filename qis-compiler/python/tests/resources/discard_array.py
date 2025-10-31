# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "guppylang ==0.21.6",
# ]
# ///

from pathlib import Path

from guppylang import guppy
from guppylang.std.builtins import array
from guppylang.std.quantum import discard_array, qubit


@guppy
def main() -> None:
    qs = array(qubit() for _ in range(10))
    discard_array(qs)

program = main.compile()
Path(__file__).with_suffix(".hugr").write_bytes(program.to_bytes())