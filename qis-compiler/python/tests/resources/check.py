# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "guppylang ==0.21.6",
# ]
# ///

from pathlib import Path

from guppylang import guppy
from guppylang.std.quantum import discard, qubit


@guppy
def main() -> None:
    q = qubit()
    discard(q)

program = main.compile()
Path(__file__).with_suffix(".hugr").write_bytes(program.to_bytes())