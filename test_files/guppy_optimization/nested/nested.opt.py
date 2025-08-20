# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "guppylang >=0.21.3",
# ]
# ///

from pathlib import Path
from sys import argv

from guppylang import guppy
from guppylang.std.quantum import measure, qubit


@guppy
def main() -> None:
    q1, q2, q3 = qubit(), qubit(), qubit()
    measure(q1)
    measure(q2)
    measure(q3)


program = main.compile()
Path(argv[0]).with_suffix(".hugr").write_bytes(program.to_bytes())
