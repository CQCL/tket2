from pathlib import Path
from sys import argv

from guppylang import guppy
from guppylang.std.quantum import cx, measure, qubit


@guppy
def main() -> None:
    q1, q2 = qubit(), qubit()
    cx(q1, q2)
    cx(q1, q2)
    measure(q1)
    measure(q2)


program = main.compile()
Path(argv[0]).with_suffix(".hugr").write_bytes(program.to_bytes())
