from pathlib import Path
from sys import argv

from guppylang import guppy
from guppylang.std.builtins import array, owned
from guppylang.std.quantum import cx, h, measure, qubit


@guppy
def main() -> None:
    q1, q2, q3, q4 = qubit(), qubit(), qubit(), qubit()
    q1, q2, q3, q4 = f(array(q1, q2, q3, q4))
    measure(q1)
    measure(q2)
    measure(q3)
    measure(q4)


@guppy
def f(qs: array[qubit, 4] @ owned) -> array[qubit, 4]:
    for i in range(4):
        h(qs[(-i) % 4])
    for i in range(3):
        cx(qs[i], qs[i + 1])
    for i in range(3):
        cx(qs[2 - i], qs[3 - i])
    for i in range(4):
        h(qs[(3 * i + 1) % 4])
    return qs


program = main.compile()
Path(argv[0]).with_suffix(".hugr").write_bytes(program.to_bytes())
