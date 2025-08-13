from pathlib import Path
from sys import argv
from hugr.envelope import EnvelopeConfig, EnvelopeFormat

from guppylang import guppy
from guppylang.std.builtins import array, owned
from guppylang.std.quantum import cz, h, measure, qubit


@guppy
def main() -> None:
    q1, q2, q3 = qubit(), qubit(), qubit()
    q1, q2, q3 = f(array(q1, q2, q3))
    measure(q1)
    measure(q2)
    measure(q3)


@guppy.comptime
def f(qs: array[qubit, 3] @ owned) -> array[qubit, 3]:
    for i in range(3):
        h(qs[i])
    for i in range(2):
        for j in range(3):
            cz(qs[j], qs[(j + 1) % 3])
    for i in range(3):
        h(qs[i])
    return qs


program = main.compile()
config = EnvelopeConfig(format=EnvelopeFormat.MODEL_WITH_EXTS, zstd=0)
Path(argv[0]).with_suffix(".hugr").write_bytes(program.to_bytes(config))
