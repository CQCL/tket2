from pathlib import Path
from sys import argv
from hugr.envelope import EnvelopeConfig, EnvelopeFormat

from guppylang import guppy
from guppylang.std.angles import angle
from guppylang.std.builtins import owned
from guppylang.std.quantum import rz, h, measure, qubit


@guppy
def main() -> None:
    q = qubit()
    q = f(q)
    measure(q)


@guppy.comptime
def g(q: qubit, x: angle) -> None:
    rz(q, x / 2 - angle(0.1))
    rz(q, x / 2 + angle(0.1))


@guppy.comptime
def f(q: qubit @ owned) -> qubit:
    h(q)
    x: angle = angle(0.0)
    for i in range(5):
        g(q, x)
        x += angle(0.2)
    h(q)
    return q


program = main.compile()
config = EnvelopeConfig(format=EnvelopeFormat.MODEL_WITH_EXTS, zstd=0)
Path(argv[0]).with_suffix(".hugr").write_bytes(program.to_bytes(config))
