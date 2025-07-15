from pathlib import Path
from sys import argv
from hugr.envelope import EnvelopeConfig, EnvelopeFormat

from guppylang import guppy
from guppylang.std.quantum import measure, qubit


@guppy
def main() -> None:
    q1, q2, q3 = qubit(), qubit(), qubit()
    measure(q1)
    measure(q2)
    measure(q3)


program = main.compile().package
config = EnvelopeConfig(format=EnvelopeFormat.MODULE, zstd=0)
Path(argv[0]).with_suffix(".hugr").write_bytes(program.to_bytes(config))
