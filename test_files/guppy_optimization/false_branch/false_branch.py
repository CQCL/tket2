from pathlib import Path
from sys import argv
from hugr.envelope import EnvelopeConfig, EnvelopeFormat

from guppylang import guppy
from guppylang.std.quantum import h, measure, qubit


@guppy
def main() -> None:
    q = qubit()
    h(q)
    if True:
        h(q)
    if False:
        h(q)
    measure(q)


program = main.compile().package
config = EnvelopeConfig(format=EnvelopeFormat.MODULE_WITH_EXTS, zstd=0)
Path(argv[0]).with_suffix(".hugr").write_bytes(program.to_bytes(config))
