from pathlib import Path
from sys import argv
from hugr.envelope import EnvelopeConfig, EnvelopeFormat

from guppylang import guppy
from guppylang.std.quantum import measure, qubit


@guppy
def main() -> None:
    q = qubit()
    measure(q)


program = main.compile()
config = EnvelopeConfig(format=EnvelopeFormat.MODEL_WITH_EXTS, zstd=0)
Path(argv[0]).with_suffix(".hugr").write_bytes(program.to_bytes(config))
