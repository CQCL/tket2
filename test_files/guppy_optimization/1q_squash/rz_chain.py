from pathlib import Path
from sys import argv

from guppylang import guppy
from guppylang.std.quantum import rz, qubit
from guppylang.std.builtins import angle


@guppy
def rz_chain(q: qubit) -> None:
    rz(q, angle(1 / 2))
    rz(q, angle(1 / 2))
    rz(q, angle(1 / 2))


program = rz_chain.compile()
Path(argv[0]).with_suffix(".hugr").write_bytes(program.to_bytes())
