from guppylang import guppy
from guppylang.std.quantum import rz, qubit
from guppylang.std.builtins import angle


@guppy
def rz_chain(q: qubit) -> None:
    rz(q, angle(3 / 2))
