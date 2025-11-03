from guppylang import guppy
from guppy.std.quantum import qubit
from guppylang.std.qsystem import rz, phased_x
from guppylang.std.builtins import angle


@guppy
def qsystem_chain(q: qubit) -> None:
    phased_x(q, angle(0.368713), angle(1.66415))
    rz(q, angle(0.870616))
