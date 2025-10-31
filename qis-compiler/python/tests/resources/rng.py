# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "guppylang ==0.21.6",
# ]
# ///

from pathlib import Path

from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.qsystem.random import RNG


@guppy
def main() -> None:
    rng = RNG(42)
    rint = rng.random_int()
    rint1 = rng.random_int()
    rfloat = rng.random_float()
    rint_bnd = rng.random_int_bounded(100)
    rng.discard()
    result("rint", rint)
    result("rint1", rint1)
    result("rfloat", rfloat)
    result("rint_bnd", rint_bnd)
    rng = RNG(84)
    rint = rng.random_int()
    rfloat = rng.random_float()
    rint_bnd = rng.random_int_bounded(200)
    rng.discard()
    result("rint2", rint)
    result("rfloat2", rfloat)
    result("rint_bnd2", rint_bnd)


program = main.compile()
Path(__file__).with_suffix(".hugr").write_bytes(program.to_bytes())
