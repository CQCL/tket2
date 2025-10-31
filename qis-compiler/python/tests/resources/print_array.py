# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "guppylang ==0.21.6",
# ]
# ///

from pathlib import Path

from guppylang import guppy
from guppylang.std.builtins import array, result
from guppylang.std.quantum import measure_array, qubit, x


@guppy
def main() -> None:
    qs = array(qubit() for _ in range(10))
    x(qs[0])
    x(qs[2])
    x(qs[3])
    x(qs[9])
    cs = measure_array(qs)
    result("cs", cs)
    result("is", array(i for i in range(100)))
    result("fs", array(i * 0.0625 for i in range(100)))

program = main.compile()
Path(__file__).with_suffix(".hugr").write_bytes(program.to_bytes())