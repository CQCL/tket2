# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "guppylang >=0.21.3",
# ]
# ///

from pathlib import Path
from sys import argv

from guppylang import guppy
from guppylang.std.builtins import result


@guppy
def main() -> None:
    result("b1", 0)
    result("b2", 0)


program = main.compile()
Path(argv[0]).with_suffix(".hugr").write_bytes(program.to_bytes())
