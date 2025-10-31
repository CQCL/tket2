# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "guppylang ==0.21.6",
# ]
# ///

from pathlib import Path

from guppylang import guppy
from guppylang.std.builtins import result


@guppy
def foo(a: int) -> None:
    result("a", a)


program = foo.compile_function()
Path(__file__).with_suffix(".hugr").write_bytes(program.to_bytes())
