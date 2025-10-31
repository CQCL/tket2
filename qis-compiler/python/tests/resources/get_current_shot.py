# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "guppylang ==0.21.6",
# ]
# ///

from pathlib import Path

from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.qsystem.utils import get_current_shot


@guppy
def main() -> None:
    result("shot", get_current_shot())


program = main.compile()
Path(__file__).with_suffix(".hugr").write_bytes(program.to_bytes())
