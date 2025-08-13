from pathlib import Path
from sys import argv

from guppylang import guppy


@guppy
def main() -> None:
    pass


program = main.compile()
Path(argv[0]).with_suffix(".hugr").write_bytes(program.to_bytes())
