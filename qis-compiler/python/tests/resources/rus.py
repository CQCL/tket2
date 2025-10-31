# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "guppylang ==0.21.6",
# ]
# ///

from pathlib import Path

from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import cx, discard, h, measure, qubit, t, tdg, x, z


@guppy
def rus(q: qubit) -> None:
    while True:
        # Prepare ancillary qubits
        a, b = qubit(), qubit()
        h(a)
        h(b)

        tdg(a)
        cx(b, a)
        t(a)
        if not measure(a):
            # First part failed; try again
            discard(b)
            continue

        t(q)
        z(q)
        cx(q, b)
        t(b)
        if measure(b):
            # Success, we are done
            break

        # Otherwise, apply correction
        x(q)


@guppy
def main() -> None:
    q = qubit()
    rus(q)
    result("result", measure(q))


program = main.compile()
Path(__file__).with_suffix(".hugr").write_bytes(program.to_bytes())
