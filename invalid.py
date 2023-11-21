from pytket import Circuit
import json
import numpy as np

from tket2.passes import greedy_depth_reduce


def compare_unitaries(a, b) -> bool:
    """Compares two unitaries for equality, up to a global phase"""
    if a[0, 0] != b[0, 0]:
        # Ignore the global phase
        angle = np.angle(b[0, 0] / a[0, 0])
        b = np.exp(-1j * angle) * b

    return np.allclose(a, b)


if __name__ == "__main__":
    # All of these fail the check
    files = [
        "fails/01e360c0-d669-11ea-8a28-38f9d36dfbf2-pre.json",
        "fails/3b9bab70-d668-11ea-a4be-38f9d36dfbf2-pre.json",
        "fails/3b6260d6-d668-11ea-a4be-38f9d36dfbf2-pre.json",
        "fails/3d13dc5c-d668-11ea-a4be-38f9d36dfbf2-pre.json",
        "fails/3d7242ec-d668-11ea-a4be-38f9d36dfbf2-pre.json",
    ]
    invalid_file = files[0]

    print("Loading circuit")
    # invalid_circ = circuit_from_qasm(invalid_file)
    with open(invalid_file, "r") as f:
        circ = Circuit.from_dict(json.load(f))

    circ_unitary = circ.get_unitary()

    print("Starting optimisation")
    (circ, _) = greedy_depth_reduce(circ)

    assert compare_unitaries(circ.get_unitary(), circ_unitary)
