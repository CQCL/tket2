# Re-export native bindings
from ._tket2.optimiser import BadgerOptimiser, PortDiffGraph
from ._tket2.circuit import Tk2Circuit

from z3 import Bool, Optimize, Implies, Not, And, sat

__all__ = ["BadgerOptimiser", "PortDiffGraph"]


def construct_z3_optimiser(diffs: PortDiffGraph, exclude_cycles=None):
    """
    Encode the PortDiffGraph into a z3 Optimize object.
    """
    bool_array = [Bool(f"rw_{i}") for i in range(diffs.n_diffs())]

    s = Optimize()

    # Add soft constraint on each node
    for i in range(diffs.n_diffs()):
        b = bool_array[i]
        s.add_soft(b, diffs.value(i))

    edges = diffs.all_edges()
    for i in range(len(edges)):
        (src, tgt, nodes) = edges[i]
        src_bool = bool_array[src]
        tgt_bool = bool_array[tgt]

        # Add tgt implies src
        s.add(Implies(tgt_bool, src_bool))

        # Add mutual exclusion constraints
        for j in range(i + 1, len(edges)):
            (src2, tgt2, nodes2) = edges[j]
            if src == src2:
                if nodes & nodes2:  # Check if nodes intersect
                    s.add(Not(And(bool_array[tgt], bool_array[tgt2])))

    for rws in exclude_cycles:
        s.add(Not(And(*(bool_array[rw] for rw in rws))))
    return s


def extract_optimal_circuit(diffs: PortDiffGraph) -> Tk2Circuit:
    """
    Find the optimal solution to the PortDiffGraph using z3.
    """
    bool_array = [Bool(f"rw_{i}") for i in range(diffs.n_diffs())]

    solution = None
    exclude_cycles: list[list[int]] = []

    while solution is None:
        s = construct_z3_optimiser(diffs, exclude_cycles)

        assert s.check() == sat
        model = s.model()
        selected = [i for (i, b) in enumerate(bool_array) if model[b]]
        try:
            solution = diffs.extract_circuit(selected)
            print(f"Selected rewrites: {selected}")
        except Exception:
            solution = None
            exclude_cycles.append(selected)
    return solution
