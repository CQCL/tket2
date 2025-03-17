"""Some utility functions for the example notebooks."""

from hugr import Hugr
from hugr.envelope import EnvelopeConfig
from tket2.passes import lower_to_pytket
from tket2.circuit import Tk2Circuit
from typing import Any


def setup_jupyter_rendering():
    """Set up hugr rendering for Jupyter notebooks."""

    def _repr_hugr(
        h: Hugr, include=None, exclude=None, **kwargs
    ) -> dict[str, bytes | str]:
        return h.render_dot()._repr_mimebundle_(include, exclude, **kwargs)

    def _repr_tk2circ(
        circ: Tk2Circuit, include=None, exclude=None, **kwargs
    ) -> dict[str, bytes | str]:
        h = Hugr.load_json(circ.to_hugr_json())
        return _repr_hugr(h, include, exclude, **kwargs)

    setattr(Hugr, "_repr_mimebundle_", _repr_hugr)
    setattr(Tk2Circuit, "_repr_mimebundle_", _repr_tk2circ)


# TODO: Should this be part of the guppy API? Or tket2?
# Takes a RawFunctionDef and converts it to a Tk2Circuit
def guppy_to_circuit(func_def: Any) -> Tk2Circuit:
    """Convert a Guppy function definition to a `Tk2Circuit`."""
    module = func_def.id.module
    assert module is not None, "Function definition must belong to a module"

    pkg = module.compile()

    json = pkg.package.to_str(EnvelopeConfig.TEXT)
    circ = Tk2Circuit.from_str(json, func_def.name)

    return lower_to_pytket(circ)
