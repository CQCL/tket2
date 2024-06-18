from guppylang.definition.function import RawFunctionDef

from tket2.circuit import Tk2Circuit


def guppy_to_circuit(func_def: RawFunctionDef) -> Tk2Circuit:
    """Convert a Guppy function definition to a `Tk2Circuit`."""
    module = func_def.id.module
    hugr = module.compile()
    json = hugr.to_raw().to_json()
    return Tk2Circuit.from_guppy_json(json, func_def.name)
