from guppylang.definition.function import RawFunctionDef

from tket2.circuit import Tk2Circuit


def guppy_to_circuit(func_def: RawFunctionDef) -> Tk2Circuit:
    """Convert a Guppy function definition to a `Tk2Circuit`."""
    module = func_def.id.module
    assert module is not None, "Function definition must belong to a module"

    hugr = module.compile()
    assert hugr is not None, "Module must be compilable"

    json = hugr.to_raw().to_json()
    return Tk2Circuit.from_guppy_json(json, func_def.name)
