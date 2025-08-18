"""HUGR extension definitions for tket circuits."""

from tket_exts.tket._util import TketExtension
from typing_extensions import deprecated
from hugr.ext import Extension
from tket_exts import tket

# This is updated by our release-please workflow, triggered by this
# annotation: x-release-please-version
__version__ = "0.10.1"

__all__ = [
    "bool",
    "debug",
    "guppy",
    "rotation",
    "futures",
    "qsystem",
    "qsystem_random",
    "qsystem_utils",
    "quantum",
    "result",
    "wasm",
]

bool: TketExtension = tket.bool.BoolExtension()
debug: TketExtension = tket.debug.DebugExtension()
guppy: TketExtension = tket.guppy.GuppyExtension()
rotation: TketExtension = tket.rotation.RotationExtension()
futures: TketExtension = tket.futures.FuturesExtension()
qsystem: TketExtension = tket.qsystem.QSystemExtension()
qsystem_random: TketExtension = tket.qsystem.QSystemRandomExtension()
qsystem_utils: TketExtension = tket.qsystem.QSystemUtilsExtension()
quantum: TketExtension = tket.quantum.QuantumExtension()
result: TketExtension = tket.result.ResultExtension()
wasm: TketExtension = tket.wasm.WasmExtension()


@deprecated("Use tket_exts.bool() instead")
def opaque_bool() -> Extension:
    return bool()
