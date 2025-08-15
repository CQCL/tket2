"""HUGR extension definitions for tket circuits."""

from typing_extensions import deprecated
from hugr.ext import Extension
from tket_exts import tket

# This is updated by our release-please workflow, triggered by this
# annotation: x-release-please-version
__version__ = "0.10.1"


@deprecated("Use tket_exts.tket.bool.extension() instead")
def opaque_bool() -> Extension:
    return tket.bool.extension()


@deprecated("Use tket_exts.tket.debug.extension() instead")
def debug() -> Extension:
    return tket.debug.extension()


@deprecated("Use tket_exts.tket.guppy.extension() instead")
def guppy() -> Extension:
    return tket.guppy.extension()


@deprecated("Use tket_exts.tket.rotation.extension() instead")
def rotation() -> Extension:
    return tket.rotation.extension()


@deprecated("Use tket_exts.tket.futures.extension() instead")
def futures() -> Extension:
    return tket.futures.extension()


@deprecated("Use tket_exts.tket.qsystem.extension() instead")
def qsystem() -> Extension:
    return tket.qsystem.extension()


@deprecated("Use tket_exts.tket.qsystem.random.extension() instead")
def qsystem_random() -> Extension:
    return tket.qsystem.random.extension()


@deprecated("Use tket_exts.tket.qsystem.utils.extension() instead")
def qsystem_utils() -> Extension:
    return tket.qsystem.utils.extension()


@deprecated("Use tket_exts.tket.quantum.extension() instead")
def quantum() -> Extension:
    return tket.quantum.extension()


@deprecated("Use tket_exts.tket.result.extension() instead")
def result() -> Extension:
    return tket.result.extension()


@deprecated("Use tket_exts.tket.wasm.extension() instead")
def wasm() -> Extension:
    return tket.wasm.extension()
