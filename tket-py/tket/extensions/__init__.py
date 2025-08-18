from tket_exts import (
    tket,
    opaque_bool,
    bool,
    debug,
    guppy,
    rotation,
    futures,
    qsystem,
    quantum,
    result,
    wasm,
)


# TODO: Remove the deprecated `opaque_bool` export in a breaking change.
__all__ = [
    "tket",
    "debug",
    "bool",
    "opaque_bool",
    "guppy",
    "rotation",
    "futures",
    "qsystem",
    "quantum",
    "result",
    "wasm",
]
