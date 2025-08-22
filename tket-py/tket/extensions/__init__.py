from tket_exts import (
    opaque_bool,
    debug,
    guppy,
    rotation,
    futures,
    qsystem,
    qsystem_random,
    qsystem_utils,
    quantum,
    result,
    wasm,
)

# TODO: Remove once tket no longer supports tket-exts 0.10.*
try:
    from tket_exts import bool
except ImportError:
    bool = opaque_bool  # type: ignore[assignment]


# TODO: Remove the deprecated `opaque_bool` export in a breaking change.
__all__ = [
    "debug",
    "bool",
    "opaque_bool",
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
