from tket2_exts import (
    opaque_bool,
    debug,
    guppy,
    rotation,
    futures,
    gpu,
    qsystem,
    qsystem_random,
    qsystem_utils,
    quantum,
    result,
    wasm,
)

import pytest


exts = [
    opaque_bool,
    debug,
    guppy,
    rotation,
    futures,
    gpu,
    qsystem,
    qsystem_random,
    qsystem_utils,
    quantum,
    result,
    wasm,
]


@pytest.mark.parametrize("ext", exts)
def test_validate_extension(ext):
    ext()
    return
