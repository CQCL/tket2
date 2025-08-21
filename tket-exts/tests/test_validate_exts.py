from tket_exts import (
    debug,
    futures,
    gpu,
    guppy,
    opaque_bool,
    qsystem,
    qsystem_random,
    qsystem_utils,
    quantum,
    result,
    rotation,
    wasm,
)

import pytest


exts = [
    debug,
    futures,
    gpu,
    guppy,
    opaque_bool,
    qsystem,
    qsystem_random,
    qsystem_utils,
    quantum,
    result,
    rotation,
    wasm,
]


@pytest.mark.parametrize("ext", exts)
def test_validate_extension(ext):
    e = ext()
    assert len(e.types) + len(e.operations) > 0
    return
