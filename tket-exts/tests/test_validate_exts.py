from typing import Callable
from hugr.ext import Extension
import tket_exts.tket as tket

import pytest


exts = [
    tket.bool.extension,
    tket.debug.extension,
    tket.guppy.extension,
    tket.futures.extension,
    tket.qsystem.extension,
    tket.qsystem.random.extension,
    tket.qsystem.utils.extension,
    tket.quantum.extension,
    tket.result.extension,
    tket.rotation.extension,
    tket.wasm.extension,
]


@pytest.mark.parametrize("ext", exts)
def test_validate_extension(ext: Callable[[], Extension]):
    e = ext()
    assert len(e.types) + len(e.operations) > 0
    return
