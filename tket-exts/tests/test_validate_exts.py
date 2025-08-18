from typing import List
import tket_exts

import pytest
from tket_exts.tket._util import TketExtension


exts: List[TketExtension] = [
    tket_exts.bool,
    tket_exts.debug,
    tket_exts.guppy,
    tket_exts.futures,
    tket_exts.qsystem,
    tket_exts.qsystem_random,
    tket_exts.qsystem_utils,
    tket_exts.quantum,
    tket_exts.result,
    tket_exts.rotation,
    tket_exts.wasm,
]


@pytest.mark.parametrize("ext", exts)
def test_validate_extension(ext: TketExtension):
    e = ext()
    assert len(e.types) + len(e.operations) > 0

    ops = ext.OPS()
    assert len(ops) == len(e.operations)

    types = ext.TYPES()
    assert len(types) == len(e.types)

    return
