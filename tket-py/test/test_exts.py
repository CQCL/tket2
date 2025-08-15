from hugr.ext import Extension
from tket.extensions import tket


def test_exts():
    exts = [
        tket.bool.extension(),
        tket.debug.extension(),
        tket.rotation.extension(),
        tket.guppy.extension(),
        tket.futures.extension(),
        tket.qsystem.extension(),
        tket.qsystem.random.extension(),
        tket.qsystem.utils.extension(),
        tket.quantum.extension(),
        tket.result.extension(),
        tket.wasm.extension(),
    ]

    for ext in exts:
        assert isinstance(ext, Extension)
