"""HUGR extension definitions for tket circuits."""

import pkgutil
import functools

from hugr.ext import Extension


# This is updated by our release-please workflow, triggered by this
# annotation: x-release-please-version
__version__ = "0.9.2"


@functools.cache
def opaque_bool() -> Extension:
    return load_extension("tket.bool")


@functools.cache
def debug() -> Extension:
    return load_extension("tket.debug")


@functools.cache
def guppy() -> Extension:
    return load_extension("tket.guppy")


@functools.cache
def rotation() -> Extension:
    return load_extension("tket.rotation")


@functools.cache
def futures() -> Extension:
    return load_extension("tket.futures")


@functools.cache
def qsystem() -> Extension:
    return load_extension("tket.qsystem")


@functools.cache
def qsystem_random() -> Extension:
    return load_extension("tket.qsystem.random")


@functools.cache
def qsystem_utils() -> Extension:
    return load_extension("tket.qsystem.utils")


@functools.cache
def quantum() -> Extension:
    return load_extension("tket.quantum")


@functools.cache
def result() -> Extension:
    return load_extension("tket.result")


@functools.cache
def wasm() -> Extension:
    return load_extension("tket.wasm")


def load_extension(name: str) -> Extension:
    replacement = name.replace(".", "/")
    json_str = pkgutil.get_data(__name__, f"data/{replacement}.json")
    assert json_str is not None, f"Could not load json for extension {name}"
    return Extension.from_json(json_str.decode())
