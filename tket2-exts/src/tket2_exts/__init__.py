"""HUGR extension definitions for tket2 circuits."""

import pkgutil
import functools

from hugr.ext import Extension


# This is updated by our release-please workflow, triggered by this
# annotation: x-release-please-version
__version__ = "0.6.0"


@functools.cache
def rotation() -> Extension:
    return load_extension("tket2.rotation")


@functools.cache
def futures() -> Extension:
    return load_extension("tket2.futures")


@functools.cache
def qsystem() -> Extension:
    return load_extension("tket2.qsystem")


@functools.cache
def qsystem_random() -> Extension:
    return load_extension("tket2.qsystem.random")


@functools.cache
def qsystem_utils() -> Extension:
    return load_extension("tket2.qsystem.utils")


@functools.cache
def quantum() -> Extension:
    return load_extension("tket2.quantum")


@functools.cache
def result() -> Extension:
    return load_extension("tket2.result")


def load_extension(name: str) -> Extension:
    replacement = name.replace(".", "/")
    json_str = pkgutil.get_data(__name__, f"data/{replacement}.json")
    assert json_str is not None, f"Could not load json for extension {name}"
    return Extension.from_json(json_str.decode())
