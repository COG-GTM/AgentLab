"""Restricted unpickling for experiment result files.

Result directories can contain files written by code the agent controls (e.g. a code
sandbox mounted inside the study directory), so `exp_args.pkl`, `step_*.pkl.gz` and
`goal_object.pkl.gz` are loaded with an allowlist instead of a bare `pickle.load`.

Only classes are accepted, never arbitrary callables (apart from a few reconstruction
helpers used by the standard library and numpy):
- any class from a trusted package (agentlab, browsergym, bgym, plus the comma-separated
  module roots in the `AGENTLAB_TRUSTED_PICKLE_MODULES` environment variable),
- subclasses of those classes defined in modules that are already imported (e.g. custom
  agent args declared in the launching script or a test module); nothing is imported for them,
- data classes (dataclasses, enums, pydantic models, omegaconf nodes) from dependencies
  whose objects end up in agent configs and LLM traces,
- plain value types from the standard library and numpy.
"""

import builtins
import collections
import copyreg
import dataclasses
import datetime
import decimal
import enum
import os
import pathlib
import pickle
import sys
import typing
import uuid

import numpy as np

TRUSTED_MODULES_ENV_VAR = "AGENTLAB_TRUSTED_PICKLE_MODULES"

_TRUSTED_ROOTS = frozenset({"agentlab", "browsergym", "bgym"})
_DATA_CLASS_ROOTS = frozenset(
    {"tapeagents", "pydantic", "pydantic_core", "omegaconf", "openai", "anthropic"}
)

_SAFE_GLOBALS = {
    *(
        getattr(builtins, name)
        for name in (
            "object", "bool", "int", "float", "complex", "str", "bytes", "bytearray",
            "list", "tuple", "dict", "set", "frozenset", "slice", "range",
        )
    ),
    datetime.datetime, datetime.date, datetime.time, datetime.timedelta, datetime.timezone,
    pathlib.Path, pathlib.PosixPath, pathlib.WindowsPath,
    pathlib.PurePath, pathlib.PurePosixPath, pathlib.PureWindowsPath,
    collections.OrderedDict, collections.defaultdict, collections.deque, collections.Counter,
    decimal.Decimal,
    uuid.UUID,
    copyreg._reconstructor,
    np.ndarray,
    typing.Any,
}  # fmt: skip

_SAFE_FUNCTIONS = {
    ("_codecs", "encode"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy.core.numeric", "_frombuffer"),
    ("numpy._core.numeric", "_frombuffer"),
}

_SAFE_MODULE_ROOTS = frozenset(
    {"builtins", "copyreg", "_codecs", "collections", "datetime", "decimal", "pathlib"}
    | {"uuid", "numpy", "PIL", "typing"}
)


class UnsafePickleError(pickle.UnpicklingError):
    """Raised when a pickle references a global that is not on the allowlist."""


def _root(module: str) -> str:
    return module.split(".", 1)[0]


def _trusted_roots() -> frozenset:
    extra = os.environ.get(TRUSTED_MODULES_ENV_VAR, "")
    return _TRUSTED_ROOTS | {m.strip() for m in extra.split(",") if m.strip()}


def _is_data_class(obj) -> bool:
    if dataclasses.is_dataclass(obj) or issubclass(obj, enum.Enum):
        return True
    pydantic = sys.modules.get("pydantic")
    if pydantic is not None and issubclass(obj, pydantic.BaseModel):
        return True
    omegaconf = sys.modules.get("omegaconf")
    return omegaconf is not None and issubclass(obj, omegaconf.Node)


def _is_allowed(obj, trusted_roots) -> bool:
    if isinstance(obj, type):
        if obj in _SAFE_GLOBALS:
            return True
        root = _root(getattr(obj, "__module__", None) or "")
        if root in trusted_roots:
            return True
        if root in _DATA_CLASS_ROOTS:
            return _is_data_class(obj)
        if root == "numpy":
            return issubclass(obj, (np.dtype, np.generic))
        if root == "PIL":
            from PIL import Image

            return obj is Image.Image
        return False
    try:
        return obj in _SAFE_GLOBALS
    except TypeError:  # unhashable
        return False


def _imported_trusted_subclass(module: str, name: str, trusted_roots) -> type | None:
    obj = sys.modules.get(module)
    for attr in name.split("."):
        obj = getattr(obj, attr, None)
    if isinstance(obj, type) and any(
        _root(getattr(base, "__module__", None) or "") in trusted_roots for base in obj.__mro__[1:]
    ):
        return obj
    return None


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        trusted_roots = _trusted_roots()
        allowed_roots = trusted_roots | _DATA_CLASS_ROOTS | _SAFE_MODULE_ROOTS
        if _root(module) in allowed_roots:
            if (module, name) in _SAFE_FUNCTIONS:
                return super().find_class(module, name)
            obj = super().find_class(module, name)
            if _is_allowed(obj, trusted_roots):
                return obj
        obj = _imported_trusted_subclass(module, name, trusted_roots)
        if obj is not None:
            return obj
        raise UnsafePickleError(
            f"Refusing to unpickle global '{module}.{name}' from an untrusted result file. "
            f"If this class belongs to your own agent or benchmark package, add its top-level "
            f"module to the {TRUSTED_MODULES_ENV_VAR} environment variable (comma-separated)."
        )


def load(file):
    """Drop-in replacement for `pickle.load` restricted to the allowlist above."""
    return RestrictedUnpickler(file).load()
