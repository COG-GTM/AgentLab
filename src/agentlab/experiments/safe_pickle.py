"""Restricted unpickling for study and experiment files.

`pickle.load` executes arbitrary code embedded in the file. Study directories can come from
shared drives or collaborators, so files such as `study.pkl.gz` and `exp_args.pkl` are loaded
through `RestrictedUnpickler`, which only resolves:

- dataclasses, Enums and pydantic models defined in trusted packages (`agentlab`, `browsergym`
  and any extra package listed in the `AGENTLAB_TRUSTED_PICKLE_MODULES` environment variable,
  comma separated, e.g. for agent args defined in your own package);
- an explicit allowlist of standard-library, numpy, pandas, pyarrow and omegaconf data types.

Any other global (functions such as `os.system`, arbitrary classes, ...) raises
`UnsafePickleError` before it can be called.
"""

import dataclasses
import enum
import io
import os
import pickle

TRUSTED_PACKAGES = ("agentlab", "browsergym", "bgym")
TRUSTED_PACKAGES_ENV_VAR = "AGENTLAB_TRUSTED_PICKLE_MODULES"

_STDLIB_GLOBALS = {
    ("builtins", name)
    for name in (
        "object",
        "set",
        "frozenset",
        "slice",
        "range",
        "complex",
        "bytes",
        "bytearray",
        "list",
        "dict",
        "tuple",
        "str",
        "int",
        "float",
        "bool",
    )
} | {
    ("copyreg", "_reconstructor"),
    ("collections", "OrderedDict"),
    ("collections", "defaultdict"),
    ("collections", "deque"),
    ("collections", "Counter"),
    ("datetime", "datetime"),
    ("datetime", "date"),
    ("datetime", "time"),
    ("datetime", "timedelta"),
    ("datetime", "timezone"),
    ("decimal", "Decimal"),
    ("fractions", "Fraction"),
    ("uuid", "UUID"),
    ("pathlib", "Path"),
    ("pathlib", "PosixPath"),
    ("pathlib", "WindowsPath"),
    ("pathlib", "PurePath"),
    ("pathlib", "PurePosixPath"),
    ("pathlib", "PureWindowsPath"),
}

_NUMPY_GLOBALS = {
    (module, name)
    for module in ("numpy.core.multiarray", "numpy._core.multiarray")
    for name in ("_reconstruct", "scalar")
} | {
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy.core.numeric", "_frombuffer"),
    ("numpy._core.numeric", "_frombuffer"),
}

_PANDAS_GLOBALS = {
    ("pandas", "DataFrame"),
    ("pandas", "Series"),
    ("pandas", "Index"),
    ("pandas", "RangeIndex"),
    ("pandas", "Timestamp"),
    ("pandas", "Timedelta"),
    ("pandas", "StringDtype"),
    ("pandas", "CategoricalDtype"),
    ("pandas", "Categorical"),
    ("pandas.core.dtypes.dtypes", "CategoricalDtype"),
    ("pandas.core.arrays.categorical", "Categorical"),
    ("pandas.core.arrays.categorical", "_reconstruct"),
    ("pandas.core.frame", "DataFrame"),
    ("pandas.core.series", "Series"),
    ("pandas.core.indexes.base", "Index"),
    ("pandas.core.indexes.base", "_new_Index"),
    ("pandas.core.indexes.range", "RangeIndex"),
    ("pandas.core.internals.managers", "BlockManager"),
    ("pandas.core.internals.managers", "SingleBlockManager"),
    ("pandas._libs.internals", "_unpickle_block"),
    ("pandas.core.internals.blocks", "new_block"),
    ("pandas.arrays", "ArrowStringArray"),
    ("pandas.arrays", "StringArray"),
    ("pandas.core.arrays.string_", "StringDtype"),
    ("pandas.core.arrays.string_", "StringArray"),
    ("pandas.core.arrays.string_arrow", "ArrowStringArray"),
    ("pandas.core.arrays.string_arrow", "ArrowStringArrayNumpySemantics"),
    ("pyarrow.lib", "_restore_array"),
    ("pyarrow.lib", "type_for_alias"),
    ("pyarrow.lib", "py_buffer"),
}

_OMEGACONF_GLOBALS = {
    ("omegaconf.dictconfig", "DictConfig"),
    ("omegaconf.listconfig", "ListConfig"),
    ("omegaconf.base", "ContainerMetadata"),
    ("omegaconf.base", "Metadata"),
    ("omegaconf.nodes", "AnyNode"),
    ("omegaconf.nodes", "StringNode"),
    ("omegaconf.nodes", "IntegerNode"),
    ("omegaconf.nodes", "FloatNode"),
    ("omegaconf.nodes", "BooleanNode"),
    ("omegaconf.nodes", "BytesNode"),
    ("omegaconf.nodes", "PathNode"),
    ("omegaconf.nodes", "EnumNode"),
    ("typing", "Any"),
}

SAFE_GLOBALS = frozenset(_STDLIB_GLOBALS | _NUMPY_GLOBALS | _PANDAS_GLOBALS | _OMEGACONF_GLOBALS)


class UnsafePickleError(pickle.UnpicklingError):
    """Raised when a pickle references a global that is not allowed."""


def _trusted_packages() -> tuple[str, ...]:
    extra = os.environ.get(TRUSTED_PACKAGES_ENV_VAR, "")
    return TRUSTED_PACKAGES + tuple(p.strip() for p in extra.split(",") if p.strip())


def _in_trusted_package(module: str) -> bool:
    return any(module == pkg or module.startswith(pkg + ".") for pkg in _trusted_packages())


def _is_data_type(obj) -> bool:
    if not isinstance(obj, type):
        return False
    if dataclasses.is_dataclass(obj) or issubclass(obj, enum.Enum):
        return True
    try:
        from pydantic import BaseModel
    except ImportError:  # pragma: no cover
        return False
    return issubclass(obj, BaseModel)


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that only resolves allowlisted data types (see module docstring)."""

    def find_class(self, module: str, name: str):
        if (module, name) in SAFE_GLOBALS:
            return super().find_class(module, name)
        if _in_trusted_package(module) and "." not in name:
            obj = super().find_class(module, name)
            if _is_data_type(obj) and _in_trusted_package(getattr(obj, "__module__", "")):
                return obj
        raise UnsafePickleError(
            f"Refusing to unpickle global '{module}.{name}': only dataclasses, Enums and pydantic "
            f"models from {', '.join(_trusted_packages())} and allowlisted data types are "
            f"permitted. If this file comes from a trusted source and defines classes in your "
            f"own package, add that package to the {TRUSTED_PACKAGES_ENV_VAR} environment variable."
        )


def load(file):
    """Safe replacement for `pickle.load` on study/experiment files."""
    return RestrictedUnpickler(file).load()


def loads(data: bytes):
    """Safe replacement for `pickle.loads` on study/experiment files."""
    return load(io.BytesIO(data))
