"""Compliant example.

Every construct here passes the Code Standards Compliance Framework checks
(``ruff check`` + ``ruff format`` with ``code-standards/ruff.toml``). It is the
Python analog of ``examples/compliant.c`` in the source framework.
"""

import json
import secrets
import subprocess

MAX_ITEMS = 100  # Rule 41: constants are UPPER_CASE


def read_config(path):
    """Read a JSON config file safely.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed config dict, or None on failure.
    """
    # Rule 23: context manager; Rule 20: specific exception handling.
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as err:
        print(f"Failed to read config {path}: {err}")
        return None


def get_name(record):
    """Return a record's name, guarding against missing values.

    Args:
        record: A mapping that may contain a ``name`` key.

    Returns:
        The stripped name, or an empty string if absent.
    """
    # Rule 22: guard the optional value before use.
    value = record.get("name")
    if value is None:
        return ""
    return value.strip()


def accumulate(item, bucket=None):
    """Append an item to a bucket without a mutable default.

    Args:
        item: The item to append.
        bucket: An optional existing list to append to.

    Returns:
        The updated list.
    """
    # Rule 24: avoid the mutable default argument trap.
    if bucket is None:
        bucket = []
    bucket.append(item)
    return bucket


def double_all(items):
    """Double every element.

    Args:
        items: An iterable of numbers.

    Returns:
        A list with each element doubled.
    """
    # Rule 35: comprehension instead of manual append loop.
    return [x * 2 for x in items]


def make_token():
    """Return a cryptographically secure token.

    Returns:
        A hex token string.
    """
    # Rule 26: use the secrets module, not random.
    return secrets.token_hex(16)


def run_listing(directory):
    """List a directory without a shell.

    Args:
        directory: The directory to list.

    Returns:
        The completed process.
    """
    # Rule 21: argument list, no shell.
    return subprocess.run(["ls", directory], shell=False, check=True)


def classify(value):
    """Classify a value's sign.

    Args:
        value: The number to classify.

    Returns:
        -1, 0, or 1.
    """
    # Rule 44: no else after return.
    if value < 0:
        return -1
    if value > 0:
        return 1
    return 0


def main():
    """Entry point demonstrating compliant usage."""
    config = read_config("config.json")
    if config is None:
        config = {}
    bucket = accumulate(1)
    print(get_name(config), double_all(bucket), make_token(), classify(-5))


if __name__ == "__main__":
    main()
