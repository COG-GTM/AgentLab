"""Examples of code violations for testing ruff / pyright detection.

WARNING: This file intentionally contains violations! DO NOT use in production.

Run: ``ruff check --config code-standards/ruff.toml code-standards/examples/violations.py``
to see the detected issues. Each function demonstrates one rule violation and
documents the expected ruff/pyright code(s). Python analog of ``violations.c``.
"""

import hashlib
import pickle
import random
import subprocess

# ==========================================================================
# CRITICAL VIOLATIONS (Rule 20-26) - should block merges in CI/CD
# ==========================================================================


def rule_20_violation_bare_except():
    """Rule 20 VIOLATION: swallowed error. Expected: E722."""
    try:
        risky = 1 / 0
        return risky
    except:  # noqa is intentionally omitted so the violation is reported
        pass


def rule_21_violation_shell_injection(user_input):
    """Rule 21 VIOLATION: shell injection. Expected: S602/S605."""
    return subprocess.run(f"ls {user_input}", shell=True)  # noqa: S602 is omitted intentionally


def rule_23_violation_resource_leak(path):
    """Rule 23 VIOLATION: unclosed file. Expected: SIM115."""
    handle = open(path)
    return handle.read()


def rule_24_violation_mutable_default(item, bucket=[]):
    """Rule 24 VIOLATION: mutable default argument. Expected: B006."""
    bucket.append(item)
    return bucket


def rule_25_violation_use_before_assignment(condition):
    """Rule 25 VIOLATION: possibly-unbound variable. Expected: F821/pyright."""
    if condition:
        count = 10
    return count  # noqa: F821 omitted intentionally


def rule_26_violation_insecure_apis(untrusted):
    """Rule 26 VIOLATION: insecure APIs. Expected: S311, S301, S324."""
    token = random.random()
    obj = pickle.loads(untrusted)
    digest = hashlib.md5(b"data").hexdigest()
    return token, obj, digest


# ==========================================================================
# MAJOR VIOLATIONS (Rule 30-35) - require review
# ==========================================================================


def rule_30_violation_type_comparison(x):
    """Rule 30 VIOLATION: type comparison / literal identity. Expected: E721, F632."""
    if type(x) == int:
        return x is "int"  # noqa: F632 omitted intentionally
    return None


def rule_32_violation_dead_code():
    """Rule 32 VIOLATION: unused local variable. Expected: F841."""
    result = compute_something()
    return 42


def rule_35_violation_performance(items):
    """Rule 35 VIOLATION: manual list build. Expected: PERF401."""
    out = []
    for x in items:
        out.append(x * 2)
    return out


# ==========================================================================
# MINOR VIOLATIONS (Rule 40-46) - style warnings only
# ==========================================================================


def RuleFortyOneBadName(BadParam):
    """Rule 41 VIOLATION: non-snake_case naming. Expected: N802, N803."""
    return BadParam


def rule_42_violation_none_comparison(x):
    """Rule 42 VIOLATION: comparison to None with ==. Expected: E711."""
    if x == None:
        return True
    return False


def rule_44_violation_else_after_return(error):
    """Rule 44 VIOLATION: else after return. Expected: RET505."""
    if error:
        return -1
    else:
        return 0


def rule_45_violation_old_syntax(name):
    """Rule 45 VIOLATION: legacy syntax. Expected: UP.*."""
    mapping = dict()
    return "%s" % name, mapping


def rule_46_violation_unused_arg(used, unused):
    """Rule 46 VIOLATION: unused argument. Expected: ARG001."""
    return used


def compute_something():
    """Helper so rule_32 references a defined name."""
    return 0
