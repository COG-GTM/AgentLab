# Rule Reference (Python)

Complete reference for all code standards rules. This is the Python
instantiation of the Code Standards Compliance Framework; rule IDs and
severities match the source C/C++ framework so findings are comparable across
languages.

## Overview

Rules are organized by severity:
- **Critical (Rule 20–29):** Safety/security issues that must be fixed
- **Major (Rule 30–39):** Quality issues that should be reviewed
- **Minor (Rule 40–49):** Style issues that improve maintainability

Each rule lists the ruff/pyright checks that enforce it.

---

## Critical Rules (Block Merge)

### Rule 20: Do Not Silently Swallow Errors
**Severity:** 🔴 Critical &nbsp; **Checks:** `E722`, `BLE001`

Never use a bare `except:` or blind `except Exception: pass`.

```python
# ❌ BAD
try:
    risky()
except:
    pass

# ✅ GOOD
try:
    risky()
except ValueError as err:
    logger.error("risky failed: %s", err)
    raise
```

### Rule 21: Prevent Injection / Unsafe Input
**Severity:** 🔴 Critical &nbsp; **Checks:** `S602`, `S603`, `S604`, `S605`, `S608`

```python
# ❌ BAD
subprocess.run(f"ls {user_input}", shell=True)
# ✅ GOOD
subprocess.run(["ls", user_input], shell=False)
```

### Rule 22: Guard Optional / None Values
**Severity:** 🔴 Critical &nbsp; **Checks:** `reportOptionalMemberAccess`, `reportOptionalSubscript` (pyright)

```python
# ❌ BAD
value = data.get("key")
return value.strip()
# ✅ GOOD
value = data.get("key")
if value is None:
    return None
return value.strip()
```

### Rule 23: Prevent Resource Leaks
**Severity:** 🔴 Critical &nbsp; **Checks:** `SIM115`

```python
# ❌ BAD
f = open(path)
data = f.read()
# ✅ GOOD
with open(path) as f:
    data = f.read()
```

### Rule 24: Avoid Dangerous Shared Mutable State
**Severity:** 🔴 Critical &nbsp; **Checks:** `B006`, `B008`

```python
# ❌ BAD
def add(item, bucket=[]):
    bucket.append(item)
# ✅ GOOD
def add(item, bucket=None):
    if bucket is None:
        bucket = []
    bucket.append(item)
```

### Rule 25: No Undefined / Use-Before-Assignment
**Severity:** 🔴 Critical &nbsp; **Checks:** `F821`, `F822`, `F823`, `reportUnboundVariable`

```python
# ❌ BAD
if cond:
    count = 10
return count
# ✅ GOOD
count = 0
if cond:
    count = 10
return count
```

### Rule 26: Avoid Insecure APIs
**Severity:** 🔴 Critical &nbsp; **Checks:** `S102`, `S307`, `S301`, `S506`, `S324`, `S311`

| Insecure | Secure |
|----------|--------|
| `eval`/`exec` | `ast.literal_eval` / explicit parsing |
| `pickle.loads(untrusted)` | `json.loads` |
| `yaml.load` | `yaml.safe_load` |
| `random` (secrets) | `secrets` |
| `md5`/`sha1` (security) | `sha256` |

---

## Major Rules (Requires Review)

### Rule 30: Avoid Unsafe Comparisons / Conversions
**Severity:** 🟡 Major &nbsp; **Checks:** `E721`, `F632`

### Rule 31: Consistent Signatures & Docstrings
**Severity:** 🟡 Major &nbsp; **Checks:** `darglint`

### Rule 32: No Redundant / Dead Code
**Severity:** 🟡 Major &nbsp; **Checks:** `F811`, `F841`

### Rule 33: Prevent Control-Flow Errors
**Severity:** 🟡 Major &nbsp; **Checks:** `F701`, `F702`, `F706`

### Rule 34: Concurrency / Shared-State Safety
**Severity:** 🟡 Major &nbsp; **Checks:** `B006`

### Rule 35: Performance Issues
**Severity:** 🟡 Major &nbsp; **Checks:** `PERF*`, `C4*`

---

## Minor Rules (Style)

### Rule 40: Consistent Formatting
**Severity:** 🟢 Minor &nbsp; **Enforced by:** `ruff format` (`ruff.toml`)

### Rule 41: Naming Conventions
**Severity:** 🟢 Minor &nbsp; **Checks:** `N` (pep8-naming)

| Kind | Convention | Example |
|------|------------|---------|
| Functions | `snake_case` | `process_data()` |
| Variables | `snake_case` | `buffer_size` |
| Constants | `UPPER_CASE` | `MAX_SIZE` |
| Classes | `PascalCase` | `DataBuffer` |

### Rule 42: Explicit Comparisons
**Severity:** 🟢 Minor &nbsp; **Checks:** `E711`, `E712`, `E713`, `E714`

### Rule 43: Simplify Boolean Expressions
**Severity:** 🟢 Minor &nbsp; **Checks:** `SIM`

### Rule 44: Avoid Else After Return
**Severity:** 🟢 Minor &nbsp; **Checks:** `RET505`, `RET506`, `RET507`, `RET508`

### Rule 45: Use Modern Syntax
**Severity:** 🟢 Minor &nbsp; **Checks:** `UP` (pyupgrade)

### Rule 46: Remove Unused Imports / Arguments
**Severity:** 🟢 Minor &nbsp; **Checks:** `F401`, `ARG`

---

## Suppression Guide

```python
result = risky_call()  # noqa: BLE001  (explain why)
```

File-level: add `# ruff: noqa: <CODE>` at the top of the file. Always document
why suppression is needed; prefer fixing over suppressing.

---

## References
- [ruff rules](https://docs.astral.sh/ruff/rules/)
- [pyright configuration](https://microsoft.github.io/pyright/#/configuration)
- [CWE](https://cwe.mitre.org/) · [OWASP Top 10](https://owasp.org/www-project-top-ten/)
