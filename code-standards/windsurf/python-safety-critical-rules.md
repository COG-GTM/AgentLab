---
trigger: always_on
---

# Python Safety-Critical Code Standards

> Copy this file to `.windsurf/rules/python-safety-critical-rules.md` in your project.

## Overview

When reviewing Python code, enforce these safety-critical rules and classify violations by severity. Report violations with their rule ID for traceability. This is the Python instantiation of the Code Standards Compliance Framework; rule IDs and severities match `rule-severity-mapping.yaml`, the ruff config, and the docs.

## Severity Levels

| Level | Icon | Action | Description |
|-------|------|--------|-------------|
| **Critical** | 🔴 | Block merge | Safety/security risk - runtime crashes, vulnerabilities, data corruption |
| **Major** | 🟡 | Require review | Quality issue - potential bugs, edge case failures |
| **Minor** | 🟢 | Warn only | Style issue - readability, maintainability |

When you find a violation: (1) detect it, (2) classify by severity, (3) report it with the Rule ID (e.g. "🔴 Critical - Rule 20 violation"), (4) explain why the rule matters, (5) suggest a compliant fix.

---

## Critical Rules (Block Merge) 🔴

### Rule 20: Do Not Silently Swallow Errors

Never use a bare `except:` or a blind `except Exception: pass`. Catch specific exceptions and handle or re-raise them.

```python
# ❌ VIOLATION - Critical (Rule 20)
try:
    result = risky()
except:
    pass  # Failure is hidden!

# ✅ COMPLIANT
try:
    result = risky()
except ValueError as err:
    logger.error("risky failed: %s", err)
    raise
```

**Why:** Swallowed exceptions hide failures, leading to undefined behavior, silent data corruption, and undebuggable production incidents.

---

### Rule 21: Prevent Injection / Unsafe Input Handling

Never pass untrusted input into a shell, SQL string, or template. Use parameterized/argument-list APIs.

```python
# ❌ VIOLATION - Critical (Rule 21)
subprocess.run(f"ls {user_input}", shell=True)

# ✅ COMPLIANT
subprocess.run(["ls", user_input], shell=False)
```

**Why:** Injection is the #1 class of security vulnerabilities (CWE-78, CWE-89).

---

### Rule 22: Guard Optional / None Values

Validate values that may be `None` before accessing attributes, items, or calling them.

```python
# ❌ VIOLATION - Critical (Rule 22)
value = data.get("key")
return value.strip()  # AttributeError if key missing!

# ✅ COMPLIANT
value = data.get("key")
if value is None:
    return None
return value.strip()
```

**Why:** `None` access is the Python analog of null-pointer dereference; it crashes at runtime (AttributeError/TypeError).

---

### Rule 23: Prevent Resource Leaks

Always use a context manager (`with`) for files, sockets, and locks.

```python
# ❌ VIOLATION - Critical (Rule 23)
f = open(path)
data = f.read()  # Never closed on error!

# ✅ COMPLIANT
with open(path) as f:
    data = f.read()
```

**Why:** Leaked descriptors accumulate and crash long-running systems.

---

### Rule 24: Avoid Dangerous Shared Mutable State

Never use a mutable default argument (`[]`, `{}`, or a function call).

```python
# ❌ VIOLATION - Critical (Rule 24)
def add(item, bucket=[]):
    bucket.append(item)  # Shared across ALL calls!

# ✅ COMPLIANT
def add(item, bucket=None):
    if bucket is None:
        bucket = []
    bucket.append(item)
```

**Why:** Mutable defaults are evaluated once and shared, causing state-corruption bugs (the Python analog of use-after-free surprises).

---

### Rule 25: No Undefined / Use-Before-Assignment

Every name must be bound on all code paths before use.

```python
# ❌ VIOLATION - Critical (Rule 25)
if condition:
    count = 10
return count  # UnboundLocalError if condition is False!

# ✅ COMPLIANT
count = 0
if condition:
    count = 10
return count
```

**Why:** Unbound names raise NameError/UnboundLocalError at runtime.

---

### Rule 26: Avoid Insecure APIs

| Insecure | Secure Alternative |
|----------|-------------------|
| `eval()`, `exec()` | explicit parsing / `ast.literal_eval` |
| `pickle.loads(untrusted)` | `json.loads` |
| `yaml.load(x)` | `yaml.safe_load(x)` |
| `random` for secrets | `secrets` module |
| `hashlib.md5`/`sha1` for security | `hashlib.sha256` |

```python
# ❌ VIOLATION - Critical (Rule 26)
token = random.random()

# ✅ COMPLIANT
import secrets
token = secrets.token_hex(16)
```

---

## Major Rules (Requires Review) 🟡

### Rule 30: Avoid Unsafe Comparisons / Conversions

```python
# ❌ VIOLATION - Major (Rule 30)
if type(x) == int:
    ...
if name is "root":   # identity check on a literal!
    ...

# ✅ COMPLIANT
if isinstance(x, int):
    ...
if name == "root":
    ...
```

---

### Rule 31: Consistent Signatures & Docstrings

Docstring parameters must match the function signature (enforced by `darglint`).

```python
# ❌ VIOLATION - Major (Rule 31) - docstring says `n`, signature says `count`
def f(count):
    """Do it.

    Args:
        n: the number.
    """
```

---

### Rule 32: No Redundant / Dead Code

```python
# ❌ VIOLATION - Major (Rule 32)
result = compute()   # assigned but never used
return other

# ✅ COMPLIANT
return other
```

---

### Rule 33: Prevent Control-Flow Errors

`break`/`continue`/`return` must appear in a valid context.

```python
# ❌ VIOLATION - Major (Rule 33)
def f():
    x = 1
    break   # break outside loop!
```

---

### Rule 34: Concurrency / Shared-State Safety

Avoid shared mutable defaults and module-level mutable globals in concurrent code (see also Rule 24).

---

### Rule 35: Avoid Performance Issues

```python
# ❌ VIOLATION - Major (Rule 35)
out = []
for x in items:
    out.append(x * 2)

# ✅ COMPLIANT
out = [x * 2 for x in items]
```

---

## Minor Rules (Style Warnings) 🟢

### Rule 40: Consistent Formatting

Use `ruff format` (Black-compatible, 100-column). Enforced by `ruff.toml`.

### Rule 41: Naming Conventions

- **Functions / variables:** `snake_case`
- **Constants:** `UPPER_CASE`
- **Classes:** `PascalCase`

```python
# ❌ VIOLATION - Minor (Rule 41)
def ProcessData(): ...
MyConstant = 100

# ✅ COMPLIANT
def process_data(): ...
MY_CONSTANT = 100
```

### Rule 42: Explicit Comparisons

```python
# ❌ VIOLATION - Minor (Rule 42)
if x == None:
if flag == True:

# ✅ COMPLIANT
if x is None:
if flag:
```

### Rule 43: Simplify Boolean Expressions

```python
# ❌ VIOLATION - Minor (Rule 43)
if flag == True:

# ✅ COMPLIANT
if flag:
```

### Rule 44: Avoid Else After Return

```python
# ❌ VIOLATION - Minor (Rule 44)
if error:
    return -1
else:
    process()

# ✅ COMPLIANT
if error:
    return -1
process()
```

### Rule 45: Use Modern Syntax

```python
# ❌ VIOLATION - Minor (Rule 45)
d = dict()
msg = "%s" % name

# ✅ COMPLIANT
d = {}
msg = f"{name}"
```

### Rule 46: Remove Unused Imports / Arguments

```python
# ❌ VIOLATION - Minor (Rule 46)
import os

def process(x, unused):
    return x

# ✅ COMPLIANT
def process(x):
    return x
```

---

## Validation Commands

```bash
# Check formatting
ruff format --config code-standards/ruff.toml --check src/

# Run static analysis
ruff check --config code-standards/ruff.toml src/

# Run validation script (format + tiered analysis)
./code-standards/scripts/validate.sh src/
```

## Suppressing Warnings

```python
result = risky_call()  # noqa: BLE001  (explain why)
```

**Always add a comment explaining why suppression is necessary.**

---

## References

- [ruff rules](https://docs.astral.sh/ruff/rules/)
- [pyright configuration](https://microsoft.github.io/pyright/#/configuration)
- [CWE - Common Weakness Enumeration](https://cwe.mitre.org/)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
