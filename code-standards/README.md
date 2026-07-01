# Python Code Standards Compliance Framework

> **Drop-in ruff + pyright configuration for safety-critical Python development with Windsurf AI.**
>
> Framework version **1.0** — the Python port of
> [`COG-GTM/-C-Code-Standards-Compliance-Framework`](https://github.com/COG-GTM/-C-Code-Standards-Compliance-Framework),
> using the same patterns, severity model, and rule-ID numbering.

---

## What Is This?

A **complete, ready-to-use code standards enforcement system** for Python. It
combines:

1. **ruff format** — automatic code formatting (consistent style; Black-compatible)
2. **ruff check** — static analysis (bug + security detection)
3. **pyright** — IDE intelligence (type diagnostics, autocomplete, go-to-def)
4. **Severity Classification** — triage issues as Critical / Major / Minor
5. **Windsurf AI Rules** — intelligent code review with fix suggestions

It is the direct analog of the C/C++ framework. The mapping:

| Framework role | C/C++ tool | Python tool | Config file |
|----------------|-----------|-------------|-------------|
| Formatting | clang-format | **ruff format** | `ruff.toml` `[format]` |
| Static analysis | clang-tidy | **ruff check** | `ruff.toml` `[lint]` |
| Language server | clangd | **pyright** | `pyrightconfig.json` |

## Severity Classification

| Severity | Icon | Rule IDs | Action | Blocks Merge? | Exit code |
|----------|------|----------|--------|---------------|-----------|
| **Critical** | 🔴 | 20–29 | Must fix immediately | Yes | 1 |
| **Major** | 🟡 | 30–39 | Requires review | Recommended | 0 |
| **Minor** | 🟢 | 40–49 | Fix when convenient | No | 0 |

The **Critical gate** is enforced by `validate.sh` running `ruff check` restricted
to the critical rule codes (the analog of clang-tidy's `WarningsAsErrors`). See
[`rule-severity-mapping.yaml`](rule-severity-mapping.yaml) for the full mapping.

## Rule Reference (summary)

| Critical (🔴) | Major (🟡) | Minor (🟢) |
|--------------|-----------|-----------|
| 20 No swallowed errors | 30 Unsafe comparisons | 40 Formatting |
| 21 No injection | 31 Signature/docstring consistency | 41 Naming |
| 22 Guard None | 32 No dead code | 42 Explicit comparisons |
| 23 No resource leaks | 33 Control-flow errors | 43 Simplify booleans |
| 24 No mutable defaults | 34 Concurrency safety | 44 No else-after-return |
| 25 No use-before-assign | 35 Performance | 45 Modern syntax |
| 26 No insecure APIs |  | 46 No unused imports/args |

Full detail with bad/good examples: [`docs/rule-reference.md`](docs/rule-reference.md).

## Quick Start

```bash
# 1. Install ruff (and optionally pyright)
pip install ruff            # or: uv add --dev ruff

# 2. Validate a directory (format + tiered analysis)
./code-standards/scripts/validate.sh src/

# 3. Auto-fix formatting + safe lint fixes
./code-standards/scripts/validate.sh src/ --fix
```

Or invoke ruff directly with the framework config:

```bash
ruff format --config code-standards/ruff.toml --check src/
ruff check  --config code-standards/ruff.toml src/
```

## Repository Contents

```
code-standards/
├── README.md                          # This documentation
├── ruff.toml                          # Formatting + static-analysis config
├── pyrightconfig.json                 # Language-server config (clangd analog)
├── rule-severity-mapping.yaml         # Severity classification (version 1.0)
├── vscode-settings.json.template      # VSCode/Windsurf editor settings
├── windsurf/
│   └── python-safety-critical-rules.md  # Windsurf AI rules (trigger: always_on)
├── examples/
│   ├── compliant.py                   # Passes all checks
│   └── violations.py                  # Intentional violations (one per rule)
├── scripts/
│   └── validate.sh                    # Validation wrapper (tiered exit codes)
├── tests/
│   ├── __init__.py
│   └── test_compliance.py             # Automated tests for the configs
└── docs/
    ├── rule-reference.md              # Detailed rule documentation
    └── setup.md                       # Setup / IDE integration guide
```

## CI/CD Integration

Add a job that runs the validator (fails on Critical + format):

```yaml
name: Code Standards
on: [push, pull_request]
jobs:
  standards:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync --frozen --extra dev
      - run: uv run ruff format --config code-standards/ruff.toml --check src/
      - run: ./code-standards/scripts/validate.sh src/
      - run: uv run pytest code-standards/tests/ -v
```

## Windsurf AI Integration

```bash
mkdir -p .windsurf/rules
cp code-standards/windsurf/python-safety-critical-rules.md .windsurf/rules/
```

Cascade then detects violations, classifies them by severity, reports them with
the Rule ID (e.g. "🔴 Critical - Rule 20 violation"), explains why, and suggests
a compliant fix.

## Adopting Repo-Wide

This framework is intentionally **additive and non-breaking**: `ruff.toml` uses
`line-length = 100` so `ruff format` output matches the repo's existing Black
config, and it is only applied when invoked with `--config`. To make it the
project's primary tooling, copy the `ruff.toml` settings into a root `ruff.toml`
or `[tool.ruff]` in `pyproject.toml`, and `pyrightconfig.json` to the root. See
[`docs/setup.md`](docs/setup.md).

## Requirements

| Tool | Minimum | Purpose |
|------|---------|---------|
| ruff | 0.6+ | Formatting + static analysis |
| pyright | 1.1+ (optional) | IDE type intelligence |
| Python | 3.10+ | Running the project/tests |
| pytest, PyYAML | — | Running compliance tests |

## License

Apache-2.0 (matches the AgentLab project license).
