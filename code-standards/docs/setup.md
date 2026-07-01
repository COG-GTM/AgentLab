# Setup Guide

How to enable the Python Code Standards Compliance Framework in a project. This
is the Python analog of the source framework's `docs/clangd-setup.md`.

## What You Need (3 Things)

| Component | What It Is | Where to Get It |
|-----------|-----------|-----------------|
| **1. ruff** | Formatter + static analyzer (the clang-format + clang-tidy analog) | `pip install ruff` or `uv add --dev ruff` |
| **2. pyright / Pylance** | Language server for IDE intelligence (the clangd analog) | Bundled with the VSCode/Windsurf Python extension, or `pip install pyright` |
| **3. This framework** | Pre-configured rules & severity mapping | `code-standards/` in this repo |

## Step 1: Install the tools

```bash
pip install ruff
# optional CLI type checker
pip install pyright
```

Or, in a `uv`-managed repo (like AgentLab):

```bash
uv add --dev ruff
uv run ruff --version
```

## Step 2: Validate your code

```bash
# One-shot validation of src/ (format + tiered analysis)
./code-standards/scripts/validate.sh src/

# Auto-fix formatting + safe lint fixes
./code-standards/scripts/validate.sh src/ --fix

# Or invoke ruff directly with the framework config
ruff format --config code-standards/ruff.toml --check src/
ruff check  --config code-standards/ruff.toml src/
```

Exit codes from `validate.sh`: `0` pass, `1` format, `2` Critical lint (block
merge), `3` both.

## Step 3: Wire up your editor

```bash
mkdir -p .vscode
cp code-standards/vscode-settings.json.template .vscode/settings.json
```

This makes Windsurf/VSCode:
- format on save with `ruff format` (Rule 40),
- show ruff diagnostics in real time (Rules 20–46),
- run pyright type analysis (Rules 22, 25, 30, 32).

## Step 4 (optional): Windsurf AI reviewer

```bash
mkdir -p .windsurf/rules
cp code-standards/windsurf/python-safety-critical-rules.md .windsurf/rules/
```

Cascade will then enforce the rules and report violations by Rule ID + severity.

## Step 5 (optional): Adopt the config repo-wide

To make ruff the project's real formatter/linter (rather than only via
`--config`), copy the settings from `code-standards/ruff.toml` into a root
`ruff.toml` or a `[tool.ruff]` block in `pyproject.toml`, and copy
`pyrightconfig.json` to the repo root. Keep `line-length = 100` so it stays
compatible with an existing Black setup.

## Verify

```bash
pytest code-standards/tests/test_compliance.py -v
```
