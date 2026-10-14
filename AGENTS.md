# AGENTS.md

Instructions for AI coding agents working in this repository.

## Unit Testing Agent

### Role

Write, run, and maintain unit tests for the code in `src/agentlab/`. Tests live under `tests/`, mirroring the source layout (`tests/llm/`, `tests/agents/`, `tests/experiments/`, `tests/benchmarks/`, `tests/analyze/`).

### 1. Discover before writing

Before adding or changing a test, read:

- This file, `README.md`, `pyproject.toml` (deps, `black` line length 100), and `pytest.ini` (markers `pricy` and `slow`; `pricy` is deselected by default).
- `.github/workflows/unit_tests.yml` and `.github/workflows/code_format.yml`. CI is the source of truth for how tests run.
- The module under test and its existing tests. Follow the naming, fixture, and mocking style already used nearby.

Don't write a test for behavior you haven't confirmed by reading the code or running it.

### 2. Reproducible environment

Match CI (Python 3.11, `uv`, locked deps):

```bash
uv python install 3.11
UV_PYTHON=3.11 uv sync --frozen --extra dev
uv run playwright install chromium --with-deps   # needed by MiniWoB-backed tests
uv run python -c 'import nltk; nltk.download("punkt_tab")'
```

For tests that need MiniWoB, run `make miniwob`, export `MINIWOB_URL="http://localhost:8080/miniwob/"`, and run `make stop-miniwob` afterwards.

Never use `--frozen`-breaking commands (`uv add`, `uv lock`, `pip install` into the project) to make a test pass.

### 3. Scope: test-only changes

By default, change files under `tests/` only. Without explicit human approval, do **not** modify:

- Production code in `src/`
- Dependencies (`pyproject.toml`, `uv.lock`)
- CI workflows, `pytest.ini`, `Makefile`, or other configuration
- Unrelated files, or local changes that someone else made and didn't commit

If a test seems to need a source change, stop and report it instead of making the change.

### 4. Fixtures and isolation

- Use small, synthetic fixtures built inline or with `tmp_path`. Never use real credentials, customer data, or network resources.
- Mock external boundaries: LLM providers (OpenAI, Anthropic, Azure, OpenRouter, Hugging Face, LiteLLM), pricing lookups, Ray/Dask, Playwright/browser, file-system and environment-variable access. Use `monkeypatch` / `unittest.mock`.
- Tests must be hermetic. Each one should pass alone, in any order, and in parallel under `pytest-xdist`. Restore any global or thread-local state the test changes (e.g. `agentlab.llm.tracking.TRACKER`).
- Mark anything that calls a paid API with `@pytest.mark.pricy`. Mark slow tests with `@pytest.mark.slow`.
- Some modules can't be imported in the default env because of optional or broken deps (e.g. `benchmarks/gaia.py`, `benchmarks/multitool_gym.py`, `agents/tapeagent`, `analyze/tapes`). Skip them, or guard them with `pytest.importorskip`. Don't add deps.

### 5. Confirmed source defects

If a test shows a real bug in `src/`, don't fix the production code. Instead:

1. Reproduce it with a minimal failing test.
2. Mark that test `@pytest.mark.xfail(strict=True, reason="<what is wrong and where>")`, so it starts failing once the bug is fixed.
3. Report the defect (file, line, expected vs actual) in your summary or PR description.

Never weaken an assertion, or assert the buggy behavior as correct, to make a test pass.

### 6. Parallel workers

When several agents or sessions write tests at the same time:

- Give each worker its own test files (e.g. one module under test per worker). Two workers must never edit the same file.
- Don't add a root `conftest.py` or other shared fixtures that later work depends on. Each PR should merge on its own, in any order.
- Base work on `main`. Ignore stale `devin/*test*` branches.

### 7. Verification

Run all of the following before you report done. Every step must pass, except `xfail`s you report.

```bash
# New/changed tests, serially
uv run pytest -p no:randomly tests/path/to/test_file.py -v

# New/changed tests, parallel (as CI runs them)
uv run pytest -n 5 tests/path/to/test_file.py

# Order independence
uv run --with pytest-randomly pytest tests/path/to/test_file.py

# Formatting (CI only gates src/, but keep tests consistent)
uv run black --check tests/path/to/test_file.py

# The relevant surrounding suite, to catch regressions
uv run pytest -n 5 -m 'not pricy' tests/<area>/

# Coverage for the module under test
uv run --with pytest-cov pytest tests/path/to/test_file.py \
  --cov=src/agentlab/<package> --cov-report=term-missing
```

Coverage notes:

- `pytest-cov` and `pytest-randomly` aren't in the dev deps. Use `uv run --with ...`, not a dependency change.
- Point `--cov` at a package or path, not at a submodule that its package `__init__` star-imports. Otherwise torch is imported twice and collection fails with `'_has_torch_function' already has a docstring`.
- Under coverage, `sys.gettrace()` is set. Code that checks for a debugger, like `loop._is_debugging()`, behaves differently, so a known-good test may fail only under `--cov`. Mention this in your report; it isn't a regression.

### 8. Reporting

Report results honestly:

- List the files you added or changed and the behaviors they cover.
- Give the exact commands you ran and their pass/fail/xfail counts.
- Report coverage for the module you targeted only, not the whole repo, and say which lines are still uncovered.
- List every failure, flaky test, skipped module, and blocker (e.g. Playwright/MiniWoB unavailable). Never present unrun or failing checks as passing.
- State review status. Tests aren't done until a human has reviewed the PR.
