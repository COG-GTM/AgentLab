---
name: writing-robust-tests
description: Write, fix, or review pytest tests in AgentLab so they are deterministic, isolated, and fail only when behavior breaks. Use when adding tests, fixing flaky/brittle/order-dependent tests, removing @flaky or sleep(), reviewing a PR that touches tests/, or when a test fails with an unhelpful message like "assert 0 == 1".
---

# Writing robust tests in AgentLab

A robust test passes every time the code is correct, fails every time it is not,
and its failure message says what went wrong. It does not depend on timing,
test order, the working directory, other xdist workers, network access, or LLM
output. CI runs `pytest -n 5 -m 'not pricy' tests/` on Python 3.11, so every
test must be safe to run in parallel with any other test.

The rules below come from real brittleness already in `tests/`. Before writing
code, read `reference.md` (same folder) for before/after examples taken from
this repo.

## 1. Set up and get a trustworthy baseline

```bash
source ~/venvs/agentlab/bin/activate      # not the default python
make miniwob                              # MiniWoB on :8080, needed by agent/launch tests
export MINIWOB_URL="http://localhost:8080/miniwob/"
pytest -m 'not pricy' tests/ -n 5 -q -rfEs
```

Before you call a failure "a flaky test", rule out the environment:

- `BrowserType.launch: Executable doesn't exist` means Playwright's browser is
  missing. Run `playwright install chromium`. (The agent tests report this only
  as `cum_reward 0 != 1`; see rule F.)
- `cannot import name 'Completion' from 'anthropic.types'` means the venv
  drifted from `uv.lock`. Check versions with `grep -A1 '^name = "<pkg>"' uv.lock`
  and pin them, or run `uv sync --frozen --extra dev`.
- Check what was **skipped**, not only what failed (`-rs`). Module-level skips
  (gaia, osworld) can hide a whole file.

## 2. Choose the cheapest layer that proves the behavior

| Layer | Use for | Rules |
|---|---|---|
| Pure unit | parsers, prompt building, `CrossProd`, tracking, cost math | No I/O, no browser, no sleep. Should run in milliseconds. |
| Mocked boundary | LLM clients (`response_api`, `chat_api`), OSWorld `DesktopEnv`, Ray graph logic | Patch where the name is looked up (`agentlab.llm.response_api.OpenAI`). Use `MagicMock(spec=...)` / `autospec=True`. |
| MiniWoB end-to-end | `run_experiments`, `GenericAgent` loop, backends | Use the `CheatMiniWoBLLM*` fakes, `task_seed` fixed, `tmp_path` for `study_dir`. |
| `@pytest.mark.pricy` | Real LLM / paid API calls | Must also `skipif` **every** key it uses. Assert structure, never exact text. Never in the default run. |

If a test needs a real WebArena or a local checkpoint, it does not belong in
`tests/`. Make it a script, or mark it `pricy` and add a skip reason.

## 3. Rules

**A. No timing.** Never use `time.sleep` to order threads or tasks. Never
assert on wall-clock values (`start_time < end_time`, "finished in < 2s"). Use
`threading.Event`/`Barrier` to force the interleaving you want. To check order,
record events in a list and assert on the sequence. Don't add `@flaky`: it hides
the bug and triples the runtime. Fix the cause instead.

**B. No shared state that leaks.** Use the `monkeypatch` fixture for
`os.environ`, module attributes, and `sys.modules`, so pytest undoes each change
after the test. Wrap `ray.init()` in `try/finally: ray.shutdown()` (or use a
fixture). Don't set logging levels or globals at import time. If a test calls
code that writes to process-global state (`WebArenaInstanceVars.init()`,
`tracking.TRACKER`), the test is responsible for restoring it. To register a key that the
code under test will write, call `monkeypatch.setenv(key, "")` first. Don't use
`monkeypatch.delenv(key, raising=False)`: on a key that isn't set, it records
nothing, so the value the code writes leaks. Prove it doesn't leak by running
the test followed by a probe test that asserts the env is clean.

**C. No hard-coded paths, and nothing that depends on the cwd.** Never write to
`/tmp/<name>`, because xdist workers will collide. Use `tmp_path`. Build paths
to fixtures from `Path(__file__).parent / "data"`, not `"tests/data/..."`.
Prefer `tmp_path` over `tempfile.TemporaryDirectory()`, because pytest keeps it
for debugging when the test fails.

**D. Never modify committed fixtures.** `tests/data/test_study` is shared. Any
code that writes caches or summaries (`get_study_summary`, relaunch,
`find_incomplete` callers) must run on a copy:
`shutil.copytree(DATA / "test_study", tmp_path / "test_study")`.

**E. Assertions must be able to fail.** `all(...)` over an empty collection is
`True`, so assert the collection is non-empty first. `isinstance` checks alone
and `assert "key" in d` alone only show that the code didn't crash. Assert the
value. Avoid `assert a or b` unless both outcomes are truly correct. **Mutation
check:** break the code under test on purpose (return `{}`, flip a condition,
skip a call). The test must go red. If it stays green, the test is useless.

**F. Failures must explain themselves.** For experiment-level tests, assert
`err_msg is None` (and print `stack_trace`) **before** asserting on reward. If
you don't, an environment problem shows up as `0 != 1`. Compare whole dicts or
records (`assert actual == expected`) so pytest prints the full diff, rather
than looping over keys and stopping at the first mismatch.

**G. Don't over-specify.** Assert the contract, not incidental formatting. For
errors, check the exception type plus a stable substring (`"OpenAIError" in
err_msg`), not the whole multi-line message. Exact strings are fine only when
the string *is* the contract (e.g. `tool_call_to_python_code` output, action
strings).

**H. One case per test.** Use `@pytest.mark.parametrize` (with `ids=`) instead
of `for case in CASES: assert ...`, so every failing case is reported, not only
the first one.

**I. No silent skips.** Never `return` early from a test, because it reports as
PASSED. Use `pytest.skip("reason")`, or `pytest.mark.xfail(strict=True,
reason=...)` for known gaps. Use `pytest.importorskip("pkg")` instead of
`try/except ModuleNotFoundError: pytest.skip(allow_module_level=True)`, which
also hides real import errors in AgentLab code. Delete commented-out tests, or
turn them into `xfail` with a reason.

**J. Deterministic inputs.** Fix seeds (`task_seed=42`). Don't put
`uuid.uuid4()` or `random` in fixtures; use `f"task-{i}"`. Fakes should be
reusable and shared (put them in `conftest.py`), not copied from file to file.

**K. LLM tests.** Unit tests never call a real model; use the `CheatMiniWoBLLM`
fakes or mocks with a spec. In `pricy` tests, assert on structure (a tool call
exists, the action parses, cost > 0) rather than exact wording.

**L. No debug leftovers.** Remove `print(...)` and `if __name__ == "__main__":`
blocks from new tests. Use `pytest path::test -s` to debug.

## 4. Prove the test is robust before opening a PR

Run all of these and paste the results into the PR description:

```bash
# 1. Runs alone, with nothing else to set up state for it
pytest tests/path/test_x.py::test_name -q -p no:cacheprovider

# 2. Stable under parallelism and repetition (CI uses -n 5)
for i in $(seq 10); do pytest tests/path/test_x.py -n 5 -q -p no:cacheprovider || break; done

# 3. Leaves no state behind: run it before a neighbour that would notice
pytest tests/path/test_x.py tests/other/test_y.py -q

# 4. Mutation check: break the code under test, confirm red, revert
# 5. Full default suite still green
pytest -m 'not pricy' tests/ -n 5 -q -rfEs
```

Also check `--durations=10`. A new non-e2e test that takes more than 1s
usually means it is sleeping or doing real I/O.

## 5. Review checklist (for PRs touching tests/)

- [ ] No `sleep`, `@flaky`, or wall-clock assertions
- [ ] Env, globals, and Ray are restored (`monkeypatch`, `finally`)
- [ ] `tmp_path` only; no `/tmp/...` and no cwd-relative paths; fixtures copied before any write
- [ ] Every assertion fails when the code is broken (mutation check done)
- [ ] `err_msg` asserted before reward; dict-equality diffs
- [ ] Parametrized cases; no early `return`; skips have reasons and use `importorskip`
- [ ] Paid calls are `pricy` and have a `skipif` for every key used
- [ ] All checks from section 4 run, with output in the PR
