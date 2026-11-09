# Before/after examples from AgentLab's own tests

Each example is real code from `tests/`. Items marked **(fixed)** were fixed in
the PR that added this skill, so the after version is in the repo now. The rest
are still open: fix them when you next touch the file.

## A. Sleeping to order threads (fixed) — `tests/llm/test_tracking.py`

Before: the test always took 2s, and on a loaded CI runner `thread_2` could
start outside `thread_1`'s tracker window.

```python
def thread_1(results=None):
    with tracking.set_tracker() as tracker:
        time.sleep(1)
        _, _ = call_llm()
        time.sleep(1)
```

After: Events force the exact interleaving. It runs in milliseconds and the
result doesn't depend on the scheduler.

```python
t1_tracker_open, t2_done = threading.Event(), threading.Event()

def thread_1(results=None):
    with tracking.set_tracker() as tracker:
        t1_tracker_open.set()
        assert t2_done.wait(timeout=10)
        _, _ = call_llm()
    results[0] = tracker.stats
```

## A/B. `@flaky` plus an unguarded `ray.init` (fixed) — `tests/experiments/test_ray.py`

Before: `@flaky(max_runs=3)` covered up failures. If an assertion failed between
`ray.init()` and `ray.shutdown()`, Ray stayed running and the next `ray.init()`
in that worker failed.

After: a fixture shuts Ray down in `finally`, and the timing assertions became
causal ones (`parent.end_time <= child.start_time`). Those can only fail if the
scheduler actually broke a dependency.

## B. Env vars leaking between tests (fixed) — `tests/experiments/test_multi_server.py`

`WebArenaInstanceVars.init()` writes `WA_*` into `os.environ`. Before the fix,
every later test in the same xdist worker could see fake Azure URLs.

```python
def test_webarena_multiserver(monkeypatch):
    for key in instance_1.make_env_vars():
        monkeypatch.setenv(key, "")  # records "was unset", so teardown deletes it again
    instance_1.init()
```

Gotcha: the first version of this fix used `monkeypatch.delenv(key, raising=False)`.
It still leaked, because `delenv` records nothing for a key that isn't already
set. A probe test that ran next and asserted no `WA_*` keys caught it.

## C. Hard-coded `/tmp` and cwd-relative paths (open) — `tests/agents/test_gaia_agent.py`

```python
exp_dir = "/tmp/gaia_unit_test"                     # shared by every xdist worker and every run
"file_path": "tests/data/32102e3e-....xlsx",        # breaks unless pytest runs from repo root
data = [{"task_id": str(uuid.uuid4()), ...}]        # different ids every run
```

Fix: take the `tmp_path` fixture, use `DATA = Path(__file__).parent.parent / "data"`,
and use `task_id=f"task-{i}"`. Also replace the `try/except ModuleNotFoundError`
module skip with `pytest.importorskip("tapeagents")`, so a broken import inside
`agentlab.agents.tapeagent` fails instead of being skipped.

## D. Shared committed fixture (pattern to copy) — `tests/analyze/test_inspect_results.py`

`get_study_summary` writes a cache file next to the study. This test does it
right: it copies `tests/data/test_study` into a temp dir first. Do the same in
any test that calls code that might write.

## E. An assertion that could not fail (fixed) — `tests/llm/test_tracking.py::test_get_pricing_openai`

`langchain-community` isn't in `uv.lock`, so `get_pricing_openai()` always
returns `{}` in CI. `all(... for model in {})` is `True`, so the test passed
without checking anything.

```python
pytest.importorskip("langchain_community", reason="OpenAI pricing comes from langchain")
pricing = tracking.get_pricing_openai()
assert pricing, "expected at least one OpenAI model in the pricing table"
```

It now reports SKIPPED in CI, which is honest. Other weak checks still open:
`test_llm_configs` (only `isinstance`), `test_get_reproducibility_info` (only
`"key" in info`), and `assert math.isnan(r) or r == 0` in `test_launch_exp`.

## F. Failure messages that hide the cause (fixed) — `tests/agents/test_agent.py`, `tests/experiments/test_launch_exp.py`

Before: with the Playwright browser missing, every agent test failed with
`assert np.int64(0) == 1`, and the real error was buried in captured logs.

After: `_assert_record` asserts `err_msg is None` first, puts `stack_trace` in
the assertion message, and compares one dict so pytest shows every mismatched
key at once:

```
AssertionError: experiment crashed:
... BrowserType.launch: Executable doesn't exist at .../chrome-linux/chrome
```

## G. Over-specified error string (fixed) — `test_agent.py::test_llm_error_no_success`

Before: an exact match on the full multi-line message
`"Exception uncaught by agent or environment in task miniwob.click-test.\nOpenAIError:\nLLM failed to respond"`.
Any change to the wording broke it. After: assert the exception type and the
cause as substrings.

## H. Loop instead of parametrize (fixed) — `test_response_api.py::test_tool_call_to_python_code`

Before: a `for` loop over `EDGE_CASES`, so only the first failing case was
reported. After: `@pytest.mark.parametrize(..., ids=[case[0] for case in EDGE_CASES])`.

## I. Silent pass via `return` (fixed) — `test_generic_prompt.py::test_main_prompt_elements_gone_one_at_a_time`

Before: `return  # TODO`, which made two untested flags report PASSED. After:
`pytest.skip("TODO ...")`, so they report SKIPPED. Still open: about 50
commented-out lines in `test_reproducibility_util.py` and `test_llm_utils.py`.
Delete them or turn them into `xfail(strict=True)`.

## J. Copy-pasted fakes (open) — `tests/agents/test_agent.py`

`CheatMiniWoBLLM_ParseRetry` and `CheatLLM_LLMError` each re-implement the
"find the button bid" regex from `CheatMiniWoBLLM`. Move one configurable fake
into `tests/conftest.py`, so a change to the AXTree format has to be fixed in
only one place.

## K. Live-LLM assertions (open) — `test_tracking.py::test_openai_chat_model` etc.

`assert "5" in answer.get("content")` depends on what the model says. These are
correctly marked `pricy`. If you touch them, assert structure (non-empty content,
cost > 0). Also, `test_response_api.py::test_multi_action_tool_calls` calls
Anthropic but only skips when `OPENAI_API_KEY` is missing. Add an
`ANTHROPIC_API_KEY` skip.

## Environment gotchas seen while writing this

- The venv's `anthropic` drifted to 2.x, while `uv.lock` pins 0.62.0. That breaks
  collection of `test_response_api.py` and `test_litellm_api.py` with
  `cannot import name 'Completion'`.
- The Playwright chromium build was missing, so all MiniWoB tests failed as
  `cum_reward 0 != 1`. Fix with `playwright install chromium`.
