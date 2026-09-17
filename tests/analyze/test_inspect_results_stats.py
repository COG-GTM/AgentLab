"""Unit tests for the statistics and error categorization helpers of inspect_results."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from agentlab.analyze import inspect_results
from agentlab.analyze.inspect_results import (
    error_report,
    get_sample_std_err,
    get_std_err,
    map_err_key,
    summarize,
)


def make_result_df(cum_reward, err_msg=None, n_steps=None, **extra):
    n = len(cum_reward)
    data = {
        "cum_reward": cum_reward,
        "err_msg": err_msg if err_msg is not None else [None] * n,
        "n_steps": n_steps if n_steps is not None else [1] * n,
    }
    data.update(extra)
    return pd.DataFrame(data)


def test_get_std_err_binary_uses_bernoulli_formula():
    df = make_result_df([1.0, 1.0, 0.0, 0.0, 1.0])
    mean, std_err = get_std_err(df, "cum_reward")

    expected_mean = 3 / 5
    assert mean == pytest.approx(expected_mean)
    assert std_err == pytest.approx(np.sqrt(expected_mean * (1 - expected_mean) / 5))


def test_get_std_err_binary_ignores_missing_values():
    df = make_result_df([1.0, 0.0, np.nan, 1.0])
    mean, std_err = get_std_err(df, "cum_reward")

    assert mean == pytest.approx(2 / 3)
    assert std_err == pytest.approx(np.sqrt((2 / 3) * (1 / 3) / 3))


def test_get_std_err_all_success_has_zero_std_err():
    df = make_result_df([1.0, 1.0, 1.0])
    mean, std_err = get_std_err(df, "cum_reward")

    assert mean == pytest.approx(1.0)
    assert std_err == pytest.approx(0.0)


def test_get_std_err_non_binary_falls_back_to_sample_std_err():
    df = make_result_df([0.2, 0.5, 1.0])

    assert get_std_err(df, "cum_reward") == get_sample_std_err(df, "cum_reward")


def test_get_sample_std_err_matches_sample_formula():
    values = [0.2, 0.5, 1.0, 0.7]
    df = make_result_df(values)
    mean, std_err = get_sample_std_err(df, "cum_reward")

    assert mean == pytest.approx(np.mean(values))
    assert std_err == pytest.approx(np.std(values, ddof=1) / np.sqrt(len(values)))


def test_get_sample_std_err_singleton_normalizes_nan_to_zero():
    df = make_result_df([0.42])
    mean, std_err = get_sample_std_err(df, "cum_reward")

    assert mean == pytest.approx(0.42)
    assert std_err == 0
    assert not np.isnan(std_err)


def test_get_std_err_singleton_non_binary_normalizes_nan_to_zero():
    df = make_result_df([0.42])
    _, std_err = get_std_err(df, "cum_reward")

    assert std_err == 0


def test_summarize_mixed_success_and_error():
    df = make_result_df(
        cum_reward=[1.0, 0.0, 0.0, 1.0],
        err_msg=[None, None, "boom", None],
        n_steps=[3, 5, 1, 7],
        terminated=[True, True, False, True],
        truncated=[False, False, False, False],
    )
    record = summarize(df)

    assert record["avg_reward"] == pytest.approx(0.5)
    assert record["std_err"] == pytest.approx(round(np.sqrt(0.5 * 0.5 / 4), 3))
    assert record["avg_steps"] == pytest.approx(4.0)
    assert record["n_completed"] == "4/4"
    assert record["n_err"] == 1


def test_summarize_counts_incomplete_episodes():
    df = make_result_df(
        cum_reward=[1.0, 0.0, 0.0],
        err_msg=[None, "boom", None],
        terminated=[True, False, False],
        truncated=[False, False, False],
    )
    record = summarize(df)

    # the errored episode counts as completed, the unfinished one does not
    assert record["n_completed"] == "2/3"
    assert record["n_err"] == 1


def test_summarize_returns_none_when_nothing_completed():
    df = make_result_df(
        cum_reward=[0.0, 0.0],
        terminated=[False, False],
        truncated=[False, False],
    )
    assert summarize(df) is None


def test_summarize_without_cum_reward_column():
    df = pd.DataFrame({"err_msg": [None, None], "n_steps": [1, 2]})
    record = summarize(df)

    assert np.isnan(record["avg_reward"])
    assert np.isnan(record["std_err"])
    assert record["n_completed"] == "0/2"
    assert record["n_err"] == 0


def test_summarize_rejects_non_zero_reward_on_error():
    df = make_result_df(
        cum_reward=[1.0, 1.0],
        err_msg=[None, "boom"],
        terminated=[True, True],
    )
    with pytest.raises(AssertionError):
        summarize(df)


def test_summarize_reports_cum_cost():
    df = make_result_df(
        cum_reward=[1.0, 0.0],
        terminated=[True, True],
        **{"stats.cum_cost": [0.125, 0.25]},
    )
    record = summarize(df)

    assert record["cum_cost"] == pytest.approx(0.375)


def test_summarize_effective_cost_replaces_cum_cost():
    df = make_result_df(
        cum_reward=[1.0, 0.0],
        terminated=[True, True],
        **{"stats.cum_cost": [0.125, 0.25], "stats.cum_effective_cost": [0.1, 0.2]},
    )
    record = summarize(df)

    assert record["cum_effective_cost"] == pytest.approx(0.3)
    assert "cum_cost" not in record


def test_map_err_key_passes_through_none():
    assert map_err_key(None) is None


def test_map_err_key_keeps_message_without_logs():
    assert map_err_key("ValueError: bad thing") == "ValueError: bad thing"


def test_map_err_key_strips_logs():
    err_msg = "ValueError: bad thing\n=== logs ===\nstep 1\nstep 2"
    assert map_err_key(err_msg) == "ValueError: bad thing"


def test_map_err_key_normalizes_token_counts():
    key_a = map_err_key("Error: your messages resulted in 12345 tokens")
    key_b = map_err_key("Error: your messages resulted in 999 tokens")

    assert key_a == key_b == "Error: your messages resulted in x tokens"


def test_map_err_key_normalizes_task_names():
    key_a = map_err_key("Exception uncaught by agent or environment in task webarena.42.")
    key_b = map_err_key("Exception uncaught by agent or environment in task webarena.7.")

    assert key_a == key_b == "Exception uncaught by agent or environment in task <task_name>."


def _fake_exp_result(task_name, task_seed, stack_trace):
    return SimpleNamespace(
        exp_dir=f"/tmp/{task_name}_{task_seed}",
        exp_args=SimpleNamespace(
            env_args=SimpleNamespace(task_name=task_name, task_seed=task_seed)
        ),
        summary_info={"stack_trace": stack_trace},
    )


def test_error_report_buckets_equivalent_errors(monkeypatch):
    exp_results = {
        "/exp/0": _fake_exp_result("task_b", 0, "trace 0"),
        "/exp/1": _fake_exp_result("task_a", 1, "trace 1"),
        "/exp/2": _fake_exp_result("task_c", 2, "trace 2"),
    }
    monkeypatch.setattr(inspect_results, "get_exp_result", lambda exp_dir: exp_results[exp_dir])

    df = pd.DataFrame(
        {
            "exp_dir": ["/exp/0", "/exp/1", "/exp/2"],
            "err_msg": [
                "Error: your messages resulted in 10 tokens\n=== logs ===\nnoise",
                "Error: your messages resulted in 20 tokens",
                "ValueError: something else",
            ],
        }
    )
    report = error_report(df)

    # the two token-count errors collapse into a single bucket
    assert "## 2x : Error: your messages resulted in x tokens" in report
    assert "## 1x : ValueError: something else" in report
    # tasks of a bucket are listed sorted by task name
    assert report.index("* task_a seed: 1") < report.index("* task_b seed: 0")
    assert "Stack Trace: \n trace 2" in report


def test_error_report_limits_stack_traces(monkeypatch):
    exp_results = {f"/exp/{i}": _fake_exp_result(f"task_{i}", i, f"trace {i}") for i in range(3)}
    monkeypatch.setattr(inspect_results, "get_exp_result", lambda exp_dir: exp_results[exp_dir])

    df = pd.DataFrame(
        {
            "exp_dir": list(exp_results.keys()),
            "err_msg": ["boom"] * 3,
        }
    )
    report = error_report(df, max_stack_trace=2)

    assert report.count("Stack Trace:") == 2
    # all tasks of the bucket are still listed
    for i in range(3):
        assert f"* task_{i} seed: {i}" in report
