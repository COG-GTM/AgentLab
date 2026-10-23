import json
import time

import numpy as np
import pandas as pd
import pytest

from agentlab.analyze import inspect_results as ir


def _result_df():
    return pd.DataFrame(
        {
            "env.task_name": ["miniwob.a", "miniwob.a", "miniwob.b", "miniwob.b"] * 2,
            "agent.agent_name": ["ag1"] * 4 + ["ag2"] * 4,
            "agent.flags.use_x": [True] * 4 + [False] * 4,
            "agent._private": [1, 2, 3, 4, 5, 6, 7, 8],
            "cum_reward": [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            "n_steps": [1, 2, 3, 4, 5, 6, 7, 8],
            "err_msg": [None] * 7 + [None],
            "stack_trace": [None] * 8,
            "truncated": [False] * 8,
            "terminated": [True] * 8,
            "stats.cum_cost": [0.1] * 8,
            "constant": ["c"] * 8,
        }
    )


def test_get_constants_and_variables():
    df = pd.DataFrame({"a": [1, 1], "b": [1, 2], "c": [np.int64(3)] * 2, "d": [[1], [2]]})
    constants, variables, out = ir.get_constants_and_variables(df, drop_constants=True)
    assert constants == {"a": 1, "c": 3}
    assert isinstance(constants["c"], int)
    assert variables == ["b", "d"]
    assert list(out.columns) == ["b", "d"]


def test_set_index_from_variables():
    df = _result_df()
    df.loc[0, "agent.flags.use_x"] = None
    ir.set_index_from_variables(df)
    assert df.index.names == [
        "env.task_name",
        "agent.agent_name",
        "env.benchmark",
        "agent.flags.use_x",
    ]
    assert set(df.index.get_level_values("env.benchmark")) == {"miniwob"}
    # None values in index variables are replaced by the string "None"
    assert "None" in set(df.index.get_level_values("agent.flags.use_x"))


def _indexed():
    df = _result_df()
    ir.set_index_from_variables(df)
    return df


def test_summarize_with_and_without_rewards():
    df = _result_df()
    s = ir.summarize(df[df["agent.agent_name"] == "ag1"])
    assert s["avg_reward"] == 0.75
    assert s["n_completed"] == "4/4"
    assert s["n_err"] == 0
    assert s["cum_cost"] == 0.4

    s = ir.summarize(pd.DataFrame({"x": [1, 2]}))
    assert s["n_completed"] == "0/2" and np.isnan(s["avg_reward"])

    not_done = df.assign(terminated=False)
    assert ir.summarize(not_done) is None


def test_summarize_effective_cost_replaces_cost():
    df = _result_df().assign(**{"stats.cum_effective_cost": 0.2})
    s = ir.summarize(df)
    assert "cum_cost" not in s and s["cum_effective_cost"] == 1.6


def test_summarize_stats():
    df = _result_df().assign(**{"stats.max_tokens": range(8)})
    s = ir.summarize_stats(df)
    assert s["cum_cost"] == pytest.approx(0.8)
    assert s["max_tokens"] == 7
    assert ir.summarize_stats(df.assign(terminated=False)) is None
    with pytest.raises(ValueError, match="Unknown stats"):
        ir.summarize_stats(df.assign(**{"stats.avg_x": 1}))


def test_std_err_helpers():
    df = pd.DataFrame({"m": [1, 0, 1, 0]})
    mean, err = ir.get_std_err(df, "m")
    assert mean == 0.5 and err == pytest.approx(0.25)

    df = pd.DataFrame({"m": [0.5, 1.5]})
    mean, err = ir.get_std_err(df, "m")
    assert mean == 1.0 and err == pytest.approx(0.5)

    _, err = ir.get_sample_std_err(pd.DataFrame({"m": [0.3]}), "m")
    assert err == 0


def test_global_report_multiple_agents_and_flag_report():
    report = ir.global_report(_indexed())
    assert list(report.index.names) == ["agent.agent_name", "env.benchmark", "use_x"]
    assert report["avg_reward"].iloc[0] >= report["avg_reward"].iloc[1]

    flags = ir.flag_report(report)
    assert list(flags.index) == ["use_x"]
    assert flags.loc["use_x", "avg_true"] == 0.75
    assert flags.loc["use_x", "avg_false"] == 0.25
    assert flags.loc["use_x", "ratio"] == 3.0


def test_global_report_single_configuration():
    df = _result_df()
    df = df[df["agent.agent_name"] == "ag1"].set_index("env.task_name")
    report = ir.global_report(df)
    assert "[ALL TASKS]" in report.index
    assert report.loc["[ALL TASKS]", "avg_reward"] == 0.75
    assert ir.flag_report(report) is None


def test_summarize_study_and_split_by_key():
    df = _indexed()
    summary = ir.summarize_study(df)
    assert len(summary) == 2

    parts = ir.split_by_key(df, "agent.agent_name")
    assert set(parts) == {"ag1", "ag2"}
    assert len(parts["ag1"]) == 4


def test_report_constant_and_variables(capsys):
    df = _result_df().assign(stack_trace=["t"] * 7 + ["u"])
    ir.report_constant_and_variables(df, show_stack_traces=False)
    out = capsys.readouterr().out
    assert "constant: c" in out
    assert "agent._private: n_unique=8" in out
    assert "stack_trace" not in out


def test_rename_bool_flags_and_shrink_columns():
    report = ir.global_report(_indexed())
    renamed = ir._rename_bool_flags(report.copy())
    assert set(renamed.index.get_level_values("use_x")) == {"✓", "-"}

    shrunk = ir.shrink_columns(
        pd.DataFrame({"avg_reward": [0.5000]}, index=pd.Index([0], name="a_b"))
    )
    assert list(shrunk.columns) == ["avg\nreward"]
    assert shrunk.index.names == ["a\nb"]
    assert shrunk.iloc[0, 0] == "0.5"


def test_find_diff():
    assert ir._find_diff((1, 2, 3), (1, 0, 4)) == [1, 2]


def test_map_err_key():
    assert ir.map_err_key(None) is None
    msg = (
        "Error: your messages resulted in 12345 tokens\n"
        "Exception uncaught by agent or environment in task miniwob.abc\n=== logs ===\nblah"
    )
    key = ir.map_err_key(msg)
    assert "x tokens" in key
    assert "<task_name>." in key
    assert "blah" not in key


def test_categorize_error_without_error():
    assert ir.categorize_error({"err_msg": None}) is None
    assert ir.categorize_error({}) is None


def test_format_log():
    assert ir._format_log(type("R", (), {"logs": None})()) == "No log found"
    short = type("R", (), {"logs": "a\nb"})()
    assert ir._format_log(short) == "a\nb"

    lines = [f"line {i}" for i in range(100)]
    long_log = type("R", (), {"logs": "\n".join(lines)})()
    out = ir._format_log(long_log, head_lines=2, tail_lines=3)
    assert out.startswith("line 0\nline 1\n...")
    assert out.endswith("line 97\nline 98\nline 99")

    with_tb = "\n".join(lines + ["action: click", "Traceback (most recent call last):", "boom"])
    out = ir._format_log(type("R", (), {"logs": with_tb})(), head_lines=2, tail_lines=3)
    assert out.endswith("action: click\nTraceback (most recent call last):\nboom")


def test_benchmark_from_task_name():
    assert ir._benchmark_from_task_name("workarena.servicenow.x") == "workarena"


def test_is_stale_tracks_modification_times(tmp_path):
    (tmp_path / "exp").mkdir()
    f = tmp_path / "exp" / "file.txt"
    f.write_text("a")
    summary = tmp_path / "study_summary.csv"

    assert ir._is_stale(tmp_path, summary)  # no summary yet
    summary.write_text("x")
    # the summary itself is excluded from the tracked modification times
    assert not ir._is_stale(tmp_path, summary)

    mtimes = json.loads((tmp_path / "_last_modification_times.json").read_text())
    assert "exp/file.txt" in mtimes and "study_summary.csv" not in mtimes

    time.sleep(0.01)
    f.write_text("b")
    import os

    os.utime(f, (time.time() + 10, time.time() + 10))
    assert ir._is_stale(tmp_path, summary)


def test_get_study_summary_uses_cache(tmp_path, monkeypatch):
    df = _indexed()
    calls = []

    def fake_load(study_dir, progress_fn=None):
        calls.append(study_dir)
        return df

    monkeypatch.setattr(ir, "load_result_df", fake_load)
    sentinel = {}
    ir.get_study_summary(tmp_path, sentinel=sentinel)
    assert sentinel["from_cache"] is False
    ir.get_study_summary(tmp_path, sentinel=sentinel)
    assert sentinel["from_cache"] is True
    ir.get_study_summary(tmp_path, ignore_cache=True, ignore_stale=True, sentinel=sentinel)
    assert sentinel["from_cache"] is False
    assert len(calls) == 2


def test_get_study_summary_none_and_get_all_summaries(tmp_path, monkeypatch):
    df = _indexed()
    (tmp_path / "_hidden").mkdir()
    (tmp_path / "empty").mkdir()
    (tmp_path / "study").mkdir()
    monkeypatch.setattr(
        ir,
        "load_result_df",
        lambda d, progress_fn=None: df if d.name == "study" else None,
    )
    assert ir.get_study_summary(tmp_path / "empty") is None
    summaries = ir.get_all_summaries(tmp_path)
    assert set(summaries.index) == {"study"}
