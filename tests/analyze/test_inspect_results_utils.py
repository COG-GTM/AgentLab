import json
import os

import numpy as np
import pandas as pd
import pytest

from agentlab.analyze.inspect_results import (
    _benchmark_from_task_name,
    _extract_ablation_study,
    _find_diff,
    _get_mtimes,
    _is_stale,
    _rename_bool_flags,
    categorize_error,
    flag_report,
    get_constants_and_variables,
    get_sample_std_err,
    get_std_err,
    map_err_key,
    shrink_columns,
    split_by_key,
    summarize,
    summarize_stats,
)


@pytest.fixture
def episodes():
    return pd.DataFrame(
        {
            "cum_reward": [1.0, 0.0, 1.0, 0.0],
            "err_msg": [None, None, None, "boom"],
            "truncated": [False, False, False, False],
            "terminated": [True, True, False, False],
            "n_steps": [3, 4, 5, 6],
            "stats.cum_cost": [0.1, 0.2, 0.3, 0.4],
        }
    )


def test_get_constants_and_variables():
    df = pd.DataFrame({"const": [np.int64(1), np.int64(1)], "var": [1, 2], "lst": [[1], [1]]})
    constants, variables, out = get_constants_and_variables(df)
    assert constants == {"const": 1}
    assert type(constants["const"]) is int
    assert variables == ["var", "lst"]  # unhashable columns are treated as variables
    assert out is df

    _, _, dropped = get_constants_and_variables(df, drop_constants=True)
    assert list(dropped.columns) == ["var", "lst"]


def test_get_std_err_binary():
    df = pd.DataFrame({"r": [1, 0, 1, 1, np.nan]})
    mean, std_err = get_std_err(df, "r")
    assert mean == pytest.approx(0.75)
    assert std_err == pytest.approx(np.sqrt(0.75 * 0.25 / 4))


def test_get_std_err_non_binary_uses_sample_std_err():
    df = pd.DataFrame({"r": [0.5, 1.5, 2.5]})
    assert get_std_err(df, "r") == get_sample_std_err(df, "r")
    mean, std_err = get_sample_std_err(df, "r")
    assert mean == pytest.approx(1.5)
    assert std_err == pytest.approx(1 / np.sqrt(3))


def test_get_sample_std_err_single_value():
    mean, std_err = get_sample_std_err(pd.DataFrame({"r": [0.3]}), "r")
    assert mean == pytest.approx(0.3)
    assert std_err == 0


def test_summarize(episodes):
    record = summarize(episodes)
    assert record["avg_reward"] == 0.5
    assert record["std_err"] == pytest.approx(0.25)
    assert record["avg_steps"] == 4.5
    assert record["n_completed"] == "3/4"  # 2 terminated + 1 error
    assert record["n_err"] == 1
    assert record["cum_cost"] == pytest.approx(1.0)


def test_summarize_effective_cost_replaces_cost(episodes):
    episodes["stats.cum_effective_cost"] = [0.5, 0.5, 0.5, 0.5]
    record = summarize(episodes)
    assert "cum_cost" not in record
    assert record["cum_effective_cost"] == 2.0


def test_summarize_without_rewards():
    record = summarize(pd.DataFrame({"n_steps": [1, 2]}))
    assert np.isnan(record["avg_reward"])
    assert record["n_completed"] == "0/2"
    assert record["n_err"] == 0


def test_summarize_nothing_completed(episodes):
    episodes["terminated"] = False
    episodes["err_msg"] = None
    assert summarize(episodes) is None


def test_summarize_error_with_reward_fails_sanity_check(episodes):
    episodes.loc[3, "cum_reward"] = 1.0
    with pytest.raises(AssertionError):
        summarize(episodes)


def test_summarize_stats(episodes):
    episodes["stats.max_tokens"] = [10, 30, 20, 5]
    record = summarize_stats(episodes)
    assert record["avg_reward"] == 0.5
    assert record["cum_cost"] == pytest.approx(1.0)
    assert record["max_tokens"] == 30


def test_summarize_stats_unknown_op(episodes):
    episodes["stats.avg_tokens"] = 1
    with pytest.raises(ValueError, match="Unknown stats operation: avg"):
        summarize_stats(episodes)


def test_summarize_stats_nothing_completed(episodes):
    episodes["terminated"] = False
    episodes["err_msg"] = None
    assert summarize_stats(episodes) is None


def test_find_diff():
    assert _find_diff((1, 2, 3), (1, 0, 3)) == [1]
    assert _find_diff((1, 2), (1, 2)) == []
    assert _find_diff((), ()) == []


@pytest.fixture
def ablation_report():
    index = pd.MultiIndex.from_tuples(
        [(True, 1), (False, 1), (True, 2)], names=["use_cot", "n_retry"]
    )
    return pd.DataFrame({"avg_reward": [0.5, 0.4, 0.6]}, index=index)


def test_extract_ablation_study(ablation_report):
    out = _extract_ablation_study(ablation_report)
    assert list(out.index) == ["Initial Configuration", "-use_cot", "n_retry←2"]
    assert list(out.columns) == ["avg_reward"]
    assert out["avg_reward"].tolist() == [0.5, 0.4, 0.6]


def test_extract_ablation_study_progression(ablation_report):
    out = _extract_ablation_study(ablation_report, progression=True)
    assert list(out.index) == [
        "Initial Configuration",
        "↳ use_cot=False",
        "↳ use_cot=True, n_retry=2",
    ]


def test_rename_bool_flags(ablation_report):
    out = _rename_bool_flags(ablation_report.copy())
    assert set(out.index.levels[0]) == {"✓", "-"}
    assert list(out.index.levels[1]) == [1, 2]


def test_rename_bool_flags_single_index():
    df = pd.DataFrame({"a": [1]}, index=[True])
    assert _rename_bool_flags(df).index.tolist() == [True]


def test_flag_report():
    index = pd.MultiIndex.from_tuples(
        [(True, True), (True, False), (False, True), (False, False)], names=["f1", "f2"]
    )
    report = pd.DataFrame({"avg_reward": [1.0, 0.6, 0.5, 0.2]}, index=index)
    out = flag_report(report)
    assert list(out.index) == ["f1", "f2"]  # sorted by ratio
    assert out.loc["f1", "avg_true"] == 0.8
    assert out.loc["f1", "avg_false"] == 0.35
    assert out.loc["f1", "ratio"] == pytest.approx(2.29)
    assert out.loc["f2", "ratio"] == pytest.approx(1.88)


def test_flag_report_single_level():
    report = pd.DataFrame({"avg_reward": [1.0]}, index=pd.Index(["a"], name="agent"))
    assert flag_report(report) is None


def test_shrink_columns():
    df = pd.DataFrame(
        {"avg_reward": [0.5000, 0.125], "n_completed": ["1/2", "2/2"]},
        index=pd.Index(["a", "b"], name="agent_name"),
    )
    out = shrink_columns(df)
    assert list(out.columns) == ["avg\nreward", "n\ncompleted"]
    assert out.index.names == ["agent\nname"]
    assert out["avg\nreward"].tolist() == ["0.5", "0.125"]
    assert out["n\ncompleted"].tolist() == ["1/2", "2/2"]
    assert list(df.columns) == ["avg_reward", "n_completed"]  # original untouched

    assert shrink_columns(df, also_wrap_index=False).index.names == ["agent_name"]


def test_map_err_key():
    assert map_err_key(None) is None
    msg = "Error: your messages resulted in 4242 tokens\n=== logs ===\nlots of logs"
    assert map_err_key(msg) == "Error: your messages resulted in x tokens"
    msg = "Exception uncaught by agent or environment in task miniwob.click-test.\n=== logs ==="
    assert map_err_key(msg) == "Exception uncaught by agent or environment in task <task_name>."


def test_categorize_error_without_error():
    assert categorize_error({"err_msg": None}) is None
    assert categorize_error({}) is None
    assert categorize_error({"err_msg": np.nan}) is None


@pytest.mark.parametrize(
    "task, expected",
    [
        ("miniwob.click-test", "miniwob"),
        ("workarena.servicenow.x", "workarena"),
        ("plain", "plain"),
    ],
)
def test_benchmark_from_task_name(task, expected):
    assert _benchmark_from_task_name(task) == expected


def test_split_by_key_column():
    df = pd.DataFrame(
        {
            "agent.agent_name": ["a", "b", "a"],
            "env.task_name": ["t1", "t1", "t2"],
            "cum_reward": [1, 0, 1],
        }
    )
    out = split_by_key(df, "agent.agent_name")
    assert set(out) == {"a", "b"}
    assert len(out["a"]) == 2 and len(out["b"]) == 1


def test_split_by_key_from_index():
    df = pd.DataFrame(
        {"agent.agent_name": ["a", "b"], "env.task_name": ["t", "t"], "cum_reward": [1, 0]}
    ).set_index("agent.agent_name")
    out = split_by_key(df, "agent.agent_name")
    assert set(out) == {"a", "b"}
    assert out["b"]["cum_reward"].tolist() == [0]


def test_get_mtimes(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "file.txt").write_text("x")
    (tmp_path / "_hidden.json").write_text("{}")
    (tmp_path / ".dot").write_text("")
    (tmp_path / "summary.csv").write_text("")
    mtimes = _get_mtimes(tmp_path, whitelist=(tmp_path / "summary.csv",))
    assert set(mtimes) == {"sub", os.path.join("sub", "file.txt")}


def test_is_stale(tmp_path):
    summary = tmp_path / "summary_df.csv"
    data = tmp_path / "exp" / "result.json"
    data.parent.mkdir()
    data.write_text("{}")

    assert _is_stale(tmp_path, summary)  # no summary yet
    summary.write_text("x")
    assert not _is_stale(tmp_path, summary)
    saved = json.loads((tmp_path / "_last_modification_times.json").read_text())
    assert os.path.join("exp", "result.json") in saved

    summary.write_text("y")  # summary changes are ignored
    assert not _is_stale(tmp_path, summary)

    os.utime(data, (1, 1))
    assert _is_stale(tmp_path, summary)
    assert not _is_stale(tmp_path, summary)
