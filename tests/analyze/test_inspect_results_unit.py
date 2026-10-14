"""Unit tests for agentlab.analyze.inspect_results using synthetic DataFrames and result dirs."""

import json
import os
import pickle
import gzip
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from agentlab.analyze import inspect_results as ir
from agentlab.experiments import loop
from agentlab.experiments.loop import AbstractAgentArgs, EnvArgs, ExpArgs, ExpResult, StepInfo

TEST_STUDY_DIR = Path(__file__).parent.parent / "data" / "test_study"


@dataclass
class FakeAgentArgs(AbstractAgentArgs):
    temperature: float = 0.1
    use_html: bool = True

    def make_agent(self):
        raise NotImplementedError


def make_exp_dir(
    root: Path,
    name: str,
    task_name: str = "miniwob.click",
    seed: int = 0,
    agent_args: AbstractAgentArgs = None,
    summary: dict = None,
    order: int = None,
    logs: str = None,
    task_messages: list = None,
) -> Path:
    """Write a minimal experiment dir: exp_args.pkl (+ summary_info.json, log, steps)."""
    exp_dir = Path(root) / name
    exp_dir.mkdir(parents=True)
    agent_args = agent_args or FakeAgentArgs(agent_name="FakeAgent")
    exp_args = ExpArgs(
        agent_args=agent_args,
        env_args=EnvArgs(task_name=task_name, task_seed=seed),
        order=order,
    )
    with open(exp_dir / "exp_args.pkl", "wb") as f:
        pickle.dump(exp_args, f)
    if summary is not None:
        (exp_dir / "summary_info.json").write_text(json.dumps(summary))
    if logs is not None:
        (exp_dir / "experiment.log").write_text(logs)
    for i, message in enumerate(task_messages or []):
        task_info = None if message is None else {"message": message}
        with gzip.open(exp_dir / f"step_{i}.pkl.gz", "wb") as f:
            pickle.dump(StepInfo(step=i, task_info=task_info), f)
    return exp_dir


def done_summary(reward=1.0, err_msg=None, stack_trace=None, n_steps=3, **extra):
    summary = {
        "n_steps": n_steps,
        "cum_reward": reward,
        "err_msg": err_msg,
        "stack_trace": stack_trace,
        "terminated": err_msg is None,
        "truncated": False,
    }
    summary.update(extra)
    return summary


def episodes_df(**columns) -> pd.DataFrame:
    """Small per-episode frame with sane defaults for missing columns."""
    n = len(next(iter(columns.values())))
    base = {
        "cum_reward": [0.0] * n,
        "err_msg": [None] * n,
        "stack_trace": [None] * n,
        "n_steps": [1] * n,
        "terminated": [True] * n,
        "truncated": [False] * n,
    }
    base.update(columns)
    return pd.DataFrame(base)


@pytest.fixture(autouse=True)
def isolated_exp_result_cache(monkeypatch):
    monkeypatch.setattr(loop, "EXP_RESULT_CACHE", {})


# ---------------------------------------------------------------------------
# get_constants_and_variables / set_index_from_variables
# ---------------------------------------------------------------------------


class TestGetConstantsAndVariables:
    def test_splits_constants_from_variables(self):
        df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3], "c": ["x", "x", "x"]})
        constants, variables, out = ir.get_constants_and_variables(df)
        assert constants == {"a": 1, "c": "x"}
        assert variables == ["b"]
        assert out is df

    def test_numpy_scalars_become_python_scalars(self):
        df = pd.DataFrame({"a": np.array([7, 7], dtype=np.int64)})
        constants, _, _ = ir.get_constants_and_variables(df)
        assert constants["a"] == 7
        assert type(constants["a"]) is int

    def test_unhashable_column_is_variable(self):
        df = pd.DataFrame({"lst": [[1], [1]], "k": [0, 0]})
        constants, variables, _ = ir.get_constants_and_variables(df)
        assert variables == ["lst"]
        assert constants == {"k": 0}

    def test_nan_only_column_is_constant(self):
        df = pd.DataFrame({"n": [np.nan, np.nan], "v": [1, 2]})
        constants, variables, _ = ir.get_constants_and_variables(df)
        assert "n" in constants and np.isnan(constants["n"])
        assert variables == ["v"]

    def test_drop_constants_returns_new_frame_without_constants(self):
        df = pd.DataFrame({"a": [1, 1], "b": [1, 2]})
        _, _, out = ir.get_constants_and_variables(df, drop_constants=True)
        assert list(out.columns) == ["b"]
        assert list(df.columns) == ["a", "b"]


class TestSetIndexFromVariables:
    def _df(self):
        return pd.DataFrame(
            {
                "env.task_name": ["miniwob.a", "miniwob.b", "workarena.c", "miniwob.a"],
                "env.task_seed": [0, 1, 2, 3],
                "agent.agent_name": ["A", "A", "B", "B"],
                "agent.flags.use_html": [True, False, True, False],
                "agent.model_url": ["u1", "u2", "u3", "u4"],
                "agent.extra_info": [1, 2, 3, 4],
                "agent._private": [1, 2, 3, 4],
                "agent.constant": [5, 5, 5, 5],
                "cum_reward": [1.0, 0.0, 0.5, 1.0],
            }
        )

    def test_default_index(self):
        df = self._df()
        ir.set_index_from_variables(df)
        assert list(df.index.names) == [
            "env.task_name",
            "agent.agent_name",
            "env.benchmark",
            "agent.flags.use_html",
        ]
        assert df.index.is_monotonic_increasing
        benchmarks = df.index.get_level_values("env.benchmark")
        assert sorted(set(benchmarks)) == ["miniwob", "workarena"]
        # Non-agent variables and constants stay as columns
        assert "env.task_seed" in df.columns
        assert "agent.constant" in df.columns

    def test_existing_benchmark_column_is_kept(self):
        df = self._df()
        df["env.benchmark"] = ["bench"] * 4
        ir.set_index_from_variables(df)
        assert set(df.index.get_level_values("env.benchmark")) == {"bench"}

    def test_without_agent_and_benchmark(self):
        df = self._df()
        ir.set_index_from_variables(df, add_agent_and_benchmark=False)
        assert list(df.index.names) == [
            "env.task_name",
            "agent.agent_name",
            "agent.flags.use_html",
        ]
        assert "env.benchmark" not in df.columns

    def test_custom_white_and_black_lists(self):
        df = self._df()
        ir.set_index_from_variables(
            df,
            index_white_list=("env.task_seed", "agent.*"),
            index_black_list=("*flags*", "*model_url*", "*extra*", "*._*"),
            add_agent_and_benchmark=False,
        )
        assert list(df.index.names) == ["env.task_name", "env.task_seed", "agent.agent_name"]

    def test_nan_in_index_variable_replaced_by_string_none(self):
        df = self._df()
        df["agent.flags.temp"] = [0.1, None, 0.1, 0.2]
        with mock.patch.object(ir, "warn") as warn:
            ir.set_index_from_variables(df)
        warn.assert_called_once()
        assert "agent.flags.temp" in warn.call_args[0][0]
        assert "None" in list(df.index.get_level_values("agent.flags.temp"))


# ---------------------------------------------------------------------------
# load_result_df
# ---------------------------------------------------------------------------


@pytest.fixture
def two_agent_study(tmp_path):
    study = tmp_path / "study"
    agents = [
        FakeAgentArgs(agent_name="FakeAgent", temperature=0.1, use_html=True),
        FakeAgentArgs(agent_name="FakeAgent", temperature=0.5, use_html=False),
    ]
    rewards = {(0, "miniwob.a"): 1.0, (0, "miniwob.b"): 0.0, (1, "miniwob.a"): 1.0}
    order = 0
    for a_idx, agent in enumerate(agents):
        for task in ["miniwob.a", "miniwob.b"]:
            summary = (
                done_summary(reward=rewards[(a_idx, task)], **{"stats.cum_cost": 0.5})
                if (a_idx, task) in rewards
                else done_summary(reward=0.0, err_msg="boom", stack_trace="Traceback")
            )
            make_exp_dir(
                study,
                f"exp_{a_idx}_{task}",
                task_name=task,
                seed=a_idx,
                agent_args=agent,
                summary=summary,
                order=order,
            )
            order += 1
    return study


class TestLoadResultDf:
    def test_empty_dir_returns_none(self, tmp_path):
        assert ir.load_result_df(tmp_path, progress_fn=None) is None

    def test_loads_and_indexes_synthetic_study(self, two_agent_study):
        df = ir.load_result_df(two_agent_study, progress_fn=None)
        assert len(df) == 4
        assert list(df.index.names) == [
            "env.task_name",
            "agent.agent_name",
            "env.benchmark",
            "agent.temperature",
            "agent.use_html",
        ]
        assert not any("_args" in c for c in df.columns)
        assert {"exp_dir", "cum_reward", "err_msg", "env.task_seed"} <= set(df.columns)
        assert df["err_msg"].notnull().sum() == 1

    def test_no_index_and_keep_suffix(self, two_agent_study):
        df = ir.load_result_df(
            two_agent_study, progress_fn=None, set_index=False, remove_args_suffix=False
        )
        assert isinstance(df.index, pd.RangeIndex)
        assert "agent_args.agent_name" in df.columns
        assert "env_args.task_name" in df.columns

    def test_progress_fn_is_used(self, two_agent_study):
        descs = []

        def progress(items, desc=None):
            descs.append(desc)
            return items

        df = ir.load_result_df(two_agent_study, progress_fn=progress)
        assert len(df) == 4
        assert descs == ["Searching experiments directories.", "Loading results"]

    def test_reuses_result_df_exp_results(self, two_agent_study):
        results = [ExpResult(d) for d in sorted(two_agent_study.iterdir())[:2]]
        result_df = pd.DataFrame({"exp_result": results})
        df = ir.load_result_df(None, progress_fn=None, result_df=result_df, set_index=False)
        assert [Path(p) for p in df["exp_dir"]] == [r.exp_dir for r in results]

    def test_loads_repo_test_study(self):
        df = ir.load_result_df(TEST_STUDY_DIR, progress_fn=None, set_index=False)
        assert len(df) == 2
        assert set(df["env.task_name"]) == {"miniwob.ascending-numbers"}
        assert set(df["env.task_seed"]) == {64, 68}
        # one exp has no summary_info.json, so its summary columns are NaN
        assert df["err_msg"].notnull().sum() == 1
        assert df["n_steps"].isnull().sum() == 1


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


class TestStdErr:
    def test_binary_metric(self):
        df = pd.DataFrame({"r": [1, 0, 1, 1, np.nan]})
        mean, se = ir.get_std_err(df, "r")
        assert mean == pytest.approx(0.75)
        assert se == pytest.approx(np.sqrt(0.75 * 0.25 / 4))

    def test_non_binary_falls_back_to_sample_std_err(self):
        df = pd.DataFrame({"r": [0.0, 0.5, 1.0]})
        mean, se = ir.get_std_err(df, "r")
        assert mean == pytest.approx(0.5)
        assert se == pytest.approx(0.5 / np.sqrt(3))

    def test_sample_std_err_known_values(self):
        df = pd.DataFrame({"r": [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]})
        mean, se = ir.get_sample_std_err(df, "r")
        assert mean == pytest.approx(5.0)
        assert se == pytest.approx(np.std(df["r"], ddof=1) / np.sqrt(8))

    def test_sample_std_err_single_value_is_zero(self):
        df = pd.DataFrame({"r": [0.3]})
        with np.errstate(all="ignore"), pytest.warns(RuntimeWarning):
            mean, se = ir.get_sample_std_err(df, "r")
        assert mean == pytest.approx(0.3)
        assert se == 0.0


class TestSummarize:
    def test_without_cum_reward(self):
        out = ir.summarize(pd.DataFrame({"x": [1, 2, 3]}))
        assert np.isnan(out["avg_reward"]) and np.isnan(out["std_err"])
        assert out["n_completed"] == "0/3"
        assert out["n_err"] == 0

    def test_returns_none_when_nothing_completed(self):
        df = episodes_df(cum_reward=[0.0, 0.0], terminated=[False, False])
        assert ir.summarize(df) is None

    def test_full_record(self):
        df = episodes_df(
            cum_reward=[1.0, 0.0, 0.0],
            err_msg=[None, None, "boom"],
            terminated=[True, False, False],
            truncated=[False, True, False],
            n_steps=[2, 4, 6],
            **{"stats.cum_cost": [0.1, 0.2, 0.3]},
        )
        out = ir.summarize(df)
        assert out["avg_reward"] == pytest.approx(0.333)
        assert out["std_err"] == pytest.approx(round(np.sqrt((1 / 3) * (2 / 3) / 3), 3))
        assert out["avg_steps"] == pytest.approx(4.0)
        assert out["n_completed"] == "3/3"
        assert out["n_err"] == 1
        assert out["cum_cost"] == pytest.approx(0.6)

    def test_effective_cost_replaces_cum_cost(self):
        df = episodes_df(
            cum_reward=[1.0, 1.0],
            **{"stats.cum_cost": [1.0, 1.0], "stats.cum_effective_cost": [0.25, 0.5]},
        )
        out = ir.summarize(df)
        assert "cum_cost" not in out
        assert out["cum_effective_cost"] == pytest.approx(0.75)

    def test_only_errors_count_when_no_done_columns(self):
        df = episodes_df(cum_reward=[0.0, 0.0], err_msg=["e", None]).drop(
            columns=["terminated", "truncated"]
        )
        out = ir.summarize(df)
        assert out["n_completed"] == "1/2"
        assert out["n_err"] == 1

    def test_errored_episode_with_reward_violates_sanity_check(self):
        df = episodes_df(cum_reward=[1.0], err_msg=["boom"])
        with pytest.raises(AssertionError):
            ir.summarize(df)


class TestSummarizeStats:
    def test_cum_and_max_ops(self):
        df = episodes_df(
            cum_reward=[1.0, 0.0],
            **{"stats.cum_tokens": [10, 20], "stats.max_tokens": [7, 3]},
        )
        out = ir.summarize_stats(df)
        assert out["avg_reward"] == pytest.approx(0.5)
        assert out["cum_tokens"] == 30
        assert out["max_tokens"] == 7

    def test_unknown_op_raises(self):
        df = episodes_df(cum_reward=[1.0], **{"stats.avg_tokens": [1]})
        with pytest.raises(ValueError, match="Unknown stats operation: avg"):
            ir.summarize_stats(df)

    def test_none_when_nothing_completed(self):
        df = episodes_df(cum_reward=[0.0], terminated=[False])
        assert ir.summarize_stats(df) is None


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def multi_agent_df():
    """Two tasks x two agents (differing by flags) x two seeds, indexed like load_result_df."""
    rows = []
    rewards = {
        ("miniwob.a", True): [1.0, 1.0],
        ("miniwob.b", True): [1.0, 0.0],
        ("miniwob.a", False): [0.0, 0.0],
        ("miniwob.b", False): [1.0, 0.0],
    }
    for (task, html), rs in rewards.items():
        for seed, r in enumerate(rs):
            rows.append(
                {
                    "env.task_name": task,
                    "agent.agent_name": "A",
                    "agent.flags.use_html": html,
                    "env.task_seed": seed,
                    "cum_reward": r,
                    "err_msg": None,
                    "stack_trace": None,
                    "n_steps": 2,
                    "terminated": True,
                    "truncated": False,
                }
            )
    df = pd.DataFrame(rows)
    ir.set_index_from_variables(df)
    return df


class TestReduceAndReport2d:
    def test_reduce_episodes_one_row_per_index(self):
        df = multi_agent_df()
        out = ir.reduce_episodes(df)
        assert len(out) == 4
        assert set(out["n_completed"]) == {"2/2"}
        assert out.loc[("miniwob.a", "A", "miniwob", True), "avg_reward"] == 1.0
        assert out.loc[("miniwob.b", "A", "miniwob", False), "avg_reward"] == 0.5

    def test_report_2d_unstacks_columns(self):
        df = multi_agent_df().reset_index().set_index(["env.task_name", "agent.flags.use_html"])
        out = ir.report_2d(df, reduce_fn=lambda d: d["cum_reward"].mean())
        assert list(out.index) == ["miniwob.a", "miniwob.b"]
        assert out.loc["miniwob.a", True] == 1.0
        assert out.loc["miniwob.a", False] == 0.0
        assert out.loc["miniwob.b", True] == 0.5


class TestReportConstantAndVariables:
    def test_prints_constants_and_top_variables(self, capsys):
        df = pd.DataFrame(
            {
                "const": ["c"] * 5,
                "var": ["a", "a", "b", "c", "d"],
                "stack_trace": ["t1", "t2", "t1", "t1", "t1"],
            }
        )
        ir.report_constant_and_variables(df, show_stack_traces=False)
        out = capsys.readouterr().out
        assert "const: c" in out
        assert "var: n_unique=4" in out
        assert "2x : a" in out
        assert "..." in out
        assert "stack_trace" not in out

    def test_stack_traces_shown_by_default(self, capsys):
        df = pd.DataFrame({"stack_trace": ["t1", "t2"]})
        ir.report_constant_and_variables(df)
        out = capsys.readouterr().out
        assert "stack_trace: n_unique=2" in out
        assert "..." not in out


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ((1, 2, 3), (1, 2, 3), []),
        ((1, 2, 3), (1, 0, 3), [1]),
        (("x", True, 0.1), ("y", False, 0.1), [0, 1]),
    ],
)
def test_find_diff(a, b, expected):
    assert ir._find_diff(a, b) == expected


class TestExtractAblationStudy:
    def _report(self):
        index = pd.MultiIndex.from_tuples(
            [(True, 0.1), (False, 0.1), (False, 0.5)], names=["use_html", "temp"]
        )
        return pd.DataFrame({"avg_reward": [0.5, 0.4, 0.3]}, index=index)

    def test_changes_relative_to_first_row(self):
        out = ir._extract_ablation_study(self._report())
        assert list(out.index) == ["Initial Configuration", "-use_html", "-use_html, temp←0.5"]
        assert list(out.columns) == ["avg_reward"]
        assert list(out["avg_reward"]) == [0.5, 0.4, 0.3]

    def test_bool_flag_turned_on_uses_plus(self):
        report = self._report().iloc[[1, 0]]
        out = ir._extract_ablation_study(report)
        assert list(out.index) == ["Initial Configuration", "+use_html"]

    def test_progression_relative_to_previous_row(self):
        out = ir._extract_ablation_study(self._report(), progression=True)
        assert list(out.index) == [
            "Initial Configuration",
            "↳ use_html=False",
            "↳ temp=0.5",
        ]


class TestGlobalReport:
    def test_single_configuration_per_task_report(self, capsys):
        df = episodes_df(
            cum_reward=[1.0, 0.0, 1.0, 1.0], **{"env.task_name": ["a.x", "a.y", "a.x", "a.y"]}
        ).set_index("env.task_name")
        report = ir.global_report(df)
        assert "Only one configuration" in capsys.readouterr().out
        assert list(report.index) == ["a.x", "a.y", "[ALL TASKS]"]
        assert report.loc["a.x", "avg_reward"] == 1.0
        assert report.loc["a.y", "avg_reward"] == 0.5
        assert report.loc["[ALL TASKS]", "avg_reward"] == 0.75
        assert report.loc["[ALL TASKS]", "n_completed"] == "4/4"

    def test_multiple_configurations_sorted_and_renamed(self, capsys):
        report = ir.global_report(multi_agent_df())
        assert "Found multiple configuration" in capsys.readouterr().out
        assert list(report.index.names) == ["agent.agent_name", "env.benchmark", "use_html"]
        assert list(report.index.get_level_values("use_html")) == [True, False]
        assert list(report["avg_reward"]) == [0.75, 0.25]

    def test_no_rename_and_no_avg_reward(self):
        report = ir.global_report(
            multi_agent_df(),
            reduce_fn=lambda d: pd.Series({"n": len(d)}),
            rename_index=None,
        )
        assert "agent.flags.use_html" in report.index.names
        assert list(report["n"]) == [4, 4]


class TestAblationReport:
    def _study(self, root, orders):
        variants = [
            FakeAgentArgs(agent_name="A", temperature=0.1, use_html=True),
            FakeAgentArgs(agent_name="A", temperature=0.1, use_html=False),
            FakeAgentArgs(agent_name="A", temperature=0.5, use_html=False),
        ]
        rewards = [1.0, 1.0, 0.0]
        for v_idx, (agent, order, reward) in enumerate(zip(variants, orders, rewards)):
            for task in ["miniwob.a", "miniwob.b"]:
                make_exp_dir(
                    root,
                    f"exp_{v_idx}_{task}",
                    task_name=task,
                    agent_args=agent,
                    summary=done_summary(reward=reward),
                    order=order,
                )
        return ir.load_result_df(root, progress_fn=None)

    def test_rows_follow_launch_order(self, tmp_path):
        df = self._study(tmp_path, orders=[0, 1, 2])
        report = ir.ablation_report(df)
        assert list(report.index) == [
            "Initial Configuration",
            "-agent.use_html",
            "agent.temperature←0.5, -agent.use_html",
        ]
        assert list(report["avg_order"]) == [0.0, 1.0, 2.0]
        assert list(report["avg_reward"]) == [1.0, 1.0, 0.0]

    def test_progression(self, tmp_path):
        df = self._study(tmp_path, orders=[2, 0, 1])
        report = ir.ablation_report(df, progression=True)
        assert list(report.index) == [
            "Initial Configuration",
            "↳ agent.temperature=0.5",
            "↳ agent.temperature=0.1, agent.use_html=True",
        ]

    def test_avg_order_none_when_orders_missing(self, tmp_path):
        df = self._study(tmp_path, orders=[None, None, None])
        report = ir.global_report(df)
        assert ir._get_avg_order(df, report.iloc[0]) is None


class TestFlagsAndDisplay:
    def _report(self):
        index = pd.MultiIndex.from_tuples(
            [("A", True, False), ("A", False, False), ("A", True, True), ("A", False, True)],
            names=["agent", "use_html", "use_ax"],
        )
        return pd.DataFrame({"avg_reward": [0.8, 0.4, 0.6, 0.2], "n_err": [0, 1, 2, 3]}, index)

    def test_rename_bool_flags_multiindex(self):
        out = ir._rename_bool_flags(self._report())
        assert set(out.index.get_level_values("use_html")) == {"✓", "-"}
        assert set(out.index.get_level_values("agent")) == {"A"}

    def test_rename_bool_flags_single_index_unchanged(self):
        df = pd.DataFrame({"x": [1]}, index=pd.Index([True], name="flag"))
        out = ir._rename_bool_flags(df)
        assert list(out.index) == [True]

    def test_flag_report_ratios(self):
        out = ir.flag_report(self._report())
        assert list(out.index) == ["use_html", "use_ax"]
        assert out.loc["use_html", "avg_true"] == 0.7
        assert out.loc["use_html", "avg_false"] == 0.3
        assert out.loc["use_html", "ratio"] == pytest.approx(2.33)
        assert out.loc["use_ax", "ratio"] == pytest.approx(0.67)

    def test_flag_report_no_rounding(self):
        out = ir.flag_report(self._report(), round_digits=None)
        assert out.loc["use_html", "ratio"] == pytest.approx(0.7 / 0.3)

    def test_flag_report_single_level(self, capsys):
        df = pd.DataFrame({"avg_reward": [1.0]}, index=pd.Index(["a"], name="x"))
        assert ir.flag_report(df) is None
        assert "Only 1 levels" in capsys.readouterr().out

    def test_shrink_columns(self):
        df = pd.DataFrame(
            {"avg_reward": [0.5, 0.125], "n_err": [1, 2]},
            index=pd.Index(["a", "b"], name="task_name"),
        )
        out = ir.shrink_columns(df)
        assert list(out.columns) == ["avg\nreward", "n\nerr"]
        assert out.index.names == ["task\nname"]
        assert list(out["avg\nreward"]) == ["0.5", "0.125"]
        assert list(out["n\nerr"]) == [1, 2]
        assert list(df.columns) == ["avg_reward", "n_err"]

    def test_shrink_columns_keep_index_names(self):
        df = pd.DataFrame({"a_b": [1.0]}, index=pd.Index(["x"], name="task_name"))
        out = ir.shrink_columns(df, also_wrap_index=False)
        assert out.index.names == ["task_name"]
        assert out.iloc[0, 0] == "1"

    def test_set_wrap_style(self):
        styler = ir.set_wrap_style(pd.DataFrame({"a": [1]}))
        assert styler.table_styles == [{"selector": "th", "props": [("white-space", "pre-wrap")]}]

    def test_display_report(self):
        with mock.patch.object(ir, "display") as display:
            ir.display_report(self._report())
        styled = display.call_args[0][0]
        data = styled.data
        assert list(data.columns) == ["agent", "use\nhtml", "use\nax", "avg\nreward", "n\nerr"]
        assert set(data["use\nhtml"]) == {"✓", "-"}

    def test_display_report_print_only_raw(self):
        with mock.patch.object(ir, "display") as display:
            ir.display_report(
                self._report(),
                apply_shrink_columns=False,
                rename_bool_flags=False,
                print_only="use_html",
            )
        data = display.call_args[0][0].data
        assert list(data.columns) == ["use_html", "avg_reward", "n_err"]
        assert set(data["use_html"]) == {True, False}


# ---------------------------------------------------------------------------
# Error utils
# ---------------------------------------------------------------------------


class TestMapErrKey:
    def test_none(self):
        assert ir.map_err_key(None) is None

    def test_strips_logs(self):
        assert ir.map_err_key("ValueError: bad\n\n=== logs ===\nline1\nline2") == "ValueError: bad"

    @pytest.mark.parametrize(
        "msg, expected",
        [
            (
                "Error: your messages resulted in 9001 tokens. === logs ===",
                "Error: your messages resulted in x tokens.",
            ),
            (
                "Exception uncaught by agent or environment in task miniwob.click-1.\n"
                "=== logs ===",
                "Exception uncaught by agent or environment in task <task_name>.",
            ),
        ],
    )
    def test_regex_normalization(self, msg, expected):
        assert ir.map_err_key(msg) == expected

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: map_err_key chops the last char when '=== logs ===' is absent (find() == -1)",
    )
    def test_message_without_logs_unchanged(self):
        assert ir.map_err_key("Some error") == "Some error"


@pytest.fixture
def error_study(tmp_path):
    root = tmp_path / "errors"
    specs = [
        ("e1", "miniwob.b", 1, "Timeout\n=== logs ===\nx", "trace-b"),
        ("e2", "miniwob.a", 2, "Timeout\n=== logs ===\ny", "trace-a"),
        ("e3", "miniwob.c", 3, "Crash\nline2\n=== logs ===", "trace-c"),
        ("ok", "miniwob.d", 4, None, None),
    ]
    rows = []
    for name, task, seed, err, trace in specs:
        exp_dir = make_exp_dir(
            root,
            name,
            task_name=task,
            seed=seed,
            summary=done_summary(reward=0.0, err_msg=err, stack_trace=trace),
            logs=f"log of {name}",
        )
        rows.append({"exp_dir": exp_dir, "err_msg": err, "stack_trace": trace})
    return pd.DataFrame(rows)


class TestErrorReport:
    def test_groups_errors_and_lists_tasks_sorted(self, error_study):
        report = ir.error_report(error_study)
        assert "## 2x : Timeout\n" in report
        assert "## 1x : Crash<br>line2\n" in report
        assert report.index("* miniwob.a seed: 2") < report.index("* miniwob.b seed: 1")
        assert report.count("Stack Trace:") == 3
        assert "trace-a" in report and "trace-c" in report
        assert "miniwob.d" not in report
        assert "err_key" in error_study.columns

    def test_max_stack_trace(self, error_study):
        report = ir.error_report(error_study, max_stack_trace=1)
        assert report.count("Stack Trace:") == 2
        assert "Showing Max 1 stack traces" in report

    def test_use_log(self, error_study):
        report = ir.error_report(error_study, use_log=True)
        assert "```bash\nlog of e2\n```" in report
        assert "Stack Trace:" not in report

    def test_precomputed_err_key_is_respected(self, error_study):
        error_study["err_key"] = ["K", "K", "K", None]
        report = ir.error_report(error_study)
        assert "## 3x : K\n" in report


class TestFormatLog:
    def test_no_log(self):
        assert ir._format_log(SimpleNamespace(logs=None)) == "No log found"

    def test_short_log_returned_unchanged(self):
        log = "\n".join(f"l{i}" for i in range(5))
        assert ir._format_log(SimpleNamespace(logs=log)) == log

    def test_long_log_without_traceback_keeps_head_and_tail(self):
        log = "\n".join(f"l{i}" for i in range(100))
        out = ir._format_log(SimpleNamespace(logs=log), head_lines=2, tail_lines=3)
        assert out == "l0\nl1\n...\n...truncated middle of the log\n...\nl97\nl98\nl99"

    def test_long_log_with_traceback_starts_at_last_action(self):
        lines = [f"l{i}" for i in range(20)]
        lines += ["action: click('1')", "more", "action: fill('2')", "noise"]
        lines += ["Traceback (most recent call last):", "  File x", "ValueError: boom"]
        log = "\n".join(lines)
        out = ir._format_log(SimpleNamespace(logs=log), head_lines=2, tail_lines=3)
        head, tail = out.split("...truncated middle of the log\n...\n")
        assert head == "l0\nl1\n...\n"
        assert tail.startswith("action: fill('2')")
        assert tail.endswith("ValueError: boom")


class TestCategorizeErrors:
    def test_no_error_returns_none(self):
        assert ir.categorize_error({"err_msg": None, "stack_trace": None}) is None
        assert ir.categorize_error({}) is None

    def test_error_report_detailed_without_errors(self):
        df = pd.DataFrame({"err_msg": [None, None], "stack_trace": [None, None]})
        assert ir.error_report_detailed(df) == ""
        assert list(df["error_category"]) == [None, None]

    def test_print_errors_chronologically_without_errors(self, capsys):
        df = pd.DataFrame(
            {"exp_date": [2, 1], "err_msg": [None, np.nan], "stack_trace": [None] * 2}
        )
        ir.print_errors_chronologically(df)
        assert capsys.readouterr().out == ""

    @pytest.mark.xfail(
        strict=True,
        raises=NameError,
        reason="BUG: categorize_error uses undefined ERR_CLASS_MAP/is_*_server_error",
    )
    def test_categorize_error_with_error(self):
        category = ir.categorize_error({"err_msg": "boom", "stack_trace": "trace"})
        assert isinstance(category, str)

    @pytest.mark.xfail(
        strict=True,
        raises=NameError,
        reason="BUG: error_report_detailed crashes on undefined ERR_CLASS_MAP",
    )
    def test_error_report_detailed_with_errors(self, error_study):
        report = ir.error_report_detailed(error_study)
        assert "Category:" in report

    @pytest.mark.xfail(
        strict=True,
        raises=NameError,
        reason="BUG: print_errors_chronologically crashes on undefined ERR_CLASS_MAP",
    )
    def test_print_errors_chronologically_with_errors(self, capsys):
        df = pd.DataFrame({"exp_date": [1, 2], "err_msg": ["a", "a"], "stack_trace": ["t", "t"]})
        ir.print_errors_chronologically(df)
        assert "2 times" in capsys.readouterr().out

    @pytest.mark.xfail(
        strict=True,
        raises=NameError,
        reason="BUG: report_different_errors crashes on undefined ERR_CLASS_MAP",
    )
    def test_report_different_errors_with_errors(self):
        df = pd.DataFrame({"err_msg": ["boom"], "stack_trace": ["t"]})
        report = ir.report_different_errors(df)
        assert report["any_err"].sum() == 1

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="BUG: report_different_errors asserts a DataFrame but .sum() yields a Series",
    )
    def test_report_different_errors_without_errors(self):
        df = pd.DataFrame({"err_msg": [None, None], "stack_trace": [None, None]})
        report = ir.report_different_errors(df)
        assert len(report) == 0


# ---------------------------------------------------------------------------
# Study-level helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "task_name, expected",
    [
        ("miniwob.click-test", "miniwob"),
        ("workarena.servicenow.order", "workarena"),
        ("plain", "plain"),
    ],
)
def test_benchmark_from_task_name(task_name, expected):
    assert ir._benchmark_from_task_name(task_name) == expected


def test_summarize_study_groups_by_agent_config():
    out = ir.summarize_study(multi_agent_df())
    assert list(out.index.names) == ["agent.agent_name", "env.benchmark", "agent.flags.use_html"]
    assert out.loc[("A", "miniwob", True), "avg_reward"] == 0.75
    assert out.loc[("A", "miniwob", False), "avg_reward"] == 0.25
    assert set(out["n_completed"]) == {"4/4"}


class TestSplitByKey:
    def test_key_in_index(self):
        out = ir.split_by_key(multi_agent_df(), "agent.flags.use_html")
        assert set(out) == {True, False}
        for value, sub_df in out.items():
            assert len(sub_df) == 4
            assert set(sub_df["agent.flags.use_html"]) == {value}
            assert list(sub_df.index.names) == [
                "env.task_name",
                "agent.agent_name",
                "env.benchmark",
            ]

    def test_key_in_columns(self):
        df = multi_agent_df()
        out = ir.split_by_key(df, "env.task_seed")
        assert set(out) == {0, 1}
        assert all(len(sub) == 4 for sub in out.values())


def make_study(root: Path, name: str, rewards=(1.0, 0.0)) -> Path:
    study_dir = root / name
    for i, r in enumerate(rewards):
        make_exp_dir(study_dir, f"exp_{i}", task_name=f"miniwob.t{i}", summary=done_summary(r))
    return study_dir


class TestGetStudySummary:
    def test_empty_study_returns_none(self, tmp_path):
        study_dir = tmp_path / "empty"
        study_dir.mkdir()
        assert ir.get_study_summary(study_dir) is None

    def test_cache_and_staleness(self, tmp_path):
        study_dir = make_study(tmp_path, "s")
        sentinel = {}
        summary = ir.get_study_summary(study_dir, sentinel=sentinel)
        assert sentinel["from_cache"] is False
        assert (study_dir / "study_summary.csv").exists()
        assert summary["n_completed"].tolist() == ["2/2"]

        cached = ir.get_study_summary(study_dir, sentinel=sentinel)
        assert sentinel["from_cache"] is True
        assert cached["n_completed"].tolist() == ["2/2"]

        summary_file = study_dir / "exp_0" / "summary_info.json"
        st = summary_file.stat()
        os.utime(summary_file, (st.st_atime, st.st_mtime + 100))
        ir.get_study_summary(study_dir, sentinel=sentinel)
        assert sentinel["from_cache"] is False

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: stale rebuild reuses cached ExpResult.summary_info, so changed results are ignored",
    )
    def test_stale_rebuild_reads_updated_results(self, tmp_path):
        study_dir = make_study(tmp_path, "s", rewards=(1.0,))
        summary = ir.get_study_summary(study_dir)
        assert summary["avg_reward"].tolist() == [1.0]

        summary_file = study_dir / "exp_0" / "summary_info.json"
        summary_file.write_text(json.dumps(done_summary(reward=0.0)))
        st = summary_file.stat()
        os.utime(summary_file, (st.st_atime, st.st_mtime + 100))

        sentinel = {}
        rebuilt = ir.get_study_summary(study_dir, sentinel=sentinel)
        assert sentinel["from_cache"] is False
        assert rebuilt["avg_reward"].tolist() == [0.0]
        persisted = pd.read_csv(study_dir / "study_summary.csv")
        assert persisted["avg_reward"].tolist() == [0.0]

    def test_ignore_stale_uses_cache_even_after_change(self, tmp_path):
        study_dir = make_study(tmp_path, "s")
        ir.get_study_summary(study_dir)
        summary_file = study_dir / "exp_0" / "summary_info.json"
        st = summary_file.stat()
        os.utime(summary_file, (st.st_atime, st.st_mtime + 100))
        sentinel = {}
        ir.get_study_summary(study_dir, ignore_stale=True, sentinel=sentinel)
        assert sentinel["from_cache"] is True


class TestMtimes:
    def test_get_mtimes_skips_hidden_and_whitelist(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "sub" / "b.txt").write_text("b")
        (tmp_path / "_hidden.json").write_text("{}")
        (tmp_path / ".dot").write_text("")
        (tmp_path / "skip.csv").write_text("")
        os.utime(tmp_path / "a.txt", (1000, 1000))
        mtimes = ir._get_mtimes(tmp_path, whitelist=(tmp_path / "skip.csv",))
        assert set(mtimes) == {"a.txt", "sub", os.path.join("sub", "b.txt")}
        assert mtimes["a.txt"] == 1000

    def test_is_stale(self, tmp_path):
        (tmp_path / "f.txt").write_text("x")
        summary_path = tmp_path / "study_summary.csv"
        assert ir._is_stale(tmp_path, summary_path) is True  # no mtimes file yet
        assert (tmp_path / "_last_modification_times.json").exists()
        assert ir._is_stale(tmp_path, summary_path) is True  # no summary yet
        summary_path.write_text("")
        assert ir._is_stale(tmp_path, summary_path) is False
        os.utime(tmp_path / "f.txt", (5, 5))
        assert ir._is_stale(tmp_path, summary_path) is True
        assert ir._is_stale(tmp_path, summary_path) is False


class TestGetAllSummaries:
    def test_collects_and_sorts_summaries(self, tmp_path, capsys):
        make_study(tmp_path, "2024-01-01_study")
        make_study(tmp_path, "2024-02-01_study", rewards=(1.0,))
        make_study(tmp_path, "_hidden_study")
        (tmp_path / "2024-03-01_empty").mkdir()
        (tmp_path / "2024-04-01_not_a_dir").write_text("")
        summaries = ir.get_all_summaries(tmp_path)
        assert list(summaries.index) == ["2024-02-01_study", "2024-01-01_study"]
        assert list(summaries["n_completed"]) == ["1/1", "2/2"]
        out = capsys.readouterr()
        assert "skip (starts with '_')" in out.out
        assert "NotADirectoryError" in out.err

    def test_include_hidden(self, tmp_path):
        make_study(tmp_path, "_hidden_study")
        summaries = ir.get_all_summaries(tmp_path, skip_hidden=False, ignore_cache=True)
        assert list(summaries.index) == ["_hidden_study"]


class TestGetAllTaskMessages:
    def test_counts_messages_per_task(self, tmp_path, capsys):
        make_exp_dir(tmp_path, "e0", task_name="miniwob.a", task_messages=["hi", "hi", "bye"])
        make_exp_dir(tmp_path, "e1", task_name="miniwob.a", task_messages=["hi", None])
        make_exp_dir(tmp_path, "e2", task_name="miniwob.b", task_messages=["yo"])
        ir.get_all_task_messages(tmp_path)
        out = capsys.readouterr().out
        block_a = out.split("miniwob.a\n")[1].split("\n\n")[0]
        assert block_a.splitlines() == ["3x : hi", "1x : bye"]
        assert "miniwob.b\n1x : yo" in out

    def test_max_n_exp_samples(self, tmp_path, capsys):
        make_exp_dir(tmp_path, "e0", task_name="miniwob.a", task_messages=["hi"])
        make_exp_dir(tmp_path, "e1", task_name="miniwob.b", task_messages=["yo"])
        with mock.patch.object(ir.random, "sample", side_effect=lambda lst, k: lst[-k:]) as s:
            ir.get_all_task_messages(tmp_path, max_n_exp=1)
        assert s.call_args[0][1] == 1
        out = capsys.readouterr().out
        assert out.count("1x :") == 1
