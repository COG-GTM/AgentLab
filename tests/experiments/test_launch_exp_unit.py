"""Hermetic unit tests for agentlab.experiments.launch_exp (no browser, no real Ray)."""

import json
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import gymnasium as gym
import pytest
from browsergym.experiments.agent import Agent

from agentlab.experiments import launch_exp, loop
from agentlab.experiments.launch_exp import (
    _hide_completed,
    _split_path,
    _split_sequential_exp,
    find_incomplete,
    import_object,
    non_dummy_count,
    noop,
    run_experiments,
)
from agentlab.experiments.loop import AbstractAgentArgs, EnvArgs, ExpArgs, ExpResult


class _ActionSet:
    def to_python_code(self, action):
        return action


class _Agent(Agent):
    action_set = _ActionSet()

    def obs_preprocessor(self, obs):
        return obs

    def get_action(self, obs):
        return "noop()", {}


@dataclass
class TrackingAgentArgs(AbstractAgentArgs):
    events: list = field(default_factory=list, repr=False)

    def prepare(self):
        self.events.append("prepare")

    def close(self):
        self.events.append("close")

    def make_agent(self):
        return _Agent()


class _OneStepEnv(gym.Env):
    def reset(self, seed=None, options=None):
        return {"goal": "g"}, {}

    def step(self, action):
        info = {"action_exec_start": 0.0, "action_exec_stop": 1.0, "action_exec_timeout": 0.0}
        return {"goal": "g"}, 1.0, True, False, info


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    monkeypatch.setattr(loop, "EXP_RESULT_CACHE", {})
    monkeypatch.setattr(loop, "_is_debugging", lambda: False)
    # ExpArgs.run() reconfigures the root logger; keep it from leaking between tests.
    monkeypatch.setattr(ExpArgs, "_set_logger", lambda self: None)
    monkeypatch.setattr(ExpArgs, "_unset_logger", lambda self: None)


def _exps(n, **kwargs):
    return [
        ExpArgs(
            agent_args=TrackingAgentArgs(),
            env_args=EnvArgs(task_name=f"task{i}", task_seed=i),
            **kwargs,
        )
        for i in range(n)
    ]


@pytest.fixture
def recorded_runs(monkeypatch):
    runs = []

    def fake_run_exp(exp_args, *deps, avg_step_timeout=60):
        runs.append((exp_args.env_args.task_name, avg_step_timeout))

    monkeypatch.setattr(launch_exp, "run_exp", fake_run_exp)
    return runs


# --------------------------------------------------------------------------- run_experiments


def test_run_experiments_empty_list(tmp_path):
    study_dir = tmp_path / "study"
    assert run_experiments(1, [], study_dir, parallel_backend="sequential") is None
    assert not study_dir.exists()


def test_run_experiments_sequential_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(EnvArgs, "make_env", lambda self, **kw: _OneStepEnv())
    exps = _exps(2)
    study_dir = tmp_path / "nested" / "study"
    run_experiments(1, exps, study_dir, parallel_backend="sequential")

    assert study_dir.is_dir()
    for exp in exps:
        assert exp.agent_args.events == ["prepare", "close"]
        assert Path(exp.exp_dir).parent == study_dir
        summary = json.loads((Path(exp.exp_dir) / "summary_info.json").read_text())
        assert summary["cum_reward"] == 1.0 and summary["err_msg"] is None
        assert ExpResult(exp.exp_dir).status == "done"


def test_run_experiments_sequential_passes_timeout(tmp_path, recorded_runs):
    run_experiments(1, _exps(3), tmp_path, parallel_backend="sequential", avg_step_timeout=7)
    assert recorded_runs == [("task0", 7), ("task1", 7), ("task2", 7)]


class _FakeParallel:
    instances = []

    def __init__(self, n_jobs, prefer):
        self.n_jobs, self.prefer = n_jobs, prefer
        _FakeParallel.instances.append(self)

    def __call__(self, jobs):
        return [fn(*args, **kwargs) for fn, args, kwargs in jobs]


def test_run_experiments_joblib_runs_sequential_first(tmp_path, monkeypatch, recorded_runs):
    import joblib

    monkeypatch.setattr(_FakeParallel, "instances", [])
    monkeypatch.setattr(joblib, "Parallel", _FakeParallel)
    exps = _exps(3)
    exps[2].sequential = True

    run_experiments(4, exps, tmp_path, parallel_backend="joblib", avg_step_timeout=5)

    assert recorded_runs == [("task2", 5), ("task0", 5), ("task1", 5)]
    (par,) = _FakeParallel.instances
    assert (par.n_jobs, par.prefer) == (4, "processes")
    assert [e.agent_args.events[0] for e in exps] == ["prepare"] * 3


@pytest.mark.xfail(
    strict=True,
    reason="BUG: joblib branch reassigns exp_args_list, so sequential agents never get close()",
)
def test_run_experiments_joblib_closes_all_agents(tmp_path, monkeypatch, recorded_runs):
    import joblib

    monkeypatch.setattr(joblib, "Parallel", _FakeParallel)
    exps = _exps(2)
    exps[0].sequential = True
    run_experiments(2, exps, tmp_path, parallel_backend="joblib")
    assert [e.agent_args.events for e in exps] == [["prepare", "close"]] * 2


@pytest.fixture
def fake_ray_module(monkeypatch):
    mod = types.ModuleType("agentlab.experiments.graph_execution_ray")
    mod.ray = mock.Mock()
    mod.execute_task_graph = mock.Mock()
    monkeypatch.setitem(sys.modules, "agentlab.experiments.graph_execution_ray", mod)
    return mod


def test_run_experiments_ray_backend_with_fake_ray(tmp_path, fake_ray_module):
    exps = _exps(2)
    run_experiments(3, exps, tmp_path, parallel_backend="ray", avg_step_timeout=9)
    fake_ray_module.ray.init.assert_called_once_with(num_cpus=3)
    fake_ray_module.execute_task_graph.assert_called_once_with(exps, avg_step_timeout=9)
    fake_ray_module.ray.shutdown.assert_called_once()


def test_run_experiments_ray_shutdown_on_error(tmp_path, fake_ray_module):
    fake_ray_module.execute_task_graph.side_effect = RuntimeError("worker died")
    exps = _exps(1)
    with pytest.raises(RuntimeError, match="worker died"):
        run_experiments(1, exps, tmp_path, parallel_backend="ray")
    fake_ray_module.ray.shutdown.assert_called_once()
    assert exps[0].agent_args.events == ["prepare", "close"]


def test_run_experiments_unknown_backend_still_closes(tmp_path, recorded_runs):
    exps = _exps(2)
    with pytest.raises(ValueError, match="Unknown parallel_backend: dask"):
        run_experiments(1, exps, tmp_path, parallel_backend="dask")
    assert recorded_runs == []
    assert all(e.agent_args.events == ["prepare", "close"] for e in exps)
    # experiments were prepared on disk before the backend was chosen
    assert all((Path(e.exp_dir) / "exp_args.pkl").exists() for e in exps)


# --------------------------------------------------------------------------- find_incomplete


def _make_study(study_dir, statuses):
    """Create prepared experiments with the given status ('done'/'error'/'incomplete')."""
    exps = []
    for i, status in enumerate(statuses):
        exp = ExpArgs(
            agent_args=TrackingAgentArgs(),
            env_args=EnvArgs(task_name=f"task{i}", task_seed=0),
            order=len(statuses) - 1 - i,
        )
        exp.prepare(study_dir)
        if status != "incomplete":
            summary = {"terminated": True, "err_msg": "boom" if status == "error" else None}
            (Path(exp.exp_dir) / "summary_info.json").write_text(json.dumps(summary))
        exps.append(exp)
    return exps


def test_find_incomplete_missing_dir(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        find_incomplete(tmp_path / "nope")


@pytest.mark.parametrize("include_errors, expected_count", [(True, 2), (False, 1)])
def test_find_incomplete(tmp_path, include_errors, expected_count):
    _make_study(tmp_path, ["done", "error", "incomplete"])
    exps = find_incomplete(tmp_path, include_errors=include_errors)

    assert non_dummy_count(exps) == expected_count
    # sorted by `order`, which was assigned in reverse creation order
    assert [e.env_args.task_name for e in exps] == ["task2", "task1", "task0"]
    assert [e.status for e in exps] == ["incomplete", "error", "done"]
    done = exps[2]
    assert done.is_dummy and done.run is noop and done.prepare is noop
    assert exps[0].is_dummy is False
    assert exps[1].is_dummy is (not include_errors)


def test_find_incomplete_all_done(tmp_path):
    _make_study(tmp_path, ["done", "done"])
    exps = find_incomplete(tmp_path)
    assert len(exps) == 2 and non_dummy_count(exps) == 0


def test_find_incomplete_orders_none_as_zero(tmp_path):
    exps = _make_study(tmp_path, ["incomplete", "incomplete"])
    for exp in exps:  # rewrite pickles with order=None for the first one
        exp.order = None if exp.env_args.task_name == "task0" else 1
        with open(Path(exp.exp_dir) / "exp_args.pkl", "wb") as f:
            import pickle

            pickle.dump(exp, f)
    found = find_incomplete(tmp_path)
    assert [e.env_args.task_name for e in found] == ["task0", "task1"]


def test_hide_completed_with_stub_result():
    exp_args = SimpleNamespace(run="run", prepare="prepare")
    res = SimpleNamespace(status="error", exp_args=exp_args)
    out = _hide_completed(res, include_errors=False)
    assert out is exp_args and out.is_dummy and out.run is noop and out.status == "error"


# --------------------------------------------------------------------------- small helpers


def test_non_dummy_count():
    exps = [SimpleNamespace(is_dummy=d) for d in (True, False, False)]
    assert non_dummy_count(exps) == 2
    assert non_dummy_count([]) == 0


def test_noop():
    assert noop() is None
    assert noop(1, 2, a=3) is None


def test_split_sequential_exp():
    a, b, c = SimpleNamespace(sequential=True), SimpleNamespace(), SimpleNamespace(sequential=False)
    assert _split_sequential_exp([a, b, c]) == ([a], [b, c])
    assert _split_sequential_exp([]) == ([], [])


@pytest.mark.parametrize(
    "path, expected",
    [
        ("pkg.mod.obj", ("pkg.mod", "obj")),
        ("pkg/mod/obj", ("pkg.mod", "obj")),
        ("mod.obj", ("mod", "obj")),
    ],
)
def test_split_path(path, expected):
    assert _split_path(path) == expected


def test_split_path_requires_a_dot():
    with pytest.raises(ValueError):
        _split_path("noseparator")


@pytest.mark.parametrize(
    "path, expected",
    [
        ("agentlab.experiments.launch_exp.noop", noop),
        ("agentlab/experiments/launch_exp/noop", noop),
        ("os.path.join", os.path.join),
    ],
)
def test_import_object(path, expected):
    assert import_object(path) is expected


@pytest.mark.parametrize(
    "path", ["agentlab.does_not_exist_xyz.obj", "agentlab.experiments.launch_exp.nope"]
)
def test_import_object_errors(path):
    with pytest.raises(ImportError, match=f"Error importing {path}"):
        import_object(path)
