"""Hermetic unit tests for agentlab.experiments.exp_utils."""

import importlib.util
import signal
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from agentlab.experiments import exp_utils
from agentlab.experiments.exp_utils import (
    MockedExpArgs,
    _episode_timeout,
    add_dependencies,
    run_exp,
    timeout_manager,
)
from agentlab.experiments.loop import EnvArgs, ExpArgs

# --------------------------------------------------------------------------- RESULTS_DIR


def _load_fresh_copy(name):
    """Execute exp_utils' source in an isolated module object (doesn't touch sys.modules)."""
    spec = importlib.util.spec_from_file_location(name, exp_utils.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "agentlab_root, copilot_root, expected",
    [
        ("agentlab_root", None, "agentlab_root"),
        ("agentlab_root", "copilot_root", "agentlab_root"),
        (None, "copilot_root", "copilot_root"),
        (None, None, "home/agentlab_results"),
    ],
)
def test_results_dir_resolution(tmp_path, monkeypatch, agentlab_root, copilot_root, expected):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for var, value in [
        ("AGENTLAB_EXP_ROOT", agentlab_root),
        ("UI_COPILOT_RESULTS_DIR", copilot_root),
    ]:
        if value is None:
            monkeypatch.delenv(var, raising=False)
        else:
            monkeypatch.setenv(var, str(tmp_path / value))

    module = _load_fresh_copy("_exp_utils_results_dir_copy")
    assert isinstance(module.RESULTS_DIR, Path)
    assert module.RESULTS_DIR == tmp_path / expected
    assert module.RESULTS_DIR.is_dir()


# --------------------------------------------------------------------------- run_exp / timeouts


def test_run_exp_delegates_to_run():
    exp = mock.Mock()
    exp.run.return_value = "result"
    assert run_exp(exp, "dep1", "dep2", avg_step_timeout=1) == "result"
    exp.run.assert_called_once_with()


@pytest.mark.parametrize(
    "max_steps, avg, episode_timeout, expected",
    [
        (None, 60, None, 10 * 60 * 60),
        (5, 10, None, 50),
        (5, 10, 20, 20),
        (5, 10, 500, 50),
        (None, 60, 100, 100),
    ],
)
def test_episode_timeout(max_steps, avg, episode_timeout, expected):
    exp = SimpleNamespace(env_args=SimpleNamespace(max_steps=max_steps))
    if episode_timeout is not None:
        exp.episode_timeout = episode_timeout
    assert _episode_timeout(exp, avg_step_timeout=avg) == expected


def test_episode_timeout_env_args_without_max_steps():
    assert _episode_timeout(SimpleNamespace(env_args=object())) == 10 * 60 * 60


def test_timeout_manager_none_is_noop(monkeypatch):
    alarm = mock.Mock()
    monkeypatch.setattr(exp_utils.signal, "alarm", alarm)
    with timeout_manager(None):
        pass
    alarm.assert_not_called()


def test_timeout_manager_windows_is_noop(monkeypatch):
    alarm = mock.Mock()
    monkeypatch.setattr(exp_utils.signal, "alarm", alarm)
    monkeypatch.setattr(exp_utils.sys, "platform", "win32")
    with timeout_manager(5):
        pass
    alarm.assert_not_called()


@pytest.mark.parametrize("seconds, expected_alarm", [(5, 5), (0.2, 1), (3.7, 3)])
def test_timeout_manager_sets_and_restores_alarm(monkeypatch, seconds, expected_alarm):
    alarm = mock.Mock()
    monkeypatch.setattr(exp_utils.signal, "alarm", alarm)
    previous = signal.getsignal(signal.SIGALRM)
    with timeout_manager(seconds):
        handler = signal.getsignal(signal.SIGALRM)
        assert handler is not previous and callable(handler)
    assert alarm.call_args_list == [mock.call(expected_alarm), mock.call(0)]
    assert signal.getsignal(signal.SIGALRM) == previous


def test_timeout_manager_handler_raises_timeout(monkeypatch):
    alarm = mock.Mock()
    monkeypatch.setattr(exp_utils.signal, "alarm", alarm)
    previous = signal.getsignal(signal.SIGALRM)
    with pytest.raises(TimeoutError, match="timed out after 4 seconds"):
        with timeout_manager(4):
            signal.getsignal(signal.SIGALRM)(signal.SIGALRM, None)  # simulate the alarm firing
    assert alarm.call_args_list[-1] == mock.call(0)
    assert signal.getsignal(signal.SIGALRM) == previous


# --------------------------------------------------------------------------- add_dependencies


def _exp(task_name, exp_id=None):
    return ExpArgs(agent_args=None, env_args=EnvArgs(task_name=task_name), exp_id=exp_id)


@pytest.mark.parametrize("deps", [None, {}, {"a": [], "b": []}])
def test_add_dependencies_noop(deps):
    exps = [_exp("a"), _exp("b")]
    assert add_dependencies(exps, deps) is exps
    assert all(e.exp_id is None and e.depends_on == () for e in exps)


def test_add_dependencies_maps_task_names_to_ids():
    exps = [_exp("a", "id-a"), _exp("b"), _exp("c", "id-c")]
    out = add_dependencies(exps, {"a": [], "b": ["a"], "c": ["a", "b"]})
    assert out is exps
    id_b = exps[1].exp_id
    assert id_b is not None
    assert exps[0].depends_on == ()
    assert exps[1].depends_on == ("id-a",)
    assert exps[2].depends_on == ("id-a", id_b)


def test_add_dependencies_rejects_duplicate_tasks():
    with pytest.raises(ValueError, match="not unique"):
        add_dependencies([_exp("a"), _exp("a")], {"a": ["b"]})


def test_add_dependencies_rejects_missing_task():
    with pytest.raises(ValueError, match="Task b is missing"):
        add_dependencies([_exp("a"), _exp("b")], {"a": ["b"]})


# --------------------------------------------------------------------------- MockedExpArgs & deprecated


def test_mocked_exp_args(monkeypatch):
    sleep = mock.Mock()
    monkeypatch.setattr(exp_utils, "sleep", sleep)
    monkeypatch.setattr(exp_utils, "time", mock.Mock(side_effect=[100.0, 103.0]))
    m = MockedExpArgs("t1", depends_on=["t0"])
    assert (m.exp_name, m.depends_on, m.env_args) == ("exp_t1", ["t0"], None)
    assert m.run() is m
    sleep.assert_called_once_with(3)
    assert (m.start_time, m.end_time) == (100.0, 103.0)
    assert MockedExpArgs("t2").depends_on == []


@pytest.mark.parametrize(
    "fn, args",
    [
        (exp_utils.make_seeds, (3,)),
        (exp_utils.order, ([],)),
        (exp_utils.hide_some_exp, ("dir", lambda e: True, True)),
    ],
)
def test_deprecated_helpers_raise(fn, args):
    with pytest.raises(DeprecationWarning, match="will be removed"):
        fn(*args)
