import signal
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agentlab.experiments import exp_utils
from agentlab.experiments.exp_utils import (
    MockedExpArgs,
    _episode_timeout,
    add_dependencies,
    run_exp,
    timeout_manager,
)


def test_run_exp_calls_run():
    exp_arg = MagicMock()
    exp_arg.run.return_value = "result"
    assert run_exp(exp_arg, "dep1", "dep2") == "result"
    exp_arg.run.assert_called_once()


def test_episode_timeout_without_max_steps():
    exp_arg = SimpleNamespace(env_args=SimpleNamespace())
    assert _episode_timeout(exp_arg) == 10 * 60 * 60


def test_episode_timeout_with_max_steps():
    exp_arg = SimpleNamespace(env_args=SimpleNamespace(max_steps=5))
    assert _episode_timeout(exp_arg, avg_step_timeout=10) == 50


def test_episode_timeout_uses_smaller_exp_timeout():
    exp_arg = SimpleNamespace(env_args=SimpleNamespace(max_steps=5), episode_timeout=20)
    assert _episode_timeout(exp_arg, avg_step_timeout=10) == 20


def test_timeout_manager_none_is_noop():
    with timeout_manager(None):
        pass


def test_timeout_manager_raises_and_restores_handler():
    previous = signal.getsignal(signal.SIGALRM)
    with pytest.raises(TimeoutError):
        with timeout_manager(0.5):  # floats are rounded up to at least 1 second
            time.sleep(3)
    assert signal.getsignal(signal.SIGALRM) == previous
    assert signal.alarm(0) == 0


def test_timeout_manager_no_timeout():
    with timeout_manager(5):
        pass
    assert signal.alarm(0) == 0


def _exp_args(task_name):
    exp_id = f"id_{task_name}"
    ea = SimpleNamespace(env_args=SimpleNamespace(task_name=task_name), exp_id=None, depends_on=())

    def make_id():
        ea.exp_id = exp_id

    ea.make_id = make_id
    return ea


@pytest.mark.parametrize("deps", [None, {"a": [], "b": []}])
def test_add_dependencies_noop(deps):
    exp_args_list = [_exp_args("a"), _exp_args("b")]
    assert add_dependencies(exp_args_list, deps) is exp_args_list
    assert all(ea.exp_id is None for ea in exp_args_list)


def test_add_dependencies_maps_task_names_to_exp_ids():
    exp_args_list = [_exp_args("a"), _exp_args("b"), _exp_args("c")]
    add_dependencies(exp_args_list, {"a": [], "b": ["a"], "c": ["a", "b"]})
    assert exp_args_list[0].depends_on == ()
    assert exp_args_list[1].depends_on == ("id_a",)
    assert exp_args_list[2].depends_on == ("id_a", "id_b")


def test_add_dependencies_duplicate_task_names():
    with pytest.raises(ValueError, match="not unique"):
        add_dependencies([_exp_args("a"), _exp_args("a")], {"a": ["a"]})


def test_add_dependencies_missing_task():
    with pytest.raises(ValueError, match="missing"):
        add_dependencies([_exp_args("a"), _exp_args("b")], {"a": ["b"]})


def test_mocked_exp_args(monkeypatch):
    monkeypatch.setattr(exp_utils, "sleep", lambda s: None)
    exp = MockedExpArgs("1", depends_on=["0"])
    assert exp.exp_name == "exp_1" and exp.depends_on == ["0"]
    assert MockedExpArgs("2").depends_on == []
    assert exp.run() is exp
    assert exp.start_time <= exp.end_time


@pytest.mark.parametrize(
    "fn, args",
    [
        (exp_utils.make_seeds, (3,)),
        (exp_utils.order, ([],)),
        (exp_utils.hide_some_exp, (".", lambda e: True, True)),
    ],
)
def test_deprecated_functions_raise(fn, args):
    with pytest.raises(DeprecationWarning):
        fn(*args)
