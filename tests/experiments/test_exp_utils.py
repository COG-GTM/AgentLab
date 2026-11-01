import signal
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agentlab.experiments import exp_utils
from agentlab.experiments.exp_utils import (
    MockedExpArgs,
    _episode_timeout,
    add_dependencies,
    hide_some_exp,
    make_seeds,
    order,
    run_exp,
    timeout_manager,
)


class FakeExpArgs:
    def __init__(self, task_name):
        self.env_args = SimpleNamespace(task_name=task_name)
        self.exp_id = None
        self.depends_on = ()

    def make_id(self):
        if self.exp_id is None:
            self.exp_id = f"id_{self.env_args.task_name}"


def test_episode_timeout_uses_max_steps():
    exp = SimpleNamespace(env_args=SimpleNamespace(max_steps=10))
    assert _episode_timeout(exp, avg_step_timeout=60) == 600


def test_episode_timeout_defaults_without_max_steps():
    exp = SimpleNamespace(env_args=SimpleNamespace())
    assert _episode_timeout(exp) == 10 * 60 * 60


def test_episode_timeout_respects_smaller_exp_timeout():
    exp = SimpleNamespace(env_args=SimpleNamespace(max_steps=10), episode_timeout=42)
    assert _episode_timeout(exp, avg_step_timeout=60) == 42
    exp.episode_timeout = 10_000
    assert _episode_timeout(exp, avg_step_timeout=60) == 600


def test_timeout_manager_none_is_noop():
    with timeout_manager(None):
        pass


def test_timeout_manager_raises_and_restores_handler():
    previous = signal.getsignal(signal.SIGALRM)
    with pytest.raises(TimeoutError):
        with timeout_manager(seconds=1):
            time.sleep(3)
    assert signal.getsignal(signal.SIGALRM) == previous
    assert signal.alarm(0) == 0  # no pending alarm


def test_timeout_manager_float_is_rounded_up_to_one_second():
    start = time.time()
    with pytest.raises(TimeoutError):
        with timeout_manager(seconds=0.1):
            time.sleep(3)
    assert time.time() - start >= 0.9


def test_timeout_manager_no_timeout_when_fast():
    with timeout_manager(seconds=5):
        pass
    assert signal.alarm(0) == 0


def test_run_exp_calls_run():
    exp = MockedExpArgs(exp_id=1)
    with patch.object(exp_utils, "sleep"):
        result = run_exp(exp, "dep1", "dep2")
    assert result is exp
    assert exp.start_time is not None and exp.end_time >= exp.start_time


def test_mocked_exp_args_defaults():
    exp = MockedExpArgs(exp_id=3, depends_on=[1, 2])
    assert exp.exp_name == "exp_3"
    assert exp.depends_on == [1, 2]
    assert MockedExpArgs(exp_id=4).depends_on == []


@pytest.mark.parametrize("deps", [None, {}, {"a": [], "b": []}])
def test_add_dependencies_noop(deps):
    exp_list = [FakeExpArgs("a"), FakeExpArgs("b")]
    assert add_dependencies(exp_list, deps) is exp_list
    assert all(e.depends_on == () for e in exp_list)
    assert all(e.exp_id is None for e in exp_list)


def test_add_dependencies_maps_task_names_to_exp_ids():
    exp_list = [FakeExpArgs("a"), FakeExpArgs("b"), FakeExpArgs("c")]
    out = add_dependencies(exp_list, {"a": [], "b": ["a"], "c": ["a", "b"]})
    by_name = {e.env_args.task_name: e for e in out}
    assert by_name["a"].depends_on == ()
    assert by_name["b"].depends_on == ("id_a",)
    assert by_name["c"].depends_on == ("id_a", "id_b")


def test_add_dependencies_duplicate_task_names():
    exp_list = [FakeExpArgs("a"), FakeExpArgs("a")]
    with pytest.raises(ValueError, match="not unique"):
        add_dependencies(exp_list, {"a": ["b"]})


def test_add_dependencies_missing_task():
    exp_list = [FakeExpArgs("a"), FakeExpArgs("b")]
    with pytest.raises(ValueError, match="b is missing"):
        add_dependencies(exp_list, {"a": ["b"]})


@pytest.mark.parametrize(
    "fn, args",
    [(make_seeds, (3,)), (order, ([],)), (hide_some_exp, ("dir", lambda e: True, True))],
)
def test_deprecated_functions_raise(fn, args):
    with pytest.raises(DeprecationWarning):
        fn(*args)


def test_results_dir_exists():
    assert exp_utils.RESULTS_DIR.exists()
