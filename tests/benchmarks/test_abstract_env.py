import pytest

from agentlab.benchmarks.abstract_env import (
    AbstractBenchmark,
    AbstractEnv,
    add_step_timing_to_env_info_decorator,
)


class _DummyEnv(AbstractEnv):
    def __init__(self, env_info=None):
        self._env_info = env_info

    def reset(self, seed=None):
        return {}, {}

    @add_step_timing_to_env_info_decorator
    def step(self, action: str):
        return {"action": action}, 1.0, True, False, self._env_info

    def close(self):
        pass


def test_abstract_benchmark_defaults():
    bench = AbstractBenchmark(name="dummy", env_args_list=[1, 2])
    assert bench.get_version() == "1"
    assert bench.prepare_backends() is None
    assert bench.dependency_graph_over_tasks() == {}


def test_abstract_env_cannot_be_instantiated():
    with pytest.raises(TypeError):
        AbstractEnv()


def test_step_timing_added_when_env_info_is_none():
    obs, reward, terminated, truncated, env_info = _DummyEnv(None).step("noop()")
    assert obs == {"action": "noop()"}
    assert reward == 1.0 and terminated and not truncated
    assert env_info["action_exec_start"] <= env_info["action_exec_stop"]
    assert env_info["action_exec_timeout"] == 0.0


def test_step_timing_does_not_override_existing_values():
    existing = {"action_exec_start": 1, "action_exec_stop": 2, "action_exec_timeout": 3}
    *_, env_info = _DummyEnv(dict(existing)).step("noop()")
    assert env_info == existing


def test_step_timing_preserves_wrapped_name():
    assert _DummyEnv.step.__name__ == "step"
