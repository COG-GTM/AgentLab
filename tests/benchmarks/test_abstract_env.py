import pytest

from agentlab.benchmarks.abstract_env import (
    AbstractBenchmark,
    AbstractEnv,
    add_step_timing_to_env_info_decorator,
)


class DummyEnv(AbstractEnv):
    def __init__(self, env_info):
        self.env_info = env_info
        self.actions = []

    def reset(self, seed=None):
        return {}, {}

    @add_step_timing_to_env_info_decorator
    def step(self, action):
        self.actions.append(action)
        return {"obs": action}, 1.0, True, False, self.env_info

    def close(self):
        pass


def test_abstract_env_cannot_be_instantiated():
    with pytest.raises(TypeError):
        AbstractEnv()


def test_decorator_adds_timing_when_env_info_none():
    env = DummyEnv(env_info=None)
    obs, reward, terminated, truncated, info = env.step("click")
    assert obs == {"obs": "click"} and reward == 1.0 and terminated and not truncated
    assert env.actions == ["click"]
    assert info["action_exec_start"] <= info["action_exec_stop"]
    assert info["action_exec_timeout"] == 0.0


def test_decorator_preserves_existing_values():
    preset = {"action_exec_start": 1, "action_exec_stop": 2, "action_exec_timeout": 3, "x": 4}
    env = DummyEnv(env_info=dict(preset))
    *_, info = env.step("a")
    assert info == preset


def test_decorator_preserves_function_metadata():
    assert DummyEnv.step.__name__ == "step"


def test_abstract_benchmark_defaults():
    bench = AbstractBenchmark(name="dummy", env_args_list=[1, 2])
    assert bench.name == "dummy"
    assert bench.env_args_list == [1, 2]
    assert bench.get_version() == "1"
    assert bench.prepare_backends() is None
    assert bench.dependency_graph_over_tasks() == {}


def test_abstract_benchmark_requires_fields():
    with pytest.raises(Exception):
        AbstractBenchmark(name="missing env args")
