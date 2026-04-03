import time

from agentlab.benchmarks.abstract_env import (
    AbstractBenchmark,
    add_step_timing_to_env_info_decorator,
)


class FakeEnv:
    """Fake environment to test the timing decorator."""

    @add_step_timing_to_env_info_decorator
    def step(self, action: str):
        time.sleep(0.01)
        return {"obs": "test"}, 1.0, False, False, {}

    @add_step_timing_to_env_info_decorator
    def step_with_existing_timing(self, action: str):
        return (
            {"obs": "test"},
            1.0,
            False,
            False,
            {"action_exec_start": 100.0, "action_exec_stop": 200.0},
        )

    @add_step_timing_to_env_info_decorator
    def step_none_env_info(self, action: str):
        return {"obs": "test"}, 0.0, True, False, None


def test_timing_decorator_adds_timing():
    """Test that the decorator adds timing info to env_info."""
    env = FakeEnv()
    obs, reward, terminated, truncated, env_info = env.step("click")

    assert "action_exec_start" in env_info
    assert "action_exec_stop" in env_info
    assert "action_exec_timeout" in env_info
    assert env_info["action_exec_timeout"] == 0.0
    assert env_info["action_exec_stop"] >= env_info["action_exec_start"]
    assert reward == 1.0
    assert terminated is False
    assert truncated is False


def test_timing_decorator_preserves_existing_timing():
    """Test that the decorator does not overwrite existing timing info."""
    env = FakeEnv()
    obs, reward, terminated, truncated, env_info = env.step_with_existing_timing("click")

    assert env_info["action_exec_start"] == 100.0
    assert env_info["action_exec_stop"] == 200.0


def test_timing_decorator_handles_none_env_info():
    """Test that the decorator handles None env_info."""
    env = FakeEnv()
    obs, reward, terminated, truncated, env_info = env.step_none_env_info("click")

    assert isinstance(env_info, dict)
    assert "action_exec_start" in env_info
    assert "action_exec_stop" in env_info


def test_abstract_benchmark_defaults():
    """Test AbstractBenchmark default methods."""
    benchmark = AbstractBenchmark(name="test_benchmark", env_args_list=[])
    assert benchmark.name == "test_benchmark"
    assert benchmark.get_version() == "1"
    assert benchmark.dependency_graph_over_tasks() == {}


def test_abstract_benchmark_prepare_backends():
    """Test that prepare_backends runs without error."""
    benchmark = AbstractBenchmark(name="test", env_args_list=[])
    benchmark.prepare_backends()  # Should not raise
