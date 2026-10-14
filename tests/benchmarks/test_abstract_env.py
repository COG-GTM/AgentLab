from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from agentlab.benchmarks import abstract_env
from agentlab.benchmarks.abstract_env import (
    AbstractBenchmark,
    AbstractEnv,
    AbstractEnvArgs,
    add_step_timing_to_env_info_decorator,
)


class _ConcreteEnv(AbstractEnv):
    def __init__(self, env_info=None):
        self.env_info = env_info
        self.actions = []

    def reset(self, seed=None):
        super().reset(seed)
        return {"seed": seed}, {}

    def step(self, action):
        super().step(action)
        self.actions.append(action)
        return {"obs": action}, 1.0, False, False, self.env_info

    def close(self):
        return super().close()


class TestAbstractBenchmark:
    def test_defaults(self):
        bench = AbstractBenchmark(name="b", env_args_list=[1, 2])
        assert bench.name == "b"
        assert bench.env_args_list == [1, 2]
        assert bench.get_version() == "1"
        assert bench.prepare_backends() is None
        assert bench.dependency_graph_over_tasks() == {}

    @pytest.mark.parametrize("kwargs", [{}, {"name": "b"}, {"env_args_list": []}])
    def test_required_fields_are_validated(self, kwargs):
        with pytest.raises(ValidationError):
            AbstractBenchmark(**kwargs)

    def test_subclass_can_override_hooks(self):
        class Bench(AbstractBenchmark):
            def dependency_graph_over_tasks(self):
                return {"a": ["b"]}

        assert Bench(name="x", env_args_list=[]).dependency_graph_over_tasks() == {"a": ["b"]}


class TestAbstractEnv:
    def test_cannot_instantiate_without_implementations(self):
        with pytest.raises(TypeError):
            AbstractEnv()

    def test_partial_subclass_is_still_abstract(self):
        class Partial(AbstractEnv):
            def reset(self, seed=None):
                return {}, {}

        with pytest.raises(TypeError, match="close|step"):
            Partial()

    def test_concrete_subclass_and_base_bodies(self):
        env = _ConcreteEnv()
        assert env.reset(seed=3) == ({"seed": 3}, {})
        assert env.close() is None
        assert AbstractEnv.step(env, "a") is None


class TestAbstractEnvArgs:
    def test_subclass_without_make_env_is_abstract(self):
        @dataclass
        class Args(AbstractEnvArgs):
            x: int = 1

        with pytest.raises(TypeError, match="make_env"):
            Args()

    def test_concrete_subclass_serialises_and_builds_env(self):
        @dataclass
        class Args(AbstractEnvArgs):
            x: int = 1

            def make_env(self, action_mapping, exp_dir, exp_task_kwargs):
                assert super().make_env(action_mapping, exp_dir, exp_task_kwargs) is None
                return _ConcreteEnv(env_info={"x": self.x})

        args = Args(x=5)
        assert args.to_dict() == {"x": 5}
        assert Args.from_dict({"x": 7}).x == 7
        env = args.make_env(None, "/tmp", {})
        assert isinstance(env, AbstractEnv)
        assert env.env_info == {"x": 5}


class TestStepTimingDecorator:
    @pytest.fixture
    def fake_clock(self, monkeypatch):
        ticks = iter([10.0, 12.5, 20.0, 21.0])
        monkeypatch.setattr(abstract_env.time, "time", lambda: next(ticks))

    def _wrapped_env(self, env_info):
        class Env(_ConcreteEnv):
            @add_step_timing_to_env_info_decorator
            def step(self, action):
                """Doc of step."""
                return super().step(action)

        return Env(env_info=env_info)

    def test_none_env_info_becomes_timed_dict(self, fake_clock):
        env = self._wrapped_env(None)
        obs, reward, terminated, truncated, info = env.step("click")
        assert (obs, reward, terminated, truncated) == ({"obs": "click"}, 1.0, False, False)
        assert info == {
            "action_exec_start": 10.0,
            "action_exec_stop": 12.5,
            "action_exec_timeout": 0.0,
        }
        assert env.actions == ["click"]

    def test_existing_timing_keys_are_not_overwritten(self, fake_clock):
        env = self._wrapped_env(
            {"action_exec_start": 1.0, "action_exec_stop": 2.0, "action_exec_timeout": 3.0, "k": 1}
        )
        *_, info = env.step("a")
        assert info == {
            "action_exec_start": 1.0,
            "action_exec_stop": 2.0,
            "action_exec_timeout": 3.0,
            "k": 1,
        }

    def test_missing_keys_are_filled_individually(self, fake_clock):
        env = self._wrapped_env({"action_exec_timeout": 5.0, "task_info": "x"})
        *_, info = env.step("a")
        assert info == {
            "action_exec_timeout": 5.0,
            "task_info": "x",
            "action_exec_start": 10.0,
            "action_exec_stop": 12.5,
        }

    def test_each_call_is_timed_separately(self, fake_clock):
        env = self._wrapped_env(None)
        *_, first = env.step("a")
        env.env_info = None
        *_, second = env.step("b")
        assert (first["action_exec_start"], first["action_exec_stop"]) == (10.0, 12.5)
        assert (second["action_exec_start"], second["action_exec_stop"]) == (20.0, 21.0)

    def test_preserves_wrapped_function_metadata(self):
        def step(self, action):
            """Doc of step."""

        wrapped = add_step_timing_to_env_info_decorator(step)
        assert wrapped.__name__ == "step"
        assert wrapped.__doc__ == "Doc of step."
        assert wrapped.__wrapped__ is step
