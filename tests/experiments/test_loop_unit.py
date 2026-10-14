"""Hermetic unit tests for agentlab.experiments.loop (fake env + fake agent, no browser)."""

import gzip
import json
import logging
import pickle
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

import gymnasium as gym
import numpy as np
import pytest
from browsergym.core.chat import Chat
from browsergym.experiments.agent import Agent
from PIL import Image

from agentlab.experiments import loop
from agentlab.experiments.loop import (
    AbstractAgentArgs,
    DataclassJSONEncoder,
    EnvArgs,
    ExpArgs,
    ExpResult,
    StepInfo,
    StepTimestamps,
)

# --------------------------------------------------------------------------- fakes


class FakeActionSet:
    def to_python_code(self, action):
        return f"# {action}"


class FakeAgent(Agent):
    action_set = FakeActionSet()

    def __init__(self, actions, raise_at=None, exc=ValueError, think="thinking"):
        self.actions = list(actions)
        self.raise_at = raise_at
        self.exc = exc
        self.think = think
        self.calls = 0
        self.task_name = None
        self.final_tape = "the-tape"

    def set_task_name(self, task_name):
        self.task_name = task_name

    def obs_preprocessor(self, obs):
        obs = dict(obs)
        obs["preprocessed"] = True
        return obs

    def get_action(self, obs):
        idx = self.calls
        self.calls += 1
        if self.raise_at is not None and idx == self.raise_at:
            raise self.exc("agent exploded")
        action = self.actions[idx] if idx < len(self.actions) else None
        info = {"stats": {"llm_cost": 0.5}}
        if self.think is not None:
            info["think"] = self.think
        return action, info


@dataclass
class FakeAgentArgs(AbstractAgentArgs):
    actions: tuple = ("click('a')", "click('b')")
    raise_at: int = None
    agents: list = field(default_factory=list, repr=False)

    def make_agent(self):
        agent = FakeAgent(self.actions, raise_at=self.raise_at)
        self.agents.append(agent)
        return agent


class FakeEnv(gym.Env):
    """Episode terminates after `n_steps` env.step() calls."""

    def __init__(self, n_steps=2, with_chat=False, goal_object=True):
        self.n_steps = n_steps
        self.t = 0
        self.closed = False
        self.actions = []
        self.reset_seed = None
        self.goal_object = goal_object
        if with_chat:
            self.chat = mock.MagicMock(spec=Chat)

    def _obs(self):
        obs = {
            "goal": "reach the end",
            "axtree_txt": "button 'ok'",
            "screenshot": np.full((4, 4, 3), self.t * 10, dtype=np.uint8),
            "screenshot_som": np.full((4, 4, 3), 200, dtype=np.uint8),
        }
        if self.goal_object:
            obs["goal_object"] = [{"type": "text", "text": "reach the end"}]
        return obs

    def reset(self, seed=None, options=None):
        self.reset_seed = seed
        self.t = 0
        return self._obs(), {"recording_start_time": 123.0}

    def step(self, action):
        self.actions.append(action)
        self.t += 1
        terminated = self.t >= self.n_steps
        info = {
            "action_exec_start": 10.0,
            "action_exec_stop": 13.0,
            "action_exec_timeout": 1.0,
            "RAW_REWARD_GLOBAL": 0.25 * self.t,
            "task_info": {"t": self.t},
        }
        return self._obs(), float(terminated), terminated, False, info

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def restore_logging():
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    openai_logger = logging.getLogger("openai._base_client")
    openai_level = openai_logger.level
    yield
    for h in list(root.handlers):
        if h not in handlers:
            root.removeHandler(h)
            h.close()
    root.handlers[:] = handlers
    root.setLevel(level)
    openai_logger.setLevel(openai_level)


@pytest.fixture
def no_debug(monkeypatch):
    monkeypatch.setattr(loop, "_is_debugging", lambda: False)


def _make_exp(tmp_path, monkeypatch, env=None, **agent_kwargs):
    env = env if env is not None else FakeEnv()
    calls = []

    def fake_make_env(self, action_mapping, exp_dir, exp_task_kwargs={}, use_raw_page_output=True):
        calls.append(dict(action_mapping=action_mapping, exp_dir=exp_dir, raw=use_raw_page_output))
        return env

    monkeypatch.setattr(EnvArgs, "make_env", fake_make_env)
    exp = ExpArgs(
        agent_args=FakeAgentArgs(**agent_kwargs),
        env_args=EnvArgs(task_name="fake.task", task_seed=7),
    )
    exp.prepare(tmp_path)
    return exp, env, calls


def _summary(exp):
    return json.loads((Path(exp.exp_dir) / "summary_info.json").read_text())


# --------------------------------------------------------------------------- EnvArgs.make_env


@pytest.fixture
def fake_gym_make(monkeypatch):
    make = mock.MagicMock(return_value="ENV")
    monkeypatch.setattr(loop.gym, "make", make)
    monkeypatch.setattr(loop, "_get_env_name", lambda name: f"browsergym/{name}")
    return make


def test_make_env_minimal_kwargs(fake_gym_make, tmp_path):
    env = EnvArgs(task_name="miniwob.click-test", max_steps=3).make_env(
        action_mapping="MAP", exp_dir=tmp_path
    )
    assert env == "ENV"
    fake_gym_make.assert_called_once_with(
        "browsergym/miniwob.click-test",
        disable_env_checker=True,
        max_episode_steps=3,
        headless=True,
        wait_for_user_message=False,
        action_mapping="MAP",
        use_raw_page_output=True,
    )


def test_make_env_all_optional_kwargs(fake_gym_make, tmp_path):
    args = EnvArgs(
        task_name="miniwob.click-test",
        record_video=True,
        viewport={"width": 10, "height": 20},
        slow_mo=0,
        pre_observation_delay=0.5,
        storage_state={"cookies": []},
        task_kwargs={"a": 1, "b": 2},
        headless=False,
        wait_for_user_message=True,
    )
    args.make_env(
        action_mapping=None,
        exp_dir=tmp_path,
        exp_task_kwargs={"b": 3, "c": 4},
        use_raw_page_output=False,
    )
    kwargs = fake_gym_make.call_args.kwargs
    assert kwargs["record_video_dir"] == tmp_path
    assert kwargs["viewport"] == {"width": 10, "height": 20}
    assert kwargs["slow_mo"] == 0  # 0 is not None -> forwarded
    assert kwargs["pre_observation_delay"] == 0.5
    assert kwargs["pw_context_kwargs"] == {"storage_state": {"cookies": []}}
    assert kwargs["task_kwargs"] == {"a": 1, "b": 3, "c": 4}
    assert kwargs["headless"] is False
    assert kwargs["wait_for_user_message"] is True
    assert kwargs["use_raw_page_output"] is False


def test_make_env_exp_task_kwargs_without_task_kwargs(fake_gym_make, tmp_path):
    EnvArgs(task_name="x").make_env(None, tmp_path, exp_task_kwargs={"z": 1})
    assert fake_gym_make.call_args.kwargs["task_kwargs"] == {"z": 1}


def test_make_env_assistantbench_output_file(fake_gym_make, tmp_path):
    EnvArgs(task_name="assistantbench.test.3", task_kwargs={"k": "v"}).make_env(None, tmp_path)
    assert fake_gym_make.call_args.kwargs["task_kwargs"] == {
        "k": "v",
        "output_file": tmp_path / "assistantbench-prediction.json",
    }


def test_make_env_assistantbench_validation_is_not_patched(fake_gym_make, tmp_path):
    EnvArgs(task_name="assistantbench.validation.3").make_env(None, tmp_path)
    assert "task_kwargs" not in fake_gym_make.call_args.kwargs


@pytest.mark.parametrize(
    "task_name, module",
    [
        ("miniwob.click-test", "browsergym.miniwob"),
        ("workarena.servicenow.x", "browsergym.workarena"),
        ("webarena.1", "browsergym.webarena"),
        ("visualwebarena.1", "browsergym.visualwebarena"),
        ("assistantbench.test.1", "browsergym.assistantbench"),
        ("weblinx.abc", "weblinx_browsergym"),
        ("openended", None),
    ],
)
def test_get_env_name_lazy_imports(monkeypatch, task_name, module):
    if module is not None:
        monkeypatch.setitem(sys.modules, module, types.ModuleType(module))
    assert loop._get_env_name(task_name) == f"browsergym/{task_name}"


# --------------------------------------------------------------------------- small helpers


def test_seed_max_is_xor_today():
    # Documents current behavior; see the xfail test below for the intended value.
    assert loop.SEED_MAX == 34


@pytest.mark.xfail(strict=True, reason="BUG: SEED_MAX = 2 ^ 32 is XOR (== 34), not 2**32")
def test_seed_max_is_two_to_the_32():
    assert loop.SEED_MAX == 2**32


def test_abstract_agent_args_name_defaults_to_class_name():
    assert FakeAgentArgs().agent_name == "FakeAgentArgs"
    assert FakeAgentArgs(agent_name="custom").agent_name == "custom"
    args = FakeAgentArgs()
    assert args.prepare() is None
    assert args.close() is None


def test_abstract_agent_args_is_abstract():
    with pytest.raises(TypeError):
        AbstractAgentArgs()


def test_save_package_versions(tmp_path):
    loop.save_package_versions(tmp_path)
    lines = (tmp_path / "package_versions.txt").read_text().splitlines()
    assert lines == sorted(lines)
    assert any(line.startswith("numpy==") for line in lines)


def test_extract_err_msg_returns_last_error():
    steps = [
        StepInfo(agent_info=None),
        StepInfo(agent_info={"err_msg": "first", "stack_trace": "st1"}),
        StepInfo(agent_info={}),
        StepInfo(agent_info={"err_msg": "second"}),
    ]
    assert loop._extract_err_msg(steps) == ("second", None)
    assert loop._extract_err_msg([StepInfo()]) == (None, None)
    assert loop._extract_err_msg([]) == (None, None)


def test_aggregate_episode_stats():
    steps = [
        StepInfo(stats=None),
        StepInfo(stats={"cost": 1.0, "tokens": np.int64(5), "gone": None}),
        StepInfo(stats={"cost": 2.5, "tokens": np.int64(7), "gone": None}),
    ]
    with pytest.warns(RuntimeWarning):  # nanmax over all-NaN column
        agg = loop._aggregate_episode_stats(steps)
    assert agg["cum_steps"] == 3
    assert agg["cum_cost"] == 3.5 and agg["max_cost"] == 2.5
    assert agg["cum_tokens"] == 12 and agg["max_tokens"] == 7
    assert type(agg["cum_tokens"]) is int
    assert agg["cum_gone"] == 0.0  # nansum of all-NaN is 0
    assert agg["max_gone"] is None  # nanmax of all-NaN is NaN -> None
    json.dumps(agg)  # must be JSON serialisable


def test_aggregate_episode_stats_empty():
    assert loop._aggregate_episode_stats([]) == {"cum_steps": 0}


def test_is_debugging_reflects_trace(monkeypatch):
    monkeypatch.setattr(loop.sys, "gettrace", lambda: None)
    assert loop._is_debugging() is False
    monkeypatch.setattr(loop.sys, "gettrace", lambda: object())
    assert loop._is_debugging() is True


def test_flatten_dict():
    assert loop._flatten_dict({"a": 1, "b": {"c": 2, "d": {"e": 3}}, "f": {}}) == {
        "a": 1,
        "b.c": 2,
        "b.d.e": 3,
    }
    assert loop._flatten_dict({"a": {"b": 1}}, sep="/") == {"a/b": 1}


@pytest.mark.parametrize(
    "agent_info, expected",
    [
        ({"think": "I should click"}, "I should click\n\naction:\nclick('a')\n"),
        ({}, "action:\nclick('a')\n"),
    ],
)
def test_send_chat_info(agent_info, expected):
    chat = mock.MagicMock()
    loop._send_chat_info(chat, "click('a')", agent_info)
    chat.add_message.assert_called_once_with(role="info", msg=expected)


def test_move_old_exp(tmp_path):
    exp_dir = tmp_path / "exp"
    loop._move_old_exp(exp_dir)  # no-op when missing
    assert not (tmp_path / "_exp").exists()
    exp_dir.mkdir()
    (exp_dir / "f.txt").write_text("x")
    loop._move_old_exp(str(exp_dir))
    assert not exp_dir.exists()
    assert (tmp_path / "_exp" / "f.txt").read_text() == "x"


def test_dataclass_json_encoder():
    payload = {
        "ts": StepTimestamps(env_start=1.5),
        "i": np.int32(3),
        "f": np.float32(0.5),
        "arr": np.array([[1, 2]]),
    }
    decoded = json.loads(json.dumps(payload, cls=DataclassJSONEncoder))
    assert decoded["ts"]["env_start"] == 1.5
    assert decoded["i"] == 3 and decoded["f"] == 0.5 and decoded["arr"] == [[1, 2]]
    with pytest.raises(TypeError):
        json.dumps({"s": {1, 2}}, cls=DataclassJSONEncoder)


# --------------------------------------------------------------------------- StepInfo


def test_step_info_from_reset_and_step_profiling():
    env = FakeEnv(n_steps=1)
    pre = mock.MagicMock(side_effect=lambda o: {**o, "pre": 1})

    s0 = StepInfo(step=0)
    s0.from_reset(env, seed=3, obs_preprocessor=pre)
    assert env.reset_seed == 3
    assert (s0.reward, s0.terminated, s0.truncated) == (0, False, False)
    assert s0.obs["pre"] == 1
    assert s0.profiling.action_exec_start == 123.0
    assert s0.profiling.action_exec_stop == s0.profiling.env_stop
    assert not s0.is_done

    s1 = StepInfo(step=1)
    s1.from_step(env, "act", obs_preprocessor=None)
    assert env.actions == ["act"]
    assert "pre" not in s1.obs
    assert s1.reward == 1.0 and s1.terminated is True and s1.is_done
    assert s1.raw_reward == 0.25
    assert s1.task_info == {"t": 1}
    t = s1.profiling
    assert t.action_exec_start == 10.0
    assert t.action_exect_after_timeout == 13.0
    assert t.action_exec_stop == 12.0
    assert t.wait_for_page_loading_start is None and t.validation_stop is None


def test_step_info_from_reset_without_recording_start_time():
    env = FakeEnv()
    env.reset = lambda seed=None: ({"goal": "g"}, {})
    s = StepInfo(step=0)
    s.from_reset(env, seed=0, obs_preprocessor=None)
    assert s.profiling.action_exec_start == s.profiling.env_start


@pytest.mark.parametrize(
    "terminated, truncated, done",
    [(False, False, False), (True, False, True), (False, True, True), (None, None, None)],
)
def test_step_info_is_done(terminated, truncated, done):
    assert StepInfo(terminated=terminated, truncated=truncated).is_done == done


def test_step_info_from_action_builds_stats():
    agent = FakeAgent(["click('x')"])
    s = StepInfo(step=0, obs={"goal": "hello world", "n": 3})
    assert s.from_action(agent) == "click('x')"
    assert s.action == "click('x')"
    assert "stats" not in s.agent_info  # popped into s.stats
    assert s.stats["llm_cost"] == 0.5
    assert s.stats["n_token_goal"] > 0
    assert "n_token_n" not in s.stats
    assert s.stats["agent_elapsed"] >= 0
    assert "step_elapsed" in s.stats


def test_step_info_make_stats_non_dict_obs():
    s = StepInfo(obs="raw text", agent_info={})
    s.make_stats()
    assert set(s.stats) == {"step_elapsed", "agent_elapsed"}


def test_save_step_info_writes_files_and_restores_screenshots(tmp_path):
    shot = np.zeros((3, 3, 3), dtype=np.uint8)
    som = np.full((3, 3, 3), 255, dtype=np.uint8)
    s = StepInfo(
        step=2,
        obs={"goal": "g", "screenshot": shot, "screenshot_som": som, "goal_object": ["G1"]},
        agent_info={"think": "t"},
    )
    s.save_step_info(tmp_path, save_json=True, save_screenshot=True, save_som=True)

    assert (tmp_path / "screenshot_step_2.png").exists()
    assert (tmp_path / "screenshot_som_step_2.png").exists()
    with gzip.open(tmp_path / "goal_object.pkl.gz", "rb") as f:
        assert pickle.load(f) == ["G1"]
    with gzip.open(tmp_path / "step_2.pkl.gz", "rb") as f:
        saved = pickle.load(f)
    assert "screenshot" not in saved.obs and "screenshot_som" not in saved.obs
    assert saved.obs["goal_object"] is None
    json_info = json.loads((tmp_path / "steps_info.json").read_text())
    assert json_info["step"] == 2 and json_info["agent_info"] == {"think": "t"}

    # screenshots are put back on the live object; goal_object stays a placeholder
    assert s.obs["screenshot"] is shot and s.obs["screenshot_som"] is som
    assert s.obs["goal_object"] is None


def test_save_step_info_goal_object_written_once(tmp_path):
    StepInfo(step=0, obs={"goal_object": ["first"]}).save_step_info(tmp_path)
    StepInfo(step=1, obs={"goal_object": ["second"]}).save_step_info(tmp_path)
    with gzip.open(tmp_path / "goal_object.pkl.gz", "rb") as f:
        assert pickle.load(f) == ["first"]


def test_save_step_info_respects_flags(tmp_path):
    s = StepInfo(
        step=0,
        obs={"screenshot": np.zeros((2, 2, 3), np.uint8), "screenshot_som": None},
    )
    s.save_step_info(tmp_path, save_screenshot=False, save_som=True)
    assert not list(tmp_path.glob("screenshot*"))
    assert not (tmp_path / "steps_info.json").exists()
    assert "screenshot_som" not in s.obs  # None is not restored


def test_save_step_info_non_dict_obs(tmp_path):
    StepInfo(step=5, obs=None).save_step_info(tmp_path)
    with gzip.open(tmp_path / "step_5.pkl.gz", "rb") as f:
        assert pickle.load(f).obs is None


# --------------------------------------------------------------------------- ExpArgs.prepare / _make_dir


def test_make_id_is_stable():
    exp = ExpArgs(agent_args=FakeAgentArgs(), env_args=EnvArgs(task_name="t"))
    exp.make_id()
    first = exp.exp_id
    assert len(first) == 36
    exp.make_id()
    assert exp.exp_id == first
    exp2 = ExpArgs(agent_args=None, env_args=None, exp_id="fixed")
    exp2.make_id()
    assert exp2.exp_id == "fixed"


def test_prepare_creates_dir_and_pickles(tmp_path):
    exp = ExpArgs(agent_args=FakeAgentArgs(), env_args=EnvArgs(task_name="t.a/b:c", task_seed=None))
    with mock.patch.object(loop.np.random, "randint", return_value=11) as randint:
        exp.prepare(tmp_path)
    randint.assert_called_once_with(0, loop.SEED_MAX)
    assert exp.env_args.task_seed == 11
    assert exp.exp_name == "FakeAgentArgs_on_t.a/b:c_11"
    assert exp.exp_dir.parent == tmp_path
    assert exp.exp_dir.name.endswith("_FakeAgentArgs_on_t.a_b_c_11")  # sanitised
    assert exp.exp_id is not None
    with open(exp.exp_dir / "exp_args.pkl", "rb") as f:
        reloaded = pickle.load(f)
    assert reloaded.exp_name == exp.exp_name and reloaded.exp_id == exp.exp_id


def test_prepare_keeps_explicit_name_and_moves_old_dir(tmp_path):
    old = tmp_path / "old_exp"
    old.mkdir()
    exp = ExpArgs(
        agent_args=FakeAgentArgs(),
        env_args=EnvArgs(task_name="t", task_seed=0),
        exp_name="my exp",
        exp_dir=old,
    )
    exp.prepare(tmp_path / "root")
    assert (tmp_path / "_old_exp").exists() and not old.exists()
    assert exp.exp_dir.name.endswith("_my exp")
    assert exp.env_args.task_seed == 0  # 0 is a valid seed and must not be resampled


def test_make_dir_adds_suffix_on_collision(tmp_path):
    from datetime import datetime

    exp = ExpArgs(agent_args=None, env_args=None, exp_name="e")
    exp.exp_date = datetime(2024, 1, 2, 3, 4, 5)
    (tmp_path / "2024-01-02_03-04-05_e").mkdir()
    (tmp_path / "2024-01-02_03-04-05_e_1").mkdir()
    exp._make_dir(tmp_path)
    assert exp.exp_dir == tmp_path / "2024-01-02_03-04-05_e_2"


def test_make_dir_gives_up_after_999(tmp_path):
    from datetime import datetime

    exp = ExpArgs(agent_args=None, env_args=None, exp_name="e")
    exp.exp_date = datetime(2024, 1, 2)
    with mock.patch.object(loop.Path, "exists", return_value=True):
        with pytest.raises(ValueError, match="unique name"):
            exp._make_dir(tmp_path)


# --------------------------------------------------------------------------- ExpArgs.run


def test_run_happy_path(tmp_path, monkeypatch, no_debug):
    env = FakeEnv(n_steps=2, with_chat=True)
    exp, env, calls = _make_exp(tmp_path, monkeypatch, env=env)
    exp.run()
    exp_dir = Path(exp.exp_dir)

    agent = exp.agent_args.agents[0]
    assert agent.task_name == "fake.task"
    assert calls[0]["exp_dir"] == exp_dir and calls[0]["raw"] is False
    assert calls[0]["action_mapping"]("x") == "# x"
    assert env.reset_seed == 7
    assert env.actions == ["click('a')", "click('b')"]
    assert env.closed

    summary = _summary(exp)
    assert summary["n_steps"] == 2
    assert summary["cum_reward"] == 1.0
    assert summary["cum_raw_reward"] == 0.75
    assert summary["err_msg"] is None and summary["stack_trace"] is None
    assert summary["terminated"] is True and summary["truncated"] is False
    assert summary["stats.cum_steps"] == 3
    assert summary["stats.cum_llm_cost"] == 1.0

    for i in range(3):
        assert (exp_dir / f"step_{i}.pkl.gz").exists()
        assert (exp_dir / f"screenshot_step_{i}.png").exists()
    assert not list(exp_dir.glob("screenshot_som_*"))  # save_som defaults to False
    assert (exp_dir / "goal_object.pkl.gz").exists()
    assert (exp_dir / "package_versions.txt").exists()
    assert "Running experiment" in (exp_dir / "experiment.log").read_text()
    assert env.chat.add_message.call_count == 2
    assert exp.err_msg is None

    # the experiment file handler is detached after run
    root = logging.getLogger()
    assert exp.logging_file_handler not in root.handlers

    result = ExpResult(exp_dir)
    assert result.status == "done"
    steps = result.steps_info
    assert [s.step for s in steps] == [0, 1, 2]
    assert steps[0].action == "click('a')" and steps[2].action is None
    assert steps[1].obs["preprocessed"] is True
    assert steps[1].obs["goal_object"] == [{"type": "text", "text": "reach the end"}]
    assert steps[1].obs["screenshot"].shape == (4, 4, 3)
    assert steps[1].obs["screenshot"][0, 0, 0] == 10
    assert "screenshot_som" not in steps[1].obs
    assert result.exp_args.exp_id == exp.exp_id


def test_run_none_action_truncates(tmp_path, monkeypatch, no_debug):
    exp, env, _ = _make_exp(tmp_path, monkeypatch, env=FakeEnv(n_steps=5), actions=())
    exp.save_som = True
    exp.run()
    summary = _summary(exp)
    assert summary["n_steps"] == 0
    assert summary["truncated"] is True and summary["terminated"] is False
    assert summary["err_msg"] is None
    assert env.actions == []
    assert (Path(exp.exp_dir) / "screenshot_som_step_0.png").exists()
    assert ExpResult(exp.exp_dir).status == "done"


def test_run_agent_exception_is_captured(tmp_path, monkeypatch, no_debug):
    exp, env, _ = _make_exp(tmp_path, monkeypatch, raise_at=1)
    exp.run()
    summary = _summary(exp)
    assert summary["err_msg"].startswith(
        "Exception uncaught by agent or environment in task fake.task.\nValueError:\nagent exploded"
    )
    assert "Traceback" in summary["stack_trace"]
    assert summary["n_steps"] == 1
    assert summary["terminated"] is False
    assert exp.err_msg == summary["err_msg"] and exp.stack_trace == summary["stack_trace"]
    assert env.closed
    # the failing step is still saved by the finally block
    assert (Path(exp.exp_dir) / "step_1.pkl.gz").exists()
    assert ExpResult(exp.exp_dir).status == "error"


def test_run_make_env_failure(tmp_path, monkeypatch, no_debug):
    exp, _, _ = _make_exp(tmp_path, monkeypatch)

    def boom(self, **kwargs):
        raise RuntimeError("no browser")

    monkeypatch.setattr(EnvArgs, "make_env", boom)
    exp.run()
    summary = _summary(exp)
    assert "RuntimeError:\nno browser" in summary["err_msg"]
    assert summary["n_steps"] == 0
    assert summary["terminated"] is None and summary["truncated"] is None
    assert not list(Path(exp.exp_dir).glob("step_*.pkl.gz"))


def test_run_agent_error_in_agent_info_is_promoted(tmp_path, monkeypatch, no_debug):
    exp, _, _ = _make_exp(tmp_path, monkeypatch)
    orig = FakeAgent.get_action

    def get_action(self, obs):
        action, info = orig(self, obs)
        if self.calls == 1:
            info["err_msg"] = "parse error"
            info["stack_trace"] = "st"
        return action, info

    monkeypatch.setattr(FakeAgent, "get_action", get_action)
    exp.run()
    summary = _summary(exp)
    assert summary["err_msg"] == "parse error" and summary["stack_trace"] == "st"


def test_run_reraises_in_debug_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "_is_debugging", lambda: True)
    exp, env, _ = _make_exp(tmp_path, monkeypatch, raise_at=0)
    with pytest.raises(ValueError, match="agent exploded"):
        exp.run()
    assert "ValueError" in _summary(exp)["err_msg"]  # finally block still ran
    assert env.closed


def test_run_debug_mode_disabled_by_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "_is_debugging", lambda: True)
    exp, _, _ = _make_exp(tmp_path, monkeypatch, raise_at=0)
    exp.enable_debug = False
    exp.run()
    assert "ValueError" in _summary(exp)["err_msg"]


def test_run_keyboard_interrupt_marks_early_termination(tmp_path, monkeypatch, no_debug):
    exp, env, _ = _make_exp(tmp_path, monkeypatch)
    monkeypatch.setattr(FakeAgent, "get_action", mock.Mock(side_effect=KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        exp.run()
    summary = _summary(exp)
    assert "KeyboardInterrupt:\nEarly termination??" in summary["err_msg"]
    assert summary["stack_trace"] is None
    assert env.closed


def test_run_finally_block_swallows_cleanup_errors(tmp_path, monkeypatch, no_debug, caplog):
    exp, env, _ = _make_exp(tmp_path, monkeypatch, env=FakeEnv(n_steps=1))
    env.close = mock.Mock(side_effect=RuntimeError("close failed"))
    monkeypatch.setattr(ExpArgs, "save_summary_info", mock.Mock(side_effect=OSError("disk")))
    monkeypatch.setattr(ExpArgs, "_unset_logger", mock.Mock(side_effect=RuntimeError("unset")))
    exp.run()  # must not raise
    env.close.assert_called_once()
    ExpArgs._unset_logger.assert_called_once()
    log = (Path(exp.exp_dir) / "experiment.log").read_text()
    assert "Error while saving experiment info: disk" in log
    assert "Error while closing the environment: close failed" in log


def test_run_finally_block_survives_save_step_failure(tmp_path, monkeypatch, no_debug):
    exp, _, _ = _make_exp(tmp_path, monkeypatch, raise_at=0)
    monkeypatch.setattr(StepInfo, "save_step_info", mock.Mock(side_effect=OSError("nope")))
    exp.run()
    assert (
        "Error while saving step info in the finally block: nope"
        in (Path(exp.exp_dir) / "experiment.log").read_text()
    )
    assert "ValueError" in _summary(exp)["err_msg"]


def test_run_saves_tape_for_tape_agents(tmp_path, monkeypatch, no_debug):
    save_tape = mock.Mock()
    monkeypatch.setattr(loop, "TapeAgent", FakeAgent)
    monkeypatch.setattr(loop, "save_tape", save_tape, raising=False)
    env = FakeEnv(n_steps=1)
    env.task = {"id": 1}
    exp, _, _ = _make_exp(tmp_path, monkeypatch, env=env)
    exp.run()
    save_tape.assert_called_once()
    exp_dir, episode_info, task, tape = save_tape.call_args.args
    assert exp_dir == exp.exp_dir and task == {"id": 1} and tape == "the-tape"
    assert [s.step for s in episode_info] == [0, 1]


def test_set_logger_configures_root_and_openai(tmp_path):
    exp = ExpArgs(agent_args=None, env_args=None, logging_level=logging.DEBUG)
    exp.exp_dir = tmp_path
    exp._set_logger()
    root = logging.getLogger()
    assert exp.logging_file_handler in root.handlers
    assert root.level == logging.DEBUG
    assert logging.getLogger("openai._base_client").level == logging.INFO
    exp._unset_logger()
    assert exp.logging_file_handler not in root.handlers
    exp.logging_file_handler.close()


@pytest.mark.xfail(
    strict=True,
    reason="BUG: _set_logger removes handlers while iterating root.handlers, skipping every other one",
)
def test_set_logger_removes_all_previous_stream_handlers(tmp_path):
    root = logging.getLogger()
    h1, h2 = logging.StreamHandler(), logging.StreamHandler()
    root.handlers[:] = [h1, h2]
    exp = ExpArgs(agent_args=None, env_args=None)
    exp.exp_dir = tmp_path
    exp._set_logger()
    try:
        assert h1 not in root.handlers and h2 not in root.handlers
    finally:
        exp.logging_file_handler.close()


# --------------------------------------------------------------------------- ExpResult


def _write_summary(exp_dir, **summary):
    (Path(exp_dir) / "summary_info.json").write_text(json.dumps(summary))


@pytest.mark.parametrize(
    "summary, status",
    [
        (None, "incomplete"),
        ({"err_msg": "boom", "terminated": True}, "error"),
        ({"err_msg": None, "terminated": True}, "done"),
        ({"truncated": True}, "done"),
        ({"terminated": False, "truncated": False}, "incomplete"),
    ],
)
def test_exp_result_status(tmp_path, summary, status):
    if summary is not None:
        _write_summary(tmp_path, **summary)
    assert ExpResult(tmp_path).status == status


def test_exp_result_empty_summary_is_incomplete(tmp_path):
    (tmp_path / "summary_info.json").write_text("")
    res = ExpResult(tmp_path)
    with pytest.raises(FileNotFoundError, match="empty"):
        res.summary_info
    assert res.status == "incomplete"


def test_exp_result_summary_is_cached(tmp_path):
    _write_summary(tmp_path, a=1)
    res = ExpResult(tmp_path)
    assert res.summary_info == {"a": 1}
    _write_summary(tmp_path, a=2)
    assert res.summary_info == {"a": 1}


def test_exp_result_exp_args_follows_moved_dir(tmp_path):
    exp = ExpArgs(agent_args=FakeAgentArgs(), env_args=EnvArgs(task_name="t", task_seed=1))
    exp.prepare(tmp_path / "a")
    moved = tmp_path / "moved"
    Path(exp.exp_dir).rename(moved)
    res = ExpResult(str(moved))
    assert res.exp_args.exp_dir == moved
    assert res.exp_args is res.exp_args
    flat = res.flat_exp_args
    assert flat["env_args.task_name"] == "t"
    assert flat["agent_args.agent_name"] == "FakeAgentArgs"
    assert res.flat_exp_args is flat


def test_exp_result_get_exp_record(tmp_path):
    res = ExpResult(tmp_path)
    assert res.get_exp_record() == {"exp_dir": tmp_path}

    exp = ExpArgs(agent_args=FakeAgentArgs(), env_args=EnvArgs(task_name="t", task_seed=1))
    exp.prepare(tmp_path / "r")
    _write_summary(exp.exp_dir, cum_reward=1.0)
    record = ExpResult(exp.exp_dir).get_exp_record()
    assert record["cum_reward"] == 1.0 and record["env_args.task_seed"] == 1


def test_exp_result_screenshots(tmp_path):
    Image.new("RGB", (2, 2), (1, 2, 3)).save(tmp_path / "screenshot_step_0.png")
    Image.new("L", (2, 2), 9).save(tmp_path / "screenshot_step_2.jpg")
    Image.new("RGB", (2, 2), (5, 5, 5)).save(tmp_path / "screenshot_som_step_0.png")
    res = ExpResult(tmp_path)

    assert res.get_screenshot_path(0) == tmp_path / "screenshot_step_0.png"
    assert res.get_screenshot_path(2) == tmp_path / "screenshot_step_2.jpg"
    with pytest.raises(FileNotFoundError, match="step 1"):
        res.get_screenshot_path(1)

    shots = res.screenshots
    assert len(shots) == 3 and shots[1] is None
    assert shots[0].getpixel((0, 0)) == (1, 2, 3)
    assert shots[2].mode == "RGB"
    assert res.get_screenshot(0) is shots[0]  # cached

    with pytest.raises(FileNotFoundError):  # som listing is driven by regular screenshots
        res.screenshots_som


def test_exp_result_step_info_without_files(tmp_path):
    StepInfo(step=0, obs={"goal": "g"}).save_step_info(tmp_path)
    StepInfo(step=1, obs=None).save_step_info(tmp_path)
    res = ExpResult(tmp_path)
    s0 = res.get_step_info(0)
    assert "screenshot" not in s0.obs and "screenshot_som" not in s0.obs
    assert res.get_step_info(0) is s0
    assert res.get_step_info(1).obs is None
    assert [s.step for s in res.steps_info] == [0, 1]


def test_exp_result_videos_and_logs(tmp_path):
    res = ExpResult(tmp_path)
    with pytest.raises(FileNotFoundError, match="chat_video"):
        res.chat_video_path
    with pytest.raises(FileNotFoundError, match="task_video"):
        res.task_video_path
    (tmp_path / "chat_video").mkdir()
    (tmp_path / "chat_video" / "c.webm").write_bytes(b"")
    (tmp_path / "task_video").mkdir()
    (tmp_path / "task_video" / "t.webm").write_bytes(b"")
    assert res.chat_video_path.name == "c.webm"
    assert res.task_video_path.name == "t.webm"
    assert res.combined_video_path == tmp_path / "combined_video.mp4"
    (tmp_path / "experiment.log").write_text("hello")
    assert res.logs == "hello"
    (tmp_path / "experiment.log").write_text("changed")
    assert res.logs == "hello"


# --------------------------------------------------------------------------- discovery helpers


@pytest.fixture
def empty_cache(monkeypatch):
    cache = {}
    monkeypatch.setattr(loop, "EXP_RESULT_CACHE", cache)
    return cache


def test_get_exp_result_caches_by_str_path(tmp_path, empty_cache):
    a = loop.get_exp_result(tmp_path)
    b = loop.get_exp_result(str(tmp_path))
    assert a is b and list(empty_cache) == [str(tmp_path)]


def _touch_exp(path):
    path.mkdir(parents=True)
    (path / "exp_args.pkl").write_bytes(b"")
    return path


def test_yield_all_exp_results(tmp_path, empty_cache):
    root1, root2 = tmp_path / "r1", tmp_path / "r2"
    visible = _touch_exp(root1 / "study" / "exp_a")
    hidden_u = _touch_exp(root1 / "_exp_b")
    hidden_d = _touch_exp(root2 / ".exp_c")
    other = _touch_exp(root2 / "exp_d")

    found = {r.exp_dir for r in loop.yield_all_exp_results([root1, root2], progress_fn=None)}
    assert found == {visible, other}

    progress = mock.Mock(side_effect=lambda it, desc: it)
    found = {
        r.exp_dir
        for r in loop.yield_all_exp_results(
            [root1, root2], progress_fn=progress, load_hidden=True, use_cache=False
        )
    }
    assert found == {visible, other, hidden_u, hidden_d}
    progress.assert_called_once()
    assert len(empty_cache) == 2  # only the cached first pass populated it

    single = list(loop.yield_all_exp_results(str(root2), progress_fn=None))
    assert [r.exp_dir for r in single] == [other]
