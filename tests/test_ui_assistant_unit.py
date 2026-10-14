"""Hermetic unit tests for agentlab.ui_assistant (no browser, no LLM)."""

import copy
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from unittest import mock

import pytest

from agentlab import ui_assistant
from agentlab.agents.generic_agent import AGENT_4o
from agentlab.experiments.loop import AbstractAgentArgs, EnvArgs, ExpArgs
from agentlab.ui_assistant import make_exp_args


@dataclass
class PlainAgentArgs(AbstractAgentArgs):
    def make_agent(self):
        raise NotImplementedError


def test_make_exp_args_generic_agent():
    agent_args = copy.deepcopy(AGENT_4o)
    agent_args.flags.action.demo_mode = "off"
    agent_args.flags.enable_chat = False

    exp_args = make_exp_args(agent_args, "https://example.test")

    assert isinstance(exp_args, ExpArgs)
    assert exp_args.agent_args is agent_args
    assert agent_args.flags.action.demo_mode == "default"
    assert agent_args.flags.enable_chat is True
    env = exp_args.env_args
    assert isinstance(env, EnvArgs)
    assert env.task_name == "openended"
    assert env.task_kwargs == {"start_url": "https://example.test"}
    assert env.task_seed is None
    assert env.max_steps == 1000
    assert env.headless is False
    assert env.record_video is True
    assert env.wait_for_user_message is True
    assert env.viewport == {"width": 1500, "height": 1280}
    assert env.slow_mo == 1000
    assert exp_args.exp_dir is None and exp_args.exp_name is None


def test_make_exp_args_non_generic_agent_with_flags():
    agent_args = SimpleNamespace(flags=SimpleNamespace(action=SimpleNamespace(demo_mode="off")))
    exp_args = make_exp_args(agent_args, "https://a.test")
    assert agent_args.flags.action.demo_mode == "default"
    assert not hasattr(agent_args.flags, "enable_chat")  # only GenericAgentArgs gets chat
    assert exp_args.env_args.task_kwargs == {"start_url": "https://a.test"}


def test_make_exp_args_agent_without_flags():
    agent_args = PlainAgentArgs()
    exp_args = make_exp_args(agent_args, "https://b.test")
    assert exp_args.agent_args is agent_args
    assert not hasattr(agent_args, "flags")


@pytest.fixture
def fake_main_env(monkeypatch, tmp_path):
    agent_args = PlainAgentArgs()
    import_object = mock.Mock(return_value=agent_args)
    prepare, run = mock.Mock(), mock.Mock()
    monkeypatch.setattr(ui_assistant, "import_object", import_object)
    monkeypatch.setattr(ui_assistant, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(ExpArgs, "prepare", prepare)
    monkeypatch.setattr(ExpArgs, "run", run)
    return SimpleNamespace(
        agent_args=agent_args, import_object=import_object, prepare=prepare, run=run
    )


def test_main_defaults(monkeypatch, tmp_path, fake_main_env):
    monkeypatch.setattr(sys, "argv", ["ui_assistant"])
    created = []
    real_make = ui_assistant.make_exp_args
    monkeypatch.setattr(
        ui_assistant,
        "make_exp_args",
        lambda a, url: created.append(real_make(a, url)) or created[-1],
    )

    ui_assistant.main()

    fake_main_env.import_object.assert_called_once_with(
        "agentlab.agents.generic_agent.AGENT_GPT5_MINI"
    )
    (exp_args,) = created
    assert exp_args.agent_args is fake_main_env.agent_args
    assert exp_args.env_args.task_kwargs == {"start_url": "https://www.google.com"}
    fake_main_env.prepare.assert_called_once_with(tmp_path / "ui_assistant_logs")
    fake_main_env.run.assert_called_once_with()


def test_main_custom_args_and_unknown_flags(monkeypatch, tmp_path, fake_main_env):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ui_assistant",
            "--agent_config",
            "my.pkg.AGENT",
            "--start_url",
            "https://custom.test",
            "--unknown",
            "ignored",
        ],
    )
    make = mock.Mock(wraps=ui_assistant.make_exp_args)
    monkeypatch.setattr(ui_assistant, "make_exp_args", make)

    ui_assistant.main()

    fake_main_env.import_object.assert_called_once_with("my.pkg.AGENT")
    make.assert_called_once_with(fake_main_env.agent_args, "https://custom.test")
    fake_main_env.prepare.assert_called_once()
    fake_main_env.run.assert_called_once()
