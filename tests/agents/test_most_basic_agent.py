from dataclasses import dataclass, field
from unittest.mock import MagicMock

import bgym
import pytest

from agentlab.agents.most_basic_agent import most_basic_agent as mba
from agentlab.agents.most_basic_agent.most_basic_agent import (
    MostBasicAgent,
    MostBasicAgentArgs,
)
from agentlab.experiments.loop import ExpArgs
from agentlab.llm.base_api import BaseModelArgs
from agentlab.llm.llm_utils import AIMessage, ParseError


class FakeChat:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, messages):
        self.calls.append([dict(m) for m in messages])
        return AIMessage(self.responses.pop(0))


@dataclass
class FakeModelArgs(BaseModelArgs):
    model_name: str = "fake/model"
    responses: list = field(default_factory=list)

    def make_model(self):
        self.chat = FakeChat(self.responses)
        return self.chat


OBS = {"goal": "Click the submit button", "pruned_html": "<button bid='a12'>Submit</button>"}
GOOD = "I should click submit.\n```\nclick('a12')\n```"


def _content_text(message) -> str:
    content = message["content"]
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content)


# ---------------------------------------------------------------- args


def test_args_defaults():
    args = MostBasicAgentArgs()
    assert args.agent_name == "BasicAgent"
    assert args.temperature == 0.1
    assert args.use_chain_of_thought is False
    assert args.chat_model_args is None


def test_make_agent_forwards_fields():
    cma = FakeModelArgs(responses=[GOOD])
    args = MostBasicAgentArgs(temperature=0.7, use_chain_of_thought=True, chat_model_args=cma)
    agent = args.make_agent()
    assert isinstance(agent, MostBasicAgent)
    assert agent.temperature == 0.7
    assert agent.use_chain_of_thought is True
    assert agent.chat_model_args is cma
    assert agent.chat is cma.chat
    assert isinstance(agent.action_set, bgym.HighLevelActionSet)


def test_set_reproducibility_mode_zeroes_temperature():
    args = MostBasicAgentArgs(temperature=0.9)
    args.set_reproducibility_mode()
    assert args.temperature == 0


def test_prepare_and_close_delegate_to_model_args():
    cma = MagicMock()
    cma.prepare_server.return_value = "prepared"
    cma.close_server.return_value = "closed"
    args = MostBasicAgentArgs(chat_model_args=cma)
    assert args.prepare() == "prepared"
    assert args.close() == "closed"
    cma.prepare_server.assert_called_once_with()
    cma.close_server.assert_called_once_with()


# ---------------------------------------------------------------- get_action


def test_get_action_parses_first_code_block():
    cma = FakeModelArgs(responses=["thinking\n```\nclick('a12')\n```\n```\nnoop()\n```"])
    agent = MostBasicAgentArgs(chat_model_args=cma).make_agent()
    action, info = agent.get_action(OBS)

    assert action == "click('a12')"
    assert info.think.startswith("thinking")
    assert "noop()" in info.think


def test_get_action_prompt_contents_and_info():
    cma = FakeModelArgs(responses=[GOOD])
    agent = MostBasicAgentArgs(chat_model_args=cma).make_agent()
    action, info = agent.get_action(OBS)

    assert action == "click('a12')"
    assert info.think == GOOD
    sent = cma.chat.calls[0]
    assert [m["role"] for m in sent] == ["system", "user"]
    assert _content_text(sent[0]) == "You are a web assistant."
    human = _content_text(sent[1])
    assert OBS["goal"] in human
    assert OBS["pruned_html"] in human
    assert "click('a314')" in human  # bid example
    assert "chain of thoughts" not in human

    # system + user + assistant answer
    assert info.stats["prompt_length"] == 3
    assert info.stats["response_length"] == len(GOOD)
    # cost tracker stats are merged in
    assert {"input_tokens", "output_tokens", "cost"} <= set(info.stats)
    assert info.extra_info["chat_model_args"]["model_name"] == "fake/model"


def test_chain_of_thought_adds_instruction():
    cma = FakeModelArgs(responses=[GOOD])
    agent = MostBasicAgentArgs(chat_model_args=cma, use_chain_of_thought=True).make_agent()
    agent.get_action(OBS)
    human = _content_text(cma.chat.calls[0][1])
    assert "chain of thoughts reasoning" in human


def test_get_action_retries_until_code_block():
    cma = FakeModelArgs(responses=["no code here", GOOD])
    agent = MostBasicAgentArgs(chat_model_args=cma).make_agent()
    action, info = agent.get_action(OBS)

    assert action == "click('a12')"
    assert len(cma.chat.calls) == 2
    retry_msgs = cma.chat.calls[1]
    assert retry_msgs[-1]["role"] == "user"
    assert "No code block found" in _content_text(retry_msgs[-1])
    assert info.stats["prompt_length"] == 5


def test_get_action_raises_after_three_failures():
    cma = FakeModelArgs(responses=["nope"] * 5)
    agent = MostBasicAgentArgs(chat_model_args=cma).make_agent()
    with pytest.raises(ParseError, match="after 3 retries"):
        agent.get_action(OBS)
    assert len(cma.chat.calls) == 3


# ---------------------------------------------------------------- module config


def test_experiment_config_and_module_constants():
    exps = mba.experiment_config()
    assert len(exps) == 2
    assert all(isinstance(e, ExpArgs) for e in exps)
    assert [e.agent_args.use_chain_of_thought for e in exps] == [True, False]
    assert {e.env_args.task_name for e in exps} == {"miniwob.click-button"}
    assert mba.env_args.max_steps == 10
    assert mba.AGENT_4o_MINI.temperature == 0.3
    assert mba.AGENT_4o_MINI.use_chain_of_thought is True
    assert mba.AGENT_4o_MINI.chat_model_args.model_name == "gpt-4o-mini-2024-07-18"
