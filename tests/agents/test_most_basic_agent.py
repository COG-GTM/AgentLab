from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from agentlab.agents.most_basic_agent.most_basic_agent import MostBasicAgentArgs
from agentlab.llm.llm_utils import ParseError


@dataclass
class _FakeModelArgs:
    model_name: str = "fake"
    answers: tuple = ()

    def make_model(self):
        answers = iter(self.answers)
        return lambda messages: {"role": "assistant", "content": next(answers)}

    def prepare_server(self):
        return "prepared"

    def close_server(self):
        return "closed"


OBS = {"goal": "Click the button", "pruned_html": "<button bid='a1'>OK</button>"}


def test_args_lifecycle():
    args = MostBasicAgentArgs(chat_model_args=_FakeModelArgs())
    assert args.prepare() == "prepared"
    assert args.close() == "closed"
    args.set_reproducibility_mode()
    assert args.temperature == 0


@pytest.mark.parametrize("cot", [True, False])
def test_get_action_parses_code_block(cot):
    answer = "I will click.\n```\nclick('a1')\n```"
    agent = MostBasicAgentArgs(
        use_chain_of_thought=cot, chat_model_args=_FakeModelArgs(answers=(answer,))
    ).make_agent()

    action, info = agent.get_action(OBS)

    assert action == "click('a1')"
    assert info.think == answer
    prompt = str(info.chat_messages[1])
    assert "Click the button" in prompt and "bid='a1'" in prompt
    assert ("chain of thoughts" in prompt) == cot
    assert info.extra_info["chat_model_args"]["model_name"] == "fake"
    assert info.stats["response_length"] == len(answer)


def test_get_action_retries_until_code_block():
    answers = ("no code", "```\nnoop()\n```")
    agent = MostBasicAgentArgs(chat_model_args=_FakeModelArgs(answers=answers)).make_agent()
    action, info = agent.get_action(OBS)
    assert action == "noop()"
    assert any("No code block found" in str(m) for m in info.chat_messages)


def test_get_action_fails_after_retries():
    agent = MostBasicAgentArgs(chat_model_args=_FakeModelArgs(answers=("x",) * 3)).make_agent()
    with pytest.raises(ParseError):
        agent.get_action(OBS)
