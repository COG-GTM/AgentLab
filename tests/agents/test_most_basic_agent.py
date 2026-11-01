from dataclasses import dataclass, field

import pytest

from agentlab.agents.most_basic_agent import most_basic_agent
from agentlab.agents.most_basic_agent.most_basic_agent import MostBasicAgent, MostBasicAgentArgs
from agentlab.llm.llm_utils import AIMessage, Discussion, ParseError

OBS = {"goal": "Click the OK button", "pruned_html": "<button bid='a1'>OK</button>"}


class FakeChat:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        return AIMessage(self.responses.pop(0))


@dataclass
class FakeModelArgs:
    model_name: str = "fake-model"
    responses: list = field(default_factory=list)
    prepared: bool = False
    closed: bool = False

    def make_model(self):
        self.chat = FakeChat(self.responses)
        return self.chat

    def prepare_server(self):
        self.prepared = True
        return "prepared"

    def close_server(self):
        self.closed = True
        return "closed"


def test_args_make_agent_and_lifecycle():
    model_args = FakeModelArgs()
    args = MostBasicAgentArgs(
        temperature=0.7, use_chain_of_thought=True, chat_model_args=model_args
    )
    agent = args.make_agent()
    assert isinstance(agent, MostBasicAgent)
    assert agent.temperature == 0.7 and agent.use_chain_of_thought
    assert agent.chat is model_args.chat

    args.set_reproducibility_mode()
    assert args.temperature == 0

    assert args.prepare() == "prepared" and model_args.prepared
    assert args.close() == "closed" and model_args.closed


def test_get_action_parses_first_code_block():
    model_args = FakeModelArgs(responses=["I will click\n```\nclick('a1')\n```\n```\nnoop()\n```"])
    agent = MostBasicAgentArgs(chat_model_args=model_args).make_agent()

    action, info = agent.get_action(OBS)

    assert action == "click('a1')"
    assert info.think.startswith("I will click")
    assert isinstance(info.chat_messages, Discussion)
    prompt = str(info.chat_messages[1])
    assert OBS["goal"] in prompt and OBS["pruned_html"] in prompt
    assert "chain of thoughts" not in prompt
    assert info.stats["prompt_length"] == 3  # system, user, assistant
    assert info.stats["response_length"] == len(info.think)
    assert {"input_tokens", "output_tokens", "cost"} <= set(info.stats)
    assert info.extra_info["chat_model_args"]["model_name"] == "fake-model"


def test_get_action_chain_of_thought_prompt():
    model_args = FakeModelArgs(responses=["```\nclick('a1')\n```"])
    agent = MostBasicAgentArgs(use_chain_of_thought=True, chat_model_args=model_args).make_agent()
    _, info = agent.get_action(OBS)
    assert "chain of thoughts" in str(info.chat_messages[1])


def test_get_action_retries_on_missing_code_block():
    model_args = FakeModelArgs(responses=["no code here", "```\nclick('a1')\n```"])
    agent = MostBasicAgentArgs(chat_model_args=model_args).make_agent()
    action, _ = agent.get_action(OBS)
    assert action == "click('a1')"
    assert len(model_args.chat.calls) == 2


def test_get_action_raises_after_retries():
    model_args = FakeModelArgs(responses=["nope"] * 3)
    agent = MostBasicAgentArgs(chat_model_args=model_args).make_agent()
    with pytest.raises(ParseError):
        agent.get_action(OBS)


def test_experiment_config():
    exps = most_basic_agent.experiment_config()
    assert len(exps) == 2
    assert [e.agent_args.use_chain_of_thought for e in exps] == [True, False]
    assert all(e.env_args.task_name == "miniwob.click-button" for e in exps)
