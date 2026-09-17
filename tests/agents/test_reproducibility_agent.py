"""Unit tests for the trace replay logic of ReproAgent and ReproChatModel."""

from dataclasses import dataclass, field

import pytest
from browsergym.experiments.agent import AgentInfo

from agentlab.agents.generic_agent import reproducibility_agent
from agentlab.agents.generic_agent.agent_configs import FLAGS_GPT_3_5
from agentlab.agents.generic_agent.generic_agent import GenericAgent
from agentlab.agents.generic_agent.reproducibility_agent import ReproAgent, ReproChatModel
from agentlab.llm.chat_api import CheatMiniWoBLLMArgs
from agentlab.llm.llm_utils import AIMessage, Discussion, HumanMessage, SystemMessage


@dataclass
class StubStepInfo:
    agent_info: dict = field(default_factory=dict)
    action: str = None


@dataclass
class StubExpResult:
    step_info: StubStepInfo
    summary_info: dict = field(default_factory=lambda: {"err_msg": "some error"})

    def get_step_info(self, step: int) -> StubStepInfo:
        return self.step_info


def make_agent(monkeypatch, step_info: StubStepInfo) -> ReproAgent:
    monkeypatch.setattr(
        reproducibility_agent, "ExpResult", lambda repro_dir: StubExpResult(step_info)
    )
    return ReproAgent(
        chat_model_args=CheatMiniWoBLLMArgs(),
        flags=FLAGS_GPT_3_5,
        repro_dir="unused",
    )


def patch_parent_get_action(monkeypatch):
    """Replay a 2-message discussion through self.chat_llm, like GenericAgent does."""

    def fake_get_action(self, obs):
        messages = Discussion([SystemMessage("system"), HumanMessage("user")])
        response = self.chat_llm(messages)
        self.actions.append(response["content"])
        return response["content"], AgentInfo(chat_messages=messages)

    monkeypatch.setattr(GenericAgent, "get_action", fake_get_action)


def recorded_discussion(answer: str = None) -> Discussion:
    messages = [SystemMessage("system"), HumanMessage("user")]
    if answer is not None:
        messages.append(AIMessage(answer))
    return Discussion(messages)


def test_get_action_without_chat_messages(monkeypatch):
    agent = make_agent(monkeypatch, StubStepInfo(agent_info={"chat_messages": None}))

    action, agent_info = agent.get_action(obs={})

    assert action is None
    assert "Agent had no chat messages" in agent_info.markdown_page
    assert "some error" in agent_info.markdown_page


def test_get_action_rebuilds_missing_assistant_message(monkeypatch):
    old_messages = recorded_discussion()
    step_info = StubStepInfo(agent_info={"chat_messages": old_messages}, action='click("42")')
    agent = make_agent(monkeypatch, step_info)
    patch_parent_get_action(monkeypatch)

    action, _ = agent.get_action(obs={})

    assert len(old_messages) == 3
    assert old_messages[2]["role"] == "assistant"
    assert old_messages[2]["content"] == '<action>click("42")</action>'
    assert isinstance(agent.chat_llm, ReproChatModel)
    assert action == '<action>click("42")</action>'


def test_get_action_without_recorded_action_keeps_messages(monkeypatch):
    old_messages = recorded_discussion()
    step_info = StubStepInfo(agent_info={"chat_messages": old_messages}, action=None)
    agent = make_agent(monkeypatch, step_info)
    patch_parent_get_action(monkeypatch)

    action, _ = agent.get_action(obs={})

    assert len(old_messages) == 2
    assert action == "<action>None</action>"  # fallback of ReproChatModel


def test_get_action_replays_recorded_answer(monkeypatch):
    old_messages = recorded_discussion('<action>click("7")</action>')
    step_info = StubStepInfo(agent_info={"chat_messages": old_messages}, action='click("7")')
    agent = make_agent(monkeypatch, step_info)
    patch_parent_get_action(monkeypatch)

    action, agent_info = agent.get_action(obs={})

    assert len(old_messages) == 3  # no reconstruction when the answer was recorded
    assert action == '<action>click("7")</action>'
    assert agent_info.stats["lines_added"] == 0
    assert agent_info.stats["lines_removed"] == 0


@pytest.mark.parametrize("n_new_messages", [2, 3])
def test_repro_chat_model(n_new_messages):
    old_messages = recorded_discussion("<action>noop()</action>")
    chat_model = ReproChatModel(old_messages, delay=0)

    response = chat_model(Discussion(list(old_messages)[:n_new_messages]))

    if n_new_messages == 2:
        assert response is old_messages[2]
        assert len(chat_model.new_messages) == 3
    else:  # the recorded answer is missing, fall back to a no-op action
        assert response["content"] == "<action>None</action>"
    assert chat_model.get_stats() == {}
