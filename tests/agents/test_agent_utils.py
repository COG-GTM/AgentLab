from dataclasses import dataclass, field

from browsergym.experiments.agent import AgentInfo

from agentlab.agents.agent_args import ChatModelAgentArgs
from agentlab.agents.agent_utils import busted_retry_ans_dict, make_agent_info


def test_busted_retry_ans_dict():
    ans_dict = busted_retry_ans_dict(max_retry=4)
    assert ans_dict == {"action": None, "n_retry": 5, "busted_retry": 1}


class FakeChatLLM:
    def __init__(self, stats):
        self._stats = stats

    def get_stats(self):
        # return a fresh copy, mirroring real models that return per-call stats
        return dict(self._stats)


@dataclass
class FakeChatModelArgs:
    model_name: str = "fake/model"
    temperature: float = 0.1


def test_make_agent_info_collects_stats_and_packs_info():
    chat_llm = FakeChatLLM({"cost": 1.0})
    ans_dict = {"action": 'click("a1")', "think": "because", "n_retry": 2, "busted_retry": 0}
    chat_messages = ["system", "human"]
    chat_model_args = FakeChatModelArgs()

    info = make_agent_info(chat_llm, ans_dict, chat_messages, chat_model_args)

    assert isinstance(info, AgentInfo)
    assert info.think == "because"
    assert info.chat_messages == chat_messages
    assert info.stats == {"cost": 1.0, "n_retry": 2, "busted_retry": 0}
    assert info.extra_info == {"chat_model_args": {"model_name": "fake/model", "temperature": 0.1}}


def test_make_agent_info_defaults_think_to_none():
    info = make_agent_info(
        FakeChatLLM({}),
        {"action": None, "n_retry": 5, "busted_retry": 1},
        [],
        FakeChatModelArgs(),
    )
    assert info.think is None


class RecordingChatModelArgs:
    """Minimal chat_model_args double recording prepare/close server calls."""

    def __init__(self):
        self.prepared = False
        self.closed = False

    def prepare_server(self):
        self.prepared = True
        return "prepared"

    def close_server(self):
        self.closed = True
        return "closed"


@dataclass
class _DummyChatModelAgentArgs(ChatModelAgentArgs):
    chat_model_args: object = None

    def make_agent(self):
        raise NotImplementedError


def test_chat_model_agent_args_prepare_close_delegate():
    chat_model_args = RecordingChatModelArgs()
    args = _DummyChatModelAgentArgs(chat_model_args=chat_model_args)

    assert args.prepare() == "prepared"
    assert chat_model_args.prepared is True

    assert args.close() == "closed"
    assert chat_model_args.closed is True
