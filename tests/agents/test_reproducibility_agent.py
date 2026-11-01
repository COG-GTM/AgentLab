from types import SimpleNamespace

import pytest
from browsergym.experiments.agent import AgentInfo
from bgym import HighLevelActionSetArgs

from agentlab.agents.generic_agent import AGENT_4o_MINI
from agentlab.agents.generic_agent.reproducibility_agent import (
    ReproAgentArgs,
    ReproChatModel,
    _diff_stats,
    _format_messages,
    _make_agent_stats,
    _make_backward_compatible,
    _make_diff,
    _set_style,
    make_repro_agent,
)
from agentlab.llm.llm_utils import AIMessage, Discussion, HumanMessage, SystemMessage

OLD_MESSAGES = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "do it"},
    {"role": "assistant", "content": "<action>click('a1')</action>"},
]


def test_repro_chat_model_replays_next_message():
    chat = ReproChatModel(OLD_MESSAGES, delay=0)
    response = chat(OLD_MESSAGES[:2])
    assert response == OLD_MESSAGES[2]
    assert chat.new_messages == OLD_MESSAGES
    assert chat.get_stats() == {}


def test_repro_chat_model_does_not_mutate_input():
    chat = ReproChatModel(OLD_MESSAGES, delay=0)
    messages = OLD_MESSAGES[:2]
    chat(messages)
    assert len(messages) == 2


def test_repro_chat_model_exhausted_returns_none_action():
    chat = ReproChatModel(OLD_MESSAGES[:2], delay=0)
    response = chat(OLD_MESSAGES[:2])
    assert response["role"] == "assistant"
    assert response["content"] == "<action>None</action>"


def test_format_messages_list_and_discussion():
    assert _format_messages(OLD_MESSAGES[:2]) == "system message:\nsys\n\nuser message:\ndo it\n"
    discussion = Discussion([SystemMessage("sys"), HumanMessage("do it")])
    assert _format_messages(discussion) == discussion.to_string()


def test_diff_stats_identical():
    assert _diff_stats("a\nb", "a\nb") == dict(lines_added=0, lines_removed=0, difference_ratio=0.0)


def test_diff_stats_changes():
    stats = _diff_stats("a\nb\nc", "a\nB\nc\nd")
    assert stats["lines_added"] == 2
    assert stats["lines_removed"] == 1
    assert stats["difference_ratio"] == pytest.approx(3 / 8)


def test_make_diff_has_style_and_no_nowrap():
    page = _make_diff("same\nold", "same\nnew")
    assert 'nowrap="nowrap"' not in page
    assert "<style" in page
    assert "Old Version" in page and "New Version" in page


def test_set_style_appends_and_prepends():
    html = "<html><head><style>a{}</style></head><body></body></html>"
    assert "a{}\nb{}" in _set_style(html, "b{}")
    assert "b{}\na{}" in _set_style(html, "b{}", prepend_previous_style=True)


def test_set_style_creates_style_tag():
    out = _set_style("<html><head></head><body></body></html>", "b{}")
    assert "<style>\nb{}</style>" in out


@pytest.mark.parametrize("agent_info", [{"think": "t"}, AgentInfo(think="t")])
def test_make_agent_stats(agent_info):
    old = OLD_MESSAGES
    new = OLD_MESSAGES[:2] + [AIMessage("<action>noop()</action>")]
    action, info = _make_agent_stats("noop()", agent_info, None, old, new)
    assert action == "noop()"
    assert isinstance(info, AgentInfo)
    assert info.think == "t"
    assert "<table" in info.html_page
    assert info.stats["lines_added"] == 1 and info.stats["lines_removed"] == 1


@pytest.mark.parametrize("action_set", ["bid+nav", ["bid", "nav"]])
def test_make_backward_compatible_converts_action_set(action_set):
    args = SimpleNamespace(
        flags=SimpleNamespace(action=SimpleNamespace(action_set=action_set, multi_actions=True))
    )
    out = _make_backward_compatible(args)
    new_set = out.flags.action.action_set
    assert isinstance(new_set, HighLevelActionSetArgs)
    assert list(new_set.subsets) == ["bid", "nav"]
    assert new_set.multiaction is True


def test_make_backward_compatible_keeps_new_format():
    action_set = HighLevelActionSetArgs(subsets=["bid"])
    args = SimpleNamespace(
        flags=SimpleNamespace(action=SimpleNamespace(action_set=action_set, multi_actions=False))
    )
    assert _make_backward_compatible(args).flags.action.action_set is action_set


def test_make_repro_agent(tmp_path):
    repro = make_repro_agent(AGENT_4o_MINI, tmp_path)
    assert isinstance(repro, ReproAgentArgs)
    assert repro._repro_dir == tmp_path
    assert repro.agent_name.startswith("Repro_")
    assert repro.chat_model_args is AGENT_4o_MINI.chat_model_args
    assert repro.flags is AGENT_4o_MINI.flags


def test_make_repro_agent_rejects_bad_inputs(tmp_path):
    with pytest.raises(AssertionError):
        make_repro_agent(SimpleNamespace(agent_name="x"), tmp_path)
    with pytest.raises(AssertionError):
        make_repro_agent(AGENT_4o_MINI, tmp_path / "missing")
