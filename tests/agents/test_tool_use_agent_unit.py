"""Hermetic unit tests for agentlab.agents.tool_use_agent.tool_use_agent.

The LLM is replaced by a scripted FakeModel; no network, browser or API key is used.
"""

import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import requests

from agentlab.agents.tool_use_agent import tool_use_agent as tua
from agentlab.agents.tool_use_agent.tool_use_agent import (
    AXTREE_NOTE,
    DEFAULT_PROMPT_CONFIG,
    SYS_MSG,
    GeneralHints,
    Goal,
    MsgGroup,
    Obs,
    PromptConfig,
    StructuredDiscussion,
    Summarizer,
    TaskHint,
    ToolUseAgent,
    ToolUseAgentArgs,
    _format_tabs,
)
from agentlab.benchmarks.osworld import OSWorldActionSet
from agentlab.llm.base_api import BaseModelArgs
from agentlab.llm.response_api import (
    AnthropicAPIMessageBuilder,
    APIPayload,
    LLMOutput,
    MessageBuilder,
    OpenAIResponseAPIMessageBuilder,
    ToolCall,
    ToolCalls,
)
from agentlab.llm.tracking import TRACKER

# --------------------------------------------------------------------------------------
# Fakes and helpers
# --------------------------------------------------------------------------------------


class FakeModel:
    """Scripted stand-in for a response-API model: returns queued LLMOutputs in order."""

    def __init__(self, responses=None, msg=None):
        self.responses = list(responses or [])
        self.payloads: list[APIPayload] = []
        self.reset_count = 0
        self.stats = SimpleNamespace(stats_dict={"n_api_calls": 0})
        if msg is not None:
            self.msg = msg

    def reset_stats(self):
        self.reset_count += 1
        self.stats.stats_dict = {"n_api_calls": 0}

    def __call__(self, payload: APIPayload) -> LLMOutput:
        assert isinstance(payload, APIPayload)
        # snapshot messages: the discussion list is mutated by the agent after the call
        self.payloads.append(
            SimpleNamespace(payload=payload, messages=list(payload.messages or []))
        )
        self.stats.stats_dict["n_api_calls"] += 1
        if not self.responses:
            raise AssertionError("FakeModel ran out of scripted responses")
        return self.responses.pop(0)


@dataclass
class FakeModelArgs(BaseModelArgs):
    model_name: str = "fake/model-1"
    api: str = "openai"
    builder: type = OpenAIResponseAPIMessageBuilder
    responses: list = field(default_factory=list)
    prepared: int = 0
    closed: int = 0

    def make_model(self):
        self._model = FakeModel(self.responses)
        return self._model

    def get_message_builder(self):
        return self.builder

    def prepare_server(self):
        self.prepared += 1
        return "prepared"

    def close_server(self):
        self.closed += 1
        return "closed"


class FakeActionSet:
    def __init__(self):
        self.apis = []

    def to_tool_description(self, api="openai"):
        self.apis.append(api)
        return [{"type": "function", "name": "click", "parameters": {}}]


def make_tool_calls(*names):
    calls = ToolCalls()
    for i, name in enumerate(names):
        calls.add_tool_call(
            ToolCall(name=name, arguments={"x": i}, raw_call=SimpleNamespace(id=f"c{i}"))
        )
    return calls


def llm_output(action="click(1)", think="thinking", tool_names=("click",)):
    return LLMOutput(think=think, action=action, tool_calls=make_tool_calls(*tool_names))


def fake_llm(builder=OpenAIResponseAPIMessageBuilder, responses=None):
    return FakeModel(responses, msg=builder)


def make_obs(**overrides):
    obs = {
        "goal_object": [{"type": "text", "text": "Book a flight to Paris"}],
        "last_action_error": "",
        "screenshot": np.zeros((4, 4, 3), dtype=np.uint8),
        "axtree_txt": "[a1] button 'Go'",
        "pruned_html": "<button bid='a1'>Go</button>",
        "open_pages_urls": ["http://a.test", "http://b.test"],
        "open_pages_titles": ["A", "B"],
        "active_page_index": 1,
        "elapsed_time": 1.5,
    }
    obs.update(overrides)
    return obs


def texts(msg: MessageBuilder) -> list[str]:
    return [c["text"] for c in msg.content if "text" in c]


def write_hint_db(path, rows):
    header = "task_name,semantic_keys,hint\n"
    body = "".join(f'{t},{k},"{h}"\n' for t, k, h in rows)
    path.write_text(header + body)
    return path


HINT_ROWS = [
    ("miniwob.book-flight", "dropdown", "select from the dropdown list"),
    ("miniwob.book-flight", "forms", "activate the field first"),
    ("miniwob.click-test", "dropdown", "select from the dropdown list"),
    ("workarena.order", "forms", "  use the form  "),
]


@pytest.fixture
def hint_csv(tmp_path):
    return write_hint_db(tmp_path / "hints.csv", HINT_ROWS)


@pytest.fixture(autouse=True)
def _reset_tracker():
    previous = getattr(TRACKER, "instance", None)
    TRACKER.instance = None
    yield
    TRACKER.instance = previous


def make_config(hint_path, **overrides):
    cfg = dict(
        goal=Goal(goal_as_system_msg=True),
        obs=Obs(use_last_error=True, use_screenshot=True, use_axtree=True),
        summarizer=Summarizer(do_summary=False),
        general_hints=GeneralHints(use_hints=False),
        task_hint=TaskHint(use_task_hint=False, hint_db_rel_path=str(hint_path)),
        keep_last_n_obs=None,
        multiaction=False,
        action_subsets=("coord",),
    )
    cfg.update(overrides)
    return PromptConfig(**cfg)


def make_agent(hint_path, responses, builder=OpenAIResponseAPIMessageBuilder, **cfg):
    model_args = FakeModelArgs(builder=builder, responses=list(responses))
    action_set = FakeActionSet()
    agent = ToolUseAgent(
        model_args=model_args, config=make_config(hint_path, **cfg), action_set=action_set
    )
    agent.set_task_name("miniwob.book-flight")
    return agent, agent.llm, action_set


# --------------------------------------------------------------------------------------
# Block.make
# --------------------------------------------------------------------------------------


def test_block_make_returns_independent_copy_and_calls_init():
    original = Obs(use_axtree=True, use_tabs=True)
    copy_ = original.make()
    assert copy_ is not original
    assert copy_ == original
    copy_.use_axtree = False
    assert original.use_axtree is True


def test_block_make_runs_init_on_copy_only(hint_csv):
    block = TaskHint(hint_db_rel_path=str(hint_csv))
    made = block.make()
    assert not hasattr(block, "hint_db")
    assert list(made.hint_db.columns) == ["task_name", "semantic_keys", "hint"]


def test_block_base_init_and_apply_are_noops():
    assert Goal()._init() is None
    with pytest.raises(TypeError):
        tua.Block()  # abstract


# --------------------------------------------------------------------------------------
# StructuredDiscussion
# --------------------------------------------------------------------------------------


def user_msg(text, builder=AnthropicAPIMessageBuilder):
    return builder.user().add_text(text)


def test_msg_group_defaults_are_not_shared():
    a, b = MsgGroup("a"), MsgGroup("b")
    a.messages.append("x")
    assert b.messages == []
    assert a.summary is None


def test_new_group_auto_names_by_index():
    d = StructuredDiscussion()
    d.new_group("goal")
    d.new_group()
    d.new_group()
    assert [g.name for g in d.groups] == ["goal", "group_1", "group_2"]


def test_append_without_group_raises():
    with pytest.raises(IndexError):
        StructuredDiscussion().append(user_msg("x"))


def test_goal_set_and_last_summary_on_empty_discussion():
    d = StructuredDiscussion()
    assert d.is_goal_set() is False
    assert d.get_last_summary() is None
    d.new_group()
    assert d.is_goal_set() is True
    assert d.get_last_summary() is None
    summary = user_msg("sum")
    d.set_last_summary(summary)
    assert d.get_last_summary() is summary


def test_flatten_without_limit_keeps_all_messages_in_order():
    d = StructuredDiscussion(keep_last_n_obs=None)
    msgs = []
    for g in range(3):
        d.new_group()
        for j in range(2):
            m = user_msg(f"{g}-{j}")
            msgs.append(m)
            d.append(m)
        d.set_last_summary(user_msg(f"summary {g}"))
    assert d.flatten() == msgs


def test_flatten_replaces_old_groups_by_summary_and_keeps_tail():
    d = StructuredDiscussion(keep_last_n_obs=1)
    d.new_group("goal")
    goal = user_msg("goal")
    d.append(goal)  # no summary -> always kept in full
    d.new_group()
    o1, s1 = user_msg("obs1"), user_msg("sum1")
    d.append(o1)
    d.set_last_summary(s1)
    d.new_group()
    o2, s2 = user_msg("obs2"), user_msg("sum2")
    d.append(o2)
    d.set_last_summary(s2)

    flat = d.flatten()
    assert flat == [goal, s1, o2]  # the tail group keeps raw messages, not its summary


def test_flatten_resets_previous_breakpoints_and_sets_a_new_one():
    d = StructuredDiscussion(keep_last_n_obs=1)
    msgs = []
    for g in range(3):
        d.new_group()
        m = user_msg(f"m{g}")
        m._cache_breakpoint = True  # stale breakpoint from a previous step
        d.append(m)
        msgs.append(m)
    flat = d.flatten()
    assert flat == msgs
    assert [m._cache_breakpoint for m in flat] == [False, False, True]


def test_flatten_with_keep_larger_than_groups_marks_nothing():
    d = StructuredDiscussion(keep_last_n_obs=5)
    d.new_group()
    m = user_msg("only")
    d.append(m)
    assert d.flatten() == [m]
    assert not hasattr(m, "_cache_breakpoint")


# --------------------------------------------------------------------------------------
# Goal block
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "builder, as_system, expected_role",
    [
        (OpenAIResponseAPIMessageBuilder, True, "developer"),
        (OpenAIResponseAPIMessageBuilder, False, "user"),
        (AnthropicAPIMessageBuilder, True, "system"),
        (AnthropicAPIMessageBuilder, False, "user"),
    ],
)
def test_goal_block_adds_system_and_goal_messages(builder, as_system, expected_role):
    llm = fake_llm(builder)
    d = StructuredDiscussion()
    d.new_group("goal")
    obs = make_obs(
        goal_object=[
            {"type": "text", "text": "Do X"},
            {"type": "image_url", "image_url": "data:image/png;base64,AAA"},
            {"type": "unknown", "foo": 1},
        ]
    )
    Goal(goal_as_system_msg=as_system).apply(llm, d, obs, sys_msg="SYSTEM")
    sys_msg, goal_msg = d.groups[0].messages
    assert sys_msg.role == ("developer" if builder is OpenAIResponseAPIMessageBuilder else "system")
    assert texts(sys_msg) == ["SYSTEM"]
    assert goal_msg.role == expected_role
    assert goal_msg.content == [
        {"text": "# Goal:\n"},
        {"text": "Do X"},
        {"image": "data:image/png;base64,AAA"},
    ]


def test_goal_block_default_sys_msg():
    llm = fake_llm()
    d = StructuredDiscussion()
    d.new_group()
    Goal().apply(llm, d, make_obs())
    assert texts(d.groups[0].messages[0]) == [SYS_MSG]


# --------------------------------------------------------------------------------------
# Obs block and _format_tabs
# --------------------------------------------------------------------------------------


def test_format_tabs_marks_active_tab():
    out = _format_tabs(make_obs())
    assert out.startswith("Currently open tabs:\n")
    assert "Tab 0:\n    Title: A\n    URL: http://a.test\n" in out
    assert "Tab 1 (active tab):\n    Title: B\n    URL: http://b.test\n" in out
    assert "Tab 0 (active tab)" not in out


def test_format_tabs_no_tabs():
    obs = make_obs(open_pages_urls=[], open_pages_titles=[])
    assert _format_tabs(obs) == "Currently open tabs:"


def test_obs_block_all_features_without_tool_calls():
    llm = fake_llm()
    d = StructuredDiscussion()
    d.new_group()
    block = Obs(use_last_error=True, use_screenshot=True, use_axtree=True, use_dom=True)
    block.use_tabs = True
    obs = make_obs(last_action_error="Boom")
    msg = block.apply(llm, d, obs, last_llm_output=LLMOutput())
    assert d.groups[0].messages == [msg]
    assert msg.role == "user"
    t = texts(msg)
    assert t[0] == "Last action error:\nBoom"
    assert t[1] == f"AXTree:\n{AXTREE_NOTE}\n[a1] button 'Go'"
    assert t[2] == "DOM:\n<button bid='a1'>Go</button>"
    assert t[3].startswith("Currently open tabs:")
    images = [c["image"] for c in msg.content if "image" in c]
    assert len(images) == 1 and images[0].startswith("data:image/png;base64,")


def test_obs_block_skips_empty_error_and_disabled_parts():
    llm = fake_llm()
    d = StructuredDiscussion()
    d.new_group()
    block = Obs(use_last_error=True, use_screenshot=False, use_axtree=False)
    msg = block.apply(llm, d, make_obs(last_action_error=""), last_llm_output=LLMOutput())
    assert msg.content == []


def test_obs_block_ignores_error_when_disabled():
    llm = fake_llm()
    d = StructuredDiscussion()
    d.new_group()
    block = Obs(use_last_error=False, use_screenshot=False)
    msg = block.apply(llm, d, make_obs(last_action_error="err"), last_llm_output=LLMOutput())
    assert msg.content == []


def test_obs_block_uses_som_screenshot(monkeypatch):
    seen = []
    monkeypatch.setattr(tua, "image_to_png_base64_url", lambda img: seen.append(img) or "IMG")
    llm = fake_llm()
    d = StructuredDiscussion()
    d.new_group()
    som = np.ones((2, 2, 3), dtype=np.uint8)
    msg = Obs(use_som=True).apply(llm, d, make_obs(screenshot_som=som), last_llm_output=LLMOutput())
    assert seen[0] is som
    assert msg.content == [{"image": "IMG"}]


def test_obs_block_responds_to_previous_tool_calls():
    llm = fake_llm()
    d = StructuredDiscussion()
    d.new_group()
    previous = llm_output(tool_names=("click", "type"))
    msg = Obs(use_screenshot=False).apply(llm, d, make_obs(), last_llm_output=previous)
    assert len(d.groups[0].messages) == 2
    obs_msg, tool_msg = d.groups[0].messages
    assert obs_msg is msg
    assert tool_msg.role == "tool"
    assert isinstance(tool_msg, OpenAIResponseAPIMessageBuilder)
    assert tool_msg.responded_tool_calls is previous.tool_calls
    assert [c.tool_response for c in previous.tool_calls] == [{"text": "See Observation"}] * 2


# --------------------------------------------------------------------------------------
# GeneralHints and Summarizer
# --------------------------------------------------------------------------------------


def test_general_hints_disabled_adds_nothing():
    d = StructuredDiscussion()
    d.new_group()
    assert GeneralHints(use_hints=False).apply(fake_llm(), d) is None
    assert d.groups[0].messages == []


def test_general_hints_enabled_adds_user_message():
    d = StructuredDiscussion()
    d.new_group()
    GeneralHints(use_hints=True).apply(fake_llm(), d)
    (msg,) = d.groups[0].messages
    assert msg.role == "user"
    assert "ControlOrMeta" in texts(msg)[0]


def test_summarizer_disabled_is_noop():
    llm = fake_llm()
    d = StructuredDiscussion()
    d.new_group()
    s = Summarizer(do_summary=False)
    assert s.apply(llm, d) is None
    assert s.apply_init(llm, d) is None
    assert d.groups[0].messages == []
    assert llm.payloads == []


@pytest.mark.parametrize(
    "high_details, expected",
    [(True, "# Summarizer instructions:"), (False, "give a semantic description")],
)
def test_summarizer_apply_init_instructions(high_details, expected):
    d = StructuredDiscussion()
    d.new_group()
    Summarizer(do_summary=True, high_details=high_details).apply_init(fake_llm(), d)
    (msg,) = d.groups[0].messages
    assert msg.role == "developer"
    assert expected in texts(msg)[0]


def test_summarizer_apply_calls_llm_and_sets_summary():
    llm = fake_llm(responses=[LLMOutput(think="the summary")])
    d = StructuredDiscussion()
    d.new_group()
    d.append(llm.msg.user().add_text("obs"))
    summary = Summarizer(do_summary=True).apply(llm, d)

    (call,) = llm.payloads
    assert [texts(m) for m in call.messages] == [["obs"], ["Summarize\n"]]
    assert call.payload.tools is None
    assert summary.role == "assistant"
    assert texts(summary) == ["the summary"]
    assert d.groups[-1].messages[-1] is summary
    assert d.get_last_summary() is summary


# --------------------------------------------------------------------------------------
# TaskHint
# --------------------------------------------------------------------------------------


def test_task_hint_relative_path_resolves_next_to_module():
    block = TaskHint().make()  # ships with hint_db.csv next to the module
    assert {"task_name", "semantic_keys", "hint"} <= set(block.hint_db.columns)
    assert len(block.hint_db) > 0


def test_task_hint_direct_uses_fnmatch_on_task_name(hint_csv):
    block = TaskHint(hint_db_rel_path=str(hint_csv)).make()
    assert block.choose_hints_direct("miniwob.book-flight") == [
        "select from the dropdown list",
        "activate the field first",
    ]
    # the runtime task name is used as the glob pattern
    assert block.choose_hints_direct("miniwob.*") == [
        "select from the dropdown list",
        "activate the field first",
        "select from the dropdown list",
    ]
    assert block.choose_hints_direct("nope") == []


def _goal_discussion(builder=OpenAIResponseAPIMessageBuilder, goal="Book flight"):
    llm = fake_llm(builder)
    d = StructuredDiscussion()
    d.new_group("goal")
    Goal().apply(llm, d, make_obs(goal_object=[{"type": "text", "text": goal}]))
    return llm, d


def test_task_hint_disabled_returns_empty(hint_csv):
    llm, d = _goal_discussion()
    block = TaskHint(use_task_hint=False, hint_db_rel_path=str(hint_csv)).make()
    assert block.apply(llm, d, "miniwob.book-flight") == {}
    assert len(d.groups[0].messages) == 2


def test_task_hint_apply_formats_and_strips_hints(hint_csv):
    llm, d = _goal_discussion()
    block = TaskHint(hint_db_rel_path=str(hint_csv)).make()
    block.apply(llm, d, "workarena.order")
    hint_msg = d.groups[0].messages[-1]
    assert hint_msg.role == "user"
    assert texts(hint_msg) == [
        "# Hints:\nHere are some hints for the task you are working on:\n- use the form"
    ]


def test_task_hint_apply_no_match_adds_nothing(hint_csv):
    llm, d = _goal_discussion()
    block = TaskHint(hint_db_rel_path=str(hint_csv)).make()
    block.apply(llm, d, "unknown.task")
    assert len(d.groups[0].messages) == 2


def test_task_hint_apply_skips_blank_hints(hint_csv):
    llm, d = _goal_discussion()
    block = TaskHint(hint_db_rel_path=str(hint_csv)).make()
    with mock.patch.object(block, "choose_hints", return_value=["   ", "real"]) as ch:
        block.apply(llm, d, "t")
    ch.assert_called_once_with(llm, "t", "# Goal:\n\nBook flight")
    assert texts(d.groups[0].messages[-1])[0].endswith(":\n- real")


def test_task_hint_choose_hints_dispatch(hint_csv):
    block = TaskHint(hint_db_rel_path=str(hint_csv)).make()
    llm = fake_llm()
    with (
        mock.patch.object(block, "choose_hints_llm", return_value=["L"]) as m_llm,
        mock.patch.object(block, "choose_hints_emb", return_value=["E"]) as m_emb,
    ):
        block.hint_retrieval_mode = "llm"
        assert block.choose_hints(llm, "t", "g") == ["L"]
        m_llm.assert_called_once_with(llm, "g")
        block.hint_retrieval_mode = "emb"
        assert block.choose_hints(llm, "t", "g") == ["E"]
        m_emb.assert_called_once_with("g")
    block.hint_retrieval_mode = "direct"
    assert block.choose_hints(llm, "workarena.order", "g") == ["  use the form  "]
    block.hint_retrieval_mode = "bogus"
    with pytest.raises(ValueError, match="Unknown hint retrieval mode: bogus"):
        block.choose_hints(llm, "t", "g")


def test_task_hint_llm_mode_selects_topic_and_dedupes(hint_csv):
    block = TaskHint(hint_db_rel_path=str(hint_csv), hint_retrieval_mode="llm").make()
    llm = fake_llm(responses=[LLMOutput(think="0")])
    hints = block.choose_hints_llm(llm, "Book a flight")
    # topic 0 == "dropdown" (first seen); its duplicate hint is collapsed
    assert hints == ["select from the dropdown list"]
    (call,) = llm.payloads
    prompt = texts(call.messages[0])[0]
    assert "Book a flight" in prompt
    assert "0. dropdown\n1. forms" in prompt


def test_task_hint_llm_mode_second_topic(hint_csv):
    block = TaskHint(hint_db_rel_path=str(hint_csv), hint_retrieval_mode="llm").make()
    llm = fake_llm(responses=[LLMOutput(think="1")])
    assert block.choose_hints_llm(llm, "g") == ["activate the field first", "  use the form  "]


@pytest.mark.parametrize("answer", ["-1", "2", "99", "not json"])
def test_task_hint_llm_mode_bad_answers_return_no_hints(hint_csv, answer):
    block = TaskHint(hint_db_rel_path=str(hint_csv), hint_retrieval_mode="llm").make()
    llm = fake_llm(responses=[LLMOutput(think=answer)])
    assert block.choose_hints_llm(llm, "g") == []


@pytest.mark.xfail(
    strict=True,
    reason="BUG: non-integer JSON answer (e.g. '\"1\"') raises TypeError instead of no hints",
)
def test_task_hint_llm_mode_quoted_number_returns_no_hints(hint_csv):
    block = TaskHint(hint_db_rel_path=str(hint_csv), hint_retrieval_mode="llm").make()
    llm = fake_llm(responses=[LLMOutput(think='"1"')])
    assert block.choose_hints_llm(llm, "g") == []


def _emb_setup(tmp_path, monkeypatch):
    rows = [
        ("t1", "k1", "hint one"),
        ("t2", "k2", "hint two"),
        ("t3", "k1", "hint one"),  # duplicate hint, dropped by encode_hints
        ("t4", "k3", "hint three"),
    ]
    csv = write_hint_db(tmp_path / "db.csv", rows)
    embs = {
        "k1: hint one": [1.0, 0.0],
        "k2: hint two": [0.0, 1.0],
        "k3: hint three": [0.5, 0.5],
    }
    np.save(f"{csv}.embs.npy", embs, allow_pickle=True)
    monkeypatch.chdir(tmp_path)
    return csv


def test_task_hint_emb_init_loads_embeddings(tmp_path, monkeypatch):
    csv = _emb_setup(tmp_path, monkeypatch)
    block = TaskHint(hint_db_rel_path=str(csv), hint_retrieval_mode="emb").make()
    assert block.uniq_hints["hint"].tolist() == ["hint one", "hint two", "hint three"]
    np.testing.assert_array_equal(block.hint_embeddings, [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])


def test_task_hint_emb_missing_embeddings_file_asserts(tmp_path, hint_csv):
    with pytest.raises(AssertionError, match="Embedding file not found"):
        TaskHint(hint_db_rel_path=str(hint_csv), hint_retrieval_mode="emb").make()


@pytest.mark.xfail(
    strict=True,
    reason="BUG: encode_hints resolves '<rel>.embs.npy' against CWD, not the CSV's directory",
)
def test_task_hint_emb_relative_path_resolves_next_to_module(tmp_path, monkeypatch):
    module_dir = tmp_path / "module"
    module_dir.mkdir()
    csv = write_hint_db(module_dir / "hint_db.csv", [("t", "k", "h")])
    np.save(f"{csv}.embs.npy", {"k: h": [1.0]}, allow_pickle=True)
    monkeypatch.setattr(tua, "__file__", str(module_dir / "tool_use_agent.py"))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    block = TaskHint(hint_db_rel_path="hint_db.csv", hint_retrieval_mode="emb").make()
    assert block.hint_embeddings.shape == (1, 1)


def _response(payload):
    return SimpleNamespace(json=lambda: payload)


def test_task_hint_choose_hints_emb_returns_top_n(tmp_path, monkeypatch):
    csv = _emb_setup(tmp_path, monkeypatch)
    block = TaskHint(hint_db_rel_path=str(csv), hint_retrieval_mode="emb", top_n=2).make()
    posts = []

    def fake_post(url, json, timeout):
        posts.append((url, json, timeout))
        if url.endswith("/encode"):
            return _response({"embeddings": [[0.1, 0.9]]})
        return _response({"similarities": [[0.2, 0.9, 0.5]]})

    monkeypatch.setattr(tua.requests, "post", fake_post)
    hints = block.choose_hints_emb("my goal")
    # top-2 by ascending argsort: index 2 (0.5) then index 1 (0.9)
    assert hints == ["hint three", "hint two"]
    assert posts[0] == (
        "http://localhost:5000/encode",
        {"texts": ["my goal"], "prompt": "task description"},
        10,
    )
    assert posts[1][0] == "http://localhost:5000/similarity"
    assert posts[1][1]["texts1"] == [[0.1, 0.9]]
    assert posts[1][1]["texts2"] == [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    assert posts[1][2] == 2


def test_task_hint_oai_embed_uses_client():
    block = TaskHint(use_task_hint=False)
    client = mock.Mock()
    client.create.return_value = SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2])])
    block._oai_emb = client
    assert block.oai_embed("hello") == [0.1, 0.2]
    client.create.assert_called_once_with(input="hello", model="text-embedding-3-small")


@pytest.mark.parametrize(
    "method, args, key, timeout",
    [
        ("_encode", (["a"],), "embeddings", 10),
        ("_similarity", ([[1.0]], [[1.0]]), "similarities", 2),
    ],
)
def test_task_hint_http_helpers_retry_then_succeed(monkeypatch, method, args, key, timeout):
    block = TaskHint(use_task_hint=False, embedder_server="http://emb.test")
    outcomes = [requests.exceptions.ConnectionError("down"), requests.exceptions.Timeout("slow")]

    def fake_post(url, json, timeout):
        if outcomes:
            raise outcomes.pop(0)
        return _response({key: [[0.5]]})

    sleeps, uniforms = [], []
    monkeypatch.setattr(tua.requests, "post", fake_post)
    monkeypatch.setattr(tua.time, "sleep", sleeps.append)
    monkeypatch.setattr(tua.random, "uniform", lambda a, b: uniforms.append((a, b)) or 0.25)

    out = getattr(block, method)(*args)
    np.testing.assert_array_equal(out, [[0.5]])
    assert sleeps == [0.25, 0.25]
    assert uniforms == [(1, timeout), (1, timeout)]


@pytest.mark.parametrize("method, args", [("_encode", (["a"],)), ("_similarity", ([1], [1]))])
def test_task_hint_http_helpers_raise_after_max_retries(monkeypatch, method, args):
    block = TaskHint(use_task_hint=False)
    calls = []

    def fake_post(url, json, timeout):
        calls.append(url)
        raise requests.exceptions.ConnectionError("down")

    sleeps = []
    monkeypatch.setattr(tua.requests, "post", fake_post)
    monkeypatch.setattr(tua.time, "sleep", sleeps.append)
    monkeypatch.setattr(tua.random, "uniform", lambda a, b: 0.0)
    with pytest.raises(requests.exceptions.ConnectionError):
        getattr(block, method)(*args, max_retries=3)
    assert len(calls) == 3
    assert len(sleeps) == 2  # no sleep after the final attempt


# --------------------------------------------------------------------------------------
# ToolUseAgentArgs
# --------------------------------------------------------------------------------------


def test_agent_args_name_from_model_name():
    args = ToolUseAgentArgs(model_args=FakeModelArgs(model_name="org/model-x"))
    assert args.agent_name == "ToolUse-org_model-x"


def test_agent_args_without_model_keeps_default_name():
    args = ToolUseAgentArgs()
    assert args.agent_name == ToolUseAgentArgs.__dataclass_fields__["agent_name"].default


def test_agent_args_prepare_and_close_delegate_to_model_args():
    model_args = FakeModelArgs()
    args = ToolUseAgentArgs(model_args=model_args)
    assert args.prepare() == "prepared"
    assert args.close() == "closed"
    assert (model_args.prepared, model_args.closed) == (1, 1)


def test_agent_args_make_agent_with_config(hint_csv):
    cfg = make_config(hint_csv)
    action_set = FakeActionSet()
    args = ToolUseAgentArgs(model_args=FakeModelArgs(), config=cfg, action_set=action_set)
    agent = args.make_agent()
    assert isinstance(agent, ToolUseAgent)
    assert agent.config is cfg
    assert agent.action_set is action_set
    assert action_set.apis == ["openai"]


def test_agent_args_make_agent_defaults_to_default_prompt_config():
    args = ToolUseAgentArgs(model_args=FakeModelArgs(), action_set=FakeActionSet())
    agent = args.make_agent()
    assert args.config is DEFAULT_PROMPT_CONFIG
    assert agent.config is DEFAULT_PROMPT_CONFIG
    assert agent.task_hint.use_task_hint is True
    assert len(agent.task_hint.hint_db) > 0  # bundled hint_db.csv
    assert agent.discussion.keep_last_n_obs is None


@pytest.mark.parametrize("name, expected", [("osworld", True), ("miniwob", False)])
def test_agent_args_set_benchmark(hint_csv, name, expected):
    args = ToolUseAgentArgs(model_args=FakeModelArgs(), config=make_config(hint_csv))
    args.set_benchmark(SimpleNamespace(name=name), demo_mode=False)
    assert args.config.obs.skip_preprocessing is expected


def test_predefined_configs_are_consistent():
    assert tua.OSWORLD_CLAUDE.config.task_hint.use_task_hint is False
    assert isinstance(tua.OSWORLD_OAI.action_set, OSWorldActionSet)
    assert tua.OSWORLD_OAI.config.keep_last_n_obs == 1
    assert tua.AGENT_CONFIG.agent_name == "ToolUse-claude-3-7-sonnet-20250219"
    assert tua.OAI_OPENROUTER_AGENT.agent_name == "ToolUse-openai_gpt-4.1"
    assert DEFAULT_PROMPT_CONFIG.action_subsets == ("coord",)


# --------------------------------------------------------------------------------------
# ToolUseAgent construction and obs_preprocessor
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "api, builder",
    [("openai", OpenAIResponseAPIMessageBuilder), ("anthropic", AnthropicAPIMessageBuilder)],
)
def test_agent_init_default_action_set_from_bgym(hint_csv, api, builder):
    model_args = FakeModelArgs(api=api, builder=builder)
    agent = ToolUseAgent(model_args=model_args, config=make_config(hint_csv))
    names = {t["name"] for t in agent.tools}
    assert "mouse_click" in names
    assert "noop" in names
    key = "input_schema" if api == "anthropic" else "parameters"
    assert all(key in t for t in agent.tools)
    assert agent.llm.msg is builder
    assert agent.msg_builder is builder
    assert agent.call_ids == []
    assert agent.last_response == LLMOutput()


def test_agent_init_with_osworld_action_set(hint_csv):
    agent = ToolUseAgent(
        model_args=FakeModelArgs(),
        config=make_config(hint_csv),
        action_set=OSWorldActionSet("computer_13"),
    )
    assert len(agent.tools) > 0
    assert all("name" in t.get("function", t) for t in agent.tools)


def test_agent_init_copies_blocks(hint_csv):
    cfg = make_config(hint_csv)
    agent = ToolUseAgent(model_args=FakeModelArgs(), config=cfg, action_set=FakeActionSet())
    assert agent.obs_block == cfg.obs and agent.obs_block is not cfg.obs
    assert agent.task_hint is not cfg.task_hint


def test_obs_preprocessor_skip_returns_shallow_copy(hint_csv):
    agent, _, _ = make_agent(hint_csv, [], obs=Obs(skip_preprocessing=True))
    obs = {"page": object(), "x": 1}
    out = agent.obs_preprocessor(obs)
    assert out == obs and out is not obs


def test_obs_preprocessor_page_extracts_screenshot(hint_csv, monkeypatch):
    agent, _, _ = make_agent(hint_csv, [], obs=Obs(use_axtree=True, use_dom=True))
    page = object()
    shot = np.zeros((1, 1, 3), dtype=np.uint8)
    extract = mock.Mock(return_value=shot)
    flatten = mock.Mock()
    monkeypatch.setattr(tua, "extract_screenshot", extract)
    monkeypatch.setattr(tua, "flatten_axtree_to_str", flatten)
    obs = {"page": page, "other": 1}
    out = agent.obs_preprocessor(obs)
    extract.assert_called_once_with(page)
    flatten.assert_not_called()  # page branch skips text extraction
    assert out == {"screenshot": shot, "other": 1}
    assert "page" in obs  # input not mutated


def test_obs_preprocessor_builds_text_and_som(hint_csv, monkeypatch):
    agent, _, _ = make_agent(
        hint_csv,
        [],
        obs=Obs(use_axtree=True, use_dom=True, use_som=True, use_zoomed_webpage=True),
    )
    calls = {}
    monkeypatch.setattr(
        tua,
        "flatten_dom_to_str",
        lambda dom, extra_properties: calls.setdefault("dom", (dom, extra_properties)) and "DOM",
    )
    monkeypatch.setattr(tua, "prune_html", lambda s: f"pruned({s})")
    monkeypatch.setattr(
        tua,
        "flatten_axtree_to_str",
        lambda ax, extra_properties: calls.setdefault("ax", (ax, extra_properties)) and "AX",
    )
    monkeypatch.setattr(
        tua,
        "overlay_som",
        lambda shot, extra_properties: calls.setdefault("som", (shot, extra_properties)) and "SOM",
    )
    extra = {"a1": {"visibility": 1}}
    obs = {
        "dom_object": "dom",
        "axtree_object": "ax",
        "screenshot": "shot",
        "extra_element_properties": extra,
    }
    out = agent.obs_preprocessor(obs)
    assert out["dom_txt"] == "DOM"
    assert out["pruned_html"] == "pruned(DOM)"
    assert out["axtree_txt"] == "AX"
    assert out["screenshot_som"] == "SOM"
    assert calls == {"dom": ("dom", extra), "ax": ("ax", extra), "som": ("shot", extra)}
    assert "axtree_txt" not in obs


def test_obs_preprocessor_with_real_browsergym_helpers(hint_csv):
    agent, _, _ = make_agent(hint_csv, [], obs=Obs(use_axtree=True))
    axtree = {
        "nodes": [
            {
                "nodeId": "1",
                "role": {"value": "RootWebArea"},
                "name": {"value": "Page"},
                "childIds": ["2"],
                "browsergym_id": "r1",
            },
            {
                "nodeId": "2",
                "role": {"value": "button"},
                "name": {"value": "Go"},
                "childIds": [],
                "browsergym_id": "a1",
            },
        ]
    }
    out = agent.obs_preprocessor({"axtree_object": axtree, "extra_element_properties": {}})
    assert "button 'Go'" in out["axtree_txt"]
    assert "[a1]" in out["axtree_txt"]


# --------------------------------------------------------------------------------------
# ToolUseAgent.get_action loop
# --------------------------------------------------------------------------------------


def test_get_action_first_step_builds_goal_and_obs(hint_csv):
    response = llm_output(action="click('a1')", think="I will click")
    agent, llm, action_set = make_agent(hint_csv, [response])
    action, info = agent.get_action(make_obs())

    assert action == "click('a1')"
    assert info.think == "I will click"
    assert llm.reset_count == 1
    assert agent.last_response is response
    assert agent._responses == [response]

    (call,) = llm.payloads
    p = call.payload
    assert p.tools == action_set.to_tool_description()
    assert p.tool_choice == "any"
    assert p.cache_tool_definition is True
    assert p.cache_complete_prompt is False
    assert p.use_cache_breakpoints is True
    roles = [m.role for m in call.messages]
    assert roles == ["developer", "developer", "user"]
    assert texts(call.messages[0])[0].endswith("You can only take one action at a time.")
    assert texts(call.messages[1]) == ["# Goal:\n", "Book a flight to Paris"]

    # chat_messages for xray: tool description first, tool-call message last
    chat = info.chat_messages
    assert chat[0].role == "tool_description"
    assert json.loads(texts(chat[0])[0]) == p.tools
    assert chat[-1].role == "tool"
    assert chat[-1].responded_tool_calls is response.tool_calls
    assert chat[1:-1] == call.messages

    # stats from model + cost tracker merged
    assert info.stats["n_api_calls"] == 1
    assert {"input_tokens", "output_tokens", "cost"} <= set(info.stats)

    # groups: goal, step-0 obs, new empty group for the next step
    assert [g.name for g in agent.discussion.groups] == ["goal", "group_1", "group_2"]
    assert agent.discussion.groups[-1].messages == []


def test_get_action_multiaction_system_prompt(hint_csv):
    agent, llm, _ = make_agent(hint_csv, [llm_output()], multiaction=True)
    agent.get_action(make_obs())
    first = texts(llm.payloads[0].messages[0])[0]
    assert first.endswith("You can take multiple actions in a single step, if needed.")


def test_get_action_second_step_reuses_goal_and_answers_tool_calls(hint_csv):
    r1 = llm_output(action="a1", think="t1", tool_names=("click",))
    r2 = llm_output(action="a2", think="t2", tool_names=("type",))
    agent, llm, _ = make_agent(hint_csv, [r1, r2])
    agent.get_action(make_obs())
    action, info = agent.get_action(make_obs(last_action_error="oops"))

    assert action == "a2"
    assert info.think == "t2"
    assert llm.reset_count == 2
    second = llm.payloads[1].messages
    assert [m.role for m in second] == ["developer", "developer", "user", "user", "tool"]
    assert texts(second[3])[0] == "Last action error:\noops"
    assert second[4].responded_tool_calls is r1.tool_calls
    assert r1.tool_calls.tool_calls[0].tool_response == {"text": "See Observation"}
    assert agent._responses == [r1, r2]
    # goal not re-added
    assert sum(1 for g in agent.discussion.groups if g.name == "goal") == 1


def test_get_action_with_summary_hints_and_task_hint(hint_csv):
    responses = [LLMOutput(think="SUMMARY-1"), llm_output(action="act", think="reason")]
    agent, llm, _ = make_agent(
        hint_csv,
        responses,
        builder=AnthropicAPIMessageBuilder,
        summarizer=Summarizer(do_summary=True, high_details=False),
        general_hints=GeneralHints(use_hints=True),
        task_hint=TaskHint(use_task_hint=True, hint_db_rel_path=str(hint_csv)),
    )
    action, info = agent.get_action(make_obs())

    assert action == "act"
    assert info.think == "SUMMARY-1\nreason"
    summary_call, action_call = llm.payloads
    assert summary_call.payload.tools is None
    assert texts(summary_call.messages[-1]) == ["Summarize\n"]

    msgs = action_call.messages
    assert [m.role for m in msgs] == [
        "system",  # sys msg
        "system",  # goal
        "system",  # summarizer instructions
        "user",  # general hints
        "user",  # task hints
        "user",  # obs
        "user",  # "Summarize"
        "assistant",  # summary
    ]
    assert "- select from the dropdown list" in texts(msgs[4])[0]
    assert "- activate the field first" in texts(msgs[4])[0]
    assert texts(msgs[-1]) == ["SUMMARY-1"]
    assert agent.discussion.groups[1].summary is msgs[-1]


def test_get_action_keep_last_n_obs_summarizes_old_steps(hint_csv):
    responses = [
        LLMOutput(think="S1"),
        llm_output(action="a1", think="t1"),
        LLMOutput(think="S2"),
        llm_output(action="a2", think="t2"),
    ]
    agent, llm, _ = make_agent(
        hint_csv,
        responses,
        builder=AnthropicAPIMessageBuilder,
        summarizer=Summarizer(do_summary=True),
        keep_last_n_obs=1,
    )
    agent.get_action(make_obs(axtree_txt="FIRST"))
    _, info = agent.get_action(make_obs(axtree_txt="SECOND"))
    assert info.think == "S2\nt2"

    final_msgs = llm.payloads[-1].messages
    all_text = "\n".join(t for m in final_msgs for t in texts(m))
    assert "SECOND" in all_text
    assert "FIRST" not in all_text  # step 1 replaced by its summary
    assert "S1" in all_text
    assert sum(getattr(m, "_cache_breakpoint", False) for m in final_msgs) == 1


def test_get_action_nested_cost_tracker_accumulates(hint_csv):
    from agentlab.llm.tracking import set_tracker

    agent, _, _ = make_agent(hint_csv, [llm_output()])
    with set_tracker() as outer:
        _, info = agent.get_action(make_obs())
    assert info.stats["cost"] == 0
    assert outer.cost == 0
    assert getattr(TRACKER, "instance", None) is None
