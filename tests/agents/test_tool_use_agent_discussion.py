from agentlab.agents.tool_use_agent.tool_use_agent import StructuredDiscussion
from agentlab.llm.response_api import AnthropicAPIMessageBuilder


def make_msg(text: str) -> AnthropicAPIMessageBuilder:
    msg = AnthropicAPIMessageBuilder("user").add_text(text)
    msg._cache_breakpoint = False
    return msg


def make_discussion(n_groups: int, n_msg_per_group: int = 2, with_summary: bool = True, **kwargs):
    discussion = StructuredDiscussion(**kwargs)
    for i in range(n_groups):
        discussion.new_group(f"group_{i}")
        for j in range(n_msg_per_group):
            discussion.append(make_msg(f"msg_{i}_{j}"))
        if with_summary:
            discussion.set_last_summary(make_msg(f"summary_{i}"))
    return discussion


def texts(messages):
    return [msg.content[0]["text"] for msg in messages]


def breakpoints(messages):
    return [getattr(msg, "_cache_breakpoint", False) for msg in messages]


def test_new_group_and_append():
    discussion = StructuredDiscussion()
    assert not discussion.is_goal_set()
    assert discussion.get_last_summary() is None

    discussion.new_group()
    discussion.append(make_msg("a"))
    discussion.append(make_msg("b"))

    assert discussion.is_goal_set()
    assert discussion.groups[0].name == "group_0"
    assert texts(discussion.groups[0].messages) == ["a", "b"]
    assert discussion.get_last_summary() is None

    summary = make_msg("summary")
    discussion.set_last_summary(summary)
    assert discussion.get_last_summary() is summary


def test_flatten_empty_discussion():
    assert StructuredDiscussion().flatten() == []


def test_flatten_without_keep_last_n_obs_keeps_everything():
    discussion = make_discussion(3)

    messages = discussion.flatten()

    assert texts(messages) == [f"msg_{i}_{j}" for i in range(3) for j in range(2)]
    assert breakpoints(messages) == [True, False, False, False, False, False]


def test_flatten_replaces_non_tail_groups_by_their_summary():
    discussion = make_discussion(4, keep_last_n_obs=2)

    messages = discussion.flatten()

    assert texts(messages) == [
        "summary_0",
        "summary_1",
        "msg_2_0",
        "msg_2_1",
        "msg_3_0",
        "msg_3_1",
    ]
    # the tail starts at group index 2, and the breakpoint is set on the message at that index
    assert breakpoints(messages) == [False, False, True, False, False, False]


def test_flatten_keeps_only_last_group_verbatim():
    discussion = make_discussion(3, keep_last_n_obs=1)

    messages = discussion.flatten()

    assert texts(messages) == ["summary_0", "summary_1", "msg_2_0", "msg_2_1"]
    assert breakpoints(messages) == [False, False, True, False]


def test_flatten_falls_back_on_messages_when_group_has_no_summary():
    discussion = make_discussion(3, keep_last_n_obs=1, with_summary=False)
    discussion.groups[1].summary = make_msg("summary_1")

    messages = discussion.flatten()

    assert texts(messages) == ["msg_0_0", "msg_0_1", "summary_1", "msg_2_0", "msg_2_1"]
    assert breakpoints(messages) == [False, False, True, False, False]


def test_flatten_unsets_previous_cache_breakpoints():
    discussion = make_discussion(3, keep_last_n_obs=1)
    for group in discussion.groups:
        group.summary._cache_breakpoint = True
        for msg in group.messages:
            msg._cache_breakpoint = True

    messages = discussion.flatten()

    assert breakpoints(messages) == [False, False, True, False]


def test_flatten_keep_last_n_obs_larger_than_number_of_groups():
    discussion = make_discussion(2, keep_last_n_obs=5)

    messages = discussion.flatten()

    assert texts(messages) == ["msg_0_0", "msg_0_1", "msg_1_0", "msg_1_1"]
    assert breakpoints(messages) == [False, False, False, False]


def test_flatten_moves_breakpoint_when_a_group_is_added():
    discussion = make_discussion(3, keep_last_n_obs=2)
    assert breakpoints(discussion.flatten()) == [False, True, False, False, False]

    discussion.new_group()
    discussion.append(make_msg("msg_3_0"))

    messages = discussion.flatten()
    assert texts(messages) == ["summary_0", "summary_1", "msg_2_0", "msg_2_1", "msg_3_0"]
    assert breakpoints(messages) == [False, False, True, False, False]
