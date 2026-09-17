from agentlab.agents.tool_use_agent.tool_use_agent import StructuredDiscussion
from agentlab.llm.response_api import AnthropicAPIMessageBuilder


def make_msg(text: str) -> AnthropicAPIMessageBuilder:
    msg = AnthropicAPIMessageBuilder.user().add_text(text)
    msg._cache_breakpoint = False
    return msg


def make_discussion(
    group_specs: list[tuple[int, bool]], keep_last_n_obs: int | None = None
) -> StructuredDiscussion:
    """Build a discussion with one group per spec of (n_messages, has_summary)."""
    discussion = StructuredDiscussion(keep_last_n_obs=keep_last_n_obs)
    for group_idx, (n_messages, has_summary) in enumerate(group_specs):
        discussion.new_group(f"group_{group_idx}")
        for msg_idx in range(n_messages):
            discussion.append(make_msg(f"g{group_idx}_m{msg_idx}"))
        if has_summary:
            discussion.set_last_summary(make_msg(f"g{group_idx}_summary"))
    return discussion


def texts(messages) -> list[str]:
    return [msg.content[0]["text"] for msg in messages]


def test_flatten_empty_discussion():
    assert StructuredDiscussion().flatten() == []


def test_flatten_without_keep_last_n_obs_keeps_everything_verbatim():
    discussion = make_discussion([(2, True), (1, True), (1, False)])

    assert texts(discussion.flatten()) == ["g0_m0", "g0_m1", "g1_m0", "g2_m0"]


def test_flatten_summarizes_all_but_the_tail():
    discussion = make_discussion([(2, True), (2, True), (2, True)], keep_last_n_obs=1)

    assert texts(discussion.flatten()) == ["g0_summary", "g1_summary", "g2_m0", "g2_m1"]


def test_flatten_keeps_the_last_n_groups_verbatim():
    discussion = make_discussion([(1, True), (1, True), (1, True), (1, True)], keep_last_n_obs=2)

    assert texts(discussion.flatten()) == ["g0_summary", "g1_summary", "g2_m0", "g3_m0"]


def test_flatten_falls_back_to_messages_when_group_has_no_summary():
    discussion = make_discussion([(2, False), (1, True), (1, True)], keep_last_n_obs=1)

    assert texts(discussion.flatten()) == ["g0_m0", "g0_m1", "g1_summary", "g2_m0"]


def test_flatten_keeps_everything_when_keep_last_n_obs_exceeds_group_count():
    discussion = make_discussion([(1, True), (1, True)], keep_last_n_obs=5)
    messages = discussion.flatten()

    assert texts(messages) == ["g0_m0", "g1_m0"]
    assert [msg._cache_breakpoint for msg in messages] == [False, False]


def test_flatten_sets_a_single_cache_breakpoint_at_the_tail_boundary():
    discussion = make_discussion([(1, True), (1, True), (2, True)], keep_last_n_obs=1)
    # pre-set a stale breakpoint on an earlier message to check it gets cleared
    discussion.groups[0].summary._cache_breakpoint = True

    messages = discussion.flatten()

    assert texts(messages) == ["g0_summary", "g1_summary", "g2_m0", "g2_m1"]
    # the breakpoint sits on the first message of the tail group, earlier ones are cleared
    assert [msg._cache_breakpoint for msg in messages] == [False, False, True, False]


def test_flatten_marks_the_first_message_when_the_whole_discussion_is_the_tail():
    discussion = make_discussion([(1, True), (1, True)], keep_last_n_obs=2)
    messages = discussion.flatten()

    assert [msg._cache_breakpoint for msg in messages] == [True, False]


def test_flatten_is_stable_across_calls():
    discussion = make_discussion([(1, True), (1, True), (1, False)], keep_last_n_obs=1)

    assert texts(discussion.flatten()) == texts(discussion.flatten())
    assert [msg._cache_breakpoint for msg in discussion.flatten()] == [False, False, True]
