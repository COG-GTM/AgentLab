"""Unit tests for the Discussion class and related HTML tag parsing utilities in llm_utils."""

import pytest

from agentlab.llm.llm_utils import (
    AIMessage,
    BaseMessage,
    Discussion,
    HumanMessage,
    ParseError,
    SystemMessage,
    extract_html_tags,
    parse_html_tags,
    parse_html_tags_raise,
)


# ---------------------------------------------------------------------------
# Discussion – construction
# ---------------------------------------------------------------------------


class TestDiscussionInit:
    def test_empty_init(self):
        d = Discussion()
        assert len(d) == 0
        assert list(d) == []

    def test_init_with_single_message(self):
        msg = SystemMessage("Hello")
        d = Discussion(msg)
        assert len(d) == 1
        assert d[0]["content"] == "Hello"

    def test_init_with_list_of_messages(self):
        msgs = [SystemMessage("A"), HumanMessage("B"), AIMessage("C")]
        d = Discussion(msgs)
        assert len(d) == 3
        assert d[0]["role"] == "system"
        assert d[1]["role"] == "user"
        assert d[2]["role"] == "assistant"


# ---------------------------------------------------------------------------
# Discussion – add_message / append
# ---------------------------------------------------------------------------


class TestDiscussionAddMessage:
    def test_add_message_from_base_message(self):
        d = Discussion()
        d.add_message(SystemMessage("sys"))
        assert len(d) == 1
        assert d[0]["role"] == "system"

    def test_add_message_from_dict(self):
        d = Discussion()
        d.add_message({"role": "user", "content": "hi"})
        assert len(d) == 1
        assert d[0]["content"] == "hi"

    def test_add_message_from_role_content(self):
        d = Discussion()
        d.add_message(role="assistant", content="reply")
        assert len(d) == 1
        assert d[0]["role"] == "assistant"
        assert d[0]["content"] == "reply"

    def test_append_delegates_to_add_message(self):
        d = Discussion()
        d.append(HumanMessage("test"))
        assert len(d) == 1
        assert d[0]["content"] == "test"


# ---------------------------------------------------------------------------
# Discussion – properties & dunder methods
# ---------------------------------------------------------------------------


class TestDiscussionProperties:
    def test_last_message(self):
        d = Discussion([SystemMessage("first"), HumanMessage("second")])
        assert d.last_message["content"] == "second"

    def test_len(self):
        d = Discussion([SystemMessage("a"), HumanMessage("b")])
        assert len(d) == 2

    def test_getitem(self):
        d = Discussion([SystemMessage("a"), HumanMessage("b")])
        assert d[0]["content"] == "a"
        assert d[1]["content"] == "b"

    def test_iter(self):
        msgs = [SystemMessage("a"), HumanMessage("b")]
        d = Discussion(msgs)
        contents = [m["content"] for m in d]
        assert contents == ["a", "b"]

    def test_str(self):
        d = Discussion([SystemMessage("hello"), HumanMessage("world")])
        s = str(d)
        assert "hello" in s
        assert "world" in s


# ---------------------------------------------------------------------------
# Discussion – merge / to_string / to_openai / to_markdown
# ---------------------------------------------------------------------------


class TestDiscussionConversions:
    def test_to_string_merges(self):
        msg = BaseMessage(
            role="user",
            content=[
                {"type": "text", "text": "part1"},
                {"type": "text", "text": "part2"},
            ],
        )
        d = Discussion([msg])
        result = d.to_string()
        assert "part1" in result
        assert "part2" in result

    def test_to_openai_returns_messages(self):
        d = Discussion([SystemMessage("sys"), HumanMessage("usr")])
        msgs = d.to_openai()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"

    def test_to_markdown(self):
        d = Discussion([SystemMessage("hello")])
        md = d.to_markdown()
        assert "Message 0" in md
        assert "hello" in md


# ---------------------------------------------------------------------------
# Discussion – add_text / add_content on last message
# ---------------------------------------------------------------------------


class TestDiscussionContentHelpers:
    def test_add_text_to_last_message(self):
        d = Discussion([HumanMessage("first")])
        d.add_text("extra")
        # After add_text the content should be a list with two text entries
        assert isinstance(d.last_message["content"], list)
        texts = [
            e["text"] for e in d.last_message["content"] if e["type"] == "text"
        ]
        assert "first" in texts
        assert "extra" in texts

    def test_add_content_to_last_message(self):
        d = Discussion([HumanMessage("msg")])
        d.add_content("text", "added")
        assert isinstance(d.last_message["content"], list)
        assert len(d.last_message["content"]) == 2


# ---------------------------------------------------------------------------
# BaseMessage helpers
# ---------------------------------------------------------------------------


class TestBaseMessage:
    def test_system_message_role(self):
        m = SystemMessage("hi")
        assert m["role"] == "system"

    def test_human_message_role(self):
        m = HumanMessage("hi")
        assert m["role"] == "user"

    def test_ai_message_role(self):
        m = AIMessage("hi")
        assert m["role"] == "assistant"

    def test_invalid_kwarg_raises(self):
        with pytest.raises(ValueError, match="Invalid attributes"):
            BaseMessage(role="user", content="x", bad_kwarg=True)

    def test_str_with_string_content(self):
        m = BaseMessage(role="user", content="hello")
        assert str(m) == "hello"

    def test_str_with_list_content(self):
        m = BaseMessage(
            role="user",
            content=[
                {"type": "text", "text": "a"},
                {"type": "text", "text": "b"},
            ],
        )
        assert str(m) == "a\nb"

    def test_to_markdown_string_content(self):
        m = BaseMessage(role="user", content="code")
        md = m.to_markdown()
        assert "```" in md
        assert "code" in md


# ---------------------------------------------------------------------------
# extract_html_tags
# ---------------------------------------------------------------------------


class TestExtractHtmlTags:
    def test_single_key(self):
        text = "<action>click(5)</action>"
        result = extract_html_tags(text, ["action"])
        assert result == {"action": ["click(5)"]}

    def test_multiple_matches(self):
        text = "<a>1</a> <a>2</a>"
        result = extract_html_tags(text, ["a"])
        assert result == {"a": ["1", "2"]}

    def test_missing_key_returns_empty(self):
        text = "<a>1</a>"
        result = extract_html_tags(text, ["b"])
        assert result == {}

    def test_multiline_content(self):
        text = "<thought>\nline1\nline2\n</thought>"
        result = extract_html_tags(text, ["thought"])
        assert len(result["thought"]) == 1
        assert "line1" in result["thought"][0]
        assert "line2" in result["thought"][0]


# ---------------------------------------------------------------------------
# parse_html_tags
# ---------------------------------------------------------------------------


class TestParseHtmlTags:
    def test_all_keys_present(self):
        text = "<action>click</action><thought>reason</thought>"
        content, valid, msg = parse_html_tags(
            text, keys=("action", "thought")
        )
        assert valid is True
        assert content["action"] == "click"
        assert content["thought"] == "reason"

    def test_missing_required_key(self):
        text = "<action>click</action>"
        content, valid, msg = parse_html_tags(
            text, keys=("action", "thought")
        )
        assert valid is False
        assert "thought" in msg

    def test_optional_key_missing_still_valid(self):
        text = "<action>click</action>"
        content, valid, msg = parse_html_tags(
            text, keys=("action",), optional_keys=("thought",)
        )
        assert valid is True

    def test_duplicate_key_without_merge(self):
        text = "<a>1</a><a>2</a>"
        content, valid, msg = parse_html_tags(text, keys=("a",))
        assert valid is False
        assert "multiple" in msg.lower()

    def test_duplicate_key_with_merge(self):
        text = "<a>1</a><a>2</a>"
        content, valid, msg = parse_html_tags(
            text, keys=("a",), merge_multiple=True
        )
        assert valid is True
        assert content["a"] == "1\n2"


# ---------------------------------------------------------------------------
# parse_html_tags_raise
# ---------------------------------------------------------------------------


class TestParseHtmlTagsRaise:
    def test_raises_on_missing_key(self):
        with pytest.raises(ParseError):
            parse_html_tags_raise("<a>1</a>", keys=("a", "b"))

    def test_returns_dict_on_success(self):
        result = parse_html_tags_raise("<a>1</a>", keys=("a",))
        assert result == {"a": "1"}
