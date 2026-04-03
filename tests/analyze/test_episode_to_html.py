from unittest.mock import MagicMock, PropertyMock, patch

from agentlab.analyze.episode_to_html import (
    _escape_html,
    _format_chat_messages_like_xray,
    _format_goal,
    _format_tool_call_like_xray,
)


def test_escape_html_basic():
    """Test that HTML special characters are escaped."""
    assert _escape_html("<div>") == "&lt;div&gt;"
    assert _escape_html('"hello"') == "&quot;hello&quot;"
    assert _escape_html("a & b") == "a &amp; b"
    assert _escape_html("it's") == "it&#x27;s"


def test_escape_html_non_string():
    """Test that non-string inputs are converted to string first."""
    assert isinstance(_escape_html(123), str)
    assert _escape_html(None) == "None"


def test_format_goal_none():
    """Test that None goal returns appropriate HTML."""
    result = _format_goal(None)
    assert "No goal specified" in result


def test_format_goal_text():
    """Test formatting a text goal."""
    goal = [{"type": "text", "text": "Click the button"}]
    result = _format_goal(goal)
    assert "Click the button" in result


def test_format_chat_messages_empty():
    """Test formatting empty chat messages."""
    result = _format_chat_messages_like_xray([])
    assert "No chat messages" in result


def test_format_chat_messages_dict():
    """Test formatting dict-style chat messages."""
    messages = [
        {"role": "system", "content": "You are an assistant"},
        {"role": "user", "content": "Hello"},
    ]
    result = _format_chat_messages_like_xray(messages)
    assert "SYSTEM" in result
    assert "USER" in result
    assert "You are an assistant" in result
    assert "Hello" in result


def test_format_chat_messages_multipart():
    """Test formatting dict messages with list content."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this"},
                {"type": "image", "image_url": "data:image/png;base64,..."},
            ],
        },
    ]
    result = _format_chat_messages_like_xray(messages)
    assert "Look at this" in result
    assert "[IMAGE]" in result


def test_format_tool_call():
    """Test formatting a tool call."""
    tool_item = {
        "name": "click",
        "input": {"bid": "a123"},
        "call_id": "call_001",
    }
    result = _format_tool_call_like_xray(tool_item)
    assert "click" in result
    assert "a123" in result
    assert "call_001" in result
