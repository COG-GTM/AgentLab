import base64
import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from agentlab.llm import llm_utils
from agentlab.llm.llm_utils import (
    AIMessage,
    BaseMessage,
    Discussion,
    HumanMessage,
    ParseError,
    SystemMessage,
    extract_html_tags,
    get_tokenizer,
    image_to_jpg_base64_url,
    image_to_png_base64_url,
    messages_to_dict,
    parse_html_tags,
    parse_html_tags_raise,
    retry_multiple,
    supports_tool_calling_for_openrouter,
)


def _decode(url, prefix):
    assert url.startswith(prefix)
    return Image.open(io.BytesIO(base64.b64decode(url[len(prefix) :])))


def test_messages_to_dict():
    out = messages_to_dict([{"role": "user", "content": "hi"}, "raw"])
    assert isinstance(out, Discussion)
    assert out[0] == {"role": "user", "content": "hi"}
    assert out[1] == {"role": "<unknown role>", "content": "raw"}
    assert all(isinstance(m, BaseMessage) for m in out)


def test_messages_to_dict_empty_and_invalid():
    assert len(messages_to_dict([])) == 0
    with pytest.raises(ValueError, match="Unknown message type"):
        messages_to_dict([42])


def test_extract_html_tags():
    text = "<a> one </a> noise <b>two</b><a>\nthree\n</a>"
    assert extract_html_tags(text, ["a", "b", "c"]) == {"a": ["one", "three"], "b": ["two"]}
    assert extract_html_tags("", ["a"]) == {}


def test_parse_html_tags_valid_and_optional():
    content, valid, msg = parse_html_tags(
        "<think>t</think><action>a</action>", ["action"], ["think", "memo"]
    )
    assert valid and msg == ""
    assert content == {"think": "t", "action": "a"}


def test_parse_html_tags_missing_and_duplicate():
    content, valid, msg = parse_html_tags(
        "<action>a</action><action>b</action>", ["action", "think"]
    )
    assert not valid
    assert "Missing the key <think>" in msg
    assert "multiple instances of the key action" in msg
    assert content["action"] == "a"


def test_parse_html_tags_merge_multiple():
    content, valid, _ = parse_html_tags("<a>1</a><a>2</a>", ["a"], merge_multiple=True)
    assert valid and content == {"a": "1\n2"}


def test_parse_html_tags_raise():
    assert parse_html_tags_raise("<a>1</a>", ["a"]) == {"a": "1"}
    with pytest.raises(ParseError, match="Missing the key <b>"):
        parse_html_tags_raise("<a>1</a>", ["a", "b"])


def _parser(text):
    if "good" not in text:
        raise ParseError(f"bad: {text}")
    return text.upper()


def test_retry_multiple_returns_parsed_answers():
    chat = MagicMock(return_value=[AIMessage("bad"), AIMessage("good 1"), AIMessage("good 2")])
    messages = Discussion([HumanMessage("q")])
    parsed, tries = retry_multiple(chat, messages, n_retry=2, parser=_parser, num_samples=3)
    assert parsed == ["GOOD 1", "GOOD 2"] and tries == 0
    chat.assert_called_once_with(messages, n_samples=3)
    assert len(messages) == 2  # only the first answer is appended


def test_retry_multiple_single_answer_retry_then_success():
    chat = MagicMock(side_effect=[AIMessage("bad"), AIMessage("good")])
    messages = Discussion([HumanMessage("q")])
    parsed, tries = retry_multiple(chat, messages, n_retry=3, parser=_parser)
    assert parsed == ["GOOD"] and tries == 1
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert "bad: bad" in messages[2]["content"]


def test_retry_multiple_exhausted():
    chat = MagicMock(return_value=AIMessage("bad"))
    with pytest.raises(ParseError, match="after 2 retries"):
        retry_multiple(chat, Discussion([HumanMessage("q")]), n_retry=2, parser=_parser, log=False)
    assert chat.call_count == 2


def test_image_to_base64_urls_from_rgba_array():
    arr = np.zeros((4, 6, 4), dtype=np.uint8)
    arr[..., 0] = 255
    arr[..., 3] = 128
    jpg = _decode(image_to_jpg_base64_url(arr), "data:image/jpeg;base64,")
    png = _decode(image_to_png_base64_url(arr), "data:image/png;base64,")
    assert jpg.size == png.size == (6, 4)
    assert png.mode == "RGB"
    assert png.getpixel((0, 0)) == (255, 0, 0)


def test_image_to_png_from_pil_grayscale():
    img = Image.new("L", (3, 3), 7)
    assert _decode(image_to_png_base64_url(img), "data:image/png;base64,").getpixel((1, 1)) == 7


def test_base_message_str_and_content_helpers():
    msg = HumanMessage("hello")
    assert str(msg) == "hello"
    assert msg.to_markdown() == "\n```\nhello\n```\n"
    msg.add_text("world")
    msg.add_image("data:image/png;base64,xyz", detail="low")
    assert msg["content"][0] == {"type": "text", "text": "hello"}
    assert msg["content"][2] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,xyz", "detail": "low"},
    }
    assert str(msg) == "hello\nworld"
    md = msg.to_markdown()
    assert "![image](data:image/png;base64,xyz)" in md and "```\nworld\n```" in md


def test_base_message_add_image_array_and_warning():
    msg = SystemMessage([{"type": "text", "text": "a"}])
    msg.add_image(np.zeros((2, 2, 3), dtype=np.uint8))
    assert msg["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert msg.__str__(warn_if_image=True) == "a"
    assert msg["role"] == "system"


def test_ai_message_log_probs_and_kwargs():
    assert AIMessage("x", log_probs=[1])["log_probs"] == [1]
    with pytest.raises(ValueError, match="Invalid attributes"):
        BaseMessage("tool", "x", tool_call_id="1")


def test_discussion_api():
    d = Discussion(SystemMessage("sys"))
    assert len(d) == 1
    d.add_message(role="user", content="q")
    d.append({"role": "assistant", "content": "a"})
    d.add_text("more")
    d.add_content("text", "even more")
    d.add_image("data:image/png;base64,abc")
    assert d.last_message["role"] == "assistant"
    assert [m["role"] for m in d] == ["system", "user", "assistant"]
    assert isinstance(d[2], BaseMessage)

    assert d.to_string() == "sys\nq\na\nmore\neven more"
    assert d[2]["content"][0] == {"type": "text", "text": "a\nmore\neven more"}
    assert d.to_openai() is d.messages
    md = d.to_markdown()
    assert md.startswith("Message 0\n") and "Message 2" in md


def test_discussion_empty():
    d = Discussion()
    assert len(d) == 0 and str(d) == ""
    with pytest.raises(IndexError):
        d.last_message


def test_get_tokenizer_falls_back_to_gpt4():
    with patch.object(llm_utils, "AutoTokenizer") as auto:
        auto.from_pretrained.side_effect = OSError("not found")
        enc = get_tokenizer("definitely/not-a-model")
    assert enc.name == llm_utils.tiktoken.encoding_for_model("gpt-4").name


def test_get_tokenizer_uses_huggingface():
    with patch.object(llm_utils, "AutoTokenizer") as auto:
        auto.from_pretrained.return_value = "hf-tokenizer"
        assert get_tokenizer("org/model") == "hf-tokenizer"


@pytest.mark.parametrize(
    "message, expected", [({"tool_calls": [{"id": "1"}]}, True), ({"content": "no"}, False)]
)
def test_supports_tool_calling_for_openrouter(message, expected):
    client = MagicMock()
    client.chat.completions.create.return_value.to_dict.return_value = {
        "choices": [{"message": message}]
    }
    with patch("openai.Client", return_value=client):
        assert supports_tool_calling_for_openrouter("org/model") is expected
    assert client.chat.completions.create.call_args.kwargs["tool_choice"] == "required"


def test_supports_tool_calling_for_openrouter_error_defaults_true():
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("network down")
    with patch("openai.Client", return_value=client):
        assert supports_tool_calling_for_openrouter("org/model") is True
