from unittest.mock import MagicMock

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
    image_to_jpg_base64_url,
    image_to_png_base64_url,
    messages_to_dict,
    parse_html_tags,
    parse_html_tags_raise,
    retry_multiple,
)


def test_messages_to_dict_from_dicts():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    discussion = messages_to_dict(messages)
    assert isinstance(discussion, Discussion)
    assert len(discussion) == 3
    assert discussion[0]["role"] == "system"


def test_messages_to_dict_from_strings():
    discussion = messages_to_dict(["raw text"])
    assert discussion[0]["role"] == "<unknown role>"


def test_messages_to_dict_unknown_type():
    with pytest.raises(ValueError):
        messages_to_dict([42])


def test_extract_html_tags():
    text = "<think>abc</think><action>click</action><think>def</think>"
    content = extract_html_tags(text, ["think", "action"])
    assert content["think"] == ["abc", "def"]
    assert content["action"] == ["click"]


def test_extract_html_tags_missing_key():
    content = extract_html_tags("no tags here", ["think"])
    assert content == {}


def test_parse_html_tags_valid():
    text = "<think>abc</think><action>click</action>"
    content, valid, retry_message = parse_html_tags(text, keys=("think", "action"))
    assert valid
    assert content["think"] == "abc"
    assert content["action"] == "click"
    assert retry_message == ""


def test_parse_html_tags_missing_key():
    content, valid, retry_message = parse_html_tags("<think>abc</think>", keys=("think", "action"))
    assert not valid
    assert "action" in retry_message


def test_parse_html_tags_optional_key():
    content, valid, _ = parse_html_tags(
        "<think>abc</think>", keys=("think",), optional_keys=("action",)
    )
    assert valid
    assert "action" not in content


def test_parse_html_tags_multiple_without_merge():
    _, valid, retry_message = parse_html_tags("<a>1</a><a>2</a>", keys=("a",))
    assert not valid
    assert "multiple instances" in retry_message


def test_parse_html_tags_merge_multiple():
    content, valid, _ = parse_html_tags("<a>1</a><a>2</a>", keys=("a",), merge_multiple=True)
    assert valid
    assert content["a"] == "1\n2"


def test_parse_html_tags_raise():
    with pytest.raises(ParseError):
        parse_html_tags_raise("<think>abc</think>", keys=("think", "action"))
    content = parse_html_tags_raise("<think>abc</think>", keys=("think",))
    assert content["think"] == "abc"


def _parser(answer):
    return parse_html_tags_raise(answer, keys=("a",))


def test_retry_multiple_success_first_try():
    chat = MagicMock(return_value={"role": "assistant", "content": "<a>ok</a>"})

    parsed_answers, tries = retry_multiple(chat, [], n_retry=2, parser=_parser, log=False)
    assert parsed_answers[0]["a"] == "ok"
    assert tries == 0
    assert chat.call_count == 1


def test_retry_multiple_retries_then_succeeds():
    chat = MagicMock(
        side_effect=[
            {"role": "assistant", "content": "bad"},
            {"role": "assistant", "content": "<a>ok</a>"},
        ]
    )

    parsed_answers, tries = retry_multiple(chat, [], n_retry=2, parser=_parser, log=False)
    assert parsed_answers[0]["a"] == "ok"
    assert tries == 1
    assert chat.call_count == 2


def test_retry_multiple_exhausts_retries():
    chat = MagicMock(return_value={"role": "assistant", "content": "bad"})

    with pytest.raises(ParseError):
        retry_multiple(chat, [], n_retry=1, parser=_parser, log=False)


def test_image_to_jpg_base64_url_from_numpy():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    url = image_to_jpg_base64_url(image)
    assert url.startswith("data:image/jpeg;base64,")


def test_image_to_png_base64_url_from_pil():
    image = Image.new("RGB", (4, 4))
    url = image_to_png_base64_url(image)
    assert url.startswith("data:image/png;base64,")


def test_image_base64_url_rgba_converted():
    image = Image.new("RGBA", (4, 4))
    url = image_to_jpg_base64_url(image)
    assert url.startswith("data:image/jpeg;base64,")


def test_base_message_str_and_add():
    msg = BaseMessage("user", "hello")
    assert msg["role"] == "user"
    assert "hello" in str(msg)
    msg.add_text("world")
    assert len(msg["content"]) == 2
    msg.add_image(Image.new("RGB", (2, 2)))
    types = [c["type"] for c in msg["content"]]
    assert "image_url" in types


def test_message_subclasses_roles():
    assert SystemMessage("s")["role"] == "system"
    assert HumanMessage("h")["role"] == "user"
    ai = AIMessage("a")
    assert ai["role"] == "assistant"


def test_discussion_basics():
    discussion = Discussion(SystemMessage("sys"))
    discussion.append(HumanMessage("question"))
    assert len(discussion) == 2
    assert discussion[0]["role"] == "system"
    assert discussion.last_message["role"] == "user"
    assert "question" in discussion.to_string()
    openai_format = discussion.to_openai()
    assert isinstance(openai_format, list) and len(openai_format) == 2


def test_discussion_add_content_and_merge():
    discussion = Discussion([SystemMessage("a")])
    discussion.add_text("b")
    discussion.merge()
    assert "a" in str(discussion)
    assert "b" in str(discussion)


def test_generic_call_api_with_retries_success():
    response = MagicMock()
    client = MagicMock(return_value=response)
    result = llm_utils.generic_call_api_with_retries(
        client,
        {"messages": []},
        is_response_valid_fn=lambda r: True,
        rate_limit_exceptions=(),
        api_error_exceptions=(),
    )
    assert result is response
    assert client.call_count == 1


def test_generic_call_api_with_retries_invalid_then_valid():
    good = MagicMock()
    bad = MagicMock()
    client = MagicMock(side_effect=[bad, good])
    result = llm_utils.generic_call_api_with_retries(
        client,
        {"messages": []},
        is_response_valid_fn=lambda r: r is good,
        rate_limit_exceptions=(),
        api_error_exceptions=(),
        max_retries=3,
        initial_retry_delay_seconds=0,
        max_retry_delay_seconds=0,
    )
    assert result is good
    assert client.call_count == 2
