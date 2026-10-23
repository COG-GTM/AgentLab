"""Unit tests for llm_utils helpers that don't require network access."""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic
import httpx
import numpy as np
import openai
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
)


@pytest.fixture
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm_utils.time, "sleep", lambda s: sleeps.append(s))
    return sleeps


class _RateLimit(Exception):
    pass


class _ApiError(Exception):
    def __init__(self, status_code=None):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def _call(client_function, **kwargs):
    defaults = dict(
        api_params={"x": 1},
        is_response_valid_fn=lambda r: r == "ok",
        rate_limit_exceptions=(_RateLimit,),
        api_error_exceptions=(_ApiError,),
        get_status_code_fn=lambda e: e.status_code,
        max_retries=3,
        initial_retry_delay_seconds=1,
        max_retry_delay_seconds=3,
    )
    defaults.update(kwargs)
    return llm_utils.generic_call_api_with_retries(client_function, **defaults)


# ---------------------------------------------------------------------------
# generic_call_api_with_retries
# ---------------------------------------------------------------------------


def test_generic_call_success_first_try(no_sleep):
    fn = MagicMock(return_value="ok")
    assert _call(fn) == "ok"
    fn.assert_called_once_with(x=1)
    assert no_sleep == []


def test_generic_call_retries_invalid_response_then_succeeds(no_sleep):
    fn = MagicMock(side_effect=["bad", "bad", "ok"])
    assert _call(fn) == "ok"
    assert fn.call_count == 3
    # first invalid retry uses exponent 0, second one exponent 1
    assert no_sleep == [1, 2]


def test_generic_call_invalid_response_exhausts_retries(no_sleep):
    fn = MagicMock(return_value="bad")
    with pytest.raises(RuntimeError, match="after 3 retries"):
        _call(fn)
    assert fn.call_count == 3


def test_generic_call_rate_limit_backoff_is_capped(no_sleep):
    fn = MagicMock(side_effect=[_RateLimit(), _RateLimit(), "ok"])
    assert _call(fn) == "ok"
    assert no_sleep == [2, 3]  # 1*2**1, min(1*2**2, 3)


def test_generic_call_rate_limit_exhausts_retries(no_sleep):
    fn = MagicMock(side_effect=_RateLimit())
    with pytest.raises(RuntimeError):
        _call(fn)
    assert fn.call_count == 3


@pytest.mark.parametrize("status", [429, 500, 503])
def test_generic_call_retriable_api_error(no_sleep, status):
    fn = MagicMock(side_effect=[_ApiError(status), "ok"])
    assert _call(fn) == "ok"
    assert len(no_sleep) == 1


def test_generic_call_retriable_api_error_exhausts_retries(no_sleep):
    fn = MagicMock(side_effect=_ApiError(500))
    with pytest.raises(RuntimeError):
        _call(fn)
    assert fn.call_count == 3


@pytest.mark.parametrize("status", [400, None])
def test_generic_call_non_retriable_api_error_is_raised(no_sleep, status):
    err = _ApiError(status)
    fn = MagicMock(side_effect=err)
    with pytest.raises(_ApiError) as exc_info:
        _call(fn)
    assert exc_info.value is err
    fn.assert_called_once()


def test_generic_call_status_code_fn_failure_is_non_retriable(no_sleep):
    def broken_status(e):
        raise RuntimeError("nope")

    fn = MagicMock(side_effect=_ApiError(500))
    with pytest.raises(_ApiError):
        _call(fn, get_status_code_fn=broken_status)


def test_generic_call_unexpected_exception_is_raised(no_sleep):
    fn = MagicMock(side_effect=KeyError("boom"))
    with pytest.raises(KeyError):
        _call(fn)
    fn.assert_called_once()


# ---------------------------------------------------------------------------
# OpenAI / Anthropic wrappers
# ---------------------------------------------------------------------------


def _openai_rate_limit_error():
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return openai.RateLimitError(
        "rate limited", response=httpx.Response(429, request=request), body=None
    )


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(error=None, choices=[1]),
        SimpleNamespace(error=None, output=[1]),
    ],
)
def test_call_openai_api_with_retries_valid(no_sleep, response):
    fn = MagicMock(return_value=response)
    assert llm_utils.call_openai_api_with_retries(fn, {"model": "m"}) is response


def test_call_openai_api_with_retries_retries_on_error_and_rate_limit(no_sleep):
    good = SimpleNamespace(error=None, choices=[1])
    fn = MagicMock(
        side_effect=[
            SimpleNamespace(error="oops"),
            SimpleNamespace(error=None, choices=[]),
            _openai_rate_limit_error(),
            good,
        ]
    )
    assert llm_utils.call_openai_api_with_retries(fn, {}, max_retries=4) is good
    assert fn.call_count == 4
    assert len(no_sleep) == 3


def test_call_openai_api_with_retries_non_retriable_api_error(no_sleep):
    request = httpx.Request("POST", "https://api.openai.com")
    err = openai.APIError("bad request", request=request, body=None)
    fn = MagicMock(side_effect=err)
    with pytest.raises(openai.APIError):
        llm_utils.call_openai_api_with_retries(fn, {})


@pytest.mark.parametrize(
    "response, valid",
    [
        (SimpleNamespace(type="message", content=[]), True),
        (SimpleNamespace(type="message", content=None), False),
        (SimpleNamespace(type="other"), False),
        ({"type": "error", "error": "x"}, False),
        (None, False),
    ],
)
def test_call_anthropic_api_with_retries_validation(no_sleep, response, valid):
    fn = MagicMock(return_value=response)
    if valid:
        assert llm_utils.call_anthropic_api_with_retries(fn, {}, max_retries=2) is response
    else:
        with pytest.raises(RuntimeError):
            llm_utils.call_anthropic_api_with_retries(fn, {}, max_retries=2)
        assert fn.call_count == 2


def test_call_anthropic_api_with_retries_server_error_then_success(no_sleep):
    request = httpx.Request("POST", "https://api.anthropic.com")
    err = anthropic.InternalServerError(
        "server", response=httpx.Response(500, request=request), body=None
    )
    good = SimpleNamespace(type="message", content=["hi"])
    fn = MagicMock(side_effect=[err, good])
    assert llm_utils.call_anthropic_api_with_retries(fn, {}) is good
    assert len(no_sleep) == 1


def test_supports_tool_calling_for_openrouter(monkeypatch):
    client = MagicMock()
    client.chat.completions.create.return_value.to_dict.return_value = {
        "choices": [{"message": {"tool_calls": [{}]}}]
    }
    monkeypatch.setattr(openai, "Client", MagicMock(return_value=client))
    assert llm_utils.supports_tool_calling_for_openrouter("some/model") is True

    client.chat.completions.create.return_value.to_dict.return_value = {
        "choices": [{"message": {"content": "no tools"}}]
    }
    assert llm_utils.supports_tool_calling_for_openrouter("some/model") is False

    client.chat.completions.create.side_effect = RuntimeError("network")
    # errors default to True so that the check never blocks a run
    assert llm_utils.supports_tool_calling_for_openrouter("some/model") is True


# ---------------------------------------------------------------------------
# retry_multiple
# ---------------------------------------------------------------------------


def _parser(content):
    if "good" not in content:
        raise ParseError("not good")
    return {"action": content}


def test_retry_multiple_returns_valid_samples():
    chat = MagicMock(
        return_value=[
            {"role": "assistant", "content": "bad"},
            {"role": "assistant", "content": "good 1"},
            {"role": "assistant", "content": "good 2"},
        ]
    )
    messages = Discussion()
    parsed, tries = llm_utils.retry_multiple(chat, messages, 3, _parser, num_samples=3)
    assert parsed == [{"action": "good 1"}, {"action": "good 2"}]
    assert tries == 0
    chat.assert_called_once_with(messages, n_samples=3)
    assert len(messages) == 1


def test_retry_multiple_wraps_single_answer_and_retries():
    chat = MagicMock(
        side_effect=[
            {"role": "assistant", "content": "bad"},
            {"role": "assistant", "content": "good"},
        ]
    )
    messages = Discussion()
    parsed, tries = llm_utils.retry_multiple(chat, messages, 3, _parser)
    assert parsed == [{"action": "good"}]
    assert tries == 1
    # assistant answer, user error feedback, assistant answer
    assert [m["role"] for m in messages] == ["assistant", "user", "assistant"]
    assert "not good" in messages[1]["content"]


def test_retry_multiple_raises_after_n_retry():
    chat = MagicMock(return_value={"role": "assistant", "content": "bad"})
    with pytest.raises(ParseError):
        llm_utils.retry_multiple(chat, Discussion(), 2, _parser)
    assert chat.call_count == 2


# ---------------------------------------------------------------------------
# messages / parsing helpers
# ---------------------------------------------------------------------------


def test_messages_to_dict_handles_dicts_and_strings():
    d = llm_utils.messages_to_dict([{"role": "user", "content": "hi"}, "raw text"])
    assert isinstance(d, Discussion)
    assert d[0]["role"] == "user"
    assert d[1] == {"role": "<unknown role>", "content": "raw text"}


def test_messages_to_dict_rejects_unknown_types():
    with pytest.raises(ValueError, match="Unknown message type"):
        llm_utils.messages_to_dict([42])


def test_messages_to_dict_langchain_fallback(monkeypatch):
    class FakeLC:
        def __init__(self, type, content):
            self.type = type
            self.content = content

    monkeypatch.setattr(llm_utils, "LangchainBaseMessage", FakeLC)
    monkeypatch.setattr(llm_utils, "convert_message_to_dict", None)
    d = llm_utils.messages_to_dict([FakeLC("human", "q"), FakeLC("ai", "a"), FakeLC("x", 1)])
    assert [m["role"] for m in d] == ["user", "assistant", "x"]
    assert d[2]["content"] == "1"


def test_messages_to_dict_langchain_converter(monkeypatch):
    class FakeLC:
        pass

    monkeypatch.setattr(llm_utils, "LangchainBaseMessage", FakeLC)
    monkeypatch.setattr(
        llm_utils, "convert_message_to_dict", lambda m: {"role": "system", "content": "c"}
    )
    d = llm_utils.messages_to_dict([FakeLC()])
    assert d[0] == {"role": "system", "content": "c"}


def test_parse_html_tags_optional_and_multiple():
    text = "<a>1</a><a>2</a><b> x </b>"
    content, valid, msg = llm_utils.parse_html_tags(text, keys=("a",), optional_keys=("b", "c"))
    assert not valid
    assert "multiple instances" in msg
    assert content == {"a": "1", "b": "x"}

    content, valid, msg = llm_utils.parse_html_tags(
        text, keys=("a",), optional_keys=("c",), merge_multiple=True
    )
    assert valid and msg == ""
    assert content["a"] == "1\n2"


def test_parse_html_tags_raise():
    assert llm_utils.parse_html_tags_raise("<k>v</k>", keys=("k",)) == {"k": "v"}
    with pytest.raises(ParseError, match="Missing the key <k>"):
        llm_utils.parse_html_tags_raise("nothing", keys=("k",))


def test_yaml_parser_invalid():
    with pytest.warns(UserWarning):
        value, valid, retry_message = llm_utils.yaml_parser("a: [unclosed")
    assert value == {} and not valid
    assert "not a valid yaml" in retry_message


def test_truncate_tokens_short_text_untouched():
    assert llm_utils.truncate_tokens("hello world", max_tokens=100) == "hello world"


def test_get_tokenizer_falls_back_to_gpt4(monkeypatch):
    llm_utils.get_tokenizer.cache_clear()

    def fail(*args, **kwargs):
        raise OSError("no hub")

    monkeypatch.setattr(llm_utils.AutoTokenizer, "from_pretrained", fail)
    try:
        enc = llm_utils.get_tokenizer("definitely-not-a-model")
        assert enc.encode("hello") == llm_utils.get_tokenizer("gpt-4").encode("hello")
    finally:
        llm_utils.get_tokenizer.cache_clear()


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_image_to_base64_urls(mode):
    img = Image.new(mode, (4, 4))
    assert llm_utils.image_to_jpg_base64_url(img).startswith("data:image/jpeg;base64,")
    assert llm_utils.image_to_png_base64_url(img).startswith("data:image/png;base64,")


def test_image_to_base64_urls_from_numpy():
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    assert llm_utils.image_to_jpg_base64_url(arr).startswith("data:image/jpeg;base64,")
    assert llm_utils.image_to_png_base64_url(arr).startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# BaseMessage / Discussion
# ---------------------------------------------------------------------------


def test_base_message_rejects_unknown_kwargs():
    with pytest.raises(ValueError, match="Invalid attributes"):
        BaseMessage("user", "x", foo=1)


def test_message_subclasses_roles():
    assert SystemMessage("s")["role"] == "system"
    assert HumanMessage("h")["role"] == "user"
    ai = AIMessage("a", log_probs=[0.1])
    assert ai["role"] == "assistant" and ai["log_probs"] == [0.1]


def test_base_message_content_is_copied():
    content = [{"type": "text", "text": "a"}]
    msg = HumanMessage(content)
    content.append({"type": "text", "text": "b"})
    assert len(msg["content"]) == 1


def test_base_message_str_with_images(caplog):
    msg = HumanMessage("hello")
    msg.add_image("data:image/png;base64,abc")
    msg["content"].append({"type": "input_text", "text": "world"})
    with caplog.at_level(logging.INFO):
        assert str(msg) == "hello\nworld"
    assert "images" in caplog.text
    assert msg.__str__(warn_if_image=True) == "hello\nworld"


def test_add_image_from_pil_with_detail():
    msg = HumanMessage("look")
    msg.add_image(Image.new("RGB", (2, 2)), detail="low")
    image_elem = msg["content"][-1]
    assert image_elem["type"] == "image_url"
    assert image_elem["image_url"]["detail"] == "low"
    assert image_elem["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_base_message_to_markdown():
    assert HumanMessage("x").to_markdown() == "\n```\nx\n```\n"
    msg = HumanMessage("text")
    msg.add_image("http://img")
    msg.add_content("image_url", "http://img2")
    md = msg.to_markdown()
    assert "```\ntext\n```" in md
    assert "![image](http://img)" in md
    assert "![image](http://img2)" in md


def test_base_message_merge_on_string_is_noop():
    msg = HumanMessage("x")
    msg.merge()
    assert msg["content"] == "x"


def test_discussion_api():
    d = Discussion(SystemMessage("sys"))
    d.add_message(role="user", content="hi")
    d.add_text("there")
    d.add_content("text", "again")
    d.add_image("http://img")
    assert len(d) == 2
    assert d.last_message is d[-1]
    assert [m["role"] for m in d] == ["system", "user"]

    md = d.to_markdown()
    assert md.startswith("Message 0")
    assert "Message 1" in md
    assert "![image](http://img)" in md

    openai_msgs = d.to_openai()
    assert openai_msgs[1]["content"][0] == {"type": "text", "text": "hi\nthere\nagain"}
    assert "sys" in d.to_string()


def test_discussion_default_empty():
    d = Discussion()
    assert len(d) == 0
    d.append({"role": "user", "content": "x"})
    assert isinstance(d[0], BaseMessage)
