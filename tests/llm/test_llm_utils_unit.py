import base64
import io
import logging
from types import SimpleNamespace
from unittest.mock import Mock

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

# ---------------------------------------------------------------------------
# helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def no_sleep(monkeypatch):
    sleep = Mock()
    monkeypatch.setattr(llm_utils.time, "sleep", sleep)
    return sleep


def _request():
    return httpx.Request("POST", "https://api.example.test/v1")


def _response(status):
    return httpx.Response(status, request=_request())


def _openai_status_error(cls, status):
    return cls(f"status {status}", response=_response(status), body=None)


def _anthropic_status_error(cls, status):
    return cls(f"status {status}", response=_response(status), body=None)


class _FakeEncoding:
    """Whitespace tokenizer standing in for a tiktoken encoding."""

    def __init__(self, name="fake"):
        self.name = name

    def encode(self, text):
        return text.split()

    def decode(self, tokens):
        return " ".join(tokens)


@pytest.fixture
def fake_tiktoken(monkeypatch):
    calls = []

    def encoding_for_model(model_name):
        calls.append(model_name)
        if model_name.startswith("unknown"):
            raise KeyError(model_name)
        return _FakeEncoding(model_name)

    monkeypatch.setattr(llm_utils.tiktoken, "encoding_for_model", encoding_for_model)
    llm_utils.get_tokenizer.cache_clear()
    llm_utils.get_tokenizer_old.cache_clear()
    yield calls
    llm_utils.get_tokenizer.cache_clear()
    llm_utils.get_tokenizer_old.cache_clear()


def _decode_data_url(url, prefix):
    assert url.startswith(prefix)
    return Image.open(io.BytesIO(base64.b64decode(url[len(prefix) :])))


# ---------------------------------------------------------------------------
# messages_to_dict
# ---------------------------------------------------------------------------


class _FakeLangchainMessage:
    def __init__(self, type, content):
        self.type = type
        self.content = content


def test_messages_to_dict_dicts_and_strings():
    out = llm_utils.messages_to_dict([{"role": "user", "content": "hi"}, "raw text"])
    assert isinstance(out, Discussion)
    assert list(out) == [
        {"role": "user", "content": "hi"},
        {"role": "<unknown role>", "content": "raw text"},
    ]
    assert all(isinstance(m, BaseMessage) for m in out)


def test_messages_to_dict_unknown_type_raises(monkeypatch):
    monkeypatch.setattr(llm_utils, "LangchainBaseMessage", None)
    with pytest.raises(ValueError, match="Unknown message type"):
        llm_utils.messages_to_dict([42])


def test_messages_to_dict_langchain_uses_convert_message_to_dict(monkeypatch):
    convert = Mock(side_effect=lambda m: {"role": "converted-" + m.type, "content": m.content})
    monkeypatch.setattr(llm_utils, "LangchainBaseMessage", _FakeLangchainMessage)
    monkeypatch.setattr(llm_utils, "convert_message_to_dict", convert)
    msg = _FakeLangchainMessage("human", "hello")
    out = llm_utils.messages_to_dict([msg])
    convert.assert_called_once_with(msg)
    assert list(out) == [{"role": "converted-human", "content": "hello"}]


@pytest.mark.parametrize(
    "lc_type, expected_role",
    [("human", "user"), ("ai", "assistant"), ("system", "system"), ("tool", "tool")],
)
def test_messages_to_dict_langchain_role_map_fallback(monkeypatch, lc_type, expected_role):
    monkeypatch.setattr(llm_utils, "LangchainBaseMessage", _FakeLangchainMessage)
    monkeypatch.setattr(llm_utils, "convert_message_to_dict", None)
    out = llm_utils.messages_to_dict([_FakeLangchainMessage(lc_type, 123)])
    assert list(out) == [{"role": expected_role, "content": "123"}]


def test_messages_to_dict_non_langchain_object_with_langchain_available(monkeypatch):
    monkeypatch.setattr(llm_utils, "LangchainBaseMessage", _FakeLangchainMessage)
    with pytest.raises(ValueError, match="Unknown message type"):
        llm_utils.messages_to_dict([object()])


# ---------------------------------------------------------------------------
# retry / retry_multiple
# ---------------------------------------------------------------------------


class _RecordingChat:
    """Fake chat model returning scripted answers and recording what it receives."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.received = []
        self.kwargs = []

    def __call__(self, messages, **kwargs):
        self.received.append([dict(m) for m in messages])
        self.kwargs.append(kwargs)
        return self.answers.pop(0)


def _int_parser(text):
    try:
        return int(text)
    except ValueError:
        raise ParseError(f"'{text}' is not an int")


def test_retry_appends_parse_error_as_user_message_and_reprompts():
    chat = _RecordingChat([AIMessage("abc"), AIMessage("42")])
    messages = Discussion([HumanMessage("give me an int")])

    assert llm_utils.retry(chat, messages, n_retry=3, parser=_int_parser) == 42

    assert chat.received[0] == [{"role": "user", "content": "give me an int"}]
    assert chat.received[1] == [
        {"role": "user", "content": "give me an int"},
        {"role": "assistant", "content": "abc", "log_probs": None},
        {"role": "user", "content": "'abc' is not an int"},
    ]
    # final answer is also appended to the discussion
    assert messages.last_message["content"] == "42"
    assert len(messages) == 4


def test_retry_exhausted_raises_parse_error_and_logs(caplog):
    chat = _RecordingChat([AIMessage("a"), AIMessage("b")])
    messages = Discussion([HumanMessage("q")])
    with caplog.at_level(logging.INFO):
        with pytest.raises(ParseError, match="after 2 retries"):
            llm_utils.retry(chat, messages, n_retry=2, parser=_int_parser)
    assert len(chat.received) == 2
    assert "Retrying 1/2" in caplog.text and "Retrying 2/2" in caplog.text
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant", "user"]


def test_retry_log_false_does_not_log(caplog):
    chat = _RecordingChat([AIMessage("a")])
    with caplog.at_level(logging.INFO):
        with pytest.raises(ParseError):
            llm_utils.retry(chat, Discussion(), n_retry=1, parser=_int_parser, log=False)
    assert "Query failed" not in caplog.text


def test_retry_zero_retries_never_calls_chat():
    chat = _RecordingChat([])
    with pytest.raises(ParseError):
        llm_utils.retry(chat, Discussion(), n_retry=0, parser=_int_parser)
    assert chat.received == []


def test_retry_multiple_returns_all_valid_samples_and_try_count():
    chat = _RecordingChat([[AIMessage("x"), AIMessage("1"), AIMessage("2")]])
    messages = Discussion([HumanMessage("q")])
    parsed, tries = llm_utils.retry_multiple(
        chat, messages, n_retry=3, parser=_int_parser, num_samples=3
    )
    assert parsed == [1, 2]
    assert tries == 0
    assert chat.kwargs == [{"n_samples": 3}]
    # a single assistant turn is recorded; which sample is kept is a known TODO in src
    assert [m["role"] for m in messages] == ["user", "assistant"]


def test_retry_multiple_wraps_single_answer_and_retries_with_all_errors(caplog):
    chat = _RecordingChat([[AIMessage("a"), AIMessage("b")], AIMessage("7")])
    messages = Discussion([HumanMessage("q")])
    with caplog.at_level(logging.INFO):
        parsed, tries = llm_utils.retry_multiple(
            chat, messages, n_retry=3, parser=_int_parser, num_samples=2
        )
    assert parsed == [7]
    assert tries == 1
    assert chat.received[1][-1] == {
        "role": "user",
        "content": str(["'a' is not an int", "'b' is not an int"]),
    }
    assert "Retrying 1/3" in caplog.text


def test_retry_multiple_exhausted_raises():
    chat = _RecordingChat([AIMessage("a"), AIMessage("b")])
    with pytest.raises(ParseError, match="after 2 retries"):
        llm_utils.retry_multiple(chat, Discussion(), n_retry=2, parser=_int_parser, log=False)
    assert len(chat.received) == 2


# ---------------------------------------------------------------------------
# generic_call_api_with_retries
# ---------------------------------------------------------------------------


class _RateLimit(Exception):
    pass


class _ApiError(Exception):
    def __init__(self, status_code=None):
        super().__init__(f"api error {status_code}")
        self.status_code = status_code


def _generic(client, valid=lambda r: r == "ok", get_status=None, **kw):
    return llm_utils.generic_call_api_with_retries(
        client_function=client,
        api_params={"x": 1},
        is_response_valid_fn=valid,
        rate_limit_exceptions=(_RateLimit,),
        api_error_exceptions=(_ApiError,),
        get_status_code_fn=get_status,
        **kw,
    )


def test_generic_success_first_try_passes_params(no_sleep):
    client = Mock(return_value="ok")
    assert _generic(client) == "ok"
    client.assert_called_once_with(x=1)
    no_sleep.assert_not_called()


def test_generic_invalid_response_backoff_then_success(no_sleep):
    client = Mock(side_effect=["bad", "bad", "ok"])
    assert _generic(client, initial_retry_delay_seconds=1, max_retry_delay_seconds=100) == "ok"
    # invalid-content retries use 2**(attempt-1)
    assert [c.args[0] for c in no_sleep.call_args_list] == [1, 2]


def test_generic_invalid_response_until_max_retries_raises(no_sleep):
    client = Mock(return_value="bad")
    with pytest.raises(RuntimeError, match="after 3 retries"):
        _generic(client, max_retries=3, initial_retry_delay_seconds=1)
    assert client.call_count == 3
    assert no_sleep.call_count == 2  # no sleep after the final attempt


def test_generic_rate_limit_backoff_is_exponential_and_capped(no_sleep):
    client = Mock(side_effect=[_RateLimit()] * 4 + ["ok"])
    assert _generic(client, initial_retry_delay_seconds=10, max_retry_delay_seconds=100) == "ok"
    # exception retries use 2**attempt, capped at max delay
    assert [c.args[0] for c in no_sleep.call_args_list] == [20, 40, 80, 100]


def test_generic_rate_limit_exhausted_raises_runtime_error(no_sleep):
    client = Mock(side_effect=_RateLimit())
    with pytest.raises(RuntimeError, match="after 2 retries"):
        _generic(client, max_retries=2)
    assert client.call_count == 2
    assert no_sleep.call_count == 1


@pytest.mark.parametrize("status", [429, 500, 502, 503, 529])
def test_generic_retriable_status_codes(no_sleep, status):
    client = Mock(side_effect=[_ApiError(status), "ok"])
    assert _generic(client, get_status=lambda e: e.status_code) == "ok"
    assert client.call_count == 2
    assert no_sleep.call_count == 1


@pytest.mark.parametrize("status", [None, 0, 400, 401, 403, 404, 422])
def test_generic_non_retriable_status_codes_reraise(no_sleep, status):
    err = _ApiError(status)
    client = Mock(side_effect=[err, "ok"])
    with pytest.raises(_ApiError) as exc_info:
        _generic(client, get_status=lambda e: e.status_code)
    assert exc_info.value is err
    assert client.call_count == 1
    no_sleep.assert_not_called()


def test_generic_api_error_without_status_fn_is_non_retriable(no_sleep):
    client = Mock(side_effect=_ApiError(503))
    with pytest.raises(_ApiError):
        _generic(client, get_status=None)
    assert client.call_count == 1


def test_generic_status_fn_failure_is_treated_as_unknown_status(no_sleep, caplog):
    def broken(e):
        raise AttributeError("nope")

    client = Mock(side_effect=_ApiError(503))
    with caplog.at_level(logging.WARNING):
        with pytest.raises(_ApiError):
            _generic(client, get_status=broken)
    assert "Could not get status code" in caplog.text


def test_generic_retriable_api_error_exhausted_raises_runtime_error(no_sleep):
    client = Mock(side_effect=_ApiError(500))
    with pytest.raises(RuntimeError, match="after 3 retries"):
        _generic(client, get_status=lambda e: e.status_code, max_retries=3)
    assert client.call_count == 3
    assert no_sleep.call_count == 2


def test_generic_unexpected_exception_reraised_immediately(no_sleep):
    client = Mock(side_effect=KeyError("surprise"))
    with pytest.raises(KeyError):
        _generic(client)
    assert client.call_count == 1
    no_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# call_openai_api_with_retries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "response, valid",
    [
        (SimpleNamespace(choices=[1]), True),
        (SimpleNamespace(output=[1]), True),
        (SimpleNamespace(choices=[1], error="boom"), False),
        (SimpleNamespace(choices=[]), False),
        (SimpleNamespace(output=[]), False),
        (SimpleNamespace(), False),
    ],
)
def test_openai_response_validity(no_sleep, response, valid):
    client = Mock(return_value=response)
    if valid:
        assert llm_utils.call_openai_api_with_retries(client, {"a": 1}, max_retries=1) is response
    else:
        with pytest.raises(RuntimeError):
            llm_utils.call_openai_api_with_retries(client, {"a": 1}, max_retries=1)
    client.assert_called_once_with(a=1)


def test_openai_rate_limit_error_is_retried(no_sleep):
    client = Mock(
        side_effect=[
            _openai_status_error(openai.RateLimitError, 429),
            SimpleNamespace(choices=["c"]),
        ]
    )
    out = llm_utils.call_openai_api_with_retries(client, {}, max_retries=3)
    assert out.choices == ["c"]
    no_sleep.assert_called_once_with(40)


def test_openai_api_error_without_http_status_is_reraised(no_sleep):
    err = openai.APIError("generic", request=_request(), body=None)
    client = Mock(side_effect=err)
    with pytest.raises(openai.APIError):
        llm_utils.call_openai_api_with_retries(client, {}, max_retries=3)
    assert client.call_count == 1


def test_openai_api_error_with_http_status_5xx_is_retried(no_sleep):
    err = openai.APIError("legacy", request=_request(), body=None)
    err.http_status = 503
    client = Mock(side_effect=[err, SimpleNamespace(choices=["c"])])
    assert llm_utils.call_openai_api_with_retries(client, {}, max_retries=3).choices == ["c"]


def test_openai_bad_request_is_fatal(no_sleep):
    client = Mock(side_effect=_openai_status_error(openai.BadRequestError, 400))
    with pytest.raises(openai.BadRequestError):
        llm_utils.call_openai_api_with_retries(client, {}, max_retries=3)
    assert client.call_count == 1


@pytest.mark.xfail(
    strict=True,
    reason="BUG: get_openai_status_code reads `http_status`, but openai>=1 exceptions expose "
    "`status_code`, so 5xx InternalServerError is never retried",
)
def test_openai_internal_server_error_is_retried(no_sleep):
    client = Mock(
        side_effect=[
            _openai_status_error(openai.InternalServerError, 500),
            SimpleNamespace(choices=["c"]),
        ]
    )
    assert llm_utils.call_openai_api_with_retries(client, {}, max_retries=3).choices == ["c"]


# ---------------------------------------------------------------------------
# call_anthropic_api_with_retries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "response, valid",
    [
        (SimpleNamespace(type="message", content=[]), True),
        (SimpleNamespace(type="message", content=[{"type": "text", "text": "hi"}]), True),
        (SimpleNamespace(type="message", content="not a list"), False),
        (SimpleNamespace(type="message"), False),
        (SimpleNamespace(type="completion", content=[]), False),
        ({"type": "error", "error": {"message": "overloaded"}}, False),
        ({"type": "message", "content": []}, False),
        (None, False),
        ({}, False),
    ],
)
def test_anthropic_response_validity(no_sleep, response, valid):
    client = Mock(return_value=response)
    if valid:
        assert llm_utils.call_anthropic_api_with_retries(client, {}, max_retries=1) is response
    else:
        with pytest.raises(RuntimeError):
            llm_utils.call_anthropic_api_with_retries(client, {}, max_retries=1)


@pytest.mark.parametrize(
    "exc_cls, status, retried",
    [
        (anthropic.RateLimitError, 429, True),
        (anthropic.InternalServerError, 500, True),
        (anthropic.InternalServerError, 529, True),
        (anthropic.BadRequestError, 400, False),
        (anthropic.AuthenticationError, 401, False),
        (anthropic.PermissionDeniedError, 403, False),
        (anthropic.NotFoundError, 404, False),
        (anthropic.UnprocessableEntityError, 422, False),
    ],
)
def test_anthropic_status_codes_retriable_vs_fatal(no_sleep, exc_cls, status, retried):
    ok = SimpleNamespace(type="message", content=[])
    err = _anthropic_status_error(exc_cls, status)
    client = Mock(side_effect=[err, ok])
    if retried:
        assert llm_utils.call_anthropic_api_with_retries(client, {}, max_retries=3) is ok
        assert client.call_count == 2
        assert no_sleep.call_count == 1
    else:
        with pytest.raises(exc_cls):
            llm_utils.call_anthropic_api_with_retries(client, {}, max_retries=3)
        assert client.call_count == 1
        no_sleep.assert_not_called()


def test_anthropic_retriable_errors_exhaust_to_runtime_error(no_sleep):
    client = Mock(side_effect=_anthropic_status_error(anthropic.InternalServerError, 503))
    with pytest.raises(RuntimeError, match="after 2 retries"):
        llm_utils.call_anthropic_api_with_retries(client, {}, max_retries=2)
    assert client.call_count == 2


@pytest.mark.xfail(
    strict=True,
    reason="BUG: anthropic.APIConnectionError is listed as a retriable API error but has no "
    "status_code, so transient network failures are re-raised instead of retried",
)
def test_anthropic_connection_error_is_retried(no_sleep):
    ok = SimpleNamespace(type="message", content=[])
    client = Mock(side_effect=[anthropic.APIConnectionError(request=_request()), ok])
    assert llm_utils.call_anthropic_api_with_retries(client, {}, max_retries=3) is ok


# ---------------------------------------------------------------------------
# supports_tool_calling_for_openrouter
# ---------------------------------------------------------------------------


def _fake_openrouter_client(monkeypatch, create):
    created = {}

    def client_factory(**kwargs):
        created.update(kwargs)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr(openai, "Client", client_factory)
    return created


@pytest.mark.parametrize(
    "message, expected",
    [({"role": "assistant", "tool_calls": [{"id": "1"}]}, True), ({"content": "nope"}, False)],
)
def test_supports_tool_calling_for_openrouter(monkeypatch, message, expected):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    response = Mock()
    response.to_dict.return_value = {"choices": [{"message": message}]}
    create = Mock(return_value=response)
    created = _fake_openrouter_client(monkeypatch, create)

    assert llm_utils.supports_tool_calling_for_openrouter("vendor/model") is expected
    assert created == {"api_key": "test-key", "base_url": "https://openrouter.ai/api/v1"}
    kwargs = create.call_args.kwargs
    assert kwargs["model"] == "vendor/model"
    assert kwargs["tool_choice"] == "required"
    assert kwargs["tools"][0]["function"]["name"] == "dummy_tool"


def test_supports_tool_calling_for_openrouter_error_defaults_to_true(monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    created = _fake_openrouter_client(monkeypatch, Mock(side_effect=RuntimeError("offline")))
    assert llm_utils.supports_tool_calling_for_openrouter("vendor/model") is True
    assert created["api_key"] is None
    assert "offline" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# tokens
# ---------------------------------------------------------------------------


def test_truncate_tokens_with_start_window(fake_tiktoken):
    text = "a b c d e f"
    assert llm_utils.truncate_tokens(text, max_tokens=2, start=1, model_name="m") == "b c"
    assert fake_tiktoken == ["m"]


def test_truncate_tokens_short_text_returned_unchanged(fake_tiktoken):
    text = "a  b\tc"
    assert llm_utils.truncate_tokens(text, max_tokens=3) is text


def test_get_tokenizer_tiktoken_hit_is_cached(fake_tiktoken):
    enc = llm_utils.get_tokenizer("gpt-x")
    assert enc.name == "gpt-x"
    assert llm_utils.get_tokenizer("gpt-x") is enc
    assert fake_tiktoken == ["gpt-x"]


def test_get_tokenizer_falls_back_to_huggingface(fake_tiktoken, monkeypatch):
    hf_tok = object()
    auto = Mock()
    auto.from_pretrained.return_value = hf_tok
    monkeypatch.setattr(llm_utils, "AutoTokenizer", auto)
    assert llm_utils.get_tokenizer("unknown/hf-model") is hf_tok
    auto.from_pretrained.assert_called_once_with("unknown/hf-model")


def test_get_tokenizer_falls_back_to_gpt4(fake_tiktoken, monkeypatch):
    auto = Mock()
    auto.from_pretrained.side_effect = OSError("not found")
    monkeypatch.setattr(llm_utils, "AutoTokenizer", auto)
    enc = llm_utils.get_tokenizer("unknown-model")
    assert enc.name == "gpt-4"
    assert fake_tiktoken == ["unknown-model", "gpt-4"]


def test_count_tokens_uses_get_tokenizer(fake_tiktoken):
    assert llm_utils.count_tokens("one two three", model="gpt-x") == 3


@pytest.mark.parametrize(
    "model_name, expected",
    [
        ("test-model", "gpt-4"),
        ("openai/gpt-4o", "gpt-4o"),
        ("azure/gpt-35-turbo/2024", "gpt-35-turbo"),
        ("reka-core", "gpt-4"),
    ],
)
def test_get_tokenizer_old_prefix_routing(fake_tiktoken, model_name, expected):
    assert llm_utils.get_tokenizer_old(model_name).name == expected


def test_get_tokenizer_old_other_uses_huggingface(fake_tiktoken, monkeypatch):
    auto = Mock()
    monkeypatch.setattr(llm_utils, "AutoTokenizer", auto)
    assert llm_utils.get_tokenizer_old("meta/llama") is auto.from_pretrained.return_value
    auto.from_pretrained.assert_called_once_with("meta/llama")


# ---------------------------------------------------------------------------
# parsers
# ---------------------------------------------------------------------------


def test_yaml_parser_joins_value_on_next_line():
    value, valid, retry_message = llm_utils.yaml_parser("action:\nclick('a')\nthought: ok")
    assert valid is True
    assert retry_message == ""
    assert value == {"action": "click('a')", "thought": "ok"}


def test_yaml_parser_invalid_yaml():
    with pytest.warns(UserWarning):
        value, valid, retry_message = llm_utils.yaml_parser("key: [unclosed")
    assert value == {}
    assert valid is False
    assert "not a valid yaml" in retry_message


def test_json_parser_non_object_json():
    assert llm_utils.json_parser("[1, 2]") == ([1, 2], True, "")


def test_compress_chunks_skip_list_and_short_chunks():
    long_a = "long paragraph A"
    short = "tiny"
    text = f"{long_a}\n\n{short}\n\n{long_a}\n\n{short}\n\nunique paragraph"
    def_dict, compressed = llm_utils._compress_chunks(text, "X", skip_list=[])
    assert def_dict == {"X-0": long_a}  # chunks <= 10 chars are never compressed
    assert compressed == "X-0\ntiny\nX-0\ntiny\nunique paragraph"

    def_dict, compressed = llm_utils._compress_chunks(text, "X", skip_list=[long_a])
    assert def_dict == {}
    assert compressed == f"{long_a}\ntiny\n{long_a}\ntiny\nunique paragraph"


def test_compress_string_without_redundancy_has_empty_definitions():
    assert llm_utils.compress_string("only once\n\nanother") == (
        "<definitions>\n</definitions>\nonly once\nanother"
    )


def test_extract_html_tags_multiple_multiline_and_case_sensitive():
    text = "<a> first </a> <b>\nmulti\nline\n</b> <a>second</a> <C>upper</C>"
    out = llm_utils.extract_html_tags(text, ["a", "b", "c", "missing"])
    assert out == {"a": ["first", "second"], "b": ["multi\nline"]}


def test_parse_html_tags_success_with_optional():
    text = "<think>hmm</think><action>click('1')</action>"
    content, valid, msg = llm_utils.parse_html_tags(
        text, keys=["action"], optional_keys=["think", "memory"]
    )
    assert valid is True
    assert msg == ""
    assert content == {"action": "click('1')", "think": "hmm"}


def test_parse_html_tags_missing_and_duplicate_keys():
    text = "<action>a1</action><action>a2</action>"
    content, valid, msg = llm_utils.parse_html_tags(text, keys=["action", "think"])
    assert valid is False
    assert content == {"action": "a1"}
    assert msg.split("\n") == [
        "Found multiple instances of the key action. You should have only one of them.",
        "Missing the key <think> in the answer.",
    ]


def test_parse_html_tags_merge_multiple():
    text = "<action>a1</action><action>a2</action>"
    content, valid, msg = llm_utils.parse_html_tags(text, keys=["action"], merge_multiple=True)
    assert valid is True
    assert content == {"action": "a1\na2"}


def test_parse_html_tags_raise():
    assert llm_utils.parse_html_tags_raise("<x>1</x>", keys=["x"]) == {"x": "1"}
    with pytest.raises(ParseError, match="Missing the key <y>"):
        llm_utils.parse_html_tags_raise("<x>1</x>", keys=["x", "y"])
    assert llm_utils.parse_html_tags_raise("<x>1</x><x>2</x>", keys=["x"], merge_multiple=True) == {
        "x": "1\n2"
    }


def test_extract_code_blocks_no_blocks():
    assert llm_utils.extract_code_blocks("no code here") == []


# ---------------------------------------------------------------------------
# images
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "image",
    [
        np.full((4, 5, 3), 200, dtype=np.uint8),
        np.full((4, 5, 4), 200, dtype=np.uint8),
        Image.new("LA", (5, 4), (200, 255)),
        Image.new("RGB", (5, 4), (200, 200, 200)),
    ],
    ids=["ndarray-rgb", "ndarray-rgba", "pil-la", "pil-rgb"],
)
def test_image_to_jpg_base64_url(image):
    img = _decode_data_url(llm_utils.image_to_jpg_base64_url(image), "data:image/jpeg;base64,")
    assert img.format == "JPEG"
    assert img.mode == "RGB"
    assert img.size == (5, 4)


def test_image_to_png_base64_url_lossless_and_rgba_dropped():
    arr = np.zeros((2, 3, 4), dtype=np.uint8)
    arr[..., 0] = 10
    arr[..., 1] = 20
    arr[..., 2] = 30
    arr[..., 3] = 128
    img = _decode_data_url(llm_utils.image_to_png_base64_url(arr), "data:image/png;base64,")
    assert img.format == "PNG"
    assert img.mode == "RGB"
    assert np.array(img)[0, 0].tolist() == [10, 20, 30]


def test_image_to_png_base64_url_pil_grayscale_kept():
    img = _decode_data_url(
        llm_utils.image_to_png_base64_url(Image.new("L", (3, 3), 7)), "data:image/png;base64,"
    )
    assert img.mode == "L"


# ---------------------------------------------------------------------------
# BaseMessage and subclasses
# ---------------------------------------------------------------------------


def test_base_message_rejects_unknown_kwargs():
    with pytest.raises(ValueError, match="Invalid attributes"):
        BaseMessage("user", "hi", foo=1)


def test_base_message_deep_copies_content():
    content = [{"type": "text", "text": "orig"}]
    msg = BaseMessage("user", content)
    content[0]["text"] = "mutated"
    assert msg["content"][0]["text"] == "orig"


@pytest.mark.parametrize(
    "cls, role", [(SystemMessage, "system"), (HumanMessage, "user"), (AIMessage, "assistant")]
)
def test_message_subclasses_roles(cls, role):
    msg = cls("c")
    assert msg["role"] == role
    assert msg["content"] == "c"


def test_ai_message_log_probs():
    assert AIMessage("c")["log_probs"] is None
    assert AIMessage("c", log_probs=[0.1])["log_probs"] == [0.1]


def test_str_only_text_elements_including_input_text(caplog):
    msg = HumanMessage(
        [
            {"type": "text", "text": "a"},
            {"type": "input_text", "text": "b"},
            {"type": "image_url", "image_url": {"url": "u"}},
        ]
    )
    with caplog.at_level(logging.INFO):
        assert str(msg) == "a\nb"
    assert "has images" in caplog.text


def test_str_warn_if_image_logs_warning(caplog):
    msg = HumanMessage([{"type": "image_url", "image_url": "u"}])
    with caplog.at_level(logging.WARNING):
        assert msg.__str__(warn_if_image=True) == ""
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_str_plain_text():
    assert str(HumanMessage("plain")) == "plain"


def test_add_text_converts_string_content_to_list():
    msg = HumanMessage("first")
    msg.add_text("second")
    assert msg["content"] == [
        {"type": "text", "text": "first"},
        {"type": "text", "text": "second"},
    ]


def test_add_image_variants():
    msg = HumanMessage("look")
    msg.add_image("data:image/png;base64,AAA")
    msg.add_image("http://img", detail="low")
    msg.add_image(np.zeros((2, 2, 3), dtype=np.uint8))
    elems = msg["content"]
    assert elems[1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}
    assert elems[2] == {"type": "image_url", "image_url": {"url": "http://img", "detail": "low"}}
    assert elems[3]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_to_markdown_string_and_mixed_content():
    assert HumanMessage("x").to_markdown() == "\n```\nx\n```\n"
    msg = HumanMessage(
        [
            {"type": "text", "text": "t"},
            {"type": "image_url", "image_url": "str-url"},
            {"type": "image_url", "image_url": {"url": "dict-url"}},
            {"type": "other", "other": "ignored"},
        ]
    )
    assert msg.to_markdown() == "\n```\nt\n```\n\n![image](str-url)\n![image](dict-url)"


def test_merge_string_content_noop():
    msg = HumanMessage("x")
    msg.merge()
    assert msg["content"] == "x"


@pytest.mark.xfail(
    strict=True,
    reason="BUG: BaseMessage.merge() collapses any single-element content via ['text'], raising "
    "KeyError when the only element is an image",
)
def test_merge_single_image_element_is_kept():
    msg = HumanMessage([{"type": "image_url", "image_url": {"url": "u"}}])
    msg.merge()
    assert msg["content"] == [{"type": "image_url", "image_url": {"url": "u"}}]


# ---------------------------------------------------------------------------
# Discussion
# ---------------------------------------------------------------------------


def test_discussion_init_variants():
    assert len(Discussion()) == 0
    single = SystemMessage("s")
    assert Discussion(single).messages == [single]
    msgs = [SystemMessage("s"), HumanMessage("h")]
    assert Discussion(msgs).messages is msgs


def test_discussion_add_message_forms_and_access():
    d = Discussion()
    d.add_message(role="system", content="s")
    d.add_message({"role": "user", "content": "u"})
    d.append(AIMessage("a"))
    assert [m["role"] for m in d] == ["system", "user", "assistant"]
    assert all(isinstance(m, BaseMessage) for m in d)
    assert d[1]["content"] == "u"
    assert d.last_message["content"] == "a"
    assert str(d) == "s\nu\na"


def test_discussion_add_content_targets_last_message():
    d = Discussion([SystemMessage("s"), HumanMessage("u")])
    d.add_text("more")
    d.add_image("http://img", detail="high")
    d.add_content("custom", "payload")
    assert d[0]["content"] == "s"
    assert d[1]["content"] == [
        {"type": "text", "text": "u"},
        {"type": "text", "text": "more"},
        {"type": "image_url", "image_url": {"url": "http://img", "detail": "high"}},
        {"type": "custom", "custom": "payload"},
    ]


def test_discussion_to_string_and_to_openai_merge_text():
    msg = HumanMessage("a")
    msg.add_text("b")
    d = Discussion([SystemMessage("s"), msg])
    assert d.to_string() == "s\na\nb"
    assert d[1]["content"] == "a\nb"
    assert d.to_openai() is d.messages


def test_discussion_to_markdown():
    d = Discussion([SystemMessage("s"), HumanMessage("u")])
    assert d.to_markdown() == ("Message 0\n\n```\ns\n```\n\n" "\n" "Message 1\n\n```\nu\n```\n\n")


# ---------------------------------------------------------------------------
# download_and_save_model
# ---------------------------------------------------------------------------


def test_download_and_save_model(monkeypatch, tmp_path, capsys):
    auto = Mock()
    monkeypatch.setattr(llm_utils, "AutoModel", auto)
    llm_utils.download_and_save_model("org/model", save_dir=str(tmp_path))
    auto.from_pretrained.assert_called_once_with("org/model")
    auto.from_pretrained.return_value.save_pretrained.assert_called_once_with(str(tmp_path))
    assert str(tmp_path) in capsys.readouterr().out
