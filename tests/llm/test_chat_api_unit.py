"""Unit tests for chat_api that never hit a real API (clients are mocked)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import openai
import pytest

from agentlab.llm import chat_api
from agentlab.llm.chat_api import (
    AnthropicChatModel,
    AnthropicModelArgs,
    ChatModel,
    ChatModelArgs,
    CheatMiniWoBLLM,
    CheatMiniWoBLLMArgs,
    OpenAIChatModel,
    OpenAIModelArgs,
    RetryError,
    SelfHostedModelArgs,
    _extract_wait_time,
    handle_error,
    make_assistant_message,
    make_system_message,
    make_user_message,
)
from agentlab.llm.llm_utils import Discussion, HumanMessage, SystemMessage
from agentlab.llm.tracking import set_tracker

MESSAGES = [make_system_message("sys"), make_user_message("hello")]


def _completion(contents=("hi",), prompt_tokens=10, completion_tokens=5, usage=True):
    return SimpleNamespace(
        usage=(
            SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
            if usage
            else None
        ),
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=c), log_probs=[-0.5]) for c in contents
        ],
    )


def _make_chat_model(responses, **kwargs):
    client = MagicMock()
    client.chat.completions.create.side_effect = responses
    client_class = MagicMock(return_value=client)
    kwargs.setdefault("min_retry_wait_time", 0)
    model = ChatModel("test-model", client_class=client_class, **kwargs)
    return model, client, client_class


@pytest.fixture
def no_sleep():
    with patch.object(chat_api.time, "sleep") as sleep:
        yield sleep


def test_message_constructors():
    assert make_system_message("a") == {"role": "system", "content": "a"}
    assert make_user_message("b") == {"role": "user", "content": "b"}
    assert make_assistant_message("c") == {"role": "assistant", "content": "c"}


@pytest.mark.parametrize(
    "msg, minimum, expected",
    [
        ("Rate limit. Please try again in 20s.", 5, 20.0),
        ("Please try again in 1.5s", 5, 5),
        ("Please try again in 90.25s", 60, 90.25),
        ("no hint here", 7, 7),
        ("", 60, 60),
    ],
)
def test_extract_wait_time(msg, minimum, expected):
    assert _extract_wait_time(msg, min_retry_wait_time=minimum) == expected


def test_handle_error_reraises_non_openai_errors(no_sleep):
    with pytest.raises(KeyError):
        handle_error(KeyError("x"), 0, 1, 3)
    no_sleep.assert_not_called()


def test_handle_error_sleeps_and_returns_message(no_sleep):
    err = openai.OpenAIError("try again in 3s")
    assert handle_error(err, 0, 1, 3) == "try again in 3s"
    no_sleep.assert_called_once_with(3.0)


def test_cheat_llm_with_message_list():
    llm = CheatMiniWoBLLMArgs().make_model()
    assert isinstance(llm, CheatMiniWoBLLM)
    answer = llm([make_system_message("s"), make_user_message("[13] button 'Click Me!'")])
    assert answer["role"] == "assistant"
    assert 'click("13")' in answer["content"]


def test_cheat_llm_with_discussion():
    llm = CheatMiniWoBLLM()
    answer = llm(Discussion([SystemMessage("s"), HumanMessage("  [7] BUTTON 'ok'")]))
    assert 'click("7")' in answer["content"]


def test_cheat_llm_without_button():
    with pytest.raises(Exception, match="Can't find the button's bid"):
        CheatMiniWoBLLM()([make_system_message("s"), make_user_message("[1] link 'x'")])


def test_cheat_llm_args_server_hooks_are_noops():
    args = CheatMiniWoBLLMArgs()
    assert args.prepare_server() is None and args.close_server() is None


def test_chat_model_init_reads_env_and_pricing(monkeypatch):
    monkeypatch.setenv("MY_KEY", "secret-from-env")
    pricing = lambda: {"test-model": {"prompt": "0.5", "completion": "1"}}
    model, _, client_class = _make_chat_model(
        [], api_key_env_var="MY_KEY", pricing_func=pricing, client_args={"base_url": "u"}
    )
    client_class.assert_called_once_with(api_key="secret-from-env", base_url="u")
    assert (model.input_cost, model.output_cost) == (0.5, 1.0)


def test_chat_model_unknown_pricing_defaults_to_zero():
    model, _, _ = _make_chat_model([], pricing_func=lambda: {})
    assert (model.input_cost, model.output_cost) == (0.0, 0.0)


def test_chat_model_requires_positive_retry():
    with pytest.raises(AssertionError):
        _make_chat_model([], max_retry=0)


def test_chat_model_call_success_and_tracking():
    model, client, _ = _make_chat_model([_completion()], temperature=0.2, max_tokens=50)
    model.input_cost, model.output_cost = 1.0, 2.0
    with set_tracker() as tracker:
        answer = model(MESSAGES)
    assert answer["role"] == "assistant" and answer["content"] == "hi"
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs == dict(
        model="test-model",
        messages=MESSAGES,
        n=1,
        temperature=0.2,
        max_completion_tokens=50,
        logprobs=False,
    )
    assert tracker.stats == {"input_tokens": 10, "output_tokens": 5, "cost": 20.0}
    assert model.get_stats() == {"n_retry_llm": 1}
    assert model.success and model.error_types == []


def test_chat_model_multiple_samples_and_temperature_override():
    model, client, _ = _make_chat_model([_completion(contents=("a", "b"))])
    answers = model(MESSAGES, n_samples=2, temperature=0.9)
    assert [a["content"] for a in answers] == ["a", "b"]
    assert client.chat.completions.create.call_args.kwargs["temperature"] == 0.9


def test_chat_model_log_probs():
    model, _, _ = _make_chat_model([_completion()], log_probs=True)
    assert model(MESSAGES)["log_probs"] == [-0.5]


def test_chat_model_retries_on_openai_errors(no_sleep):
    model, client, _ = _make_chat_model(
        [openai.OpenAIError("try again in 2s"), _completion(usage=False), _completion()],
        max_retry=3,
    )
    assert model(MESSAGES)["content"] == "hi"
    assert model.retries == 3
    assert model.error_types[0] == "try again in 2s"
    assert "usage information" in model.error_types[1]
    assert no_sleep.call_count == 2


def test_chat_model_raises_retry_error(no_sleep):
    model, _, _ = _make_chat_model([openai.OpenAIError("down")] * 2, max_retry=2)
    with pytest.raises(RetryError, match="after 2 retries"):
        model(MESSAGES)
    assert not model.success
    assert model.error_types == ["down", "down"]


def test_chat_model_non_openai_error_propagates():
    model, _, _ = _make_chat_model([ValueError("bad")])
    with pytest.raises(ValueError, match="bad"):
        model(MESSAGES)


def test_openai_model_args_make_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    with patch.object(chat_api.tracking, "get_model_info", side_effect=Exception("offline")):
        model = OpenAIModelArgs(model_name="gpt-x", max_new_tokens=None).make_model()
    assert isinstance(model, OpenAIChatModel)
    assert model.max_tokens is chat_api.NOT_GIVEN
    assert model.api_key == "fake"


def test_self_hosted_unknown_backend():
    with pytest.raises(ValueError, match="Backend foo is not supported"):
        SelfHostedModelArgs(model_name="m", backend="foo").make_model()


def test_chat_model_args_is_deprecated():
    with pytest.warns(DeprecationWarning):
        args = ChatModelArgs(model_name="legacy")
    assert args.make_model() is None


@pytest.fixture
def anthropic_client():
    client = MagicMock()
    with patch.object(chat_api.anthropic, "Anthropic", return_value=client) as cls:
        yield client, cls


def _anthropic_response(text="ok"):
    return SimpleNamespace(
        content=[SimpleNamespace(text=text)],
        usage=SimpleNamespace(input_tokens=3, output_tokens=4),
    )


def test_anthropic_model_args_make_model(anthropic_client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    _, cls = anthropic_client
    model = AnthropicModelArgs(
        model_name="claude-x", max_new_tokens=12, temperature=0.1
    ).make_model()
    assert isinstance(model, AnthropicChatModel)
    assert model.max_tokens == 12 and model.temperature == 0.1
    cls.assert_called_once_with(api_key="fake")


def test_anthropic_call_converts_messages(anthropic_client):
    client, _ = anthropic_client
    client.messages.create.return_value = _anthropic_response("done")
    model = AnthropicChatModel("claude-x", api_key="k", max_tokens=8, temperature=0.4)
    with set_tracker() as tracker:
        answer = model(MESSAGES + [make_assistant_message("prev")])
    assert answer["content"] == "done"
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["system"] == "sys"
    assert kwargs["messages"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "prev"},
    ]
    assert kwargs["max_tokens"] == 8 and kwargs["temperature"] == 0.4
    assert tracker.stats["input_tokens"] == 3 and tracker.stats["output_tokens"] == 4


def test_anthropic_call_without_system(anthropic_client):
    client, _ = anthropic_client
    client.messages.create.return_value = _anthropic_response()
    model = AnthropicChatModel("claude-x", api_key="k")
    with set_tracker():
        model([make_user_message("hi")], temperature=0.0)
    kwargs = client.messages.create.call_args.kwargs
    assert "system" not in kwargs
    assert kwargs["temperature"] == 0.0


def test_anthropic_retry_then_success(anthropic_client, no_sleep):
    client, _ = anthropic_client
    client.messages.create.side_effect = [RuntimeError("overloaded"), _anthropic_response("ok")]
    model = AnthropicChatModel("claude-x", api_key="k", max_retry=2)
    with set_tracker():
        assert model(MESSAGES)["content"] == "ok"
    no_sleep.assert_called_once_with(60)


def test_anthropic_retry_exhausted(anthropic_client, no_sleep):
    client, _ = anthropic_client
    client.messages.create.side_effect = RuntimeError("overloaded")
    model = AnthropicChatModel("claude-x", api_key="k", max_retry=3)
    with pytest.raises(RuntimeError, match="overloaded"):
        model(MESSAGES)
    assert client.messages.create.call_count == 3
    assert no_sleep.call_count == 2
