"""Offline unit tests for chat_api (no network calls)."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

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
    RetryError,
    SelfHostedModelArgs,
    _extract_wait_time,
    handle_error,
    make_assistant_message,
    make_system_message,
    make_user_message,
)
from agentlab.llm.llm_utils import Discussion, HumanMessage, SystemMessage


@pytest.fixture(autouse=True)
def fresh_tracker(monkeypatch):
    # other tests may leave TRACKER.instance set to None in this thread
    monkeypatch.setattr(chat_api.tracking, "TRACKER", threading.local())


@pytest.fixture(autouse=True)
def fake_api_keys(monkeypatch):
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setenv(var, "test-key")


@pytest.fixture
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(chat_api.time, "sleep", lambda s: sleeps.append(s))
    return sleeps


def test_make_messages():
    assert make_system_message("s") == {"role": "system", "content": "s"}
    assert make_user_message("u") == {"role": "user", "content": "u"}
    assert make_assistant_message("a") == {"role": "assistant", "content": "a"}


PROMPT = "Some header\n[12] button 'Click me'\n"


@pytest.mark.parametrize(
    "messages",
    [
        [make_system_message("sys"), make_user_message(PROMPT)],
        Discussion([SystemMessage("sys"), HumanMessage(PROMPT)]),
    ],
)
def test_cheat_miniwob_llm(messages):
    answer = CheatMiniWoBLLM()(messages)
    assert answer["role"] == "assistant"
    assert 'click("12")' in answer["content"]


def test_cheat_miniwob_llm_waits_and_fails_without_button(monkeypatch, no_sleep):
    llm = CheatMiniWoBLLMArgs(wait_time=2).make_model()
    with pytest.raises(Exception, match="Can't find the button"):
        llm([make_system_message("sys"), make_user_message("no buttons here")])
    assert no_sleep == [2]


def test_cheat_miniwob_llm_args_server_noops():
    args = CheatMiniWoBLLMArgs()
    assert args.prepare_server() is None
    assert args.close_server() is None


@pytest.mark.parametrize(
    "msg, expected",
    [
        ("Rate limit. Please try again in 90.5s.", 90.5),
        ("Rate limit. Please try again in 5s.", 60),
        ("unrelated error", 60),
    ],
)
def test_extract_wait_time(msg, expected):
    assert _extract_wait_time(msg) == expected


def test_handle_error(no_sleep):
    err = openai.OpenAIError("please try again in 2s")
    assert handle_error(err, 0, min_retry_wait_time=1, max_retry=3) == "please try again in 2s"
    assert no_sleep == [2.0]


def test_handle_error_reraises_non_openai_errors():
    with pytest.raises(KeyError):
        handle_error(KeyError("x"), 0, 1, 3)


def _completion(contents, usage=True):
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5) if usage else None,
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=c), log_probs="lp") for c in contents
        ],
    )


def _chat_model(create, **kwargs):
    client = MagicMock()
    client.chat.completions.create.side_effect = create
    model = ChatModel("m", client_class=MagicMock(return_value=client), **kwargs)
    return model, client


def test_chat_model_pricing():
    model, _ = _chat_model(None, pricing_func=lambda: {"m": {"prompt": "0.1", "completion": "0.2"}})
    assert (model.input_cost, model.output_cost) == (0.1, 0.2)

    model, _ = _chat_model(None, pricing_func=lambda: {})
    assert (model.input_cost, model.output_cost) == (0.0, 0.0)


def test_chat_model_api_key_from_env(monkeypatch):
    monkeypatch.setenv("MY_KEY", "secret")
    model, _ = _chat_model(None, api_key_env_var="MY_KEY")
    assert model.api_key == "secret"


def test_chat_model_single_sample_with_log_probs():
    model, client = _chat_model([_completion(["hello"])], log_probs=True, temperature=0.3)
    res = model([make_user_message("hi")])
    assert res["content"] == "hello" and res["log_probs"] == "lp"
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0.3 and kwargs["n"] == 1
    assert model.get_stats() == {"n_retry_llm": 1}


def test_chat_model_multiple_samples():
    model, _ = _chat_model([_completion(["a", "b"])])
    res = model([make_user_message("hi")], n_samples=2, temperature=0)
    assert [r["content"] for r in res] == ["a", "b"]


def test_chat_model_retries_on_missing_usage(no_sleep):
    model, _ = _chat_model(
        [_completion(["x"], usage=False), _completion(["ok"])], min_retry_wait_time=0
    )
    assert model([make_user_message("hi")])["content"] == "ok"
    assert model.retries == 2 and model.success
    assert len(model.error_types) == 1


def test_chat_model_raises_retry_error(no_sleep):
    model, _ = _chat_model(openai.OpenAIError("down"), max_retry=2, min_retry_wait_time=0)
    with pytest.raises(RetryError, match="after 2 retries"):
        model([make_user_message("hi")])
    assert model.error_types == ["down", "down"]


def test_chat_model_rejects_zero_retry():
    with pytest.raises(AssertionError):
        _chat_model(None, max_retry=0)


def test_azure_chat_model_requires_env(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    with pytest.raises(AssertionError, match="AZURE_OPENAI_API_KEY"):
        chat_api.AzureChatModel("gpt-4o")
    with pytest.raises(AssertionError, match="AZURE_OPENAI_ENDPOINT"):
        chat_api.AzureChatModel("gpt-4o", api_key="k")


def test_azure_chat_model_client_args(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.azure.com")
    monkeypatch.setattr(chat_api.tracking, "get_pricing_openai", lambda: {})
    model = chat_api.AzureChatModel("gpt-4o", api_key="k", deployment_name="old")
    assert str(model.client.base_url).startswith("https://example.azure.com")


def test_openrouter_and_vllm_models(monkeypatch):
    monkeypatch.setattr(chat_api.tracking, "get_pricing_openrouter", lambda: {})
    router = chat_api.OpenRouterChatModel("some/model", api_key="k")
    assert "openrouter.ai" in str(router.client.base_url)

    vllm = chat_api.VLLMChatModel("some/model", api_key="k", n_retry_server=2)
    assert "0.0.0.0:8000" in str(vllm.client.base_url)
    assert vllm.max_retry == 2


def test_self_hosted_model_args_vllm_and_unknown_backend():
    args = SelfHostedModelArgs(model_name="m", backend="vllm", max_new_tokens=10)
    model = args.make_model()
    assert isinstance(model, chat_api.VLLMChatModel)
    assert model.max_tokens == 10

    with pytest.raises(ValueError, match="not supported"):
        SelfHostedModelArgs(model_name="m", backend="other").make_model()


def test_chat_model_args_is_deprecated():
    with pytest.warns(DeprecationWarning):
        args = ChatModelArgs(model_name="m")
    assert args.make_model() is None


def _anthropic_response(text):
    return SimpleNamespace(
        content=[SimpleNamespace(text=text)],
        usage=SimpleNamespace(input_tokens=1, output_tokens=2),
    )


def test_anthropic_chat_model_converts_messages():
    model = AnthropicModelArgs(model_name="claude", max_new_tokens=20, temperature=0.1).make_model()
    model.client = MagicMock()
    model.client.messages.create.return_value = _anthropic_response("hey")

    res = model([make_system_message("sys"), make_user_message("hi")])

    assert res["content"] == "hey"
    kwargs = model.client.messages.create.call_args.kwargs
    assert kwargs["system"] == "sys"
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert kwargs["max_tokens"] == 20 and kwargs["temperature"] == 0.1


def test_anthropic_chat_model_retries(no_sleep):
    model = AnthropicChatModel("claude", api_key="k", max_retry=2)
    model.client = MagicMock()
    model.client.messages.create.side_effect = [RuntimeError("x"), _anthropic_response("ok")]
    res = model([make_user_message("hi")], temperature=0)
    assert res["content"] == "ok"
    assert "system" not in model.client.messages.create.call_args.kwargs
    assert no_sleep == [60]

    model.client.messages.create.side_effect = RuntimeError("down")
    with pytest.raises(RuntimeError, match="down"):
        model([make_user_message("hi")])
