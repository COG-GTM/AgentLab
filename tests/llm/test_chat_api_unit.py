"""Hermetic unit tests for agentlab.llm.chat_api (all SDK clients are mocked)."""

import logging
import warnings
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest
from openai.types.chat import ChatCompletion

import agentlab.llm.chat_api as chat_api
import agentlab.llm.tracking as tracking
from agentlab.llm.chat_api import (
    AnthropicChatModel,
    AnthropicModelArgs,
    AzureChatModel,
    AzureModelArgs,
    ChatModel,
    ChatModelArgs,
    CheatMiniWoBLLM,
    CheatMiniWoBLLMArgs,
    HuggingFaceURLChatModel,
    OpenAIChatModel,
    OpenAIModelArgs,
    OpenRouterChatModel,
    OpenRouterError,
    OpenRouterModelArgs,
    RetryError,
    SelfHostedModelArgs,
    VLLMChatModel,
    _extract_wait_time,
    handle_error,
    make_assistant_message,
    make_system_message,
    make_user_message,
)
from agentlab.llm.llm_utils import AIMessage, Discussion, HumanMessage, SystemMessage

ENV_VARS = [
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "ANTHROPIC_API_KEY",
    "VLLM_API_KEY",
    "TGI_TOKEN",
    "AGENTLAB_MODEL_URL",
    "AGENTLAB_MODEL_TOKEN",
]

_MISSING = object()


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch):
    """Clear API-key env vars, block real sleeps, and isolate the thread-local tracker."""
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    sleep = MagicMock()
    monkeypatch.setattr(chat_api.time, "sleep", sleep)
    previous = getattr(tracking.TRACKER, "instance", _MISSING)
    if previous is not _MISSING:
        del tracking.TRACKER.instance
    yield sleep
    if hasattr(tracking.TRACKER, "instance"):
        del tracking.TRACKER.instance
    if previous is not _MISSING:
        tracking.TRACKER.instance = previous


@pytest.fixture
def sleep_mock(_hermetic):
    return _hermetic


@pytest.fixture
def openai_cls(monkeypatch):
    """Replace the OpenAI client class used by every ChatModel subclass."""
    cls = MagicMock(name="OpenAI")
    monkeypatch.setattr(chat_api, "OpenAI", cls)
    return cls


def make_completion(contents=("hello",), prompt_tokens=10, completion_tokens=5, usage=True):
    return ChatCompletion(
        id="cmpl-1",
        created=0,
        model="fake-model",
        object="chat.completion",
        choices=[
            {
                "index": i,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": c},
                "logprobs": {"content": []},
            }
            for i, c in enumerate(contents)
        ],
        usage=(
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            if usage
            else None
        ),
    )


def make_chat_model(responses, pricing=None, **kwargs):
    """A ChatModel whose client.chat.completions.create yields `responses` in order."""
    client_class = MagicMock(name="client_class")
    pricing_func = (lambda: pricing) if pricing is not None else None
    defaults = dict(
        model_name="fake-model",
        api_key="sk-test",
        client_class=client_class,
        pricing_func=pricing_func,
        min_retry_wait_time=3,
    )
    defaults.update(kwargs)
    model = ChatModel(**defaults)
    model.client.chat.completions.create.side_effect = list(responses)
    return model


MESSAGES = [make_system_message("sys"), make_user_message("hi")]


# ---------------------------------------------------------------- message helpers


@pytest.mark.parametrize(
    "factory, role",
    [
        (make_system_message, "system"),
        (make_user_message, "user"),
        (make_assistant_message, "assistant"),
    ],
)
def test_make_message_helpers(factory, role):
    assert factory("some content") == {"role": role, "content": "some content"}


# ---------------------------------------------------------------- CheatMiniWoBLLM

AXTREE_PROMPT = """
RootWebArea 'Click Test Task'
[10] heading 'Click on the widget below'
  [12] button 'Click Me!'
"""


def test_cheat_llm_clicks_button_bid_from_list_messages(sleep_mock):
    llm = CheatMiniWoBLLM()
    answer = llm([make_system_message("sys"), make_user_message(AXTREE_PROMPT)])
    assert answer["role"] == "assistant"
    assert '<action>\nclick("12")\n</action>' in answer["content"]
    sleep_mock.assert_not_called()


def test_cheat_llm_accepts_discussion():
    llm = CheatMiniWoBLLM()
    discussion = Discussion([SystemMessage("sys"), HumanMessage(AXTREE_PROMPT)])
    answer = llm(discussion)
    assert 'click("12")' in answer["content"]


def test_cheat_llm_raises_without_button():
    llm = CheatMiniWoBLLM()
    with pytest.raises(Exception, match="Can't find the button's bid"):
        llm([make_system_message("sys"), make_user_message("[3] link 'home'")])


def test_cheat_llm_waits_when_wait_time_set(sleep_mock):
    llm = CheatMiniWoBLLM(wait_time=2)
    llm([make_system_message("sys"), make_user_message(AXTREE_PROMPT)])
    sleep_mock.assert_called_once_with(2)


def test_cheat_llm_args_make_model_and_server_hooks():
    args = CheatMiniWoBLLMArgs(wait_time=3)
    model = args.make_model()
    assert isinstance(model, CheatMiniWoBLLM)
    assert model.wait_time == 3
    assert args.model_name == "test/cheat_miniwob_click_test"
    assert args.prepare_server() is None
    assert args.close_server() is None


# ---------------------------------------------------------------- model args


@pytest.mark.parametrize(
    "args_cls, model_cls_name, expect_log_probs",
    [
        (OpenRouterModelArgs, "OpenRouterChatModel", True),
        (OpenAIModelArgs, "OpenAIChatModel", True),
        (AzureModelArgs, "AzureChatModel", True),
        (AnthropicModelArgs, "AnthropicChatModel", False),
    ],
)
def test_model_args_make_model_forwards_fields(
    monkeypatch, args_cls, model_cls_name, expect_log_probs
):
    fake_cls = MagicMock(name=model_cls_name)
    monkeypatch.setattr(chat_api, model_cls_name, fake_cls)
    args = args_cls(model_name="m", temperature=0.3, max_new_tokens=42, log_probs=True)

    assert args.make_model() is fake_cls.return_value

    expected = dict(model_name="m", temperature=0.3, max_tokens=42)
    if expect_log_probs:
        expected["log_probs"] = True
    fake_cls.assert_called_once_with(**expected)


def test_self_hosted_args_huggingface_reads_url_and_token_from_env(monkeypatch):
    fake_cls = MagicMock(name="HuggingFaceURLChatModel")
    monkeypatch.setattr(chat_api, "HuggingFaceURLChatModel", fake_cls)
    monkeypatch.setenv("AGENTLAB_MODEL_URL", "http://tgi.local")
    monkeypatch.setenv("AGENTLAB_MODEL_TOKEN", "tok-env")
    args = SelfHostedModelArgs(model_name="m", max_new_tokens=7, temperature=0.2, n_retry_server=2)

    assert args.make_model() is fake_cls.return_value

    assert args.model_url == "http://tgi.local"
    assert args.token == "tok-env"
    fake_cls.assert_called_once_with(
        model_name="m",
        model_url="http://tgi.local",
        token="tok-env",
        temperature=0.2,
        max_new_tokens=7,
        n_retry_server=2,
        log_probs=False,
    )


def test_self_hosted_args_huggingface_explicit_values_win_over_env(monkeypatch):
    fake_cls = MagicMock()
    monkeypatch.setattr(chat_api, "HuggingFaceURLChatModel", fake_cls)
    monkeypatch.setenv("AGENTLAB_MODEL_URL", "http://env")
    monkeypatch.setenv("AGENTLAB_MODEL_TOKEN", "env-tok")
    SelfHostedModelArgs(model_name="m", model_url="http://explicit", token="t").make_model()
    kwargs = fake_cls.call_args.kwargs
    assert kwargs["model_url"] == "http://explicit"
    assert kwargs["token"] == "t"


def test_self_hosted_args_huggingface_missing_env_raises(monkeypatch):
    monkeypatch.setattr(chat_api, "HuggingFaceURLChatModel", MagicMock())
    with pytest.raises(KeyError, match="AGENTLAB_MODEL_URL"):
        SelfHostedModelArgs(model_name="m").make_model()


@pytest.mark.xfail(
    strict=True,
    reason="BUG: SelfHostedModelArgs.make_model omits required base_model_name -> TypeError",
)
def test_self_hosted_args_huggingface_builds_real_model(monkeypatch):
    monkeypatch.setattr(chat_api, "InferenceClient", MagicMock())
    with patch("agentlab.llm.huggingface_utils.AutoTokenizer") as tok:
        tok.from_pretrained.return_value = MagicMock()
        model = SelfHostedModelArgs(
            model_name="m", model_url="http://tgi.local", token="t"
        ).make_model()
    assert isinstance(model, HuggingFaceURLChatModel)


def test_self_hosted_args_vllm_backend(openai_cls, monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "vllm-key")
    args = SelfHostedModelArgs(
        model_name="m", backend="vllm", max_new_tokens=11, temperature=0.4, n_retry_server=3
    )
    model = args.make_model()
    assert isinstance(model, VLLMChatModel)
    assert model.max_retry == 3
    assert model.max_tokens == 11
    assert model.temperature == 0.4
    openai_cls.assert_called_once_with(api_key="vllm-key", base_url="http://0.0.0.0:8000/v1")


def test_self_hosted_args_unknown_backend():
    with pytest.raises(ValueError, match="Backend tensorrt is not supported"):
        SelfHostedModelArgs(model_name="m", backend="tensorrt").make_model()


def test_chat_model_args_is_deprecated():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        args = ChatModelArgs(model_name="m")
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert "deprecated" in str(caught[0].message)
    assert args.make_model() is None
    assert args.n_retry_server == 4


# ---------------------------------------------------------------- retry helpers


@pytest.mark.parametrize(
    "message, min_wait, expected",
    [
        ("Rate limit reached. Please try again in 20s.", 5, 20.0),
        ("Rate limit reached. Please try again in 7.25s.", 1, 7.25),
        ("Rate limit reached. Please try again in 1.5s.", 5, 5),
        ("Rate limit reached. Please try again in 20ms.", 5, 5),
        ("Some unrelated error", 60, 60),
        ("", 0, 0),
    ],
)
def test_extract_wait_time(message, min_wait, expected):
    assert _extract_wait_time(message, min_retry_wait_time=min_wait) == expected


def test_extract_wait_time_default_minimum():
    assert _extract_wait_time("nothing to parse") == 60


def test_handle_error_reraises_non_openai_errors(sleep_mock):
    err = ValueError("boom")
    with pytest.raises(ValueError, match="boom"):
        handle_error(err, itr=0, min_retry_wait_time=1, max_retry=3)
    sleep_mock.assert_not_called()


def test_handle_error_sleeps_and_returns_message(sleep_mock, caplog):
    err = openai.OpenAIError("Rate limit. Please try again in 12s.")
    with caplog.at_level(logging.WARNING):
        result = handle_error(err, itr=1, min_retry_wait_time=2, max_retry=4)
    assert result == "Rate limit. Please try again in 12s."
    sleep_mock.assert_called_once_with(12.0)
    assert "Retrying... (2/4)" in caplog.text


def test_openrouter_error_is_openai_error():
    assert issubclass(OpenRouterError, openai.OpenAIError)


# ---------------------------------------------------------------- ChatModel.__init__


def test_chat_model_requires_positive_max_retry():
    with pytest.raises(AssertionError, match="max_retry"):
        ChatModel(model_name="m", max_retry=0, client_class=MagicMock())


def test_chat_model_reads_api_key_from_env_var(monkeypatch):
    monkeypatch.setenv("MY_KEY", "env-key")
    client_class = MagicMock()
    model = ChatModel(
        model_name="m",
        api_key_env_var="MY_KEY",
        client_class=client_class,
        client_args={"base_url": "http://x"},
    )
    assert model.api_key == "env-key"
    client_class.assert_called_once_with(api_key="env-key", base_url="http://x")
    assert model.client is client_class.return_value


def test_chat_model_explicit_api_key_beats_env_var(monkeypatch):
    monkeypatch.setenv("MY_KEY", "env-key")
    model = ChatModel(
        model_name="m", api_key="explicit", api_key_env_var="MY_KEY", client_class=MagicMock()
    )
    assert model.api_key == "explicit"


def test_chat_model_pricing_found():
    pricing = {"m": {"prompt": "0.001", "completion": "0.002"}}
    model = ChatModel(model_name="m", client_class=MagicMock(), pricing_func=lambda: pricing)
    assert model.input_cost == pytest.approx(0.001)
    assert model.output_cost == pytest.approx(0.002)


def test_chat_model_pricing_missing_model_defaults_to_zero(caplog):
    with caplog.at_level(logging.WARNING):
        model = ChatModel(model_name="m", client_class=MagicMock(), pricing_func=lambda: {})
    assert (model.input_cost, model.output_cost) == (0.0, 0.0)
    assert "not found in the pricing information" in caplog.text


def test_chat_model_without_pricing_func_is_free():
    model = ChatModel(model_name="m", client_class=MagicMock())
    assert (model.input_cost, model.output_cost) == (0.0, 0.0)


# ---------------------------------------------------------------- ChatModel.__call__


def test_chat_model_call_success_builds_request_and_tracks_cost(sleep_mock):
    pricing = {"fake-model": {"prompt": 0.01, "completion": 0.1}}
    model = make_chat_model(
        [make_completion(["the answer"], prompt_tokens=100, completion_tokens=20)],
        pricing=pricing,
        temperature=0.7,
        max_tokens=55,
    )
    with tracking.set_tracker() as tracker:
        res = model(MESSAGES)

    assert isinstance(res, AIMessage)
    assert res["content"] == "the answer"
    assert res.get("log_probs") is None
    model.client.chat.completions.create.assert_called_once_with(
        model="fake-model",
        messages=MESSAGES,
        n=1,
        temperature=0.7,
        max_completion_tokens=55,
        logprobs=False,
    )
    assert tracker.input_tokens == 100
    assert tracker.output_tokens == 20
    assert tracker.cost == pytest.approx(100 * 0.01 + 20 * 0.1)
    assert model.get_stats() == {"n_retry_llm": 1}
    assert model.success is True
    assert model.error_types == []
    sleep_mock.assert_not_called()


def test_chat_model_call_temperature_override_and_multiple_samples():
    model = make_chat_model([make_completion(["a", "b", "c"])], temperature=0.7)
    res = model(MESSAGES, n_samples=3, temperature=0.0)
    assert [m["content"] for m in res] == ["a", "b", "c"]
    assert all(isinstance(m, AIMessage) for m in res)
    kwargs = model.client.chat.completions.create.call_args.kwargs
    assert kwargs["n"] == 3
    assert kwargs["temperature"] == 0.0


def test_chat_model_call_without_tracker_instance_does_not_fail():
    tracking.TRACKER.instance = None
    model = make_chat_model([make_completion()])
    assert model(MESSAGES)["content"] == "hello"


def test_chat_model_call_retries_on_openai_error_then_succeeds(sleep_mock):
    request = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
    rate_limit = openai.RateLimitError(
        "Rate limit reached. Please try again in 9s.",
        response=httpx.Response(429, request=request),
        body=None,
    )
    model = make_chat_model([rate_limit, make_completion(["ok"])], max_retry=3)
    res = model(MESSAGES)
    assert res["content"] == "ok"
    assert model.retries == 2
    assert model.success is True
    assert model.error_types == ["Rate limit reached. Please try again in 9s."]
    sleep_mock.assert_called_once_with(9.0)


def test_chat_model_call_missing_usage_is_retried_as_openrouter_error(sleep_mock):
    model = make_chat_model([make_completion(usage=False), make_completion(["fine"])], max_retry=2)
    assert model(MESSAGES)["content"] == "fine"
    assert model.retries == 2
    assert "does not contain usage information" in model.error_types[0]
    sleep_mock.assert_called_once_with(3)


def test_chat_model_call_raises_retry_error_after_exhausting_retries(sleep_mock):
    errors = [openai.OpenAIError(f"failure {i}") for i in range(3)]
    model = make_chat_model(errors, max_retry=3)
    with pytest.raises(RetryError, match="after 3 retries\nLast error: failure 2"):
        model(MESSAGES)
    assert model.retries == 3
    assert model.success is False
    assert model.error_types == ["failure 0", "failure 1", "failure 2"]
    assert sleep_mock.call_count == 3
    assert model.get_stats() == {"n_retry_llm": 3}


def test_chat_model_call_non_openai_error_propagates_without_retry(sleep_mock):
    model = make_chat_model([KeyError("bad"), make_completion()], max_retry=3)
    with pytest.raises(KeyError):
        model(MESSAGES)
    assert model.retries == 1
    sleep_mock.assert_not_called()


@pytest.mark.xfail(
    strict=True,
    reason="BUG: ChatModel reads choices[0].log_probs but the OpenAI SDK field is `logprobs`",
)
def test_chat_model_call_returns_log_probs():
    completion = make_completion(["x"])
    model = make_chat_model([completion], log_probs=True)
    res = model(MESSAGES)
    assert res["log_probs"] == completion.choices[0].logprobs


# ---------------------------------------------------------------- ChatModel subclasses


def test_openai_chat_model(openai_cls, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    pricing = MagicMock(return_value={"gpt-x": {"prompt": 1e-6, "completion": 2e-6}})
    monkeypatch.setattr(tracking, "get_pricing_litellm", pricing)

    model = OpenAIChatModel(model_name="gpt-x", max_tokens=None, log_probs=True)

    pricing.assert_called_once_with(model_name="gpt-x")
    openai_cls.assert_called_once_with(api_key="sk-env")
    assert model.max_tokens is openai.NOT_GIVEN
    assert model.log_probs is True
    assert (model.input_cost, model.output_cost) == (1e-6, 2e-6)


def test_openrouter_chat_model(openai_cls, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-env")
    monkeypatch.setattr(
        tracking,
        "get_pricing_openrouter",
        MagicMock(return_value={"meta/llama": {"prompt": 0.5, "completion": 1.5}}),
    )
    model = OpenRouterChatModel(model_name="meta/llama", max_tokens=12)
    openai_cls.assert_called_once_with(api_key="or-env", base_url="https://openrouter.ai/api/v1")
    assert model.max_tokens == 12
    assert (model.input_cost, model.output_cost) == (0.5, 1.5)


@pytest.mark.parametrize(
    "env, missing",
    [
        ({"AZURE_OPENAI_ENDPOINT": "https://az"}, "AZURE_OPENAI_API_KEY"),
        ({"AZURE_OPENAI_API_KEY": "az-key"}, "AZURE_OPENAI_ENDPOINT"),
    ],
)
def test_azure_chat_model_requires_env(openai_cls, monkeypatch, env, missing):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(AssertionError, match=missing):
        AzureChatModel(model_name="gpt-4o")
    openai_cls.assert_not_called()


def test_azure_chat_model_builds_client(openai_cls, monkeypatch, caplog):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://my.azure.endpoint")
    monkeypatch.setattr(
        tracking,
        "get_pricing_openai",
        MagicMock(return_value={"gpt-4o": {"prompt": 2.0, "completion": 4.0}}),
    )
    with caplog.at_level(logging.INFO):
        model = AzureChatModel(model_name="gpt-4o", deployment_name="legacy-deployment")

    openai_cls.assert_called_once_with(
        api_key="az-key",
        base_url="https://my.azure.endpoint",
        default_query={"api-version": "preview"},
    )
    assert model.api_key == "az-key"
    assert (model.input_cost, model.output_cost) == (2.0, 4.0)
    assert "Deployment name is deprecated" in caplog.text


def test_vllm_chat_model(openai_cls):
    model = VLLMChatModel(model_name="local", api_key="k", n_retry_server=6)
    assert model.max_retry == 6
    assert (model.input_cost, model.output_cost) == (0.0, 0.0)
    openai_cls.assert_called_once_with(api_key="k", base_url="http://0.0.0.0:8000/v1")


@pytest.fixture
def hf_patches(monkeypatch):
    inference_client = MagicMock(name="InferenceClient")
    monkeypatch.setattr(chat_api, "InferenceClient", inference_client)
    with patch("agentlab.llm.huggingface_utils.AutoTokenizer") as tok:
        tok.from_pretrained.return_value = MagicMock(name="tokenizer")
        yield inference_client, tok


def test_huggingface_url_chat_model_uses_tgi_token_env(hf_patches, monkeypatch, caplog):
    inference_client, tok = hf_patches
    monkeypatch.setenv("TGI_TOKEN", "tgi-env")
    with caplog.at_level(logging.WARNING):
        model = HuggingFaceURLChatModel(
            model_name="org/model",
            base_model_name="org/base",
            model_url="http://tgi",
            temperature=1e-4,
            max_new_tokens=33,
            log_probs=True,
        )
    tok.from_pretrained.assert_called_once_with("org/base")
    inference_client.assert_called_once_with(model="http://tgi", token="tgi-env")
    assert model.llm.func is inference_client.return_value.text_generation
    assert model.llm.keywords == {"max_new_tokens": 33, "details": True}
    assert model.temperature == 1e-4
    assert "temperature is too low" in caplog.text


def test_huggingface_url_chat_model_explicit_token(hf_patches, caplog):
    inference_client, _ = hf_patches
    with caplog.at_level(logging.WARNING):
        HuggingFaceURLChatModel(
            model_name="org/model", base_model_name=None, model_url="http://tgi", token="t"
        )
    inference_client.assert_called_once_with(model="http://tgi", token="t")
    assert "temperature is too low" not in caplog.text


# ---------------------------------------------------------------- AnthropicChatModel


def make_anthropic_response(text="claude says hi", input_tokens=8, output_tokens=3):
    return MagicMock(
        content=[MagicMock(text=text)],
        usage=MagicMock(input_tokens=input_tokens, output_tokens=output_tokens),
    )


@pytest.fixture
def anthropic_cls(monkeypatch):
    cls = MagicMock(name="Anthropic")
    monkeypatch.setattr(chat_api.anthropic, "Anthropic", cls)
    return cls


def test_anthropic_model_reads_env_key(anthropic_cls, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-env")
    model = AnthropicChatModel(model_name="claude")
    anthropic_cls.assert_called_once_with(api_key="ant-env")
    assert model.client is anthropic_cls.return_value


def test_anthropic_model_converts_messages_and_tracks_usage(anthropic_cls):
    model = AnthropicChatModel(model_name="claude", api_key="k", temperature=0.2, max_tokens=64)
    model.client.messages.create.return_value = make_anthropic_response()
    messages = [
        make_system_message("be nice"),
        make_user_message("hello"),
        make_assistant_message("hi"),
        make_user_message("again"),
    ]
    with tracking.set_tracker() as tracker:
        res = model(messages)

    assert res == AIMessage("claude says hi")
    model.client.messages.create.assert_called_once_with(
        model="claude",
        messages=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "again"},
        ],
        max_tokens=64,
        temperature=0.2,
        system="be nice",
    )
    assert (tracker.input_tokens, tracker.output_tokens, tracker.cost) == (8, 3, 0)


def test_anthropic_model_without_system_message_and_temperature_override(anthropic_cls):
    model = AnthropicChatModel(model_name="claude", api_key="k", temperature=0.2)
    model.client.messages.create.return_value = make_anthropic_response("ok")
    assert model([make_user_message("q")], temperature=0.9)["content"] == "ok"
    kwargs = model.client.messages.create.call_args.kwargs
    assert "system" not in kwargs
    assert kwargs["temperature"] == 0.9


def test_anthropic_model_retries_then_succeeds(anthropic_cls, sleep_mock, caplog):
    model = AnthropicChatModel(model_name="claude", api_key="k", max_retry=3)
    model.client.messages.create.side_effect = [
        RuntimeError("overloaded"),
        make_anthropic_response("second time lucky"),
    ]
    with caplog.at_level(logging.WARNING):
        res = model([make_user_message("q")])
    assert res["content"] == "second time lucky"
    sleep_mock.assert_called_once_with(60)
    assert "Anthropic API error (attempt 1): overloaded" in caplog.text


def test_anthropic_model_raises_last_error_after_max_retry(anthropic_cls, sleep_mock):
    model = AnthropicChatModel(model_name="claude", api_key="k", max_retry=2)
    model.client.messages.create.side_effect = [RuntimeError("e1"), RuntimeError("e2")]
    with pytest.raises(RuntimeError, match="e2"):
        model([make_user_message("q")])
    assert model.client.messages.create.call_count == 2
    sleep_mock.assert_called_once_with(60)


@pytest.mark.xfail(
    strict=True,
    reason="BUG: AnthropicChatModel calls TRACKER.instance even when it is None (after set_tracker exits)",
)
def test_anthropic_model_with_none_tracker_instance(anthropic_cls, sleep_mock):
    with tracking.set_tracker():
        pass  # leaves TRACKER.instance = None, as after any finished agent step
    model = AnthropicChatModel(model_name="claude", api_key="k", max_retry=2)
    model.client.messages.create.return_value = make_anthropic_response("ok")
    assert model([make_user_message("q")])["content"] == "ok"
    sleep_mock.assert_not_called()
