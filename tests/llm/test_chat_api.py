import os
from types import SimpleNamespace

import openai
import pytest

import agentlab.llm.chat_api as chat_api
import agentlab.llm.tracking as tracking
from agentlab.llm.chat_api import (
    AnthropicModelArgs,
    AzureModelArgs,
    ChatModel,
    OpenAIModelArgs,
    RetryError,
    _extract_wait_time,
    handle_error,
    make_system_message,
    make_user_message,
)

# TODO(optimass): figure out a good model for all tests


if "AGENTLAB_LOCAL_TEST" in os.environ:
    skip_tests = os.environ["AGENTLAB_LOCAL_TEST"] != "1"
else:
    skip_tests = False


@pytest.mark.pricy
@pytest.mark.skipif(skip_tests, reason="Skipping on remote as Azure is pricy")
@pytest.mark.skipif(
    not os.getenv("AZURE_OPENAI_API_KEY"), reason="Skipping as Azure API key not set"
)
def test_api_model_args_azure():
    model_args = AzureModelArgs(
        model_name="gpt-4.1-nano",
        deployment_name="gpt-4.1-nano",
        max_total_tokens=8192,
        max_input_tokens=8192 - 512,
        max_new_tokens=512,
        temperature=1e-1,
    )
    model = model_args.make_model()

    messages = [
        make_system_message("You are an helpful virtual assistant"),
        make_user_message("Give the third prime number"),
    ]
    answer = model(messages)

    assert "5" in answer.get("content")


@pytest.mark.pricy
@pytest.mark.skipif(skip_tests, reason="Skipping on remote as Azure is pricy")
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Skipping as OpenAI API key not set")
def test_api_model_args_openai():
    model_args = OpenAIModelArgs(
        model_name="gpt-4o-mini",
        max_total_tokens=8192,
        max_input_tokens=8192 - 512,
        max_new_tokens=512,
        temperature=1e-1,
    )
    model = model_args.make_model()

    messages = [
        make_system_message("You are an helpful virtual assistant"),
        make_user_message("Give the third prime number"),
    ]
    answer = model(messages)

    assert "5" in answer.get("content")


@pytest.mark.pricy
@pytest.mark.skipif(skip_tests, reason="Skipping on remote as Anthropic is pricy")
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="Skipping as Anthropic API key not set"
)
def test_api_model_args_anthropic():
    model_args = AnthropicModelArgs(
        model_name="claude-3-haiku-20240307",
        max_total_tokens=8192,
        max_input_tokens=8192 - 512,
        max_new_tokens=512,
        temperature=1e-1,
    )
    model = model_args.make_model()

    messages = [
        make_system_message("You are an helpful virtual assistant"),
        make_user_message("Give the third prime number. Just the number, no explanation."),
    ]
    answer = model(messages)
    assert "5" in answer.get("content")


class _FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _FakeClient:
    def __init__(self, api_key=None, responses=(), **kwargs):
        self.api_key = api_key
        self.chat = SimpleNamespace(completions=_FakeCompletions(responses))


def _make_completion(content="hello", prompt_tokens=10, completion_tokens=5):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def _make_chat_model(responses, monkeypatch, input_cost=0.1, output_cost=0.2, max_retry=4):
    """ChatModel wired to a fake client, with sleeps recorded instead of performed."""
    sleeps = []
    monkeypatch.setattr(chat_api.time, "sleep", sleeps.append)
    model = ChatModel(
        model_name="fake-model",
        api_key="fake-key",
        max_retry=max_retry,
        min_retry_wait_time=1,
        client_class=_FakeClient,
        client_args={"responses": responses},
        pricing_func=lambda: {
            "fake-model": {"prompt": str(input_cost), "completion": str(output_cost)}
        },
    )
    return model, sleeps


def test_extract_wait_time_uses_parsed_value():
    message = "Rate limit reached. Please try again in 90.5s. Contact us."
    assert _extract_wait_time(message, min_retry_wait_time=60) == 90.5


def test_extract_wait_time_clamps_to_min():
    message = "Rate limit reached. Please try again in 2s."
    assert _extract_wait_time(message, min_retry_wait_time=60) == 60


def test_extract_wait_time_without_match():
    assert _extract_wait_time("Some other error", min_retry_wait_time=42) == 42


def test_handle_error_reraises_non_openai_error():
    error = ValueError("not an openai error")
    with pytest.raises(ValueError):
        handle_error(error, itr=0, min_retry_wait_time=1, max_retry=4)


def test_handle_error_sleeps_and_returns_error_type(monkeypatch):
    sleeps = []
    monkeypatch.setattr(chat_api.time, "sleep", sleeps.append)
    error = openai.OpenAIError("Rate limit reached. Please try again in 90s.")

    error_type = handle_error(error, itr=0, min_retry_wait_time=1, max_retry=4)

    assert error_type == "Rate limit reached. Please try again in 90s."
    assert sleeps == [90.0]


def test_chat_model_retries_then_succeeds(monkeypatch):
    responses = [
        openai.OpenAIError("Rate limit reached. Please try again in 3s."),
        openai.OpenAIError("Rate limit reached. Please try again in 3s."),
        _make_completion(content="the answer"),
    ]
    model, sleeps = _make_chat_model(responses, monkeypatch)

    answer = model([make_user_message("hi")])

    assert answer.get("content") == "the answer"
    assert model.success
    assert model.retries == 3
    assert len(model.error_types) == 2
    assert sleeps == [3.0, 3.0]


def test_chat_model_raises_retry_error_when_retries_exhausted(monkeypatch):
    responses = [openai.OpenAIError("boom") for _ in range(3)]
    model, sleeps = _make_chat_model(responses, monkeypatch, max_retry=3)

    with pytest.raises(RetryError, match="after 3 retries"):
        model([make_user_message("hi")])

    assert not model.success
    assert model.retries == 3
    assert sleeps == [1, 1, 1]


def test_chat_model_retries_when_usage_is_missing(monkeypatch):
    no_usage = _make_completion()
    no_usage.usage = None
    model, _ = _make_chat_model([no_usage, _make_completion()], monkeypatch)

    model([make_user_message("hi")])

    assert model.retries == 2
    assert isinstance(model.error_types[0], str)


def test_chat_model_cost_is_tracked(monkeypatch):
    completion = _make_completion(prompt_tokens=10, completion_tokens=5)
    model, _ = _make_chat_model([completion], monkeypatch, input_cost=0.1, output_cost=0.2)

    with tracking.set_tracker() as tracker:
        model([make_user_message("hi")])

    assert tracker.input_tokens == 10
    assert tracker.output_tokens == 5
    assert tracker.cost == pytest.approx(10 * 0.1 + 5 * 0.2)


def test_chat_model_returns_all_samples(monkeypatch):
    completion = _make_completion()
    completion.choices = [
        SimpleNamespace(message=SimpleNamespace(content="a")),
        SimpleNamespace(message=SimpleNamespace(content="b")),
    ]
    model, _ = _make_chat_model([completion], monkeypatch)

    answers = model([make_user_message("hi")], n_samples=2)

    assert [a.get("content") for a in answers] == ["a", "b"]


if __name__ == "__main__":
    test_api_model_args_anthropic()
