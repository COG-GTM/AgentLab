import os
import time
from functools import partial

import pytest

import agentlab.llm.tracking as tracking
from agentlab.llm.chat_api import (
    AzureChatModel,
    OpenAIChatModel,
    OpenRouterChatModel,
    make_system_message,
    make_user_message,
)


def test_get_action_decorator():
    action, agent_info = tracking.cost_tracker_decorator(lambda x, y: call_llm())(None, None)
    assert action == "action"
    assert agent_info["stats"] == {
        "input_tokens": 1,
        "output_tokens": 1,
        "cost": 1.0,
    }


OPENROUTER_API_KEY_AVAILABLE = os.environ.get("OPENROUTER_API_KEY") is not None

OPENROUTER_MODELS = (
    "anthropic/claude-3.5-sonnet",
    "meta-llama/llama-3.1-405b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "google/gemini-pro-1.5",
)


@pytest.mark.skipif(not OPENROUTER_API_KEY_AVAILABLE, reason="OpenRouter API key is not available")
def test_get_pricing_openrouter():
    pricing = tracking.get_pricing_openrouter()
    assert isinstance(pricing, dict)
    assert all(isinstance(v, dict) for v in pricing.values())
    for model in OPENROUTER_MODELS:
        assert model in pricing
        assert isinstance(pricing[model], dict)
        assert all(isinstance(v, float) for v in pricing[model].values())


def test_get_pricing_openai():
    pricing = tracking.get_pricing_openai()
    assert isinstance(pricing, dict)
    assert all("prompt" in pricing[model] and "completion" in pricing[model] for model in pricing)
    assert all(isinstance(pricing[model]["prompt"], float) for model in pricing)
    assert all(isinstance(pricing[model]["completion"], float) for model in pricing)


def call_llm():
    if hasattr(tracking.TRACKER, "instance") and isinstance(
        tracking.TRACKER.instance, tracking.LLMTracker
    ):
        tracking.TRACKER.instance(1, 1, 1)
    return "action", {"stats": {}}


def test_tracker():
    with tracking.set_tracker() as tracker:
        _, _ = call_llm()

    assert tracker.stats["cost"] == 1


def test_imbricate_trackers():
    with tracking.set_tracker() as tracker4:
        with tracking.set_tracker() as tracker1:
            _, _ = call_llm()
        with tracking.set_tracker() as tracker3:
            _, _ = call_llm()
            _, _ = call_llm()
            with tracking.set_tracker() as tracker1bis:
                _, _ = call_llm()

    assert tracker1.stats["cost"] == 1
    assert tracker1bis.stats["cost"] == 1
    assert tracker3.stats["cost"] == 3
    assert tracker4.stats["cost"] == 4


def test_threaded_trackers():
    """thread_2 occurs in the middle of thread_1, results should be separate."""
    import threading

    def thread_1(results=None):
        with tracking.set_tracker() as tracker:
            time.sleep(1)
            _, _ = call_llm()
            time.sleep(1)
        results[0] = tracker.stats

    def thread_2(results=None):
        time.sleep(1)
        with tracking.set_tracker() as tracker:
            _, _ = call_llm()
        results[1] = tracker.stats

    results = [None] * 2
    threads = [
        threading.Thread(target=partial(thread_1, results=results)),
        threading.Thread(target=partial(thread_2, results=results)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert all(result["cost"] == 1 for result in results)


OPENAI_API_KEY_AVAILABLE = os.environ.get("OPENAI_API_KEY") is not None


@pytest.mark.pricy
@pytest.mark.skipif(not OPENAI_API_KEY_AVAILABLE, reason="OpenAI API key is not available")
def test_openai_chat_model():
    chat_model = OpenAIChatModel("gpt-4o-mini")
    assert chat_model.input_cost > 0
    assert chat_model.output_cost > 0

    messages = [
        make_system_message("You are an helpful virtual assistant"),
        make_user_message("Give the third prime number"),
    ]
    with tracking.set_tracker() as tracker:
        answer = chat_model(messages)
    assert "5" in answer.get("content")
    assert tracker.stats["cost"] > 0


AZURE_OPENAI_API_KEY_AVAILABLE = (
    os.environ.get("AZURE_OPENAI_API_KEY") is not None
    and os.environ.get("AZURE_OPENAI_ENDPOINT") is not None
)


@pytest.mark.pricy
@pytest.mark.skipif(
    not AZURE_OPENAI_API_KEY_AVAILABLE, reason="Azure OpenAI API key is not available"
)
def test_azure_chat_model():
    chat_model = AzureChatModel(model_name="gpt-4.1-nano", deployment_name="gpt-4.1-nano")
    assert chat_model.input_cost > 0
    assert chat_model.output_cost > 0

    messages = [
        make_system_message("You are an helpful virtual assistant"),
        make_user_message("Give the third prime number"),
    ]
    with tracking.set_tracker() as tracker:
        answer = chat_model(messages)
    assert "5" in answer.get("content")
    assert tracker.stats["cost"] > 0


@pytest.mark.pricy
@pytest.mark.skipif(not OPENROUTER_API_KEY_AVAILABLE, reason="OpenRouter API key is not available")
def test_openrouter_chat_model():
    chat_model = OpenRouterChatModel("openai/gpt-4o-mini")
    assert chat_model.input_cost > 0
    assert chat_model.output_cost > 0

    messages = [
        make_system_message("You are an helpful virtual assistant"),
        make_user_message("Give the third prime number"),
    ]
    with tracking.set_tracker() as tracker:
        answer = chat_model(messages)
    assert "5" in answer.get("content")
    assert tracker.stats["cost"] > 0


class _FakeUsage:
    """Usage object that supports both attribute access and dict()."""

    def __init__(self, **fields):
        self.__dict__.update(fields)

    def __iter__(self):
        return iter(self.__dict__.items())


class _FakeResponse:
    def __init__(self, usage=None):
        if usage is not None:
            self.usage = usage


class _CostModel(tracking.TrackAPIPricingMixin):
    """Minimal model exposing the effective cost logic without hitting any provider."""

    def __init__(self, pricing_api, input_cost, output_cost, response=None):
        self._pricing_api = pricing_api
        self.input_cost = input_cost
        self.output_cost = output_cost
        self._response = response
        self.reset_stats()

    def _call_api(self, *args, **kwargs):
        return self._response

    def _parse_response(self, response):
        return response


def test_effective_cost_anthropic_with_cache_tokens():
    model = _CostModel("anthropic", input_cost=1e-5, output_cost=2e-5)
    usage = _FakeUsage(
        input_tokens=100,
        output_tokens=50,
        cache_read_input_tokens=1000,
        cache_creation_input_tokens=200,
    )

    cost = model.get_effective_cost(_FakeResponse(usage))

    expected = (
        100 * 1e-5
        + 50 * 2e-5
        + 1000 * 1e-5 * tracking.ANTHROPIC_CACHE_PRICING_FACTOR["cache_read_tokens"]
        + 200 * 1e-5 * tracking.ANTHROPIC_CACHE_PRICING_FACTOR["cache_write_tokens"]
    )
    assert cost == pytest.approx(expected)
    # cache tokens must contribute, otherwise cached runs would be reported as cheaper than they are
    assert cost > 100 * 1e-5 + 50 * 2e-5


def test_effective_cost_anthropic_without_cache_tokens():
    model = _CostModel("anthropic", input_cost=1e-5, output_cost=2e-5)
    usage = _FakeUsage(input_tokens=100, output_tokens=50)

    cost = model.get_effective_cost(_FakeResponse(usage))

    assert cost == pytest.approx(100 * 1e-5 + 50 * 2e-5)


def test_effective_cost_openai_chat_completion_with_cached_tokens():
    model = _CostModel("openai", input_cost=1e-5, output_cost=2e-5)
    usage = _FakeUsage(
        prompt_tokens=1000,
        completion_tokens=50,
        prompt_tokens_details=_FakeUsage(cached_tokens=800),
    )

    cost = model.get_effective_cost(_FakeResponse(usage))

    expected = (
        200 * 1e-5
        + 800 * 1e-5 * tracking.OPENAI_CACHE_PRICING_FACTOR["cache_read_tokens"]
        + 50 * 2e-5
    )
    assert cost == pytest.approx(expected)
    # cached input tokens are cheaper than new ones
    assert cost < 1000 * 1e-5 + 50 * 2e-5


def test_effective_cost_openai_chat_completion_without_details():
    model = _CostModel("openai", input_cost=1e-5, output_cost=2e-5)
    usage = _FakeUsage(prompt_tokens=1000, completion_tokens=50, prompt_tokens_details=None)

    cost = model.get_effective_cost(_FakeResponse(usage))

    assert cost == pytest.approx(1000 * 1e-5 + 50 * 2e-5)


def test_effective_cost_openai_response_api_with_cached_tokens():
    model = _CostModel("openai", input_cost=1e-5, output_cost=2e-5)
    usage = _FakeUsage(
        input_tokens=1000,
        output_tokens=50,
        input_tokens_details=_FakeUsage(cached_tokens=400),
    )

    cost = model.get_effective_cost(_FakeResponse(usage))

    expected = (
        600 * 1e-5
        + 400 * 1e-5 * tracking.OPENAI_CACHE_PRICING_FACTOR["cache_read_tokens"]
        + 50 * 2e-5
    )
    assert cost == pytest.approx(expected)


def test_effective_cost_openai_without_usage():
    model = _CostModel("openai", input_cost=1e-5, output_cost=2e-5)

    assert model.get_effective_cost(_FakeResponse()) == 0.0


def test_effective_cost_unsupported_provider():
    model = _CostModel("some-unknown-provider", input_cost=1e-5, output_cost=2e-5)
    usage = _FakeUsage(input_tokens=100, output_tokens=50)

    assert model.get_effective_cost(_FakeResponse(usage)) == 0.0


def test_effective_cost_litellm_delegates_to_completion_cost(monkeypatch):
    model = _CostModel("litellm", input_cost=1e-5, output_cost=2e-5)
    response = _FakeResponse(_FakeUsage(input_tokens=100, output_tokens=50))
    monkeypatch.setattr(tracking, "completion_cost", lambda resp: 0.42)

    assert model.get_effective_cost(response) == 0.42


def test_call_accumulates_effective_cost_in_stats():
    usage = _FakeUsage(
        input_tokens=100,
        output_tokens=50,
        cache_read_input_tokens=1000,
        cache_creation_input_tokens=200,
    )
    model = _CostModel(
        "anthropic", input_cost=1e-5, output_cost=2e-5, response=_FakeResponse(usage)
    )
    expected = model.get_effective_cost(_FakeResponse(usage))

    model()
    model()

    assert model.stats.stats_dict["effective_cost"] == pytest.approx(2 * expected)
    assert model.stats.stats_dict["n_api_calls"] == 2
