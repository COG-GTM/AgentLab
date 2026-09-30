import os
import time
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from anthropic.types import Usage as AnthropicUsage
from openai.types.completion_usage import CompletionUsage, PromptTokensDetails
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseUsage,
)

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


INPUT_COST = 2e-6
OUTPUT_COST = 1e-5


def make_pricing_mixin(pricing_api):
    mixin = tracking.TrackAPIPricingMixin()
    mixin._pricing_api = pricing_api
    mixin.input_cost = INPUT_COST
    mixin.output_cost = OUTPUT_COST
    return mixin


def make_chat_completion_usage(prompt_tokens, completion_tokens, cached_tokens):
    details = None if cached_tokens is None else PromptTokensDetails(cached_tokens=cached_tokens)
    return CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=details,
    )


def make_response_api_usage(input_tokens, output_tokens, cached_tokens):
    return ResponseUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=cached_tokens),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )


def test_anthropic_effective_cost_with_cache_read_and_write():
    mixin = make_pricing_mixin("anthropic")
    usage = AnthropicUsage(
        input_tokens=100,
        output_tokens=50,
        cache_read_input_tokens=1000,
        cache_creation_input_tokens=200,
    )
    response = SimpleNamespace(usage=usage)

    # 100 new input + 50 output + 1000 cache reads at 0.1x + 200 cache writes at 1.25x
    expected = 100 * 2e-6 + 50 * 1e-5 + 1000 * 2e-7 + 200 * 2.5e-6
    assert expected == pytest.approx(1.4e-3)
    assert mixin.get_effective_cost_from_antrophic_api(response) == pytest.approx(expected)
    assert mixin.get_effective_cost(response) == pytest.approx(expected)


def test_anthropic_effective_cost_cache_read_is_discounted():
    mixin = make_pricing_mixin("anthropic")
    uncached = SimpleNamespace(
        usage=AnthropicUsage(
            input_tokens=1000,
            output_tokens=0,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
    )
    cached = SimpleNamespace(
        usage=AnthropicUsage(
            input_tokens=0,
            output_tokens=0,
            cache_read_input_tokens=1000,
            cache_creation_input_tokens=0,
        )
    )

    assert mixin.get_effective_cost(uncached) == pytest.approx(1000 * INPUT_COST)
    assert mixin.get_effective_cost(cached) == pytest.approx(
        1000 * INPUT_COST * tracking.ANTHROPIC_CACHE_PRICING_FACTOR["cache_read_tokens"]
    )


def test_anthropic_effective_cost_without_cache_fields():
    mixin = make_pricing_mixin("anthropic")
    response = SimpleNamespace(usage=SimpleNamespace(input_tokens=100, output_tokens=50))

    assert mixin.get_effective_cost(response) == pytest.approx(100 * 2e-6 + 50 * 1e-5)


def test_openai_chat_completion_effective_cost_with_cached_tokens():
    mixin = make_pricing_mixin("openai")
    response = SimpleNamespace(
        usage=make_chat_completion_usage(
            prompt_tokens=1000, completion_tokens=100, cached_tokens=600
        )
    )

    # prompt_tokens includes cached tokens: 400 new at 1x, 600 cached at 0.5x
    expected = 400 * 2e-6 + 600 * 1e-6 + 100 * 1e-5
    assert expected == pytest.approx(2.4e-3)
    assert mixin.get_effective_cost_from_openai_api(response) == pytest.approx(expected)
    assert mixin.get_effective_cost(response) == pytest.approx(expected)


def test_openai_chat_completion_effective_cost_without_prompt_tokens_details():
    mixin = make_pricing_mixin("openai")
    response = SimpleNamespace(
        usage=make_chat_completion_usage(
            prompt_tokens=1000, completion_tokens=100, cached_tokens=None
        )
    )

    assert mixin.get_effective_cost(response) == pytest.approx(1000 * 2e-6 + 100 * 1e-5)


def test_openai_response_api_effective_cost_with_cached_tokens():
    mixin = make_pricing_mixin("openai")
    usage = make_response_api_usage(input_tokens=1000, output_tokens=100, cached_tokens=600)
    assert not hasattr(usage, "prompt_tokens_details")
    response = SimpleNamespace(usage=usage)

    expected = 400 * 2e-6 + 600 * 1e-6 + 100 * 1e-5
    assert mixin.get_effective_cost_from_openai_api(response) == pytest.approx(expected)
    assert mixin.get_effective_cost(response) == pytest.approx(expected)


def test_openai_effective_cost_without_usage_is_zero():
    mixin = make_pricing_mixin("openai")

    assert mixin.get_effective_cost(SimpleNamespace(usage=None)) == 0.0
    assert mixin.get_effective_cost(SimpleNamespace()) == 0.0


def test_litellm_effective_cost_delegates_to_completion_cost():
    mixin = make_pricing_mixin("litellm")
    response = SimpleNamespace(usage=None)

    with patch.object(tracking, "completion_cost", return_value=0.0123) as mock_completion_cost:
        assert mixin.get_effective_cost(response) == 0.0123
    mock_completion_cost.assert_called_once_with(response)


@pytest.mark.parametrize("pricing_api", ["openrouter", "unknown", None])
def test_unsupported_provider_effective_cost_is_zero(pricing_api):
    mixin = make_pricing_mixin(pricing_api)
    response = SimpleNamespace(
        usage=make_chat_completion_usage(
            prompt_tokens=1000, completion_tokens=100, cached_tokens=600
        )
    )

    assert mixin.get_effective_cost(response) == 0.0


class FakePricedModel(tracking.TrackAPIPricingMixin):
    def __init__(self, pricing_api, response):
        self._pricing_api = pricing_api
        self.input_cost = INPUT_COST
        self.output_cost = OUTPUT_COST
        self._response = response
        self.reset_stats()

    def _call_api(self, *args, **kwargs):
        return self._response

    def _parse_response(self, response):
        return "parsed"


def test_call_accumulates_cache_aware_effective_cost_in_stats():
    response = SimpleNamespace(
        usage=make_chat_completion_usage(
            prompt_tokens=1000, completion_tokens=100, cached_tokens=600
        )
    )
    model = FakePricedModel("openai", response)

    with tracking.set_tracker() as tracker:
        assert model() == "parsed"
        assert model() == "parsed"

    stats = model.stats.stats_dict
    assert stats["n_api_calls"] == 2
    assert stats["usage_cached_tokens"] == 1200
    assert stats["effective_cost"] == pytest.approx(2 * 2.4e-3)
    # The tracker cost ignores caching and bills all prompt tokens at the full input price.
    assert tracker.stats["cost"] == pytest.approx(2 * (1000 * 2e-6 + 100 * 1e-5))
