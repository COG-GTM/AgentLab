from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agentlab.llm import tracking
from agentlab.llm.tracking import (
    LLMTracker,
    Stats,
    TrackAPIPricingMixin,
    _remove_version_suffix,
    cost_tracker_decorator,
    get_pricing_anthropic,
    get_pricing_litellm,
    get_pricing_openai,
    get_pricing_openrouter,
    set_tracker,
)


class Usage:
    """Mimics pydantic usage objects: attribute access + iteration over (key, value)."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __iter__(self):
        return iter(self.__dict__.items())


class DummyModel(TrackAPIPricingMixin):
    def __init__(self, model_name="dummy-model", pricing_api="openai", response=None):
        self.model_name = model_name
        self.response = response
        self.init_pricing_tracker(pricing_api=pricing_api)

    def _call_api(self, *args, **kwargs):
        return self.response

    def _parse_response(self, response):
        return "parsed"


@pytest.fixture
def no_litellm_info():
    with patch.object(tracking, "get_model_info", side_effect=Exception("unknown model")) as m:
        yield m


def test_llm_tracker_accumulates_with_suffix():
    tracker = LLMTracker(suffix="agent")
    tracker(10, 5, 0.1)
    tracker(1, 1, 0.2)
    assert tracker.stats == {
        "input_tokens_agent": 11,
        "output_tokens_agent": 6,
        "cost_agent": pytest.approx(0.3),
    }
    other = LLMTracker()
    other.add_tracker(tracker)
    assert other.stats == {"input_tokens": 11, "output_tokens": 6, "cost": pytest.approx(0.3)}
    assert "input_tokens=11" in repr(other)


def test_set_tracker_nested_propagates_to_parent():
    with set_tracker("outer") as outer:
        outer(1, 1, 1.0)
        with set_tracker("inner") as inner:
            inner(2, 3, 0.5)
        assert inner.stats["cost_inner"] == 0.5
    assert outer.stats == {"input_tokens_outer": 3, "output_tokens_outer": 4, "cost_outer": 1.5}


def test_cost_tracker_decorator_updates_agent_info_stats():
    class Agent:
        @cost_tracker_decorator
        def get_action(self, obs):
            tracking.TRACKER.instance(4, 2, 0.25)
            return "noop()", {"stats": {"existing": 1}}

    action, info = Agent().get_action({})
    assert action == "noop()"
    assert info["stats"] == {"existing": 1, "input_tokens": 4, "output_tokens": 2, "cost": 0.25}


@pytest.mark.parametrize(
    "name, expected",
    [
        ("anthropic.claude-3-sonnet-v1:0", "claude-3-sonnet"),
        ("anthropic.claude-v2", "claude"),
        ("claude-instant-v1.2", "claude-instant"),
        ("gpt-4o", "gpt-4o"),
    ],
)
def test_remove_version_suffix(name, expected):
    assert _remove_version_suffix(name) == expected


def test_get_pricing_openai_pairs_prompt_and_completion():
    fake = SimpleNamespace(
        MODEL_COST_PER_1K_TOKENS={"gpt-x": 1.0, "gpt-x-completion": 2.0, "lonely": 3.0}
    )
    with patch.object(tracking, "openai_info", fake):
        assert get_pricing_openai() == {"gpt-x": {"prompt": 0.001, "completion": 0.002}}


def test_get_pricing_openai_without_langchain():
    with patch.object(tracking, "openai_info", None):
        assert get_pricing_openai() == {}


def test_get_pricing_anthropic():
    fake = SimpleNamespace(
        MODEL_COST_PER_1K_INPUT_TOKENS={"anthropic.claude-3-v1:0": 3.0},
        MODEL_COST_PER_1K_OUTPUT_TOKENS={"anthropic.claude-3-v1:0": 15.0, "anthropic.other": 1.0},
    )
    with patch.object(tracking, "bedrock_anthropic_callback", fake):
        assert get_pricing_anthropic() == {
            "claude-3": {"prompt": 0.003, "completion": 0.015},
            "other": {"completion": 0.001},
        }


def test_get_pricing_anthropic_without_langchain():
    with patch.object(tracking, "bedrock_anthropic_callback", None):
        assert get_pricing_anthropic() == {}


def test_get_pricing_litellm():
    info = {"input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6}
    with patch.object(tracking, "get_model_info", return_value=info):
        assert get_pricing_litellm("m") == {"m": {"prompt": 1e-6, "completion": 2e-6}}


def test_get_pricing_litellm_unknown_model(no_litellm_info):
    assert get_pricing_litellm("m") == {"m": {"prompt": 0.0, "completion": 0.0}}


def test_get_pricing_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    response = MagicMock(status_code=200)
    response.json.return_value = {
        "data": [{"id": "org/model", "pricing": {"prompt": "0.1", "completion": "0.2"}}]
    }
    get_pricing_openrouter.cache_clear()
    try:
        with patch.object(tracking.requests, "get", return_value=response) as get:
            assert get_pricing_openrouter() == {"org/model": {"prompt": 0.1, "completion": 0.2}}
        assert get.call_args.kwargs["headers"] == {"Authorization": "Bearer fake-key"}
    finally:
        get_pricing_openrouter.cache_clear()


def test_get_pricing_openrouter_http_error(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    get_pricing_openrouter.cache_clear()
    try:
        with patch.object(tracking.requests, "get", return_value=MagicMock(status_code=500)):
            with pytest.raises(ValueError, match="Failed to get model metadata"):
                get_pricing_openrouter()
    finally:
        get_pricing_openrouter.cache_clear()


def test_get_pricing_openrouter_requires_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    get_pricing_openrouter.cache_clear()
    with pytest.raises(AssertionError):
        get_pricing_openrouter()


def test_set_pricing_attributes_from_provider():
    with patch.object(
        tracking, "get_pricing_openai", return_value={"m": {"prompt": "1", "completion": "2"}}
    ):
        model = DummyModel(model_name="m", pricing_api="openai")
    assert (model.input_cost, model.output_cost) == (1.0, 2.0)
    assert isinstance(model.stats, Stats)


def test_set_pricing_attributes_falls_back_to_litellm():
    info = {"input_cost_per_token": 0.5, "output_cost_per_token": 0.25}
    with patch.object(tracking, "get_model_info", return_value=info):
        model = DummyModel(pricing_api="unsupported")
    assert (model.input_cost, model.output_cost) == (0.5, 0.25)


def test_set_pricing_attributes_unknown_everywhere(no_litellm_info):
    with patch.object(tracking, "get_pricing_anthropic", return_value={}):
        model = DummyModel(pricing_api="anthropic")
    assert (model.input_cost, model.output_cost) == (0.0, 0.0)


@pytest.fixture
def priced_model(no_litellm_info):
    model = DummyModel(pricing_api=None)
    model.input_cost, model.output_cost = 1.0, 2.0
    return model


@pytest.mark.parametrize(
    "response, expected",
    [
        (SimpleNamespace(usage=SimpleNamespace(input_tokens=3, output_tokens=4)), (3, 4)),
        (SimpleNamespace(usage=SimpleNamespace(prompt_tokens=5, completion_tokens=6)), (5, 6)),
        ({"usage": {"prompt_tokens": 7, "completion_tokens": 8}}, (7, 8)),
        ({"usage": {"input_tokens": 1}}, (0, 0)),
        (SimpleNamespace(), (0, 0)),
    ],
)
def test_get_tokens_counts_from_response(priced_model, response, expected):
    assert priced_model.get_tokens_counts_from_response(response) == expected


def test_update_pricing_tracker(priced_model):
    response = SimpleNamespace(usage=SimpleNamespace(input_tokens=10, output_tokens=5))
    with set_tracker() as tracker:
        priced_model.update_pricing_tracker(response)
    assert tracker.stats == {"input_tokens": 10, "output_tokens": 5, "cost": 20.0}


def test_effective_cost_anthropic(priced_model):
    priced_model._pricing_api = "anthropic"
    usage = SimpleNamespace(
        input_tokens=100, output_tokens=10, cache_input_tokens=50, cache_creation_input_tokens=20
    )
    # 100*1 + 10*2 + 50*0.1 + 20*1.25
    assert priced_model.get_effective_cost(SimpleNamespace(usage=usage)) == pytest.approx(150)


def test_effective_cost_openai_chat_completion(priced_model):
    priced_model._pricing_api = "openai"
    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=10,
        prompt_tokens_details=SimpleNamespace(cached_tokens=40),
    )
    # 60 new * 1 + 40 cached * 0.5 + 10 * 2
    assert priced_model.get_effective_cost(SimpleNamespace(usage=usage)) == pytest.approx(100)
    usage.prompt_tokens_details = None
    assert priced_model.get_effective_cost(SimpleNamespace(usage=usage)) == pytest.approx(120)


def test_effective_cost_openai_responses_api(priced_model):
    priced_model._pricing_api = "openai"
    usage = SimpleNamespace(
        input_tokens=100, output_tokens=10, input_tokens_details=SimpleNamespace(cached_tokens=20)
    )
    assert priced_model.get_effective_cost(SimpleNamespace(usage=usage)) == pytest.approx(110)
    usage.input_tokens_details = None
    assert priced_model.get_effective_cost(SimpleNamespace(usage=usage)) == pytest.approx(120)


def test_effective_cost_openai_no_usage(priced_model):
    priced_model._pricing_api = "openai"
    assert priced_model.get_effective_cost(SimpleNamespace()) == 0.0


def test_effective_cost_litellm(priced_model):
    priced_model._pricing_api = "litellm"
    with patch.object(tracking, "completion_cost", return_value=0.42) as cost:
        assert priced_model.get_effective_cost("resp") == 0.42
    cost.assert_called_once_with("resp")


def test_effective_cost_unsupported_provider(priced_model):
    assert priced_model.get_effective_cost(SimpleNamespace()) == 0.0


def test_call_updates_stats_and_tracker(no_litellm_info):
    usage = Usage(
        prompt_tokens=100,
        completion_tokens=10,
        total_tokens=110,
        prompt_tokens_details=SimpleNamespace(cached_tokens=40),
        model="ignored-non-numeric",
    )
    model = DummyModel(pricing_api="openai", response=SimpleNamespace(usage=usage))
    model.input_cost, model.output_cost = 1.0, 2.0
    with set_tracker() as tracker:
        assert model() == "parsed"
        assert model() == "parsed"
    stats = model.stats.stats_dict
    assert stats["usage_prompt_tokens"] == 200
    assert stats["usage_completion_tokens"] == 20
    assert stats["usage_cached_tokens"] == 80
    assert stats["n_api_calls"] == 2
    assert stats["effective_cost"] == pytest.approx(200)
    assert "usage_model" not in stats
    assert tracker.stats["cost"] == pytest.approx(240)

    model.reset_stats()
    assert dict(model.stats.stats_dict) == {}


def test_call_with_responses_api_details(no_litellm_info):
    usage = Usage(
        input_tokens=10, output_tokens=1, input_tokens_details=SimpleNamespace(cached_tokens=4)
    )
    model = DummyModel(pricing_api=None, response=SimpleNamespace(usage=usage))
    model()
    assert model.stats.stats_dict["usage_cached_tokens"] == 4
    assert model.stats.stats_dict["effective_cost"] == 0.0


def test_stats_increment():
    stats = Stats()
    stats.increment_stats_dict({"a": 1, "b": 2.5})
    stats.increment_stats_dict({"a": 1})
    assert dict(stats.stats_dict) == {"a": 2, "b": 2.5}
