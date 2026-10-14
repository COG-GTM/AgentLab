"""Hermetic unit tests for agentlab.llm.tracking (no network, no API keys)."""

import importlib.util
import logging
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import agentlab.llm.tracking as tracking
from agentlab.llm.tracking import (
    ANTHROPIC_CACHE_PRICING_FACTOR,
    OPENAI_CACHE_PRICING_FACTOR,
    LLMTracker,
    Stats,
    TrackAPIPricingMixin,
    set_tracker,
)


@pytest.fixture(autouse=True)
def _reset_thread_local_tracker():
    previous = getattr(tracking.TRACKER, "instance", None)
    tracking.TRACKER.instance = None
    tracking.get_pricing_openrouter.cache_clear()
    yield
    tracking.TRACKER.instance = previous
    tracking.get_pricing_openrouter.cache_clear()


class FakePricedModel(TrackAPIPricingMixin):
    """Minimal concrete user of the mixin with a canned raw response."""

    def __init__(self, model_name="fake-model", pricing_api="openai", response=None):
        self.model_name = model_name
        self._pricing_api = pricing_api
        self.input_cost = 0.0
        self.output_cost = 0.0
        self.reset_stats()
        self._response = response
        self.calls = []

    def _call_api(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self._response

    def _parse_response(self, response):
        return ("parsed", response)


def _priced(pricing_api, input_cost, output_cost):
    model = FakePricedModel(pricing_api=pricing_api)
    model.input_cost = input_cost
    model.output_cost = output_cost
    return model


# --------------------------------------------------------------------------- LLMTracker


class TestLLMTracker:
    def test_default_keys_and_accumulation(self):
        tracker = LLMTracker()
        tracker(10, 5, 0.25)
        tracker(1, 2, 0.5)
        assert tracker.stats == {"input_tokens": 11, "output_tokens": 7, "cost": 0.75}

    def test_suffix_changes_stat_keys(self):
        tracker = LLMTracker(suffix="judge")
        tracker(3, 4, 1.5)
        assert tracker.stats == {
            "input_tokens_judge": 3,
            "output_tokens_judge": 4,
            "cost_judge": 1.5,
        }

    def test_add_tracker_merges_counts(self):
        a, b = LLMTracker(), LLMTracker("x")
        a(1, 1, 1.0)
        b(2, 3, 0.5)
        a.add_tracker(b)
        assert (a.input_tokens, a.output_tokens, a.cost) == (3, 4, 1.5)
        # the merged-in tracker is untouched
        assert (b.input_tokens, b.output_tokens, b.cost) == (2, 3, 0.5)

    def test_repr(self):
        tracker = LLMTracker()
        tracker(2, 3, 0.125)
        assert repr(tracker) == "LLMTracker(input_tokens=2, output_tokens=3, cost=0.125)"


# --------------------------------------------------------------------------- set_tracker


class TestSetTracker:
    def test_installs_and_restores_none(self):
        with set_tracker() as tracker:
            assert tracking.TRACKER.instance is tracker
        assert tracking.TRACKER.instance is None

    def test_creates_instance_attribute_when_missing(self):
        del tracking.TRACKER.instance
        with set_tracker("s") as tracker:
            assert tracker.cost_key == "cost_s"
        assert tracking.TRACKER.instance is None

    def test_nested_trackers_propagate_to_parent(self):
        with set_tracker() as outer:
            outer(1, 1, 1.0)
            with set_tracker("inner") as inner:
                inner(10, 20, 2.0)
                assert tracking.TRACKER.instance is inner
            assert tracking.TRACKER.instance is outer
            assert inner.stats == {
                "input_tokens_inner": 10,
                "output_tokens_inner": 20,
                "cost_inner": 2.0,
            }
        assert (outer.input_tokens, outer.output_tokens, outer.cost) == (11, 21, 3.0)

    def test_restores_previous_on_exception_and_still_merges(self):
        with set_tracker() as outer:
            with pytest.raises(RuntimeError):
                with set_tracker() as inner:
                    inner(5, 5, 0.5)
                    raise RuntimeError("boom")
            assert tracking.TRACKER.instance is outer
        assert outer.cost == 0.5

    def test_threads_do_not_bleed(self):
        barrier = threading.Barrier(2)
        results = {}
        errors = []

        def worker(name, n_calls, cost):
            try:
                with set_tracker() as tracker:
                    barrier.wait(timeout=5)  # both trackers are alive at the same time
                    for _ in range(n_calls):
                        tracking.TRACKER.instance(1, 2, cost)
                    barrier.wait(timeout=5)
                    results[name] = (tracker.stats, tracking.TRACKER.instance is tracker)
                results[name + "_after"] = tracking.TRACKER.instance
            except Exception as e:  # pragma: no cover - surfaced below
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=("a", 3, 1.0)),
            threading.Thread(target=worker, args=("b", 5, 10.0)),
        ]
        with set_tracker() as main_tracker:
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        assert not errors
        assert results["a"] == ({"input_tokens": 3, "output_tokens": 6, "cost": 3.0}, True)
        assert results["b"] == ({"input_tokens": 5, "output_tokens": 10, "cost": 50.0}, True)
        assert results["a_after"] is None and results["b_after"] is None
        assert main_tracker.stats == {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}


# --------------------------------------------------------------------------- decorator


def test_cost_tracker_decorator_with_suffix_updates_stats():
    class Agent:
        def get_action(self, obs):
            tracking.TRACKER.instance(7, 3, 0.75)
            return f"act({obs})", {"stats": {"existing": 1}}

    wrapped = tracking.cost_tracker_decorator(Agent.get_action, suffix="sub")
    action, info = wrapped(Agent(), "o")
    assert action == "act(o)"
    assert info["stats"] == {
        "existing": 1,
        "input_tokens_sub": 7,
        "output_tokens_sub": 3,
        "cost_sub": 0.75,
    }
    assert tracking.TRACKER.instance is None


# --------------------------------------------------------------------------- pricing getters


class TestGetPricingOpenRouter:
    def test_parses_and_casts_pricing(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        response = MagicMock(status_code=200)
        response.json.return_value = {
            "data": [
                {"id": "a/model-1", "pricing": {"prompt": "0.000001", "completion": "0.000002"}},
                {"id": "b/model-2", "pricing": {"prompt": "0", "completion": "1"}},
            ]
        }
        with patch.object(tracking.requests, "get", return_value=response) as get:
            prices = tracking.get_pricing_openrouter()
            # second call is served from functools.cache
            assert tracking.get_pricing_openrouter() is prices
        get.assert_called_once_with(
            "https://openrouter.ai/api/v1/models", headers={"Authorization": "Bearer sk-test"}
        )
        assert prices == {
            "a/model-1": {"prompt": 1e-06, "completion": 2e-06},
            "b/model-2": {"prompt": 0.0, "completion": 1.0},
        }

    def test_non_200_raises(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        with patch.object(tracking.requests, "get", return_value=MagicMock(status_code=500)):
            with pytest.raises(ValueError, match="Failed to get model metadata"):
                tracking.get_pricing_openrouter()

    def test_missing_key_asserts(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with patch.object(tracking.requests, "get") as get:
            with pytest.raises(AssertionError, match="OpenRouter API key is required"):
                tracking.get_pricing_openrouter()
        get.assert_not_called()


class TestGetPricingOpenAI:
    def test_pairs_prompt_and_completion_per_token(self, monkeypatch):
        fake = SimpleNamespace(
            MODEL_COST_PER_1K_TOKENS={
                "gpt-x": 1.0,
                "gpt-x-completion": 2.0,
                "orphan": 5.0,  # no completion entry -> dropped
                "lonely-completion": 3.0,  # no prompt entry -> dropped
            }
        )
        monkeypatch.setattr(tracking, "openai_info", fake)
        assert tracking.get_pricing_openai() == {"gpt-x": {"prompt": 0.001, "completion": 0.002}}

    def test_missing_langchain_returns_empty(self, monkeypatch, caplog):
        monkeypatch.setattr(tracking, "openai_info", None)
        with caplog.at_level(logging.WARNING):
            assert tracking.get_pricing_openai() == {}
        assert "Failed to get OpenAI pricing" in caplog.text


@pytest.mark.parametrize(
    "name, expected",
    [
        ("anthropic.claude-v2", "claude"),
        ("anthropic.claude-v2:1", "claude"),
        ("anthropic.claude-3-sonnet-20240229-v1:0", "claude-3-sonnet-20240229"),
        ("anthropic.claude-instant-v1", "claude-instant"),
        ("claude-3-haiku-v1.5", "claude-3-haiku"),
        ("claude-3-haiku", "claude-3-haiku"),
        ("model-v2-mini", "model-v2-mini"),  # version only stripped at the end
    ],
)
def test_remove_version_suffix(name, expected):
    assert tracking._remove_version_suffix(name) == expected


class TestGetPricingAnthropic:
    def test_merges_input_and_output_tables(self, monkeypatch):
        fake = SimpleNamespace(
            MODEL_COST_PER_1K_INPUT_TOKENS={
                "anthropic.claude-v2": 8.0,
                "anthropic.claude-3-v1:0": 3.0,
            },
            MODEL_COST_PER_1K_OUTPUT_TOKENS={
                "anthropic.claude-v2": 24.0,
                "anthropic.claude-3-v1:0": 15.0,
                "anthropic.output-only-v1": 1.0,
            },
        )
        monkeypatch.setattr(tracking, "bedrock_anthropic_callback", fake)
        assert tracking.get_pricing_anthropic() == {
            "claude": {"prompt": 0.008, "completion": 0.024},
            "claude-3": {"prompt": 0.003, "completion": 0.015},
            "output-only": {"completion": 0.001},
        }

    def test_missing_langchain_returns_empty(self, monkeypatch, caplog):
        monkeypatch.setattr(tracking, "bedrock_anthropic_callback", None)
        with caplog.at_level(logging.WARNING):
            assert tracking.get_pricing_anthropic() == {}
        assert "Failed to get Anthropic pricing" in caplog.text


class TestGetPricingLiteLLM:
    def test_uses_model_info(self):
        info = {"input_cost_per_token": 1e-6, "output_cost_per_token": 4e-6}
        with patch.object(tracking, "get_model_info", return_value=info) as gmi:
            assert tracking.get_pricing_litellm("m") == {"m": {"prompt": 1e-6, "completion": 4e-6}}
        gmi.assert_called_once_with("m")

    def test_partial_info_defaults_to_zero(self):
        with patch.object(tracking, "get_model_info", return_value={"input_cost_per_token": 2.0}):
            assert tracking.get_pricing_litellm("m") == {"m": {"prompt": 2.0, "completion": 0.0}}

    def test_error_logged_and_zero(self, caplog):
        with patch.object(tracking, "get_model_info", side_effect=Exception("unknown model")):
            with caplog.at_level(logging.ERROR):
                assert tracking.get_pricing_litellm("m") == {
                    "m": {"prompt": 0.0, "completion": 0.0}
                }
        assert "unknown model" in caplog.text


# --------------------------------------------------------------------------- mixin: pricing setup


class TestPricingSetup:
    @pytest.mark.parametrize(
        "api, attr",
        [
            ("openai", "get_pricing_openai"),
            ("anthropic", "get_pricing_anthropic"),
            ("openrouter", "get_pricing_openrouter"),
        ],
    )
    def test_fetch_dispatches_per_provider(self, monkeypatch, api, attr):
        sentinel = {"fake-model": {"prompt": 1, "completion": 2}}
        monkeypatch.setattr(tracking, attr, lambda: sentinel)
        model = FakePricedModel(pricing_api=api)
        assert model.fetch_pricing_information_from_provider() is sentinel

    def test_fetch_litellm_passes_model_name(self, monkeypatch):
        seen = []
        monkeypatch.setattr(tracking, "get_pricing_litellm", lambda name: seen.append(name) or {})
        FakePricedModel(
            model_name="lite/m", pricing_api="litellm"
        ).fetch_pricing_information_from_provider()
        assert seen == ["lite/m"]

    def test_fetch_unsupported_returns_none(self, caplog):
        model = FakePricedModel(pricing_api="vllm")
        with caplog.at_level(logging.WARNING):
            assert model.fetch_pricing_information_from_provider() is None
        assert "Unsupported provider: vllm" in caplog.text

    def test_init_pricing_tracker_uses_provider_table(self, monkeypatch):
        monkeypatch.setattr(
            tracking,
            "get_pricing_openai",
            lambda: {"fake-model": {"prompt": "0.5", "completion": 2}},
        )
        model = FakePricedModel(pricing_api=None)
        model.stats.increment_stats_dict({"x": 1})
        model.init_pricing_tracker(pricing_api="openai")
        assert model._pricing_api == "openai"
        assert (model.input_cost, model.output_cost) == (0.5, 2.0)
        assert isinstance(model.input_cost, float)
        assert dict(model.stats.stats_dict) == {}  # stats are reset

    def test_falls_back_to_litellm_model_info(self, monkeypatch):
        monkeypatch.setattr(tracking, "get_pricing_openai", lambda: {"other": {}})
        info = {"input_cost_per_token": 3e-6, "output_cost_per_token": 6e-6}
        with patch.object(tracking, "get_model_info", return_value=info):
            model = FakePricedModel(pricing_api="openai")
            model.set_pricing_attributes()
        assert (model.input_cost, model.output_cost) == (3e-6, 6e-6)

    def test_unsupported_provider_falls_back_to_litellm(self):
        with patch.object(tracking, "get_model_info", return_value={}):
            model = FakePricedModel(pricing_api="nope")
            model.set_pricing_attributes()
        assert (model.input_cost, model.output_cost) == (0.0, 0.0)

    def test_fallback_failure_zeroes_costs(self, monkeypatch, caplog):
        monkeypatch.setattr(tracking, "get_pricing_openai", lambda: {})
        model = FakePricedModel(pricing_api="openai")
        model.input_cost, model.output_cost = 9.0, 9.0
        with patch.object(tracking, "get_model_info", side_effect=Exception("not mapped")):
            with caplog.at_level(logging.WARNING):
                model.set_pricing_attributes()
        assert (model.input_cost, model.output_cost) == (0.0, 0.0)
        assert "Failed to fetch pricing for fake-model" in caplog.text


# --------------------------------------------------------------------------- token counts


class TestTokenCounts:
    @pytest.mark.parametrize(
        "response, expected",
        [
            (SimpleNamespace(usage=SimpleNamespace(input_tokens=3, output_tokens=4)), (3, 4)),
            (SimpleNamespace(usage=SimpleNamespace(prompt_tokens=5, completion_tokens=6)), (5, 6)),
            ({"usage": {"input_tokens": 7, "output_tokens": 8}}, (7, 8)),
            ({"usage": {"prompt_tokens": 9, "completion_tokens": 10}}, (9, 10)),
        ],
        ids=["obj-responses", "obj-chat", "dict-responses", "dict-chat"],
    )
    def test_extraction(self, response, expected):
        assert FakePricedModel().get_tokens_counts_from_response(response) == expected

    @pytest.mark.parametrize(
        "response",
        [
            SimpleNamespace(usage=None),
            SimpleNamespace(usage=SimpleNamespace(input_tokens=3)),
            {"usage": {"input_tokens": 1}},
            {"no_usage": True},
            object(),
        ],
        ids=["usage-none", "obj-missing-output", "dict-missing-output", "dict-no-usage", "bare"],
    )
    def test_unextractable_defaults_to_zero(self, response, caplog):
        with caplog.at_level(logging.WARNING):
            assert FakePricedModel().get_tokens_counts_from_response(response) == (0, 0)
        assert "Unable to extract input and output tokens" in caplog.text

    def test_update_pricing_tracker_feeds_active_tracker(self):
        model = _priced("openai", 0.5, 2.0)
        response = SimpleNamespace(usage=SimpleNamespace(input_tokens=10, output_tokens=3))
        with set_tracker() as tracker:
            model.update_pricing_tracker(response)
        assert tracker.stats == {"input_tokens": 10, "output_tokens": 3, "cost": 11.0}

    def test_update_pricing_tracker_without_tracker_is_noop(self):
        model = _priced("openai", 0.5, 2.0)
        response = SimpleNamespace(usage=SimpleNamespace(input_tokens=10, output_tokens=3))
        model.update_pricing_tracker(response)  # TRACKER.instance is None
        del tracking.TRACKER.instance
        model.update_pricing_tracker(response)  # attribute missing entirely
        assert not hasattr(tracking.TRACKER, "instance")


# --------------------------------------------------------------------------- effective cost


def _anthropic_usage(input_tokens, output_tokens, cache_read, cache_write):
    # attribute names as read by the current implementation
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
    )


class TestEffectiveCostAnthropic:
    @pytest.mark.parametrize(
        "usage, expected",
        [
            # input*1e-3 + output*5e-3 + read*1e-3*0.1 + write*1e-3*1.25
            (_anthropic_usage(1000, 200, 0, 0), 1.0 + 1.0),
            (_anthropic_usage(1000, 200, 4000, 0), 1.0 + 1.0 + 0.4),
            (_anthropic_usage(1000, 200, 0, 800), 1.0 + 1.0 + 1.0),
            (_anthropic_usage(100, 0, 2000, 400), 0.1 + 0.2 + 0.5),
            (SimpleNamespace(), 0.0),  # all attributes default to 0
        ],
        ids=["no-cache", "cache-read", "cache-write", "read+write", "empty-usage"],
    )
    def test_table(self, usage, expected):
        model = _priced("anthropic", 1e-3, 5e-3)
        cost = model.get_effective_cost(SimpleNamespace(usage=usage))
        assert cost == pytest.approx(expected, rel=1e-12)

    def test_factors(self):
        assert ANTHROPIC_CACHE_PRICING_FACTOR == {
            "cache_read_tokens": 0.1,
            "cache_write_tokens": 1.25,
        }

    def test_no_usage_attribute_costs_zero(self):
        assert _priced("anthropic", 1.0, 1.0).get_effective_cost(object()) == 0.0

    def test_negative_cost_warns(self, caplog):
        model = _priced("anthropic", -1.0, 0.0)
        with caplog.at_level(logging.WARNING):
            cost = model.get_effective_cost(SimpleNamespace(usage=_anthropic_usage(1, 0, 0, 0)))
        assert cost == -1.0
        assert "Negative effective cost" in caplog.text

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: cache reads are read from usage.cache_input_tokens; Anthropic's field is "
        "cache_read_input_tokens, so cache-read tokens are never billed",
    )
    def test_real_anthropic_usage_bills_cache_reads(self):
        from anthropic.types import Usage

        usage = Usage(
            input_tokens=1000,
            output_tokens=200,
            cache_read_input_tokens=4000,
            cache_creation_input_tokens=0,
        )
        cost = _priced("anthropic", 1e-3, 5e-3).get_effective_cost(SimpleNamespace(usage=usage))
        assert cost == pytest.approx(2.4, rel=1e-12)

    @pytest.mark.xfail(
        strict=True,
        raises=TypeError,
        reason="BUG: anthropic Usage leaves cache_*_input_tokens as None when unused; "
        "None * float raises TypeError instead of costing 0",
    )
    def test_real_anthropic_usage_without_cache_fields(self):
        from anthropic.types import Usage

        usage = Usage(input_tokens=1000, output_tokens=200)
        cost = _priced("anthropic", 1e-3, 5e-3).get_effective_cost(SimpleNamespace(usage=usage))
        assert cost == pytest.approx(2.0, rel=1e-12)


class _ChatUsage:
    """Chat Completions-style usage object (has prompt_tokens_details)."""

    def __init__(self, prompt, completion, cached):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.prompt_tokens_details = (
            None if cached is None else SimpleNamespace(cached_tokens=cached)
        )


class _ResponsesUsage:
    """Responses API-style usage object (has input_tokens_details, no prompt_tokens_details)."""

    def __init__(self, inp, out, cached):
        self.input_tokens = inp
        self.output_tokens = out
        self.input_tokens_details = (
            None if cached is None else SimpleNamespace(cached_tokens=cached)
        )


class TestEffectiveCostOpenAI:
    @pytest.mark.parametrize(
        "usage, expected",
        [
            # new_input*1e-3 + cached*1e-3*0.5 + output*4e-3
            (_ChatUsage(1000, 250, 0), 1.0 + 1.0),
            (_ChatUsage(1000, 250, 600), 0.4 + 0.3 + 1.0),
            (_ChatUsage(1000, 250, None), 1.0 + 1.0),
            (_ResponsesUsage(2000, 500, 0), 2.0 + 2.0),
            (_ResponsesUsage(2000, 500, 2000), 0.0 + 1.0 + 2.0),
            (_ResponsesUsage(2000, 500, None), 2.0 + 2.0),
        ],
        ids=[
            "chat-no-cache",
            "chat-cached",
            "chat-details-none",
            "responses-no-cache",
            "responses-all-cached",
            "responses-details-none",
        ],
    )
    def test_table(self, usage, expected):
        model = _priced("openai", 1e-3, 4e-3)
        cost = model.get_effective_cost(SimpleNamespace(usage=usage))
        assert cost == pytest.approx(expected, rel=1e-12)

    def test_factors(self):
        assert OPENAI_CACHE_PRICING_FACTOR == {"cache_read_tokens": 0.5, "cache_write_tokens": 1}

    def test_real_sdk_usage_objects(self):
        from openai.types.chat import ChatCompletion
        from openai.types.responses import ResponseUsage

        chat = ChatCompletion.model_validate(
            {
                "id": "x",
                "object": "chat.completion",
                "created": 1,
                "model": "m",
                "choices": [],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 10,
                    "total_tokens": 110,
                    "prompt_tokens_details": {"cached_tokens": 40},
                },
            }
        )
        responses_usage = ResponseUsage.model_validate(
            {
                "input_tokens": 100,
                "output_tokens": 10,
                "total_tokens": 110,
                "input_tokens_details": {"cached_tokens": 40},
                "output_tokens_details": {"reasoning_tokens": 0},
            }
        )
        model = _priced("openai", 0.01, 0.1)
        expected = 60 * 0.01 + 40 * 0.005 + 10 * 0.1
        assert model.get_effective_cost(chat) == pytest.approx(expected, rel=1e-12)
        assert model.get_effective_cost(SimpleNamespace(usage=responses_usage)) == pytest.approx(
            expected, rel=1e-12
        )

    def test_missing_usage_costs_zero(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert _priced("openai", 1.0, 1.0).get_effective_cost(SimpleNamespace()) == 0.0
        assert "No usage information" in caplog.text

    def test_negative_cost_warns(self, caplog):
        model = _priced("openai", 1.0, 1.0)
        with caplog.at_level(logging.WARNING):
            # more cached tokens than prompt tokens -> negative new_input
            cost = model.get_effective_cost(SimpleNamespace(usage=_ChatUsage(1, 0, 10)))
        assert cost == pytest.approx(-9 + 5.0)
        assert "OpenAI: Negative effective cost" in caplog.text


class TestEffectiveCostDispatch:
    def test_litellm_uses_completion_cost(self):
        response = object()
        with patch.object(tracking, "completion_cost", return_value=0.42) as cc:
            assert _priced("litellm", 0, 0).get_effective_cost(response) == 0.42
        cc.assert_called_once_with(response)

    def test_unknown_provider_zero(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert _priced("openrouter", 1, 1).get_effective_cost(object()) == 0.0
        assert "No effective cost calculated" in caplog.text


# --------------------------------------------------------------------------- mixin __call__ / Stats


class TestMixinCall:
    def test_chat_completion_response_updates_stats_and_tracker(self):
        usage = _IterableUsage(
            {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "prompt_tokens_details": SimpleNamespace(cached_tokens=4),
            }
        )
        response = SimpleNamespace(usage=usage)
        model = _priced("openai", 0.1, 1.0)
        model._response = response
        with set_tracker() as tracker:
            out = model("a", k=1)
            model("b")
        assert out == ("parsed", response)
        assert model.calls == [(("a",), {"k": 1}), (("b",), {})]
        effective = (6 * 0.1 + 4 * 0.05 + 5 * 1.0) * 2
        stats = dict(model.stats.stats_dict)
        assert stats.pop("effective_cost") == pytest.approx(effective)
        assert stats == {
            "usage_prompt_tokens": 20,
            "usage_completion_tokens": 10,
            "usage_total_tokens": 30,
            "usage_cached_tokens": 8,
            "n_api_calls": 2,
        }
        assert tracker.input_tokens == 20 and tracker.output_tokens == 10
        assert tracker.cost == pytest.approx(2 * (10 * 0.1 + 5 * 1.0))

    def test_responses_api_cached_tokens(self):
        usage = _IterableUsage(
            {
                "input_tokens": 8,
                "output_tokens": 2,
                "input_tokens_details": SimpleNamespace(cached_tokens=3),
                "prompt_tokens_details": None,
            }
        )
        model = FakePricedModel(pricing_api="anthropic", response=SimpleNamespace(usage=usage))
        model("x")
        stats = dict(model.stats.stats_dict)
        assert stats["usage_cached_tokens"] == 3
        assert stats["usage_input_tokens"] == 8
        assert "usage_input_tokens_details" not in stats

    def test_response_without_usage(self):
        model = FakePricedModel(pricing_api="nope", response=SimpleNamespace())
        assert model() == ("parsed", model._response)
        assert dict(model.stats.stats_dict) == {"n_api_calls": 1, "effective_cost": 0.0}


class _IterableUsage(SimpleNamespace):
    """Usage object that supports dict(obj) like pydantic models and exposes attributes."""

    def __init__(self, fields):
        super().__init__(**fields)

    def __iter__(self):
        return iter(vars(self).items())


def test_stats_increment_accumulates_and_defaults():
    stats = Stats()
    stats.increment_stats_dict({"a": 1, "b": 0.5})
    stats.increment_stats_dict({"a": 2, "c": 3})
    assert dict(stats.stats_dict) == {"a": 3, "b": 0.5, "c": 3}
    assert stats.stats_dict["missing"] == 0.0
    assert Stats().stats_dict is not stats.stats_dict


# --------------------------------------------------------------------------- import fallbacks


def _exec_tracking_copy(name):
    path = Path(tracking.__file__)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_without_langchain_community(monkeypatch):
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: (
            None if name == "langchain_community" else real_find_spec(name, *a, **k)
        ),
    )
    module = _exec_tracking_copy("_tracking_copy_no_langchain")
    assert module.openai_info is None and module.bedrock_anthropic_callback is None
    assert module.get_pricing_openai() == {}


def test_import_with_broken_langchain_callbacks(monkeypatch):
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: (
            object() if name == "langchain_community" else real_find_spec(name, *a, **k)
        ),
    )
    monkeypatch.setitem(sys.modules, "langchain_community.callbacks", None)
    module = _exec_tracking_copy("_tracking_copy_broken_langchain")
    assert module.openai_info is None and module.bedrock_anthropic_callback is None
    assert module.get_pricing_anthropic() == {}
