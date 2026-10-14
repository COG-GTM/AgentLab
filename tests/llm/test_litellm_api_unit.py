"""Hermetic unit tests for agentlab.llm.litellm_api (litellm.completion is never called)."""

import json
import logging
import os
import runpy
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import litellm
import pytest
from litellm.types.utils import Choices, Message, ModelResponse, Usage

import agentlab.llm.litellm_api as litellm_api
import agentlab.llm.tracking as tracking
from agentlab.llm.litellm_api import LiteLLMAPIMessageBuilder, LiteLLMModel, LiteLLMModelArgs
from agentlab.llm.response_api import APIPayload, LLMOutput, ToolCalls

MODEL = "openai/fake-model"
PRICING = {MODEL: {"prompt": 1e-3, "completion": 2e-3}}

RESPONSES_API_TOOLS = [
    {
        "type": "function",
        "name": "click",
        "description": "Click an element.",
        "parameters": {"type": "object", "properties": {"bid": {"type": "string"}}},
    },
    {
        "type": "function",
        "name": "fill",
        "description": "Fill a field.",
        "parameters": {
            "type": "object",
            "properties": {"bid": {"type": "string"}, "value": {"type": "string"}},
        },
    },
]

_MISSING = object()


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch):
    """Stub pricing/model-info lookups and isolate the thread-local tracker."""
    monkeypatch.setattr(tracking, "get_pricing_litellm", MagicMock(return_value=PRICING))
    monkeypatch.setattr(litellm, "get_model_info", MagicMock(return_value={"max_tokens": 1000}))
    previous = getattr(tracking.TRACKER, "instance", _MISSING)
    if previous is not _MISSING:
        del tracking.TRACKER.instance
    yield
    if hasattr(tracking.TRACKER, "instance"):
        del tracking.TRACKER.instance
    if previous is not _MISSING:
        tracking.TRACKER.instance = previous


def make_model(**kwargs):
    model = LiteLLMModel(model_name=MODEL, **kwargs)
    model.client = MagicMock(name="litellm.completion")
    return model


def tool_call(call_id, name, arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def make_response(content=None, tool_calls=None, prompt_tokens=10, completion_tokens=4):
    return ModelResponse(
        choices=[Choices(message=Message(content=content, tool_calls=tool_calls))],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


class FakeMessage:
    def __init__(self, **fields):
        self._fields = fields

    def to_dict(self):
        return dict(self._fields)


def fake_response(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def user_msg(*texts):
    msg = LiteLLMAPIMessageBuilder.user()
    for t in texts:
        msg.add_text(t)
    return msg


# ---------------------------------------------------------------- __init__


def test_init_defaults_and_pricing():
    model = LiteLLMModel(model_name=MODEL)
    assert model.client.func is litellm_api.completion
    assert model.client.keywords == {}
    assert model.max_tokens == 100
    assert model.temperature is None
    assert model.action_space_as_tools is True
    assert model.use_only_first_toolcall is False
    assert model.litellm_info == {"max_tokens": 1000}
    assert (model.input_cost, model.output_cost) == (1e-3, 2e-3)
    tracking.get_pricing_litellm.assert_called_once_with(MODEL)


def test_init_forwards_base_url_and_api_key_to_client():
    model = LiteLLMModel(model_name=MODEL, base_url="http://proxy", api_key="sk-x")
    assert model.client.keywords == {"base_url": "http://proxy", "api_key": "sk-x"}


def test_init_tolerates_model_info_failure(monkeypatch, caplog):
    monkeypatch.setattr(litellm, "get_model_info", MagicMock(side_effect=Exception("unknown")))
    with caplog.at_level(logging.ERROR):
        model = LiteLLMModel(model_name=MODEL)
    assert not hasattr(model, "litellm_info")
    assert "Failed to get litellm model info: unknown" in caplog.text


# ---------------------------------------------------------------- _call_api


def test_call_api_minimal_payload():
    model = make_model(temperature=0.3, max_tokens=50)
    payload = APIPayload(messages=[user_msg("hello")])

    result = model._call_api(payload)

    assert result is model.client.return_value
    model.client.assert_called_once_with(
        model=MODEL,
        messages=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        temperature=0.3,
        max_completion_tokens=50,
        num_retries=5,
    )


def test_call_api_omits_none_temperature_and_max_tokens():
    model = make_model(temperature=None, max_tokens=None)
    model._call_api(APIPayload(messages=[user_msg("hi")]))
    kwargs = model.client.call_args.kwargs
    assert "temperature" not in kwargs
    assert "max_completion_tokens" not in kwargs
    assert "tools" not in kwargs
    assert "tool_choice" not in kwargs
    assert "reasoning_effort" not in kwargs


def test_call_api_converts_responses_api_tools():
    model = make_model()
    model._call_api(APIPayload(messages=[user_msg("hi")], tools=RESPONSES_API_TOOLS))
    tools = model.client.call_args.kwargs["tools"]
    assert tools == [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in RESPONSES_API_TOOLS
    ]


def test_call_api_passes_chat_completion_tools_through_unchanged():
    model = make_model()
    chat_tools = LiteLLMModel.format_tools_for_chat_completion(RESPONSES_API_TOOLS)
    model._call_api(APIPayload(messages=[user_msg("hi")], tools=chat_tools))
    assert model.client.call_args.kwargs["tools"] is chat_tools


@pytest.mark.parametrize(
    "tool_choice, expected",
    [("required", "required"), ("any", "required"), ("auto", "auto"), ("none", "none")],
)
def test_call_api_tool_choice_mapping(tool_choice, expected):
    model = make_model()
    model._call_api(
        APIPayload(messages=[user_msg("hi")], tools=RESPONSES_API_TOOLS, tool_choice=tool_choice)
    )
    assert model.client.call_args.kwargs["tool_choice"] == expected


def test_call_api_force_call_tool():
    model = make_model()
    model._call_api(
        APIPayload(messages=[user_msg("hi")], tools=RESPONSES_API_TOOLS, force_call_tool="fill")
    )
    assert model.client.call_args.kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "fill"},
    }


def test_call_api_reasoning_effort():
    model = make_model()
    model._call_api(APIPayload(messages=[user_msg("hi")], reasoning_effort="high"))
    assert model.client.call_args.kwargs["reasoning_effort"] == "high"


def test_call_api_cache_flags_mark_last_tool_and_last_content_item():
    model = make_model()
    payload = APIPayload(
        messages=[user_msg("first"), user_msg("a", "b")],
        tools=RESPONSES_API_TOOLS,
        cache_tool_definition=True,
        cache_complete_prompt=True,
    )
    model._call_api(payload)
    kwargs = model.client.call_args.kwargs
    assert "cache_control" not in kwargs["tools"][0]
    assert kwargs["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert kwargs["messages"][-1]["content"][-1] == {
        "type": "text",
        "text": "b",
        "cache_control": {"type": "ephemeral"},
    }
    assert "cache_control" not in kwargs["messages"][-1]["content"][0]
    assert "cache_control" not in kwargs["messages"][0]["content"][-1]


def test_call_api_cache_tool_definition_without_tools_is_noop():
    model = make_model()
    model._call_api(APIPayload(messages=[user_msg("hi")], cache_tool_definition=True))
    assert "tools" not in model.client.call_args.kwargs


# ---------------------------------------------------------------- response parsing


def test_parse_response_single_tool_call():
    model = make_model()
    response = make_response(content="", tool_calls=[tool_call("c1", "click", {"bid": "12"})])
    out = model._parse_response(response)
    assert isinstance(out, LLMOutput)
    assert out.raw_response is response
    assert out.action == "click(bid='12')"
    assert len(out.tool_calls) == 1
    tc = out.tool_calls.tool_calls[0]
    assert (tc.name, tc.arguments) == ("click", {"bid": "12"})
    assert tc.raw_call["id"] == "c1"
    assert out.tool_calls.raw_calls is response


def test_parse_response_multiple_tool_calls_joined_as_multiaction():
    model = make_model()
    response = make_response(
        content="",
        tool_calls=[
            tool_call("c1", "click", {"bid": "12"}),
            tool_call("c2", "fill", {"bid": "3", "value": "hi"}),
        ],
    )
    out = model._parse_response(response)
    assert out.action == "click(bid='12')\nfill(bid='3', value='hi')"
    assert [tc.name for tc in out.tool_calls] == ["click", "fill"]


def test_parse_response_use_only_first_toolcall():
    model = make_model(use_only_first_toolcall=True)
    response = make_response(
        content="",
        tool_calls=[tool_call("c1", "click", {"bid": "1"}), tool_call("c2", "noop", {})],
    )
    out = model._parse_response(response)
    assert out.action == "click(bid='1')"
    assert len(out.tool_calls) == 1


def test_parse_response_tool_call_without_arguments():
    model = make_model()
    out = model._parse_response(make_response(content="", tool_calls=[tool_call("c", "noop", {})]))
    assert out.action == "noop()"


def test_parse_response_text_only():
    model = make_model()
    out = model._parse_response(make_response(content="Just thinking out loud"))
    assert out.think == "Just thinking out loud"
    assert out.action is None
    assert out.tool_calls is None


def test_parse_response_action_space_not_as_tools_returns_no_action():
    model = make_model()
    model.action_space_as_tools = False
    response = make_response(content="", tool_calls=[tool_call("c1", "click", {"bid": "12"})])
    out = model._parse_response(response)
    assert out.action is None
    assert len(out.tool_calls) == 1


def test_extract_env_actions_from_empty_toolcalls():
    model = make_model()
    assert model._extract_env_actions_from_toolcalls(ToolCalls()) is None
    assert model._extract_env_actions_from_toolcalls(None) is None


@pytest.mark.parametrize(
    "fields, wrap_tag, expected",
    [
        ({"content": "answer"}, "think", "answer"),
        ({"reasoning": "because", "content": "answer"}, "think", "<think>because</think>\nanswer"),
        ({"reasoning": "why", "content": "x"}, "reason", "<reason>why</reason>\nx"),
        ({"text": "router text ", "content": "content"}, "think", "router text content"),
        ({"reasoning": "", "content": "c"}, "think", "c"),
    ],
)
def test_extract_thinking_content(fields, wrap_tag, expected):
    model = make_model()
    response = fake_response(FakeMessage(**fields))
    assert model._extract_thinking_content_from_response(response, wrap_tag=wrap_tag) == expected


def test_extract_thinking_content_accepts_plain_dict_message():
    model = make_model()
    response = fake_response({"reasoning": "r", "content": "c"})
    assert model._extract_thinking_content_from_response(response) == "<think>r</think>\nc"


@pytest.mark.xfail(
    strict=True,
    reason="BUG: think output is the string 'None' when a tool-call message has content=None",
)
def test_parse_response_tool_call_with_none_content_has_empty_think():
    model = make_model()
    response = make_response(content=None, tool_calls=[tool_call("c1", "click", {"bid": "12"})])
    assert model._parse_response(response).think == ""


def test_format_tools_for_chat_completion_none():
    assert LiteLLMModel.format_tools_for_chat_completion(None) is None


def test_format_tools_for_chat_completion_drops_extra_keys():
    tool = dict(RESPONSES_API_TOOLS[0], strict=True)
    formatted = LiteLLMModel.format_tools_for_chat_completion([tool])
    assert formatted == [
        {
            "type": "function",
            "function": {
                "name": "click",
                "description": "Click an element.",
                "parameters": tool["parameters"],
            },
        }
    ]


# ---------------------------------------------------------------- full __call__ via pricing mixin


def test_model_call_tracks_cost_and_stats(monkeypatch):
    monkeypatch.setattr(tracking, "completion_cost", MagicMock(return_value=0.25))
    model = make_model()
    response = make_response(
        content="",
        tool_calls=[tool_call("c1", "click", {"bid": "7"})],
        prompt_tokens=100,
        completion_tokens=50,
    )
    model.client.return_value = response

    with tracking.set_tracker() as tracker:
        out = model(APIPayload(messages=[user_msg("go")]))

    assert out.action == "click(bid='7')"
    assert tracker.input_tokens == 100
    assert tracker.output_tokens == 50
    assert tracker.cost == pytest.approx(100 * 1e-3 + 50 * 2e-3)
    assert model.stats.stats_dict["n_api_calls"] == 1
    assert model.stats.stats_dict["effective_cost"] == 0.25
    assert model.stats.stats_dict["usage_prompt_tokens"] == 100
    tracking.completion_cost.assert_called_once_with(response)


# ---------------------------------------------------------------- message builder


def test_message_builder_prepare_user_message_with_text_and_image():
    msg = LiteLLMAPIMessageBuilder.user().add_text("look").add_image("data:image/png;base64,AA")
    assert msg.prepare_message() == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
            ],
        }
    ]


def test_message_builder_rejects_unknown_content():
    msg = LiteLLMAPIMessageBuilder.user()
    msg.content.append({"audio": "x"})
    with pytest.raises(ValueError, match="Unsupported content type"):
        msg.prepare_message()


def _responded_tool_calls(response):
    calls = make_model()._extract_tool_calls_from_response(response)
    for i, tc in enumerate(calls):
        tc.response_text(f"result {i}")
    return calls


def test_message_builder_tool_role_replays_raw_message_and_tool_outputs():
    response = make_response(
        content="",
        tool_calls=[tool_call("c1", "click", {"bid": "1"}), tool_call("c2", "fill", {"bid": "2"})],
    )
    msg = LiteLLMAPIMessageBuilder.add_responded_tool_calls(_responded_tool_calls(response))

    output = msg.prepare_message()

    assert output[0] is response.choices[0].message
    assert output[1:] == [
        {"name": "click", "role": "tool", "tool_call_id": "c1", "content": "result 0"},
        {"name": "fill", "role": "tool", "tool_call_id": "c2", "content": "result 1"},
    ]


def test_message_builder_handle_tool_call_use_only_first_truncates_raw_message():
    response = make_response(
        content="",
        tool_calls=[tool_call("c1", "click", {"bid": "1"}), tool_call("c2", "fill", {"bid": "2"})],
    )
    calls = _responded_tool_calls(response)
    msg = LiteLLMAPIMessageBuilder.add_responded_tool_calls(calls)

    output = msg.prepare_message(use_only_first_toolcall=True)

    assert [tc.id for tc in output[0].tool_calls] == ["c1"]


def test_message_builder_handle_tool_call_requires_tool_calls():
    msg = LiteLLMAPIMessageBuilder("tool")
    with pytest.raises(ValueError, match="No tool calls found"):
        msg.prepare_message()


def test_message_builder_rejects_image_tool_response():
    response = make_response(content="", tool_calls=[tool_call("c1", "screenshot", {})])
    calls = make_model()._extract_tool_calls_from_response(response)
    calls.tool_calls[0].response_image("data:image/png;base64,AA")
    msg = LiteLLMAPIMessageBuilder.add_responded_tool_calls(calls)
    with pytest.raises(AssertionError, match="Image output is not supported"):
        msg.prepare_message()


@pytest.mark.xfail(
    strict=True,
    reason="BUG: use_only_first_toolcall is not forwarded to prepare_message, so the replayed "
    "assistant message keeps every tool call but only the first gets a tool response",
)
def test_use_only_first_toolcall_replay_has_matching_tool_responses():
    model = make_model(use_only_first_toolcall=True)
    response = make_response(
        content="",
        tool_calls=[tool_call("c1", "click", {"bid": "1"}), tool_call("c2", "fill", {"bid": "2"})],
    )
    calls = model._parse_response(response).tool_calls
    for tc in calls:
        tc.response_text("done")
    payload = APIPayload(
        messages=[user_msg("go"), LiteLLMAPIMessageBuilder.add_responded_tool_calls(calls)]
    )

    model._call_api(payload)

    sent = model.client.call_args.kwargs["messages"]
    assistant, tool_outputs = sent[1], sent[2:]
    assert [tc.id for tc in assistant.tool_calls] == [t["tool_call_id"] for t in tool_outputs]


# ---------------------------------------------------------------- model args


def test_model_args_make_model_forwards_fields():
    args = LiteLLMModelArgs(
        model_name=MODEL,
        max_new_tokens=321,
        temperature=0.6,
        base_url="http://proxy",
        api_key="sk",
        use_only_first_toolcall=True,
    )
    model = args.make_model()
    assert isinstance(model, LiteLLMModel)
    assert model.model_name == MODEL
    assert model.max_tokens == 321
    assert model.temperature == 0.6
    assert model.use_only_first_toolcall is True
    assert model.client.keywords == {"base_url": "http://proxy", "api_key": "sk"}


def test_model_args_message_builder_and_api():
    args = LiteLLMModelArgs(model_name=MODEL)
    assert args.get_message_builder() is LiteLLMAPIMessageBuilder
    assert args.api == "openai"
    assert (args.base_url, args.api_key, args.use_only_first_toolcall) == (None, None, False)


def test_module_enables_litellm_modify_params():
    assert litellm.modify_params is True


# ---------------------------------------------------------------- __main__ smoke script


def test_main_block_builds_one_agent_per_model_and_runs_study(monkeypatch):
    monkeypatch.setenv("LITELLM_LOG", "DEBUG")
    tool_use = types.ModuleType("agentlab.agents.tool_use_agent")
    tool_use.DEFAULT_PROMPT_CONFIG = object()
    tool_use.ToolUseAgentArgs = MagicMock(name="ToolUseAgentArgs")
    study_mod = types.ModuleType("agentlab.experiments.study")
    study_mod.Study = MagicMock(name="Study")

    with patch.dict(
        sys.modules,
        {"agentlab.agents.tool_use_agent": tool_use, "agentlab.experiments.study": study_mod},
    ):
        runpy.run_path(litellm_api.__file__, run_name="__main__")

    calls = tool_use.ToolUseAgentArgs.call_args_list
    model_args = [c.kwargs["model_args"] for c in calls]
    assert [m.model_name for m in model_args][:2] == ["openai/gpt-4.1", "openai/gpt-4.1-mini"]
    assert len(model_args) == 6
    assert all(m.max_new_tokens == 2000 and m.temperature is None for m in model_args)
    assert all(c.kwargs["config"] is tool_use.DEFAULT_PROMPT_CONFIG for c in calls)
    study_args, study_kwargs = study_mod.Study.call_args
    assert study_args[1] == "miniwob_tiny_test"
    assert len(study_args[0]) == 6
    study_mod.Study.return_value.run.assert_called_once_with(
        n_jobs=5, parallel_backend="ray", strict_reproducibility=False, n_relaunch=3
    )
    assert os.environ["LITELLM_LOG"] == "WARNING"
