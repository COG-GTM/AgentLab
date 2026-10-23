"""Offline unit tests for LiteLLMModel (the litellm client is replaced by a fake)."""

import json

import pytest
from litellm import ModelResponse

from agentlab.llm.litellm_api import LiteLLMAPIMessageBuilder, LiteLLMModel, LiteLLMModelArgs
from agentlab.llm.response_api import APIPayload, ToolCall, ToolCalls

RESPONSE_API_TOOL = {
    "type": "function",
    "name": "click",
    "description": "Click an element.",
    "parameters": {"type": "object", "properties": {"bid": {"type": "string"}}},
}


def _response(content="", tool_calls=None, reasoning=None):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
            for i, (name, args) in enumerate(tool_calls)
        ]
    if reasoning is not None:
        message["reasoning"] = reasoning
    return ModelResponse(
        choices=[{"index": 0, "message": message, "finish_reason": "stop"}],
        model="gpt-4o-mini",
        usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    )


@pytest.fixture
def model():
    m = LiteLLMModelArgs(
        model_name="openai/gpt-4o-mini", max_new_tokens=50, temperature=0.2
    ).make_model()
    calls = []

    def fake_client(**kwargs):
        calls.append(kwargs)
        return m._next_response

    m.client = fake_client
    m._next_response = _response("hello")
    m.calls = calls
    return m


def _user_payload(**kwargs):
    return APIPayload(messages=[LiteLLMAPIMessageBuilder.user().add_text("hi")], **kwargs)


def test_model_args_builder():
    assert LiteLLMModelArgs(model_name="x").get_message_builder() is LiteLLMAPIMessageBuilder


def test_call_api_basic_params(model):
    model._call_api(_user_payload())
    params = model.calls[-1]
    assert params["model"] == "openai/gpt-4o-mini"
    assert params["temperature"] == 0.2
    assert params["max_completion_tokens"] == 50
    assert params["num_retries"] == 5
    assert params["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert "tools" not in params and "tool_choice" not in params


def test_call_api_omits_unset_params(model):
    model.temperature = None
    model.max_tokens = None
    model._call_api(_user_payload())
    assert "temperature" not in model.calls[-1]
    assert "max_completion_tokens" not in model.calls[-1]


def test_call_api_converts_tools_and_caching(model):
    model._call_api(
        _user_payload(
            tools=[dict(RESPONSE_API_TOOL)],
            tool_choice="any",
            reasoning_effort="low",
            cache_tool_definition=True,
            cache_complete_prompt=True,
        )
    )
    params = model.calls[-1]
    assert params["tools"][0]["function"]["name"] == "click"
    assert params["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert params["tool_choice"] == "required"
    assert params["reasoning_effort"] == "low"
    assert params["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}


def test_call_api_keeps_chat_completion_tools_and_forces_tool(model):
    chat_tool = {"type": "function", "function": {"name": "click"}}
    model._call_api(_user_payload(tools=[chat_tool], force_call_tool="click"))
    params = model.calls[-1]
    assert params["tools"] == [chat_tool]
    assert params["tool_choice"] == {"type": "function", "function": {"name": "click"}}

    model._call_api(_user_payload(tool_choice="auto"))
    assert model.calls[-1]["tool_choice"] == "auto"


def test_api_payload_rejects_tool_choice_with_force():
    with pytest.raises(ValueError, match="mutually exclusive"):
        APIPayload(tool_choice="auto", force_call_tool="click")


def test_parse_response_single_tool_call(model):
    model._next_response = _response("ok", tool_calls=[("click", {"bid": "a1"})])
    out = model(_user_payload())
    assert out.think == "ok"
    assert out.action == "click(bid='a1')"
    assert len(out.tool_calls) == 1
    assert model.stats.stats_dict["n_api_calls"] == 1


def test_parse_response_multiple_tool_calls(model):
    calls = [("click", {"bid": "a1"}), ("fill", {"bid": "a2", "value": "x"})]
    model._next_response = _response(tool_calls=calls)
    out = model._parse_response(model._next_response)
    assert out.action == "click(bid='a1')\nfill(bid='a2', value='x')"

    model.use_only_first_toolcall = True
    out = model._parse_response(model._next_response)
    assert out.action == "click(bid='a1')"
    assert len(out.tool_calls) == 1


def test_parse_response_without_tool_calls(model):
    out = model._parse_response(_response("just text"))
    assert out.action is None and out.tool_calls is None

    model.action_space_as_tools = False
    assert model._parse_response(_response("x")).action is None


def test_thinking_content_is_wrapped(model):
    think = model._extract_thinking_content_from_response(_response("answer", reasoning="because"))
    assert think == "<think>because</think>\nanswer"


def test_format_tools_for_chat_completion():
    assert LiteLLMModel.format_tools_for_chat_completion(None) is None
    (tool,) = LiteLLMModel.format_tools_for_chat_completion([RESPONSE_API_TOOL])
    assert tool == {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Click an element.",
            "parameters": RESPONSE_API_TOOL["parameters"],
        },
    }


def _responded_tool_calls(n=2):
    response = _response(tool_calls=[("click", {"bid": str(i)}) for i in range(n)])
    raw = response.choices[0].message.to_dict()["tool_calls"]
    calls = []
    for i, raw_call in enumerate(raw):
        tc = ToolCall(name="click", arguments={"bid": str(i)}, raw_call=raw_call)
        tc.response_text(f"clicked {i}")
        calls.append(tc)
    return ToolCalls(tool_calls=calls, raw_calls=response)


def test_message_builder_tool_responses():
    builder = LiteLLMAPIMessageBuilder.add_responded_tool_calls(_responded_tool_calls())
    out = builder.prepare_message()
    assert len(out[0].tool_calls) == 2
    assert out[1:] == [
        {"name": "click", "role": "tool", "tool_call_id": "call_0", "content": "clicked 0"},
        {"name": "click", "role": "tool", "tool_call_id": "call_1", "content": "clicked 1"},
    ]


def test_message_builder_only_first_tool_call():
    builder = LiteLLMAPIMessageBuilder.add_responded_tool_calls(_responded_tool_calls())
    out = builder.prepare_message(use_only_first_toolcall=True)
    assert len(out[0].tool_calls) == 1


def test_message_builder_tool_without_calls():
    builder = LiteLLMAPIMessageBuilder("tool")
    with pytest.raises(ValueError, match="No tool calls"):
        builder.handle_tool_call()
