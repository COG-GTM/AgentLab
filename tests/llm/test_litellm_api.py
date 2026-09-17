import json
import os
from functools import partial
from types import SimpleNamespace

import pytest
from agentlab.llm.litellm_api import LiteLLMAPIMessageBuilder, LiteLLMModel, LiteLLMModelArgs
from agentlab.llm.response_api import APIPayload, LLMOutput, ToolCall, ToolCalls

chat_api_tools = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather in a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location to get the weather for.",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The unit of temperature.",
                },
            },
            "required": ["location"],
        },
    },
    {
        "type": "function",
        "name": "get_time",
        "description": "Get the current time in a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location to get the time for.",
                }
            },
            "required": ["location"],
        },
    },
]


# test_config (setting name, BaseModelArgs, model_name, tools)
tool_test_configs = [
    ("gpt-4.1", LiteLLMModelArgs, "openai/gpt-4.1-2025-04-14", chat_api_tools),
    # ("claude-3", LiteLLMModelArgs, "anthropic/claude-3-haiku-20240307", anthropic_tools),   # fails for parallel tool calls
    # ("claude-3.7", LiteLLMModelArgs, "anthropic/claude-3-7-sonnet-20250219", anthropic_tools), # fails for parallel tool calls
    ("claude-4-sonnet", LiteLLMModelArgs, "anthropic/claude-sonnet-4-20250514", chat_api_tools),
    # ("gpt-o3", LiteLLMModelArgs, "openai/o3-2025-04-16", chat_api_tools), # fails for parallel tool calls
    # add more models as needed
]


def add_user_messages(msg_builder):
    return [
        msg_builder.user().add_text("What is the weather in Paris and Delhi?"),
        msg_builder.user().add_text("You must call multiple tools to achieve the task."),
    ]


## Test multiaction
@pytest.mark.pricy
def test_multi_action_tool_calls():
    """
    Test that the model can produce multiple tool calls in parallel.
    Note: Remove assert and Uncomment commented lines to see the full behaviour of models and tool choices.
    """
    res_df = []
    for tool_choice in [
        # "none",
        "required",  # fails for Responses API
        "any",  # fails for Responses API
        "auto",
        # "get_weather",  # force a specific tool call
    ]:
        for name, llm_class, checkpoint_name, tools in tool_test_configs:
            model_args = llm_class(model_name=checkpoint_name, max_new_tokens=200, temperature=None)
            llm, msg_builder = model_args.make_model(), model_args.get_message_builder()
            messages = add_user_messages(msg_builder)
            if tool_choice == "get_weather":  # force a specific tool call
                response: LLMOutput = llm(
                    APIPayload(messages=messages, tools=tools, force_call_tool=tool_choice)
                )
            else:
                response: LLMOutput = llm(
                    APIPayload(messages=messages, tools=tools, tool_choice=tool_choice)
                )
            num_tool_calls = len(response.tool_calls) if response.tool_calls else 0
            row = {
                "model": name,
                "checkpoint": checkpoint_name,
                "tool_choice": tool_choice,
                "num_tool_calls": num_tool_calls,
                "action": response.action,
            }
            res_df.append(row)
            assert (
                num_tool_calls == 2
            ), f"Expected 2 tool calls, but got {num_tool_calls} for {name} with tool choice {tool_choice}"
    # import pandas as pd
    # print(pd.DataFrame(res_df))


@pytest.mark.pricy
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Skipping as OpenAI API key not set")
def test_single_tool_call():
    """
    Test that the LLMOutput contains only one tool call when use_only_first_toolcall is True.
    """
    for tool_choice in [
        # 'none',
        "required",
        "any",
        "auto",
    ]:
        for name, llm_class, checkpoint_name, tools in tool_test_configs:
            print(name, "tool choice:", tool_choice, "\n", "**" * 10)
            llm_class = partial(llm_class, use_only_first_toolcall=True)
            model_args = llm_class(model_name=checkpoint_name, max_new_tokens=200, temperature=None)
            llm, msg_builder = model_args.make_model(), model_args.get_message_builder()
            messages = add_user_messages(msg_builder)
            if tool_choice == "get_weather":  # force a specific tool call
                response: LLMOutput = llm(
                    APIPayload(messages=messages, tools=tools, force_call_tool=tool_choice)
                )
            else:
                response: LLMOutput = llm(
                    APIPayload(messages=messages, tools=tools, tool_choice=tool_choice)
                )
            num_tool_calls = len(response.tool_calls) if response.tool_calls else 0
            assert (
                num_tool_calls == 1
            ), f"Expected 1 tool calls, but got {num_tool_calls} for {name} with tool choice {tool_choice }"


@pytest.mark.pricy
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Skipping as OpenAI API key not set")
def test_force_tool_call():
    """
    Test that the model can produce a specific tool call when requested.
    The user message asks the 'weather' but we force call tool "get_time".
    We test if 'get_time' is present in the tool calls.
    Note: Model can have other tool calls as well.
    """
    force_call_tool = "get_time"
    for name, llm_class, checkpoint_name, tools in tool_test_configs:
        model_args = llm_class(model_name=checkpoint_name, max_new_tokens=200, temperature=None)
        llm, msg_builder = model_args.make_model(), model_args.get_message_builder()
        messages = add_user_messages(msg_builder)  # asks weather in Paris and Delhi
        response: LLMOutput = llm(
            APIPayload(messages=messages, tools=tools, force_call_tool=force_call_tool)
        )
        called_fn_names = [call.name for call in response.tool_calls] if response.tool_calls else []
        assert response.tool_calls is not None
        assert any(
            fn_name == "get_time" for fn_name in called_fn_names
        ), f"Model:{name},Expected all tool calls to be 'get_time', but got {called_fn_names} with force call {force_call_tool}"


class FakeMessage:
    """Minimal stand-in for a litellm/OpenAI chat completion message."""

    def __init__(self, data: dict, tool_calls=None):
        self._data = dict(data)
        self.tool_calls = tool_calls

    def to_dict(self):
        data = dict(self._data)
        if self.tool_calls is not None:
            data["tool_calls"] = self.tool_calls
        return data


def make_raw_tool_call(call_id: str, name: str, arguments, as_json: bool = True):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments) if as_json else arguments,
        },
    }


def make_response(content="", reasoning=None, text=None, tool_calls=None):
    data = {"content": content}
    if reasoning is not None:
        data["reasoning"] = reasoning
    if text is not None:
        data["text"] = text
    message = FakeMessage(data, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture
def model():
    return LiteLLMModel(model_name="openai/gpt-4.1", api_key="fake-key")


def test_extract_thinking_wraps_reasoning(model):
    response = make_response(content="answer", reasoning="because", text="prefix ")
    assert (
        model._extract_thinking_content_from_response(response)
        == "<think>because</think>\nprefix answer"
    )


def test_extract_thinking_without_reasoning(model):
    response = make_response(content="answer")
    assert model._extract_thinking_content_from_response(response) == "answer"


def test_extract_thinking_custom_wrap_tag(model):
    response = make_response(content="", reasoning="because")
    assert (
        model._extract_thinking_content_from_response(response, wrap_tag="reason")
        == "<reason>because</reason>\n"
    )


def test_extract_tool_calls_parses_arguments(model):
    response = make_response(
        tool_calls=[
            make_raw_tool_call("call_1", "get_weather", {"location": "Paris"}),
            make_raw_tool_call("call_2", "get_time", {"location": "Delhi"}),
        ]
    )
    tool_calls = model._extract_tool_calls_from_response(response)
    assert [call.name for call in tool_calls] == ["get_weather", "get_time"]
    assert tool_calls.tool_calls[0].arguments == {"location": "Paris"}
    assert tool_calls.raw_calls is response


def test_extract_tool_calls_returns_none_without_tool_calls(model):
    assert model._extract_tool_calls_from_response(make_response(content="hi")) is None


def test_extract_tool_calls_keeps_only_first_when_configured():
    model = LiteLLMModel(
        model_name="openai/gpt-4.1", api_key="fake-key", use_only_first_toolcall=True
    )
    response = make_response(
        tool_calls=[
            make_raw_tool_call("call_1", "get_weather", {"location": "Paris"}),
            make_raw_tool_call("call_2", "get_time", {"location": "Delhi"}),
        ]
    )
    tool_calls = model._extract_tool_calls_from_response(response)
    assert len(tool_calls) == 1
    assert tool_calls.tool_calls[0].name == "get_weather"


def test_extract_tool_calls_malformed_json_raises(model):
    response = make_response(
        tool_calls=[make_raw_tool_call("call_1", "get_weather", "{not json", as_json=False)]
    )
    with pytest.raises(json.JSONDecodeError):
        model._extract_tool_calls_from_response(response)


def test_extract_env_actions_single_toolcall(model):
    tool_calls = ToolCalls(
        tool_calls=[ToolCall(name="get_weather", arguments={"location": "Paris"})]
    )
    assert model._extract_env_actions_from_toolcalls(tool_calls) == "get_weather(location='Paris')"


def test_extract_env_actions_multiple_toolcalls_are_joined(model):
    tool_calls = ToolCalls(
        tool_calls=[
            ToolCall(name="get_weather", arguments={"location": "Paris"}),
            ToolCall(name="get_time", arguments={}),
        ]
    )
    assert (
        model._extract_env_actions_from_toolcalls(tool_calls)
        == "get_weather(location='Paris')\nget_time()"
    )


def test_extract_env_actions_empty_toolcalls(model):
    assert model._extract_env_actions_from_toolcalls(ToolCalls()) is None
    assert model._extract_env_actions_from_toolcalls(None) is None


def test_extract_env_actions_from_text_response_is_unsupported(model):
    assert model._extract_env_actions_from_text_response(make_response(content="click(1)")) is None


def test_parse_response_with_tool_calls(model):
    response = make_response(
        content="here you go",
        reasoning="thinking",
        tool_calls=[make_raw_tool_call("call_1", "get_weather", {"location": "Paris"})],
    )
    output = model._parse_response(response)
    assert isinstance(output, LLMOutput)
    assert output.raw_response is response
    assert output.think == "<think>thinking</think>\nhere you go"
    assert output.action == "get_weather(location='Paris')"
    assert len(output.tool_calls) == 1


def test_parse_response_without_tool_calls(model):
    output = model._parse_response(make_response(content="no tools"))
    assert output.action is None
    assert output.tool_calls is None
    assert output.think == "no tools"


def test_parse_response_text_action_space_has_no_action(model):
    model.action_space_as_tools = False
    output = model._parse_response(make_response(content="click(1)"))
    assert output.action is None


def make_responded_tool_calls(n_calls: int = 2, image_response: bool = False):
    raw_calls = [
        make_raw_tool_call(f"call_{i}", f"tool_{i}", {"arg": i}) for i in range(1, n_calls + 1)
    ]
    response = make_response(tool_calls=raw_calls)
    tool_calls = []
    for raw_call in raw_calls:
        call = ToolCall(name=raw_call["function"]["name"], arguments={}, raw_call=raw_call)
        if image_response:
            call.response_image("data:image/png;base64,abc")
        else:
            call.response_text(f"result of {raw_call['id']}")
        tool_calls.append(call)
    return ToolCalls(tool_calls=tool_calls, raw_calls=response)


def test_handle_tool_call_builds_provider_messages():
    responded = make_responded_tool_calls(n_calls=2)
    builder = LiteLLMAPIMessageBuilder.add_responded_tool_calls(responded)
    output = builder.prepare_message()

    assert output[0] is responded.raw_calls.choices[0].message
    assert output[1:] == [
        {
            "name": "tool_1",
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "result of call_1",
        },
        {
            "name": "tool_2",
            "role": "tool",
            "tool_call_id": "call_2",
            "content": "result of call_2",
        },
    ]


def test_handle_tool_call_truncates_raw_calls_to_first():
    responded = make_responded_tool_calls(n_calls=2)
    builder = LiteLLMAPIMessageBuilder.add_responded_tool_calls(responded)
    output = builder.prepare_message(use_only_first_toolcall=True)

    assert len(output[0].tool_calls) == 1
    assert output[0].tool_calls[0]["id"] == "call_1"


def test_handle_tool_call_rejects_image_response():
    responded = make_responded_tool_calls(n_calls=1, image_response=True)
    builder = LiteLLMAPIMessageBuilder.add_responded_tool_calls(responded)
    with pytest.raises(AssertionError, match="Image output is not supported"):
        builder.prepare_message()


def test_handle_tool_call_without_responded_tool_calls():
    builder = LiteLLMAPIMessageBuilder("tool")
    with pytest.raises(ValueError, match="No tool calls found"):
        builder.handle_tool_call()


def test_format_tools_for_chat_completion():
    formatted = LiteLLMModel.format_tools_for_chat_completion(chat_api_tools)
    assert formatted[0] == {
        "type": "function",
        "function": {
            "name": chat_api_tools[0]["name"],
            "description": chat_api_tools[0]["description"],
            "parameters": chat_api_tools[0]["parameters"],
        },
    }
    assert LiteLLMModel.format_tools_for_chat_completion(None) is None


if __name__ == "__main__":
    test_multi_action_tool_calls()
    test_force_tool_call()
    test_single_tool_call()
