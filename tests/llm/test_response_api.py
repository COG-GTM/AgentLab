import os
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import anthropic
import openai
import pytest

from agentlab.llm import tracking
from agentlab.llm.response_api import (
    AnthropicAPIMessageBuilder,
    APIPayload,
    ClaudeResponseModelArgs,
    LLMOutput,
    OpenAIChatCompletionAPIMessageBuilder,
    OpenAIChatModelArgs,
    OpenAIResponseAPIMessageBuilder,
    OpenAIResponseModelArgs,
    ToolCall,
    ToolCalls,
)


# Helper to create a mock OpenAI ChatCompletion response
def create_mock_openai_chat_completion(
    content=None, tool_calls=None, prompt_tokens=10, completion_tokens=20
):
    completion = MagicMock(spec=openai.types.chat.ChatCompletion)
    choice = MagicMock()
    message = MagicMock(spec=openai.types.chat.ChatCompletionMessage)
    message.content = content
    message.tool_calls = None
    if tool_calls:
        message.tool_calls = []
        for tc in tool_calls:
            tool_call_mock = MagicMock(
                spec=openai.types.chat.chat_completion_message_tool_call.ChatCompletionMessageToolCall
            )
            tool_call_mock.id = tc["id"]
            tool_call_mock.type = tc["type"]
            tool_call_mock.function = MagicMock()
            tool_call_mock.function.name = tc["function"]["name"]
            tool_call_mock.function.arguments = tc["function"]["arguments"]
            message.tool_calls.append(tool_call_mock)

    choice.message = message
    completion.choices = [choice]

    completion.usage = MagicMock()
    # Explicitly set the attributes that get_tokens_counts_from_response will try first.
    # These are the generic names.
    completion.usage.input_tokens = prompt_tokens
    completion.usage.output_tokens = completion_tokens

    # Also set the OpenAI-specific names if any other part of the code might look for them directly,
    # or if get_tokens_counts_from_response had different fallback logic.
    completion.usage.prompt_tokens = prompt_tokens
    completion.usage.completion_tokens = completion_tokens
    prompt_tokens_details_mock = MagicMock()
    prompt_tokens_details_mock.cached_tokens = 0
    completion.usage.prompt_tokens_details = prompt_tokens_details_mock

    completion.model_dump.return_value = {
        "id": "chatcmpl-xxxx",
        "choices": [
            {"message": {"role": "assistant", "content": content, "tool_calls": tool_calls}}
        ],
        # Ensure the usage dict in model_dump also reflects the token counts accurately.
        # The get_tokens_counts_from_response also has a path for dict style.
        "usage": {
            "input_tokens": prompt_tokens,  # Generic name
            "output_tokens": completion_tokens,  # Generic name
            "prompt_tokens": prompt_tokens,  # OpenAI specific
            "completion_tokens": completion_tokens,  # OpenAI specific
            "prompt_tokens_details": {"cached_tokens": 0},
        },
    }
    message.to_dict.return_value = {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls,
    }
    return completion


responses_api_tools = [
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
    }
]

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
    }
]
anthropic_tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather in a given location.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location to get the weather for.",
                },
            },
            "required": ["location"],
        },
    }
]


# Helper to create a mock Anthropic response
def create_mock_anthropic_response(
    text_content=None, tool_use=None, input_tokens=15, output_tokens=25
):

    response = MagicMock(spec=anthropic.types.Message)
    response.type = "message"  # Explicitly set the type attribute
    response.content = []
    response.content = []
    if text_content:
        text_block = MagicMock(spec=anthropic.types.TextBlock)
        text_block.type = "text"
        text_block.text = text_content
        response.content.append(text_block)
    if tool_use:
        tool_use_block = MagicMock(spec=anthropic.types.ToolUseBlock)
        tool_use_block.type = "tool_use"
        tool_use_block.id = tool_use["id"]
        tool_use_block.name = tool_use["name"]
        tool_use_block.input = tool_use["input"]
        response.content.append(tool_use_block)
    response.usage = MagicMock()
    response.usage.input_tokens = input_tokens
    response.usage.output_tokens = output_tokens
    response.usage.cache_input_tokens = 0
    response.usage.cache_creation_input_tokens = 0
    return response


def create_mock_openai_responses_api_response(
    outputs: Optional[List[Dict[str, Any]]] = None, input_tokens: int = 10, output_tokens: int = 20
) -> MagicMock:
    """
    Helper to create a mock response object similar to what
    openai.resources.Responses.create() would return.
    Compatible with OpenAIResponseModel and TrackAPIPricingMixin.
    """

    response_mock = MagicMock(spec=openai.types.responses.response.Response)
    response_mock.type = "response"
    response_mock.output = []

    if outputs:
        for out_data in outputs:
            output_item_mock = MagicMock()
            output_item_mock.type = out_data.get("type")

            if output_item_mock.type == "function_call":
                # You can adapt this depending on your expected object structure
                output_item_mock.name = out_data.get("name")
                output_item_mock.arguments = out_data.get("arguments")
                output_item_mock.call_id = out_data.get("call_id")
            elif output_item_mock.type == "reasoning":
                output_item_mock.summary = []
                for text_content in out_data.get("summary", []):
                    summary_text_mock = MagicMock()
                    summary_text_mock.text = text_content
                    output_item_mock.summary.append(summary_text_mock)

            response_mock.output.append(output_item_mock)

    # Token usage for pricing tracking
    response_mock.usage = MagicMock(spec=openai.types.responses.response.ResponseUsage)
    response_mock.usage.input_tokens = input_tokens
    response_mock.usage.output_tokens = output_tokens
    response_mock.usage.prompt_tokens = input_tokens
    response_mock.usage.completion_tokens = output_tokens
    input_tokens_details_mock = MagicMock()
    input_tokens_details_mock.cached_tokens = 0
    response_mock.usage.input_tokens_details = input_tokens_details_mock

    return response_mock


# --- Test MessageBuilders ---


def test_openai_response_api_message_builder_text():
    builder = OpenAIResponseAPIMessageBuilder.user()
    builder.add_text("Hello, world!")
    messages = builder.prepare_message()
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == [{"type": "input_text", "text": "Hello, world!"}]


def test_openai_response_api_message_builder_image():
    builder = OpenAIResponseAPIMessageBuilder.user()
    builder.add_image("data:image/png;base64,SIMPLEBASE64STRING")
    messages = builder.prepare_message()
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == [
        {"type": "input_image", "image_url": "data:image/png;base64,SIMPLEBASE64STRING"}
    ]


def test_anthropic_api_message_builder_text():
    builder = AnthropicAPIMessageBuilder.user()
    builder.add_text("Hello, Anthropic!")
    messages = builder.prepare_message()
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == [{"type": "text", "text": "Hello, Anthropic!"}]


def test_anthropic_api_message_builder_image():
    builder = AnthropicAPIMessageBuilder.user()
    builder.add_image("data:image/png;base64,ANTHROPICBASE64")
    messages = builder.prepare_message()
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert len(messages[0]["content"]) == 1
    image_content = messages[0]["content"][0]
    assert image_content["type"] == "image"
    assert image_content["source"]["type"] == "base64"
    assert image_content["source"]["media_type"] == "image/png"
    assert image_content["source"]["data"] == "ANTHROPICBASE64"  # Base64 prefix should be stripped


def test_openai_chat_completion_api_message_builder_text():
    builder = OpenAIChatCompletionAPIMessageBuilder.user()
    builder.add_text("Hello, ChatCompletion!")
    messages = builder.prepare_message()

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == [{"type": "text", "text": "Hello, ChatCompletion!"}]


def test_openai_chat_completion_api_message_builder_image():
    builder = OpenAIChatCompletionAPIMessageBuilder.user()
    builder.add_image("data:image/jpeg;base64,CHATCOMPLETIONBASE64")
    messages = builder.prepare_message()

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,CHATCOMPLETIONBASE64"}}
    ]


def test_openai_chat_completion_model_parse_and_cost():
    args = OpenAIChatModelArgs(model_name="gpt-3.5-turbo")
    with patch("agentlab.llm.response_api.OpenAI") as mock_openai_class:
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        model = args.make_model()

    mock_response = create_mock_openai_chat_completion(
        content="This is a test thought.",
        tool_calls=[
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
            }
        ],
        prompt_tokens=50,
        completion_tokens=30,
    )

    with patch.object(
        model.client.chat.completions, "create", return_value=mock_response
    ) as mock_create:
        with tracking.set_tracker() as global_tracker:
            messages = [
                OpenAIChatCompletionAPIMessageBuilder.user().add_text(
                    "What's the weather in Paris?"
                )
            ]
            payload = APIPayload(messages=messages)
            parsed_output = model(payload)

    mock_create.assert_called_once()
    assert parsed_output.raw_response.choices[0].message.content == "This is a test thought."
    assert parsed_output.action == """get_weather(location='Paris')"""
    assert parsed_output.raw_response.choices[0].message.tool_calls[0].id == "call_123"
    # Check cost tracking (token counts)
    assert global_tracker.stats["input_tokens"] == 50
    assert global_tracker.stats["output_tokens"] == 30
    assert global_tracker.stats["cost"] > 0


def test_claude_response_model_parse_and_cost():
    args = ClaudeResponseModelArgs(model_name="claude-3-haiku-20240307")
    model = args.make_model()

    mock_anthropic_api_response = create_mock_anthropic_response(
        text_content="Thinking about the request.",
        tool_use={"id": "tool_abc", "name": "search_web", "input": {"query": "latest news"}},
        input_tokens=40,
        output_tokens=20,
    )

    with patch.object(
        model.client.messages, "create", return_value=mock_anthropic_api_response
    ) as mock_create:
        with tracking.set_tracker() as global_tracker:
            messages = [AnthropicAPIMessageBuilder.user().add_text("Search for latest news")]
            payload = APIPayload(messages=messages)
            parsed_output = model(payload)

    mock_create.assert_called_once()
    fn_call = next(iter(parsed_output.tool_calls))

    assert "Thinking about the request." in parsed_output.think
    assert parsed_output.action == """search_web(query='latest news')"""
    assert fn_call.name == "search_web"
    assert global_tracker.stats["input_tokens"] == 40
    assert global_tracker.stats["output_tokens"] == 20


def test_openai_response_model_parse_and_cost():
    args = OpenAIResponseModelArgs(model_name="gpt-4.1")

    mock_function_call_output = {
        "type": "function_call",
        "name": "get_current_weather",
        "arguments": '{"location": "Boston, MA", "unit": "celsius"}',
        "call_id": "call_abc123",
    }

    mock_api_resp = create_mock_openai_responses_api_response(
        outputs=[mock_function_call_output],
        input_tokens=70,
        output_tokens=40,
    )

    with patch("agentlab.llm.response_api.OpenAI") as mock_openai_class:
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        model = args.make_model()

    with patch.object(
        model.client.responses, "create", return_value=mock_api_resp
    ) as mock_create_method:
        with tracking.set_tracker() as global_tracker:
            messages = [
                OpenAIResponseAPIMessageBuilder.user().add_text("What's the weather in Boston?")
            ]
            payload = APIPayload(messages=messages)
            parsed_output = model(payload)

    mock_create_method.assert_called_once()
    fn_calls = [
        content
        for content in parsed_output.tool_calls.raw_calls.output
        if content.type == "function_call"
    ]
    assert parsed_output.action == "get_current_weather(location='Boston, MA', unit='celsius')"
    assert fn_calls[0].call_id == "call_abc123"
    assert parsed_output.raw_response == mock_api_resp
    assert global_tracker.stats["input_tokens"] == 70
    assert global_tracker.stats["output_tokens"] == 40


# --- Test multi-turn tool-call formatting (mocked, non-pricy) ---


def _make_openai_responses_computer_call_tool_calls(call_id="cu_call_1"):
    computer_call = MagicMock()
    computer_call.type = "computer_call"
    computer_call.call_id = call_id
    raw_response = create_mock_openai_responses_api_response()
    raw_response.output.append(computer_call)
    tool_call = ToolCall(name="click", arguments={"x": 1, "y": 2}, raw_call=computer_call)
    return ToolCalls(tool_calls=[tool_call], raw_calls=raw_response)


def test_openai_response_model_multi_turn_tool_call_formatting():
    args = OpenAIResponseModelArgs(model_name="gpt-4.1")
    first_response = create_mock_openai_responses_api_response(
        outputs=[
            {"type": "reasoning", "summary": ["Need the weather tool."]},
            {
                "type": "function_call",
                "name": "get_weather",
                "arguments": '{"location": "Paris"}',
                "call_id": "call_paris",
            },
        ]
    )
    second_response = create_mock_openai_responses_api_response(
        outputs=[
            {
                "type": "function_call",
                "name": "get_weather",
                "arguments": '{"location": "Delhi"}',
                "call_id": "call_delhi",
            }
        ]
    )

    with patch("agentlab.llm.response_api.OpenAI") as mock_openai_class:
        mock_openai_class.return_value = MagicMock()
        model = args.make_model()
    builder = args.get_message_builder()
    assert builder is OpenAIResponseAPIMessageBuilder

    messages = [builder.user().add_text("What is the weather in Paris?")]
    with patch.object(
        model.client.responses, "create", side_effect=[first_response, second_response]
    ) as mock_create:
        first_output = model(APIPayload(messages=messages, tools=responses_api_tools))
        assert len(first_output.tool_calls) == 1
        for tool_call in first_output.tool_calls:
            tool_call.response_text("It's sunny! 25°C")
        messages += [
            builder.add_responded_tool_calls(first_output.tool_calls),
            builder.user().add_text("What is the weather in Delhi?"),
        ]
        second_output = model(APIPayload(messages=messages, tools=responses_api_tools))

    assert mock_create.call_count == 2
    sent_input = mock_create.call_args_list[1].kwargs["input"]
    assert sent_input == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "What is the weather in Paris?"}],
        },
        first_response.output[0],
        first_response.output[1],
        {
            "type": "function_call_output",
            "call_id": "call_paris",
            "output": "It's sunny! 25°C",
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "What is the weather in Delhi?"}],
        },
    ]
    assert second_output.action == "get_weather(location='Delhi')"


def test_openai_response_api_message_builder_computer_call_output():
    tool_calls = _make_openai_responses_computer_call_tool_calls(call_id="cu_call_1")
    next(iter(tool_calls)).response_image("data:image/png;base64,SCREENSHOT")

    messages = OpenAIResponseAPIMessageBuilder.add_responded_tool_calls(
        tool_calls
    ).prepare_message()

    assert messages == [
        tool_calls.raw_calls.output[0],
        {
            "type": "computer_call_output",
            "call_id": "cu_call_1",
            "output": {"type": "input_image", "image_url": "data:image/png;base64,SCREENSHOT"},
        },
    ]


def test_openai_response_api_message_builder_rejects_image_in_function_call_response():
    raw_response = create_mock_openai_responses_api_response(
        outputs=[
            {
                "type": "function_call",
                "name": "get_weather",
                "arguments": '{"location": "Paris"}',
                "call_id": "call_paris",
            }
        ]
    )
    tool_call = ToolCall(
        name="get_weather", arguments={"location": "Paris"}, raw_call=raw_response.output[0]
    ).response_image("data:image/png;base64,IMG")
    tool_calls = ToolCalls(tool_calls=[tool_call], raw_calls=raw_response)

    msg = OpenAIResponseAPIMessageBuilder.add_responded_tool_calls(tool_calls)
    with pytest.raises(AssertionError, match="Image output is not supported"):
        msg.prepare_message()


def test_openai_response_api_message_builder_rejects_text_in_computer_call_response():
    tool_calls = _make_openai_responses_computer_call_tool_calls()
    next(iter(tool_calls)).response_text("not a screenshot")

    msg = OpenAIResponseAPIMessageBuilder.add_responded_tool_calls(tool_calls)
    with pytest.raises(AssertionError, match="Text output is not supported"):
        msg.prepare_message()


def test_claude_model_multi_turn_tool_call_formatting():
    args = ClaudeResponseModelArgs(model_name="claude-3-haiku-20240307")
    model = args.make_model()
    builder = args.get_message_builder()
    assert builder is AnthropicAPIMessageBuilder

    first_response = create_mock_anthropic_response(
        text_content="Let me check.",
        tool_use={"id": "toolu_paris", "name": "get_weather", "input": {"location": "Paris"}},
    )
    second_tool_use = MagicMock(spec=anthropic.types.ToolUseBlock)
    second_tool_use.type = "tool_use"
    second_tool_use.id = "toolu_rome"
    second_tool_use.name = "get_weather"
    second_tool_use.input = {"location": "Rome"}
    first_response.content.append(second_tool_use)
    second_response = create_mock_anthropic_response(
        tool_use={"id": "toolu_delhi", "name": "get_weather", "input": {"location": "Delhi"}},
    )

    messages = [builder.user().add_text("What is the weather in Paris and Rome?")]
    with patch.object(
        model.client.messages, "create", side_effect=[first_response, second_response]
    ) as mock_create:
        first_output = model(APIPayload(messages=messages, tools=anthropic_tools))
        assert len(first_output.tool_calls) == 2
        paris_call, rome_call = first_output.tool_calls
        paris_call.response_text("Sunny, 25°C")
        rome_call.response_text("Cloudy, 18°C")
        messages += [
            builder.add_responded_tool_calls(first_output.tool_calls),
            builder.user().add_text("What is the weather in Delhi?"),
        ]
        second_output = model(APIPayload(messages=messages, tools=anthropic_tools))

    assert mock_create.call_count == 2
    sent_messages = mock_create.call_args_list[1].kwargs["messages"]
    assert sent_messages == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "What is the weather in Paris and Rome?"}],
        },
        {"role": "assistant", "content": first_response.content},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_paris", "content": "Sunny, 25°C"},
                {"type": "tool_result", "tool_use_id": "toolu_rome", "content": "Cloudy, 18°C"},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "What is the weather in Delhi?"}],
        },
    ]
    assert second_output.action == "get_weather(location='Delhi')"


def test_anthropic_api_message_builder_rejects_image_in_tool_response():
    raw_response = create_mock_anthropic_response(
        tool_use={"id": "toolu_1", "name": "get_weather", "input": {"location": "Paris"}},
    )
    tool_call = ToolCall(
        name="get_weather", arguments={"location": "Paris"}, raw_call=raw_response.content[0]
    ).response_image("data:image/png;base64,IMG")
    tool_calls = ToolCalls(tool_calls=[tool_call], raw_calls=raw_response)

    msg = AnthropicAPIMessageBuilder.add_responded_tool_calls(tool_calls)
    with pytest.raises(AssertionError, match="Image output is not supported"):
        msg.prepare_message()


def test_openai_chat_completion_model_multi_turn_tool_call_formatting():
    args = OpenAIChatModelArgs(model_name="gpt-4.1")
    with patch("agentlab.llm.response_api.OpenAI") as mock_openai_class:
        mock_openai_class.return_value = MagicMock()
        model = args.make_model()
    builder = args.get_message_builder()
    assert builder is OpenAIChatCompletionAPIMessageBuilder

    first_response = create_mock_openai_chat_completion(
        tool_calls=[
            {
                "id": "call_paris",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
            },
            {
                "id": "call_rome",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"location": "Rome"}'},
            },
        ]
    )
    second_response = create_mock_openai_chat_completion(
        tool_calls=[
            {
                "id": "call_delhi",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"location": "Delhi"}'},
            }
        ]
    )

    messages = [builder.user().add_text("What is the weather in Paris and Rome?")]
    with patch.object(
        model.client.chat.completions, "create", side_effect=[first_response, second_response]
    ) as mock_create:
        first_output = model(APIPayload(messages=messages, tools=chat_api_tools))
        assert len(first_output.tool_calls) == 2
        paris_call, rome_call = first_output.tool_calls
        paris_call.response_text("Sunny, 25°C")
        rome_call.response_text("Cloudy, 18°C")
        messages += [
            builder.add_responded_tool_calls(first_output.tool_calls),
            builder.user().add_text("What is the weather in Delhi?"),
        ]
        second_output = model(APIPayload(messages=messages, tools=chat_api_tools))

    assert mock_create.call_count == 2
    sent_messages = mock_create.call_args_list[1].kwargs["messages"]
    assert sent_messages == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "What is the weather in Paris and Rome?"}],
        },
        first_response.choices[0].message,
        {
            "name": "get_weather",
            "role": "tool",
            "tool_call_id": "call_paris",
            "content": "Sunny, 25°C",
        },
        {
            "name": "get_weather",
            "role": "tool",
            "tool_call_id": "call_rome",
            "content": "Cloudy, 18°C",
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "What is the weather in Delhi?"}],
        },
    ]
    assert second_output.action == "get_weather(location='Delhi')"


def test_openai_chat_completion_api_message_builder_rejects_image_in_tool_response():
    raw_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
    }
    raw_response = create_mock_openai_chat_completion(tool_calls=[raw_call])
    tool_call = ToolCall(
        name="get_weather", arguments={"location": "Paris"}, raw_call=raw_call
    ).response_image("data:image/png;base64,IMG")
    tool_calls = ToolCalls(tool_calls=[tool_call], raw_calls=raw_response)

    msg = OpenAIChatCompletionAPIMessageBuilder.add_responded_tool_calls(tool_calls)
    with pytest.raises(AssertionError, match="Image output is not supported"):
        msg.prepare_message()


TOOL_MESSAGE_BUILDERS = [
    OpenAIResponseAPIMessageBuilder,
    AnthropicAPIMessageBuilder,
    OpenAIChatCompletionAPIMessageBuilder,
]


@pytest.mark.parametrize("builder_cls", TOOL_MESSAGE_BUILDERS)
def test_add_responded_tool_calls_requires_all_responses(builder_cls):
    answered = ToolCall(name="get_weather", arguments={"location": "Paris"}).response_text("Sunny")
    unanswered = ToolCall(name="get_weather", arguments={"location": "Rome"})
    tool_calls = ToolCalls(tool_calls=[answered, unanswered])

    with pytest.raises(AssertionError, match="All tool calls must have a response"):
        builder_cls.add_responded_tool_calls(tool_calls)


@pytest.mark.parametrize("builder_cls", TOOL_MESSAGE_BUILDERS)
def test_add_responded_tool_calls_builds_tool_message(builder_cls):
    tool_call = ToolCall(name="get_weather", arguments={"location": "Paris"}).response_text("Sunny")
    tool_calls = ToolCalls(tool_calls=[tool_call])

    msg = builder_cls.add_responded_tool_calls(tool_calls)

    assert isinstance(msg, builder_cls)
    assert msg.role == "tool"
    assert msg.responded_tool_calls is tool_calls
    assert msg.content == []


@pytest.mark.parametrize("builder_cls", TOOL_MESSAGE_BUILDERS)
def test_tool_message_without_responded_tool_calls_raises(builder_cls):
    with pytest.raises(ValueError, match="No tool calls found"):
        builder_cls("tool").prepare_message()


# --- Test Response Models (Pricy - require API keys and actual calls) ---


@pytest.mark.pricy
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_openai_chat_completion_model_pricy_call():
    """Tests OpenAIChatCompletionModel with a real API call."""
    args = OpenAIChatModelArgs(
        model_name="gpt-4.1",
        temperature=1e-5,
        max_new_tokens=100,
    )

    tools = chat_api_tools
    model = args.make_model()

    with tracking.set_tracker() as global_tracker:
        messages = [
            OpenAIChatCompletionAPIMessageBuilder.user().add_text("What is the weather in Paris?")
        ]
        payload = APIPayload(messages=messages, tools=tools, tool_choice="required")
        parsed_output = model(payload)

    assert parsed_output.raw_response is not None
    assert (
        parsed_output.action == "get_weather(location='Paris')"
    ), f""" Expected get_weather(location='Paris') but got {parsed_output.action}"""
    assert global_tracker.stats["input_tokens"] > 0
    assert global_tracker.stats["output_tokens"] > 0
    assert global_tracker.stats["cost"] > 0


@pytest.mark.pricy
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
def test_claude_response_model_pricy_call():
    """Tests ClaudeResponseModel with a real API call."""

    args = ClaudeResponseModelArgs(
        model_name="claude-3-haiku-20240307",
        temperature=1e-5,
        max_new_tokens=100,
    )
    tools = anthropic_tools
    model = args.make_model()

    with tracking.set_tracker() as global_tracker:
        messages = [AnthropicAPIMessageBuilder.user().add_text("What is the weather in Paris?")]
        payload = APIPayload(messages=messages, tools=tools)
        parsed_output = model(payload)

    assert parsed_output.raw_response is not None
    assert (
        parsed_output.action == "get_weather(location='Paris')"
    ), f"""Expected get_weather('Paris') but got {parsed_output.action}"""
    assert global_tracker.stats["input_tokens"] > 0
    assert global_tracker.stats["output_tokens"] > 0
    assert global_tracker.stats["cost"] > 0


@pytest.mark.pricy
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_openai_response_model_pricy_call():
    """
    Tests OpenAIResponseModel output parsing and cost tracking with both
    function_call and reasoning outputs.
    """
    args = OpenAIResponseModelArgs(model_name="gpt-4.1", temperature=1e-5, max_new_tokens=100)

    tools = responses_api_tools
    model = args.make_model()

    with tracking.set_tracker() as global_tracker:
        messages = [
            OpenAIResponseAPIMessageBuilder.user().add_text("What is the weather in Paris?")
        ]
        payload = APIPayload(messages=messages, tools=tools)
        parsed_output = model(payload)

    assert parsed_output.raw_response is not None
    assert (
        parsed_output.action == """get_weather(location='Paris', unit='celsius')"""
    ), f""" Expected get_weather(location='Paris', unit='celsius') but got {parsed_output.action}"""
    assert global_tracker.stats["input_tokens"] > 0
    assert global_tracker.stats["output_tokens"] > 0
    assert global_tracker.stats["cost"] > 0


@pytest.mark.pricy
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_openai_response_model_with_multiple_messages_and_cost_tracking():
    """
    Test OpenAIResponseModel's output parsing and cost tracking
    with a tool-using assistant and follow-up interaction.
    """
    args = OpenAIResponseModelArgs(model_name="gpt-4.1", temperature=1e-5, max_new_tokens=100)

    tools = responses_api_tools
    model = args.make_model()
    builder = args.get_message_builder()

    messages = [builder.user().add_text("What is the weather in Paris?")]

    with tracking.set_tracker() as tracker:
        payload = APIPayload(messages=messages, tools=tools, tool_choice="required")
        parsed = model(payload)
        prev_input = tracker.stats["input_tokens"]
        prev_output = tracker.stats["output_tokens"]
        prev_cost = tracker.stats["cost"]

        assert parsed.tool_calls, "Expected tool calls in the response"
        # Set tool responses
        for tool_call in parsed.tool_calls:
            tool_call.response_text("Its sunny! 25°C")
        # Simulate tool execution and user follow-up
        messages += [
            builder.add_responded_tool_calls(parsed.tool_calls),
            builder.user().add_text("What is the weather in Delhi?"),
        ]

        payload = APIPayload(messages=messages, tools=tools, tool_choice="required")
        parsed = model(payload)

        delta_input = tracker.stats["input_tokens"] - prev_input
        delta_output = tracker.stats["output_tokens"] - prev_output
        delta_cost = tracker.stats["cost"] - prev_cost

    assert prev_input > 0
    assert prev_output > 0
    assert prev_cost > 0
    assert parsed.raw_response is not None
    assert (
        parsed.action == """get_weather(location='Delhi', unit='celsius')"""
    ), f"Unexpected action: {parsed.action}"
    assert delta_input > 0
    assert delta_output > 0
    assert delta_cost > 0
    assert tracker.stats["input_tokens"] == prev_input + delta_input
    assert tracker.stats["output_tokens"] == prev_output + delta_output
    assert tracker.stats["cost"] == pytest.approx(prev_cost + delta_cost)


@pytest.mark.pricy
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_openai_chat_completion_model_with_multiple_messages_and_cost_tracking():
    """
    Test OpenAIResponseModel's output parsing and cost tracking
    with a tool-using assistant and follow-up interaction.
    """
    args = OpenAIChatModelArgs(model_name="gpt-4.1", temperature=1e-5, max_new_tokens=100)

    tools = [
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
        }
    ]

    model = args.make_model()
    builder = args.get_message_builder()

    messages = [builder.user().add_text("What is the weather in Paris?")]

    with tracking.set_tracker() as tracker:
        payload = APIPayload(messages=messages, tools=tools, tool_choice="required")
        parsed = model(payload)
        prev_input = tracker.stats["input_tokens"]
        prev_output = tracker.stats["output_tokens"]
        prev_cost = tracker.stats["cost"]

        for tool_call in parsed.tool_calls:
            tool_call.response_text("Its sunny! 25°C")
        # Simulate tool execution and user follow-up
        messages += [
            builder.add_responded_tool_calls(parsed.tool_calls),
            builder.user().add_text("What is the weather in Delhi?"),
        ]
        # Set tool responses

        payload = APIPayload(messages=messages, tools=tools, tool_choice="required")
        parsed = model(payload)

        delta_input = tracker.stats["input_tokens"] - prev_input
        delta_output = tracker.stats["output_tokens"] - prev_output
        delta_cost = tracker.stats["cost"] - prev_cost

    assert prev_input > 0
    assert prev_output > 0
    assert prev_cost > 0
    assert parsed.raw_response is not None
    assert (
        parsed.action == """get_weather(location='Delhi')"""
    ), f"Unexpected action: {parsed.action}"
    assert delta_input > 0
    assert delta_output > 0
    assert delta_cost > 0
    assert tracker.stats["input_tokens"] == prev_input + delta_input
    assert tracker.stats["output_tokens"] == prev_output + delta_output
    assert tracker.stats["cost"] == pytest.approx(prev_cost + delta_cost)


@pytest.mark.pricy
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
def test_claude_model_with_multiple_messages_pricy_call():
    model_factory = ClaudeResponseModelArgs(
        model_name="claude-3-haiku-20240307", temperature=1e-5, max_new_tokens=100
    )
    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location.",
            "input_schema": {
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
        }
    ]
    model = model_factory.make_model()
    msg_builder = model_factory.get_message_builder()
    messages = []

    messages.append(msg_builder.user().add_text("What is the weather in Paris?"))
    with tracking.set_tracker() as global_tracker:
        payload = APIPayload(messages=messages, tools=tools)
        llm_output1 = model(payload)

        prev_input = global_tracker.stats["input_tokens"]
        prev_output = global_tracker.stats["output_tokens"]
        prev_cost = global_tracker.stats["cost"]

        for tool_call in llm_output1.tool_calls:
            tool_call.response_text("It's sunny! 25°C")
        messages += [
            msg_builder.add_responded_tool_calls(llm_output1.tool_calls),
            msg_builder.user().add_text("What is the weather in Delhi?"),
        ]
        payload = APIPayload(messages=messages, tools=tools)
        llm_output2 = model(payload)
        delta_input = global_tracker.stats["input_tokens"] - prev_input
        delta_output = global_tracker.stats["output_tokens"] - prev_output
        delta_cost = global_tracker.stats["cost"] - prev_cost

    assert prev_input > 0, "Expected previous input tokens to be greater than 0"
    assert prev_output > 0, "Expected previous output tokens to be greater than 0"
    assert prev_cost > 0, "Expected previous cost value to be greater than 0"
    assert llm_output2.raw_response is not None
    assert (
        llm_output2.action == """get_weather(location='Delhi', unit='celsius')"""
    ), f"""Expected get_weather('Delhi') but got {llm_output2.action}"""
    assert delta_input > 0, "Expected new input tokens to be greater than 0"
    assert delta_output > 0, "Expected new output tokens to be greater than 0"
    assert delta_cost > 0, "Expected new cost value to be greater than 0"
    assert global_tracker.stats["input_tokens"] == prev_input + delta_input
    assert global_tracker.stats["output_tokens"] == prev_output + delta_output
    assert global_tracker.stats["cost"] == pytest.approx(prev_cost + delta_cost)


## Test multiaction
@pytest.mark.pricy
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Skipping as OpenAI API key not set")
def test_multi_action_tool_calls():
    """
    Test that the model can produce multiple tool calls in parallel.
    Uncomment commented lines to see the full behaviour of models and tool choices.
    """
    # test_config (setting name, BaseModelArgs, model_name, tools)
    tool_test_configs = [
        (
            "gpt-4.1-responses API",
            OpenAIResponseModelArgs,
            "gpt-4.1-2025-04-14",
            responses_api_tools,
        ),
        ("gpt-4.1-chat Completions API", OpenAIChatModelArgs, "gpt-4.1-2025-04-14", chat_api_tools),
        # ("claude-3", ClaudeResponseModelArgs, "claude-3-haiku-20240307", anthropic_tools),   # fails
        # ("claude-3.7", ClaudeResponseModelArgs, "claude-3-7-sonnet-20250219", anthropic_tools), # fails
        ("claude-4-sonnet", ClaudeResponseModelArgs, "claude-sonnet-4-20250514", anthropic_tools),
        # add more models as needed
    ]

    def add_user_messages(msg_builder):
        return [
            msg_builder.user().add_text("What is the weather in Paris and Delhi?"),
            msg_builder.user().add_text("You must call multiple tools to achieve the task."),
        ]

    res_df = []

    for tool_choice in [
        # 'none',
        # 'required', # fails for Responses API
        # 'any',  # fails for Responses API
        "auto",
        # 'get_weather'
    ]:
        for name, llm_class, checkpoint_name, tools in tool_test_configs:
            print(name, "tool choice:", tool_choice, "\n", "**" * 10)
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
            res_df.append(
                {
                    "model": name,
                    "checkpoint": checkpoint_name,
                    "tool_choice": tool_choice,
                    "num_tool_calls": num_tool_calls,
                    "action": response.action,
                }
            )
            assert (
                num_tool_calls == 2
            ), f"Expected 2 tool calls, but got {num_tool_calls} for {name} with tool choice {tool_choice}"
        # import pandas as pd
        # print(pd.DataFrame(res_df))


EDGE_CASES = [
    # 1. Empty kwargs dict
    ("valid_function", {}, "valid_function()"),
    # 2. Kwargs with problematic string values (quotes, escapes, unicode)
    (
        "send_message",
        {
            "text": 'He said "Hello!" and used a backslash: \\',
            "unicode": "Café naïve résumé 🚀",
            "newlines": "Line1\nLine2\tTabbed",
        },
        "send_message(text='He said \"Hello!\" and used a backslash: \\\\', unicode='Café naïve résumé 🚀', newlines='Line1\\nLine2\\tTabbed')",
    ),
    # 3. Mixed types including problematic float values
    (
        "complex_call",
        {
            "infinity": float("inf"),
            "nan": float("nan"),
            "negative_zero": -0.0,
            "scientific": 1.23e-45,
        },
        "complex_call(infinity=inf, nan=nan, negative_zero=-0.0, scientific=1.23e-45)",
    ),
    # 4. Deeply nested structures that could stress repr()
    (
        "process_data",
        {
            "nested": {"level1": {"level2": {"level3": [1, 2, {"deep": True}]}}},
            "circular_ref_like": {"a": {"b": {"c": "back_to_start"}}},
        },
        "process_data(nested={'level1': {'level2': {'level3': [1, 2, {'deep': True}]}}}, circular_ref_like={'a': {'b': {'c': 'back_to_start'}}})",
    ),
]


def test_tool_call_to_python_code():
    from agentlab.llm.response_api import tool_call_to_python_code

    for edge_case in EDGE_CASES:
        func_name, kwargs, expected = edge_case
        result = tool_call_to_python_code(func_name, kwargs)
        print(result)
        assert result == expected, f"Expected {expected} but got {result}"


if __name__ == "__main__":
    test_tool_call_to_python_code()
    # test_openai_chat_completion_model_parse_and_cost()
