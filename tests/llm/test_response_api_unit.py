"""Hermetic unit tests for agentlab.llm.response_api (no network, no API keys)."""

import json
import logging
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from anthropic.types import Usage as AnthropicUsage
from openai.types.chat import ChatCompletion
from openai.types.responses import ResponseUsage

import agentlab.llm.response_api as ra
import agentlab.llm.tracking as tracking
from agentlab.llm.response_api import (
    AgentlabAction,
    AnthropicAPIMessageBuilder,
    APIPayload,
    BaseResponseModel,
    ClaudeResponseModel,
    ClaudeResponseModelArgs,
    LLMOutput,
    MessageBuilder,
    OpenAIChatCompletionAPIMessageBuilder,
    OpenAIChatCompletionModel,
    OpenAIChatModelArgs,
    OpenAIResponseAPIMessageBuilder,
    OpenAIResponseModel,
    OpenAIResponseModelArgs,
    OpenRouterModelArgs,
    ToolCall,
    ToolCalls,
    tool_call_to_python_code,
)

PRICES = {"prompt": 1e-3, "completion": 2e-3}


@pytest.fixture(autouse=True)
def _offline_pricing(monkeypatch):
    """Keep model construction hermetic: no pricing lookups, no stray env keys."""
    table = {name: PRICES for name in ("gpt-test", "claude-test", "router/test")}
    monkeypatch.setattr(tracking, "get_pricing_openai", lambda: table)
    monkeypatch.setattr(tracking, "get_pricing_anthropic", lambda: table)
    previous = getattr(tracking.TRACKER, "instance", None)
    tracking.TRACKER.instance = None
    yield
    tracking.TRACKER.instance = previous


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)


# --------------------------------------------------------------------------- ToolCall / ToolCalls


class TestToolCall:
    def test_response_setters_and_flag(self):
        call = ToolCall(name="click", arguments={"bid": "a1"})
        assert not call.is_response_set
        assert call.response_text("ok") is call
        assert call.tool_response == {"text": "ok"} and call.is_response_set
        call.response_image("data:img")
        assert call.tool_response == {"image": "data:img"}

    def test_repr_hides_raw_call(self):
        call = ToolCall(name="fill", arguments={"text": "x"}, raw_call=object())
        assert repr(call) == "ToolCall(name=fill, arguments={'text': 'x'})"

    def test_collection_protocol(self):
        calls = ToolCalls()
        assert not calls and len(calls) == 0 and calls.all_responses_set  # vacuous truth
        a, b = ToolCall(name="a"), ToolCall(name="b")
        assert calls.add_tool_call(a).add_tool_call(b) is calls
        assert bool(calls) and len(calls) == 2 and list(calls) == [a, b]
        a.response_text("done")
        assert not calls.all_responses_set
        b.response_image("img")
        assert calls.all_responses_set

    def test_defaults_not_shared(self):
        assert ToolCalls().tool_calls is not ToolCalls().tool_calls
        assert ToolCall().arguments is not ToolCall().arguments
        out = LLMOutput()
        assert (out.raw_response, out.think, out.action, out.tool_calls) == (None, "", None, None)


# --------------------------------------------------------------------------- MessageBuilder base


class TestMessageBuilderBase:
    @pytest.mark.parametrize(
        "factory, role", [("system", "system"), ("user", "user"), ("assistant", "assistant")]
    )
    def test_role_factories(self, factory, role):
        msg = getattr(MessageBuilder, factory)()
        assert msg.role == role and msg.content == [] and msg.responded_tool_calls is None

    def test_add_text_and_image_chain(self):
        msg = MessageBuilder.user().add_text("hi").add_image("data:x")
        assert msg.content == [{"text": "hi"}, {"image": "data:x"}]

    def test_add_image_url_encodes_png(self):
        msg = MessageBuilder.user().add_image_url(np.zeros((2, 2, 3), dtype=np.uint8))
        assert msg.content[0]["image"].startswith("data:image/png;base64,")

    def test_base_prepare_and_caching_not_implemented(self):
        with pytest.raises(NotImplementedError):
            MessageBuilder.user().prepare_message()
        with pytest.raises(NotImplementedError):
            MessageBuilder.user().mark_all_previous_msg_for_caching()

    def test_add_responded_tool_calls_requires_all_responses(self):
        calls = ToolCalls(tool_calls=[ToolCall(name="a")])
        with pytest.raises(AssertionError, match="must have a response"):
            MessageBuilder.add_responded_tool_calls(calls)
        calls.tool_calls[0].response_text("ok")
        msg = OpenAIChatCompletionAPIMessageBuilder.add_responded_tool_calls(calls)
        assert isinstance(msg, OpenAIChatCompletionAPIMessageBuilder)
        assert msg.role == "tool" and msg.responded_tool_calls is calls

    def test_to_markdown_text_and_image(self):
        md = MessageBuilder.assistant().add_text("hello").add_image("http://img").to_markdown()
        assert md == "### Assistant\n\n```\nhello\n```\n\n![Image](http://img)"

    def test_to_markdown_tool_calls(self):
        calls = ToolCalls(
            tool_calls=[
                ToolCall(name="click", arguments={"bid": "a1"}).response_text("clicked"),
                ToolCall(name="noop", arguments={}).response_image("shot.png"),
            ]
        )
        md = MessageBuilder.add_responded_tool_calls(calls).to_markdown()
        assert md.startswith("### Tool\n")
        assert "**Tool Call 1**: click(bid='a1')" in md
        assert "**Tool Response 1:**\n```\nclicked\n```" in md
        assert "**Tool Call 2**: noop()" in md
        assert "![Tool Response Image](shot.png)" in md

    def test_to_markdown_tool_call_without_response(self):
        msg = MessageBuilder("tool")
        msg.responded_tool_calls = ToolCalls(tool_calls=[ToolCall(name="x", arguments=None)])
        md = msg.to_markdown()
        assert "**Tool Call 1**: x()" in md and "Tool Response" not in md


# --------------------------------------------------------------------------- OpenAI Responses builder


def _responded(raw_calls, *calls):
    return MessageBuilder.add_responded_tool_calls(
        ToolCalls(tool_calls=list(calls), raw_calls=raw_calls)
    )


class TestOpenAIResponseAPIMessageBuilder:
    def test_system_uses_developer_role(self):
        msg = OpenAIResponseAPIMessageBuilder.system().add_text("rules")
        assert msg.prepare_message() == [
            {"role": "developer", "content": [{"type": "input_text", "text": "rules"}]}
        ]

    def test_assistant_text_is_output_text(self):
        msg = OpenAIResponseAPIMessageBuilder.assistant().add_text("a").add_image("img")
        assert msg.prepare_message()[0]["content"] == [
            {"type": "output_text", "text": "a"},
            {"type": "input_image", "image_url": "img"},
        ]

    def test_unsupported_content(self):
        msg = OpenAIResponseAPIMessageBuilder.user()
        msg.content.append({"audio": "x"})
        with pytest.raises(ValueError, match="Unsupported content type"):
            msg.prepare_message()

    def test_tool_role_without_calls_raises(self):
        with pytest.raises(ValueError, match="No tool calls"):
            OpenAIResponseAPIMessageBuilder("tool").prepare_message()

    def test_function_and_computer_call_outputs(self):
        prior = [_ns(type="reasoning"), _ns(type="function_call")]
        fn = ToolCall(name="f", raw_call=_ns(type="function_call", call_id="c1")).response_text("r")
        cu = ToolCall(name="cu", raw_call=_ns(type="computer_call", call_id="c2")).response_image(
            "i"
        )
        other = ToolCall(name="o", raw_call=_ns(type="custom", call_id="c3")).response_text("z")
        msg = OpenAIResponseAPIMessageBuilder.add_responded_tool_calls(
            ToolCalls(tool_calls=[fn, cu, other], raw_calls=_ns(output=prior))
        )
        out = msg.prepare_message()
        assert out[:2] == prior
        assert out[2:] == [
            {"type": "function_call_output", "call_id": "c1", "output": "r"},
            {
                "type": "computer_call_output",
                "call_id": "c2",
                "output": {"type": "input_image", "image_url": "i"},
            },
        ]

    @pytest.mark.parametrize(
        "call_type, response, match",
        [
            ("function_call", {"image": "i"}, "Image output"),
            ("computer_call", {"text": "t"}, "Text output"),
        ],
    )
    def test_wrong_response_kind_asserts(self, call_type, response, match):
        call = ToolCall(name="f", raw_call=_ns(type=call_type, call_id="c"), tool_response=response)
        msg = OpenAIResponseAPIMessageBuilder.add_responded_tool_calls(
            ToolCalls(tool_calls=[call], raw_calls=_ns(output=[]))
        )
        with pytest.raises(AssertionError, match=match):
            msg.prepare_message()

    def test_mark_caching_is_noop(self):
        msg = OpenAIResponseAPIMessageBuilder.user().add_text("x")
        msg.mark_all_previous_msg_for_caching()
        assert not hasattr(msg, "_cache_breakpoint")


# --------------------------------------------------------------------------- Anthropic builder


class TestAnthropicAPIMessageBuilder:
    def test_assistant_text_is_stripped(self):
        msg = AnthropicAPIMessageBuilder.assistant().add_text("  answer \n").add_image("raw")
        out = msg.prepare_message()
        assert out[0]["content"][0] == {"type": "text", "text": "answer"}
        assert msg.content[0]["text"] == "  answer \n"  # source content untouched

    def test_user_text_not_stripped(self):
        msg = AnthropicAPIMessageBuilder.user().add_text(" keep ")
        assert msg.prepare_message() == [
            {"role": "user", "content": [{"type": "text", "text": " keep "}]}
        ]

    @pytest.mark.parametrize(
        "img, data", [("data:image/png;base64,QUJD", "QUJD"), ("QUJD", "QUJD")]
    )
    def test_image_prefix_removed(self, img, data):
        assert AnthropicAPIMessageBuilder.user().transform_content({"image": img}) == {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": data},
        }

    def test_unsupported_content(self):
        with pytest.raises(ValueError, match="Unsupported content type"):
            AnthropicAPIMessageBuilder.user().transform_content({"video": "v"})

    def test_tool_role_without_calls_raises(self):
        with pytest.raises(ValueError, match="No tool calls"):
            AnthropicAPIMessageBuilder("tool").prepare_message()

    def test_tool_results(self):
        blocks = [_ns(type="text"), _ns(type="tool_use")]
        calls = ToolCalls(
            tool_calls=[
                ToolCall(name="a", raw_call=_ns(id="tu_1")).response_text("r1"),
                ToolCall(name="b", raw_call=_ns(id="tu_2")).response_text("r2"),
            ],
            raw_calls=_ns(content=blocks),
        )
        out = AnthropicAPIMessageBuilder.add_responded_tool_calls(calls).prepare_message()
        assert out == [
            {"role": "assistant", "content": blocks},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu_1", "content": "r1"},
                    {"type": "tool_result", "tool_use_id": "tu_2", "content": "r2"},
                ],
            },
        ]

    def test_tool_image_response_asserts(self):
        calls = ToolCalls(
            tool_calls=[ToolCall(name="a", raw_call=_ns(id="t")).response_image("i")],
            raw_calls=_ns(content=[]),
        )
        with pytest.raises(AssertionError, match="Image output"):
            AnthropicAPIMessageBuilder.add_responded_tool_calls(calls).prepare_message()

    def test_mark_caching_sets_breakpoint(self):
        msg = AnthropicAPIMessageBuilder.user()
        assert not getattr(msg, "_cache_breakpoint", False)
        msg.mark_all_previous_msg_for_caching()
        assert msg._cache_breakpoint is True


# --------------------------------------------------------------------------- Chat Completion builder


class TestOpenAIChatCompletionAPIMessageBuilder:
    def test_unsupported_content(self):
        with pytest.raises(ValueError, match="Unsupported content type"):
            OpenAIChatCompletionAPIMessageBuilder.user().convert_content_to_expected_format(
                {"x": 1}
            )

    def test_tool_role_without_calls_raises(self):
        with pytest.raises(ValueError, match="No tool calls"):
            OpenAIChatCompletionAPIMessageBuilder("tool").prepare_message()

    def test_tool_outputs(self):
        assistant_msg = {"role": "assistant", "tool_calls": ["..."]}
        raw = _ns(choices=[_ns(message=assistant_msg)])
        call = ToolCall(
            name="click", raw_call={"id": "call_1", "function": {"name": "click"}}
        ).response_text("done")
        out = OpenAIChatCompletionAPIMessageBuilder.add_responded_tool_calls(
            ToolCalls(tool_calls=[call], raw_calls=raw)
        ).prepare_message()
        assert out == [
            assistant_msg,
            {"name": "click", "role": "tool", "tool_call_id": "call_1", "content": "done"},
        ]

    def test_tool_image_response_asserts(self):
        call = ToolCall(name="c", raw_call={"id": "1", "function": {"name": "c"}}).response_image(
            "i"
        )
        msg = OpenAIChatCompletionAPIMessageBuilder.add_responded_tool_calls(
            ToolCalls(tool_calls=[call], raw_calls=_ns(choices=[_ns(message={})]))
        )
        with pytest.raises(AssertionError, match="Image output"):
            msg.prepare_message()

    def test_mark_caching_is_noop(self):
        msg = OpenAIChatCompletionAPIMessageBuilder.user()
        assert msg.mark_all_previous_msg_for_caching() is None


# --------------------------------------------------------------------------- APIPayload / helpers


class TestAPIPayload:
    def test_tool_choice_and_force_call_are_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            APIPayload(tool_choice="auto", force_call_tool="click")

    def test_reasoning_effort_logs_info(self, caplog):
        with caplog.at_level(logging.INFO, logger=ra.logger.name):
            payload = APIPayload(reasoning_effort="high")
        assert payload.reasoning_effort == "high"
        assert "reasoning_effort is used by LiteLLM API only" in caplog.text

    def test_defaults(self):
        p = APIPayload(force_call_tool="x")
        assert (p.messages, p.tools, p.tool_choice, p.use_cache_breakpoints) == (
            None,
            None,
            None,
            False,
        )


@pytest.mark.parametrize(
    "name, kwargs, expected",
    [
        ("noop", None, "noop()"),
        ("noop", {}, "noop()"),
        ("click", {"bid": "a1", "button": "left"}, "click(bid='a1', button='left')"),
        ("scroll", {"dx": 0, "dy": -1.5, "flag": True}, "scroll(dx=0, dy=-1.5, flag=True)"),
        ("fill", {"text": 'it\'s "q"\n'}, "fill(text='it\\'s \"q\"\\n')"),
        ("multi", {"keys": ["a", "b"]}, "multi(keys=['a', 'b'])"),
    ],
)
def test_tool_call_to_python_code(name, kwargs, expected):
    assert tool_call_to_python_code(name, kwargs) == expected


def test_agentlab_action_helpers():
    call = ToolCall(name="go_back", arguments={})
    assert AgentlabAction.convert_toolcall_to_agentlab_action_format(call) == "go_back()"
    assert (
        AgentlabAction.convert_multiactions_to_agentlab_action_format(["a()", "b()"]) == "a()\nb()"
    )
    assert AgentlabAction.convert_multiactions_to_agentlab_action_format([]) is None


def test_base_response_model_call_chains_call_and_parse():
    class Echo(BaseResponseModel):
        def _call_api(self, payload):
            return {"raw": payload}

        def _parse_response(self, response):
            return LLMOutput(raw_response=response, think="t")

    model = Echo("m", api_key="k", temperature=0.3, max_tokens=7)
    assert (model.model_name, model.api_key, model.temperature, model.max_tokens) == (
        "m",
        "k",
        0.3,
        7,
    )
    out = model("payload")
    assert out.raw_response == {"raw": "payload"} and out.think == "t"


# --------------------------------------------------------------------------- OpenAI Responses model


def _responses_obj(outputs, inp=10, out=5):
    usage = _ns(input_tokens=inp, output_tokens=out, input_tokens_details=_ns(cached_tokens=0))
    return _ns(output=outputs, usage=usage)


class TestOpenAIResponseModel:
    def _model(self, **kw):
        return OpenAIResponseModel(model_name="gpt-test", api_key="sk-test", **kw)

    def test_init(self):
        model = self._model(base_url="http://local/v1", temperature=0.2)
        assert model.client.base_url.host == "local"
        assert model.client.api_key == "sk-test"
        assert (model.input_cost, model.output_cost) == (1e-3, 2e-3)
        assert model._pricing_api == "openai" and model.max_tokens == 100

    @pytest.mark.parametrize(
        "payload_kw, expected_choice",
        [
            ({}, None),
            ({"tool_choice": "auto"}, "auto"),
            ({"tool_choice": "any"}, "required"),
            ({"tool_choice": "required"}, "required"),
            ({"force_call_tool": "click"}, {"type": "function", "name": "click"}),
        ],
    )
    def test_call_api_params(self, payload_kw, expected_choice):
        model = self._model(temperature=0.5, max_tokens=20)
        msgs = [
            OpenAIResponseAPIMessageBuilder.system().add_text("sys"),
            OpenAIResponseAPIMessageBuilder.user().add_text("hi"),
        ]
        tools = [{"type": "function", "name": "click"}]
        with patch.object(ra, "call_openai_api_with_retries", return_value="RAW") as call:
            assert model._call_api(APIPayload(messages=msgs, tools=tools, **payload_kw)) == "RAW"
        fn, params = call.call_args.args
        assert fn == model.client.responses.create
        assert params["model"] == "gpt-test"
        assert params["input"] == [
            {"role": "developer", "content": [{"type": "input_text", "text": "sys"}]},
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        ]
        assert params["temperature"] == 0.5 and params["max_output_tokens"] == 20
        assert params["tools"] == tools
        assert params.get("tool_choice") == expected_choice

    def test_call_api_omits_unset_params(self):
        model = self._model(max_tokens=None)
        with patch.object(ra, "call_openai_api_with_retries", return_value="RAW") as call:
            model._call_api(APIPayload(messages=[]))
        assert call.call_args.args[1] == {"model": "gpt-test", "input": []}

    def test_parse_function_calls_and_thinking(self):
        fc1 = _ns(type="function_call", name="click", arguments=json.dumps({"bid": "a1"}))
        fc2 = _ns(type="function_call", name="noop", arguments="{}")
        response = _responses_obj(
            [
                _ns(type="reasoning", summary=[_ns(text="plan")]),
                _ns(type="reasoning", summary=[]),
                _ns(type="message", content=[_ns(text="hello")]),
                _ns(type="message", content=[]),
                _ns(type="other", output_text="extra"),
                _ns(type="other", output_text=""),
                fc1,
                fc2,
            ]
        )
        out = self._model()._parse_response(response)
        assert out.think == "plan\nhello\nextra\n"
        assert out.action == "click(bid='a1')\nnoop()"
        assert [c.name for c in out.tool_calls] == ["click", "noop"]
        assert out.tool_calls.tool_calls[0].raw_call is fc1
        assert out.tool_calls.raw_calls is response and out.raw_response is response

    def test_parse_single_call_and_no_call(self):
        model = self._model()
        one = model._parse_response(
            _responses_obj([_ns(type="function_call", name="go_back", arguments="{}")])
        )
        assert one.action == "go_back()"
        none = model._parse_response(_responses_obj([_ns(type="message", content=[_ns(text="x")])]))
        assert none.action is None and len(none.tool_calls) == 0

    def test_computer_call_requires_subclass(self):
        response = _responses_obj([_ns(type="computer_call", action=_ns(type="click"))])
        with pytest.raises(NotImplementedError):
            self._model()._parse_response(response)

        class CUA(OpenAIResponseModel):
            def cua_action_to_env_tool_name_and_args(self, action):
                return "mouse_click", {"kind": action.type}

        model = CUA(model_name="gpt-test", api_key="sk-test")
        assert model._parse_response(response).action == "mouse_click(kind='click')"

    def test_text_action_space_returns_none(self):
        model = self._model()
        model.action_space_as_tools = False
        response = _responses_obj([_ns(type="function_call", name="a", arguments="{}")])
        out = model._parse_response(response)
        assert out.action is None and len(out.tool_calls) == 1

    def test_full_call_tracks_cost(self):
        model = self._model()
        response = _responses_obj([_ns(type="function_call", name="a", arguments="{}")], 100, 10)
        response.usage = ResponseUsage(
            input_tokens=100,
            output_tokens=10,
            total_tokens=110,
            input_tokens_details={"cached_tokens": 0},
            output_tokens_details={"reasoning_tokens": 0},
        )
        with patch.object(ra, "call_openai_api_with_retries", return_value=response):
            with tracking.set_tracker() as tracker:
                out = model(
                    APIPayload(messages=[OpenAIResponseAPIMessageBuilder.user().add_text("q")])
                )
        assert out.action == "a()"
        assert tracker.stats == {
            "input_tokens": 100,
            "output_tokens": 10,
            "cost": pytest.approx(0.12),
        }
        assert model.stats.stats_dict["n_api_calls"] == 1


# --------------------------------------------------------------------------- Chat Completion model


def _chat_completion(message, prompt=10, completion=5):
    return ChatCompletion.model_validate(
        {
            "id": "c",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-test",
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
            "usage": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        }
    )


def _tc(id_, name, args):
    return {
        "id": id_,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


class TestOpenAIChatCompletionModel:
    def _model(self, **kw):
        return OpenAIChatCompletionModel(model_name="gpt-test", api_key="sk-test", **kw)

    def test_init(self):
        model = self._model(base_url="http://router/v1")
        assert model.client.base_url.host == "router" and model.client.api_key == "sk-test"
        assert model.action_space_as_tools is True
        assert (model.input_cost, model.output_cost) == (1e-3, 2e-3)

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: OpenAIChatCompletionModel.__init__ does not forward api_key to super()",
    )
    def test_init_keeps_api_key_attribute(self):
        assert self._model().api_key == "sk-test"

    def test_format_tools_for_chat_completion(self):
        tools = [
            {
                "type": "function",
                "name": "click",
                "description": "d",
                "parameters": {"p": 1},
                "strict": True,
            }
        ]
        assert OpenAIChatCompletionModel.format_tools_for_chat_completion(tools) == [
            {
                "type": "function",
                "function": {"name": "click", "description": "d", "parameters": {"p": 1}},
            }
        ]
        assert OpenAIChatCompletionModel.format_tools_for_chat_completion(None) is None

    @pytest.mark.parametrize(
        "payload_kw, expected_choice",
        [
            ({}, None),
            ({"tool_choice": "none"}, "none"),
            ({"tool_choice": "any"}, "required"),
            ({"force_call_tool": "f"}, {"type": "function", "function": {"name": "f"}}),
        ],
    )
    def test_call_api_params(self, payload_kw, expected_choice):
        model = self._model(temperature=0.0, max_tokens=33)
        tools = [{"type": "function", "name": "f", "description": "d", "parameters": {}}]
        msgs = [OpenAIChatCompletionAPIMessageBuilder.user().add_text("hi").add_image("u")]
        with patch.object(ra, "call_openai_api_with_retries", return_value="RAW") as call:
            model._call_api(APIPayload(messages=msgs, tools=tools, **payload_kw))
        fn, params = call.call_args.args
        assert fn == model.client.chat.completions.create
        assert params["messages"] == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "image_url", "image_url": {"url": "u"}},
                ],
            }
        ]
        assert params["temperature"] == 0.0 and params["max_completion_tokens"] == 33
        assert params["tools"][0]["function"]["name"] == "f"
        assert params.get("tool_choice") == expected_choice

    def test_call_api_minimal(self):
        model = self._model(max_tokens=None)
        with patch.object(ra, "call_openai_api_with_retries", return_value="RAW") as call:
            model._call_api(APIPayload(messages=[]))
        assert call.call_args.args[1] == {"model": "gpt-test", "messages": []}

    def test_parse_multiple_tool_calls(self):
        response = _chat_completion(
            {
                "role": "assistant",
                "content": "thinking",
                "tool_calls": [_tc("c1", "click", {"bid": "a"}), _tc("c2", "noop", {})],
            }
        )
        out = self._model()._parse_response(response)
        assert out.action == "click(bid='a')\nnoop()"
        assert out.tool_calls.tool_calls[0].raw_call["id"] == "c1"
        assert out.tool_calls.raw_calls is response
        assert out.think == "thinking"

    def test_parse_no_tool_calls(self):
        out = self._model()._parse_response(
            _chat_completion({"role": "assistant", "content": "hi"})
        )
        assert out.action is None and out.tool_calls is None and out.think == "hi"

    def test_text_action_space_returns_none(self):
        model = self._model()
        model.action_space_as_tools = False
        response = _chat_completion(
            {"role": "assistant", "content": "", "tool_calls": [_tc("c", "a", {})]}
        )
        assert model._parse_response(response).action is None

    def test_thinking_wraps_reasoning_dict_message(self):
        response = _ns(choices=[_ns(message={"reasoning": "why", "text": "T", "content": "C"})])
        model = self._model()
        assert model._extract_thinking_content_from_response(response) == "<think>why</think>\nTC"
        assert (
            model._extract_thinking_content_from_response(response, wrap_tag="r")
            == "<r>why</r>\nTC"
        )
        empty = _ns(choices=[_ns(message={"reasoning": ""})])
        assert model._extract_thinking_content_from_response(empty) == ""

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: tool-call-only chat responses have content=None, which is rendered as the "
        "literal string 'None' in the extracted thinking",
    )
    def test_thinking_ignores_null_content(self):
        response = _chat_completion(
            {"role": "assistant", "content": None, "tool_calls": [_tc("c", "a", {})]}
        )
        assert self._model()._parse_response(response).think == ""

    def test_tool_response_round_trip(self):
        model = self._model()
        response = _chat_completion(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_tc("call_9", "click", {"bid": "b"})],
            }
        )
        calls = model._parse_response(response).tool_calls
        for c in calls:
            c.response_text("clicked")
        out = OpenAIChatCompletionAPIMessageBuilder.add_responded_tool_calls(
            calls
        ).prepare_message()
        assert out[0] is response.choices[0].message
        assert out[1] == {
            "name": "click",
            "role": "tool",
            "tool_call_id": "call_9",
            "content": "clicked",
        }


# --------------------------------------------------------------------------- Claude model


def _claude_response(blocks, inp=10, out=5):
    usage = _ns(
        input_tokens=inp, output_tokens=out, cache_input_tokens=0, cache_creation_input_tokens=0
    )
    return _ns(content=blocks, usage=usage)


class TestClaudeResponseModel:
    def _model(self, **kw):
        return ClaudeResponseModel(model_name="claude-test", api_key="sk-ant", **kw)

    def test_init(self):
        model = self._model(base_url="http://claude.local")
        assert model.client.api_key == "sk-ant" and "claude.local" in str(model.client.base_url)
        assert model._pricing_api == "anthropic"
        assert (model.input_cost, model.output_cost) == (1e-3, 2e-3)

    def test_filter_system_messages(self):
        sys1 = AnthropicAPIMessageBuilder.system().add_text("a")
        user = AnthropicAPIMessageBuilder.user().add_text("u")
        raw = {"role": "system", "content": "dict messages are not filtered"}
        sys_msgs, others = ClaudeResponseModel.filter_system_messages([sys1, user, raw])
        assert sys_msgs == [sys1] and others == [user, raw]

    def test_filter_system_messages_rejects_typed_image(self):
        sys_msg = AnthropicAPIMessageBuilder.system()
        sys_msg.content.append({"type": "image"})
        with pytest.raises(TypeError, match="cannot contain images"):
            ClaudeResponseModel.filter_system_messages([sys_msg])

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: filter_system_messages checks c.get('type') but builder content is "
        "{'image': ...}, so images added via add_image are never rejected",
    )
    def test_filter_system_messages_rejects_builder_image(self):
        sys_msg = AnthropicAPIMessageBuilder.system().add_text("a").add_image("data:x")
        with pytest.raises(TypeError):
            ClaudeResponseModel.filter_system_messages([sys_msg])

    @pytest.mark.parametrize(
        "payload_kw, expected_choice",
        [
            ({}, None),
            ({"tool_choice": "auto"}, {"type": "auto"}),
            ({"tool_choice": "required"}, {"type": "any"}),
            ({"tool_choice": "any"}, {"type": "any"}),
            ({"force_call_tool": "click"}, {"type": "tool", "name": "click"}),
        ],
    )
    def test_call_api_params(self, payload_kw, expected_choice):
        model = self._model(temperature=0.7, max_tokens=50)
        msgs = [
            AnthropicAPIMessageBuilder.system().add_text("s1"),
            AnthropicAPIMessageBuilder.system().add_text("s2"),
            AnthropicAPIMessageBuilder.user().add_text("q"),
        ]
        tools = [{"name": "click"}]
        with patch.object(ra, "call_anthropic_api_with_retries", return_value="RAW") as call:
            assert model._call_api(APIPayload(messages=msgs, tools=tools, **payload_kw)) == "RAW"
        fn, params = call.call_args.args
        assert fn == model.client.messages.create
        assert params["system"] == "s1\ns2"
        assert params["messages"] == [{"role": "user", "content": [{"type": "text", "text": "q"}]}]
        assert params["temperature"] == 0.7 and params["max_tokens"] == 50
        assert params["tools"] == tools
        assert params.get("tool_choice") == expected_choice

    def test_call_api_minimal(self):
        model = self._model(max_tokens=None)
        with patch.object(ra, "call_anthropic_api_with_retries", return_value="RAW") as call:
            model._call_api(APIPayload(messages=[]))
        assert call.call_args.args[1] == {"model": "claude-test", "messages": [], "system": ""}

    def test_cache_breakpoints_and_flags(self):
        model = self._model()
        first = AnthropicAPIMessageBuilder.user().add_text("old1").add_text("old2")
        first.mark_all_previous_msg_for_caching()
        second = AnthropicAPIMessageBuilder.user().add_text("new")
        tools = [{"name": "a"}, {"name": "b"}]
        payload = APIPayload(
            messages=[first, second],
            tools=tools,
            use_cache_breakpoints=True,
            cache_tool_definition=True,
            cache_complete_prompt=True,
        )
        with patch.object(ra, "call_anthropic_api_with_retries", return_value="RAW") as call:
            model._call_api(payload)
        params = call.call_args.args[1]
        eph = {"type": "ephemeral"}
        assert "cache_control" not in params["messages"][0]["content"][0]
        assert params["messages"][0]["content"][1]["cache_control"] == eph
        assert params["messages"][1]["content"][0]["cache_control"] == eph
        assert "cache_control" not in params["tools"][0]
        assert params["tools"][1]["cache_control"] == eph

    def test_breakpoint_ignored_when_disabled(self):
        msg = AnthropicAPIMessageBuilder.user().add_text("x")
        msg.mark_all_previous_msg_for_caching()
        with patch.object(ra, "call_anthropic_api_with_retries", return_value="RAW") as call:
            self._model()._call_api(APIPayload(messages=[msg]))
        assert "cache_control" not in call.call_args.args[1]["messages"][0]["content"][0]

    def test_apply_cache_breakpoints_without_marker(self):
        prepared = [{"content": [{"type": "text", "text": "x"}]}]
        assert self._model().apply_cache_breakpoints(object(), prepared) == [
            {"content": [{"type": "text", "text": "x"}]}
        ]

    def test_parse_response(self):
        tool1 = _ns(type="tool_use", name="click", input={"bid": "z"}, id="t1")
        tool2 = _ns(type="tool_use", name="noop", input={}, id="t2")
        response = _claude_response(
            [_ns(type="text", text="I will "), _ns(type="text", text="click"), tool1, tool2]
        )
        out = self._model()._parse_response(response)
        assert out.think == "I will click"
        assert out.action == "click(bid='z')\nnoop()"
        assert out.tool_calls.tool_calls[0].raw_call is tool1

    def test_parse_single_and_none(self):
        model = self._model()
        one = model._parse_response(
            _claude_response([_ns(type="tool_use", name="a", input={}, id="t")])
        )
        assert one.action == "a()" and one.think == ""
        none = model._parse_response(_claude_response([_ns(type="text", text="no tools")]))
        assert none.action is None and len(none.tool_calls) == 0

    def test_text_action_space_returns_none(self):
        model = self._model()
        model.action_space_as_tools = False
        out = model._parse_response(
            _claude_response([_ns(type="tool_use", name="a", input={}, id="t")])
        )
        assert out.action is None

    def test_full_call_tracks_effective_cost(self):
        model = self._model()
        response = _claude_response([_ns(type="tool_use", name="a", input={}, id="t")], 1000, 100)
        response.usage = AnthropicUsage(
            input_tokens=1000,
            output_tokens=100,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        with patch.object(ra, "call_anthropic_api_with_retries", return_value=response):
            with tracking.set_tracker() as tracker:
                out = model(APIPayload(messages=[AnthropicAPIMessageBuilder.user().add_text("q")]))
        assert out.action == "a()"
        assert tracker.cost == pytest.approx(1.2)
        assert model.stats.stats_dict["effective_cost"] == pytest.approx(1.2)


# --------------------------------------------------------------------------- ModelArgs factories


@pytest.mark.parametrize(
    "args_cls, model_cls, builder_cls, model_name",
    [
        (OpenAIResponseModelArgs, OpenAIResponseModel, OpenAIResponseAPIMessageBuilder, "gpt-test"),
        (
            OpenAIChatModelArgs,
            OpenAIChatCompletionModel,
            OpenAIChatCompletionAPIMessageBuilder,
            "gpt-test",
        ),
        (ClaudeResponseModelArgs, ClaudeResponseModel, AnthropicAPIMessageBuilder, "claude-test"),
    ],
)
def test_model_args_factories(monkeypatch, args_cls, model_cls, builder_cls, model_name):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
    args = args_cls(model_name=model_name, temperature=0.4, max_new_tokens=64)
    model = args.make_model()
    assert type(model) is model_cls
    assert (model.model_name, model.temperature, model.max_tokens) == (model_name, 0.4, 64)
    assert args.get_message_builder() is builder_cls
    assert args.api in ("openai", "anthropic")


def test_openrouter_model_args(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    args = OpenRouterModelArgs(model_name="router/test", max_new_tokens=8)
    model = args.make_model()
    assert isinstance(model, OpenAIChatCompletionModel)
    assert str(model.client.base_url).rstrip("/") == "https://openrouter.ai/api/v1"
    assert model.client.api_key == "sk-or"
    assert model.max_tokens == 8 and args.api == "openai"
    assert args.get_message_builder() is OpenAIChatCompletionAPIMessageBuilder
