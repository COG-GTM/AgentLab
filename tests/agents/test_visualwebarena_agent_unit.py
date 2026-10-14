import base64
import io
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import numpy as np
import PIL.Image
import pytest
from bgym import HighLevelActionSetArgs

from agentlab.agents.visualwebarena import agent as vwa
from agentlab.agents.visualwebarena.agent import (
    VisualWebArenaAgent,
    VisualWebArenaAgentArgs,
    WebArenaAgent,
    image_data_to_uri,
    parser,
)
from agentlab.agents.visualwebarena.prompts import TEMPLATES
from agentlab.llm.base_api import BaseModelArgs
from agentlab.llm.llm_utils import ParseError

GOOD = "Let's think step-by-step. ```click('a7')```"


class FakeChat:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, messages):
        self.calls.append([dict(m) for m in messages])
        return {"role": "assistant", "content": self.responses.pop(0)}


@dataclass
class FakeModelArgs(BaseModelArgs):
    model_name: str = "org/fake-model"
    responses: list = field(default_factory=list)

    def make_model(self):
        self.chat = FakeChat(self.responses)
        return self.chat


def _decode(uri: str, fmt: str) -> PIL.Image.Image:
    prefix = f"data:image/{fmt};base64,"
    assert uri.startswith(prefix)
    return PIL.Image.open(io.BytesIO(base64.b64decode(uri[len(prefix) :])))


def _png_bytes(mode="RGB", size=(5, 3)):
    buf = io.BytesIO()
    PIL.Image.new(mode, size).save(buf, format="PNG")
    return buf.getvalue()


def _action_set():
    return HighLevelActionSetArgs(subsets=["bid"]).make_action_set()


def _agent(observation_type="axtree", few_shot=False, responses=()):
    chat = FakeChat(responses)
    agent = VisualWebArenaAgent(
        temperature=0.1,
        chat_model=chat,
        action_set=_action_set(),
        observation_type=observation_type,
        with_few_shot_examples=few_shot,
    )
    return agent, chat


def _obs(**overrides):
    obs = {
        "active_page_index": np.array([1]),
        "open_pages_titles": ["Home", "Shop"],
        "axtree_txt": "[a7] button 'Buy'",
        "url": "http://shop.test/item",
        "screenshot": np.zeros((6, 6, 3), np.uint8),
        "extra_element_properties": {},
        "goal_object": [{"type": "text", "text": "Buy the item"}],
    }
    obs.update(overrides)
    return obs


def _texts(content):
    return [c["text"] for c in content if c["type"] == "text"]


def _images(content):
    return [c for c in content if c["type"] == "image_url"]


# ---------------------------------------------------------------- image_data_to_uri


@pytest.mark.parametrize("fmt", ["png", "jpeg"])
def test_image_data_to_uri_from_ndarray(fmt):
    arr = np.zeros((4, 7, 3), np.uint8)
    img = _decode(image_data_to_uri(arr, fmt), fmt)
    assert img.size == (7, 4)
    assert img.format == fmt.upper()


def test_image_data_to_uri_from_bytes_defaults_to_png():
    img = _decode(image_data_to_uri(_png_bytes(size=(5, 3))), "png")
    assert img.size == (5, 3)


@pytest.mark.parametrize("mode", ["RGBA", "LA"])
def test_image_data_to_uri_drops_alpha(mode):
    img = _decode(image_data_to_uri(_png_bytes(mode=mode), "png"), "png")
    assert img.mode == "RGB"


def test_image_data_to_uri_rgba_array_to_jpeg():
    arr = np.zeros((3, 3, 4), np.uint8)
    img = _decode(image_data_to_uri(arr, "jpeg"), "jpeg")
    assert img.mode == "RGB"


def test_image_data_to_uri_rejects_unknown_format():
    with pytest.raises(AssertionError):
        image_data_to_uri(np.zeros((2, 2, 3), np.uint8), "gif")


# ---------------------------------------------------------------- parser


def test_parser_returns_first_block_and_full_response():
    resp = "thinking\n```\nclick('a7')\n```\nthen\n```\nnoop()\n```"
    assert parser(resp) == {"action": "click('a7')", "think": resp}


def test_parser_inline_backticks():
    assert parser(GOOD)["action"] == "click('a7')"


def test_parser_without_code_raises():
    with pytest.raises(ParseError, match="No code block"):
        parser("I would click the button")


# ---------------------------------------------------------------- args


def test_args_agent_name_format():
    args = VisualWebArenaAgentArgs(chat_model_args=FakeModelArgs())
    assert args.agent_name == "VisualWebArenaAgent-axtree_som-org_fake-model"
    assert args.with_few_shot_examples is True


def test_webarena_partial_uses_axtree():
    args = WebArenaAgent(chat_model_args=FakeModelArgs())
    assert args.observation_type == "axtree"
    assert args.agent_name == "WebArenaAgent-axtree-org_fake-model"


def test_module_level_configs():
    assert vwa.WA_AGENT_4O_MINI.agent_name.startswith("WebArenaAgent-axtree-")
    assert vwa.VWA_AGENT_4O_MINI.observation_type == "axtree_som"
    for args in (vwa.WA_AGENT_SONNET, vwa.VWA_AGENT_SONNET, vwa.VWA_AGENT_4O):
        assert "/" not in args.agent_name


def test_args_set_benchmark_deepcopies_action_set():
    asa = HighLevelActionSetArgs(subsets=["bid", "nav"])
    bench = MagicMock(high_level_action_set_args=asa)
    args = VisualWebArenaAgentArgs(chat_model_args=FakeModelArgs())
    args.set_benchmark(bench, demo_mode=False)
    assert args.action_set_args == asa
    assert args.action_set_args is not asa


def test_args_reproducibility_prepare_close():
    cma = MagicMock()
    cma.model_name = "x"
    cma.prepare_server.return_value = "p"
    cma.close_server.return_value = "c"
    args = VisualWebArenaAgentArgs(chat_model_args=cma, temperature=0.8)
    args.set_reproducibility_mode()
    assert args.temperature == 0.0
    assert args.prepare() == "p"
    assert args.close() == "c"


def test_args_make_agent():
    cma = FakeModelArgs()
    args = VisualWebArenaAgentArgs(
        chat_model_args=cma,
        action_set_args=HighLevelActionSetArgs(subsets=["bid"]),
        observation_type="axtree",
        with_few_shot_examples=False,
        temperature=0.3,
    )
    agent = args.make_agent()
    assert isinstance(agent, VisualWebArenaAgent)
    assert agent.chat_model is cma.chat
    assert agent.temperature == 0.3
    assert agent.observation_type == "axtree"
    assert agent.few_shot_messages == []


# ---------------------------------------------------------------- agent init


@pytest.mark.parametrize("obs_type", ["axtree", "axtree_screenshot", "axtree_som"])
def test_intro_message_contains_action_space(obs_type):
    agent, _ = _agent(obs_type)
    (intro,) = agent.intro_messages
    assert intro["type"] == "text"
    assert "click(bid" in intro["text"]
    assert "{action_space_description}" not in intro["text"]
    assert agent.action_history == ["None"]
    assert agent.few_shot_messages == []


def test_text_only_few_shot_examples():
    agent, _ = _agent("axtree", few_shot=True)
    n = len(TEMPLATES["axtree"]["examples"])
    assert len(agent.few_shot_messages) == n
    assert agent.few_shot_messages[0]["text"].startswith(f"Example 1/{n}:")
    assert "ACTION: " in agent.few_shot_messages[0]["text"]
    assert _images(agent.few_shot_messages) == []


@pytest.mark.parametrize("obs_type", ["axtree_screenshot", "axtree_som"])
def test_multimodal_few_shot_examples_embed_screenshots(obs_type):
    agent, _ = _agent(obs_type, few_shot=True)
    examples = TEMPLATES[obs_type]["examples"]
    n_img = sum(len(e) == 3 for e in examples)
    n_txt = sum(len(e) == 2 for e in examples)
    assert n_img > 0
    imgs = _images(agent.few_shot_messages)
    assert len(imgs) == n_img
    assert all(i["image_url"]["url"].startswith("data:image/png;base64,") for i in imgs)
    assert len(agent.few_shot_messages) == 4 * n_img + n_txt


def test_bad_few_shot_example_format_raises(monkeypatch):
    monkeypatch.setitem(
        TEMPLATES, "axtree", {"intro": "{action_space_description}", "examples": [("only-one",)]}
    )
    with pytest.raises(ValueError, match="Unexpected format"):
        _agent("axtree", few_shot=True)


# ---------------------------------------------------------------- get_action


def test_get_action_axtree_builds_messages():
    agent, chat = _agent("axtree", responses=[GOOD])
    action, info = agent.get_action(_obs())

    assert action == "click('a7')"
    assert info.think == GOOD
    system, user, answer = info.chat_messages
    assert answer["content"] == GOOD
    assert system["role"] == "system" and system["content"] == agent.intro_messages
    assert user["role"] == "user"
    texts = _texts(user["content"])
    obs_text = texts[0]
    assert obs_text.startswith("OBSERVATION:")
    assert "Tab 0: Home | Tab 1 (current): Shop" in obs_text
    assert "[a7] button 'Buy'" in obs_text
    assert "URL: http://shop.test/item" in obs_text
    assert "PREVIOUS ACTION: None" in obs_text
    assert texts[-2].startswith("OBJECTIVE:")
    assert user["content"][-1] == {"type": "text", "text": "Buy the item"}
    assert _images(user["content"]) == []
    assert agent.action_history == ["None", "click('a7')"]
    assert {"input_tokens", "output_tokens", "cost"} <= set(info.stats)


def test_get_action_tracks_previous_action():
    agent, chat = _agent("axtree", responses=[GOOD, "```scroll(0, 100)```"])
    agent.get_action(_obs())
    action, info = agent.get_action(_obs())
    assert action == "scroll(0, 100)"
    assert "PREVIOUS ACTION: click('a7')" in _texts(info.chat_messages[1]["content"])[0]


def test_get_action_screenshot_without_som(monkeypatch):
    som = MagicMock()
    monkeypatch.setattr(vwa, "overlay_som", som)
    agent, _ = _agent("axtree_screenshot", responses=[GOOD])
    _, info = agent.get_action(_obs())
    content = info.chat_messages[1]["content"]
    assert len(_images(content)) == 1
    assert "SCREENSHOT:\n" in _texts(content)
    som.assert_not_called()


def test_get_action_screenshot_with_som(monkeypatch):
    marked = np.full((6, 6, 3), 255, np.uint8)
    som = MagicMock(return_value=marked)
    monkeypatch.setattr(vwa, "overlay_som", som)
    agent, _ = _agent("axtree_som", responses=[GOOD])
    obs = _obs(extra_element_properties={"a7": {"som": True}})
    _, info = agent.get_action(obs)
    som.assert_called_once_with(obs["screenshot"], {"a7": {"som": True}})
    (img,) = _images(info.chat_messages[1]["content"])
    assert img["image_url"]["url"] == image_data_to_uri(marked)


def test_get_action_includes_few_shots_before_observation():
    agent, _ = _agent("axtree", few_shot=True, responses=[GOOD])
    _, info = agent.get_action(_obs())
    content = info.chat_messages[1]["content"]
    n = len(agent.few_shot_messages)
    assert content[:n] == agent.few_shot_messages
    assert content[n]["text"].startswith("OBSERVATION:")


def test_get_action_retries_then_raises():
    agent, chat = _agent("axtree", responses=["no code"] * 3)
    with pytest.raises(ParseError):
        agent.get_action(_obs())
    assert len(chat.calls) == 3
    assert agent.action_history == ["None"]
