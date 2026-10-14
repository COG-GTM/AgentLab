import logging
from copy import deepcopy
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import numpy as np
import pytest
from bgym import HighLevelActionSetArgs

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.generic_agent import tmlr_config
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.agents.visual_agent import agent_configs
from agentlab.agents.visual_agent import visual_agent_prompts as vap
from agentlab.agents.visual_agent.visual_agent import VisualAgent, VisualAgentArgs
from agentlab.agents.visual_agent.visual_agent_prompts import (
    History,
    MainPrompt,
    Observation,
    PromptFlags,
    SystemPrompt,
    make_instructions,
)
from agentlab.llm.base_api import BaseModelArgs
from agentlab.llm.llm_utils import AIMessage, ParseError

VALID = "<think>\nI will click the button.\n</think>\n<action>\nmouse_click(10, 20)\n</action>"


class FakeChat:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, messages):
        self.calls += 1
        return AIMessage(self.responses.pop(0))

    def get_stats(self):
        return {"n_calls": self.calls}


@dataclass
class FakeModelArgs(BaseModelArgs):
    model_name: str = "fake/vision-model"
    responses: list = field(default_factory=list)

    def make_model(self):
        self.chat = FakeChat(self.responses)
        return self.chat


def _flags(**overrides) -> PromptFlags:
    flags = deepcopy(agent_configs.DEFAULT_PROMPT_FLAGS)
    for k, v in overrides.items():
        setattr(flags, k, v)
    return flags


def _obs(**overrides):
    obs = {
        "goal_object": [{"type": "text", "text": "Buy the red shoes"}],
        "chat_messages": [],
        "last_action_error": "",
        "open_pages_urls": ["http://a.test", "http://b.test"],
        "open_pages_titles": ["Page A", "Page B"],
        "active_page_index": 1,
        "screenshot": np.zeros((8, 8, 3), np.uint8),
        "screenshot_som": np.full((8, 8, 3), 255, np.uint8),
    }
    obs.update(overrides)
    return obs


def _text(prompt) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        return "".join(p.get("text", "") for p in prompt)
    content = prompt["content"]
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if p["type"] == "text")


def _images(message):
    return [p for p in message["content"] if p["type"] == "image_url"]


# ---------------------------------------------------------------- prompt pieces


def test_prompt_flags_defaults():
    flags = PromptFlags()
    assert flags.obs is None and flags.action is None
    assert flags.use_thinking is True
    assert flags.use_concrete_example is False
    assert flags.use_abstract_example is True
    assert flags.enable_chat is False
    assert flags.extra_instructions is None


def test_system_prompt():
    text = SystemPrompt().prompt
    assert text.startswith("You are an agent trying to solve a web task")
    assert "send messages to the user" in text


class TestMakeInstructions:
    def test_goal_mode(self):
        instr = make_instructions(_obs(), from_chat=False, extra_instructions=None)
        assert isinstance(instr, dp.GoalInstructions)
        text = _text(instr.prompt)
        assert "## Goal:" in text and "Buy the red shoes" in text
        assert "Extra instructions" not in text

    def test_goal_mode_with_extra_instructions(self):
        instr = make_instructions(_obs(), from_chat=False, extra_instructions="Be quick.")
        text = _text(instr.prompt)
        assert "## Extra instructions:" in text and "Be quick." in text

    def test_goal_mode_without_chat_messages_key(self, caplog):
        obs = _obs()
        del obs["chat_messages"]
        with caplog.at_level(logging.WARNING):
            instr = make_instructions(obs, from_chat=False, extra_instructions=None)
        assert isinstance(instr, dp.GoalInstructions)
        assert "goal mode" not in caplog.text

    @pytest.mark.parametrize("n_user,warns", [(0, False), (1, False), (2, True), (3, True)])
    def test_warns_on_multiple_user_messages_in_goal_mode(self, caplog, n_user, warns):
        msgs = [{"role": "user", "message": "hi", "timestamp": 0}] * n_user
        msgs += [{"role": "assistant", "message": "yo", "timestamp": 0}] * 2
        with caplog.at_level(logging.WARNING):
            make_instructions(_obs(chat_messages=msgs), from_chat=False, extra_instructions=None)
        assert ("enable_chat=True" in caplog.text) is warns

    def test_chat_mode(self):
        msgs = [
            {"role": "user", "message": "Find my order", "timestamp": 0},
            {"role": "assistant", "message": "Sure", "timestamp": 1},
        ]
        instr = make_instructions(_obs(chat_messages=msgs), True, extra_instructions="Polite.")
        assert isinstance(instr, dp.ChatInstructions)
        text = instr.prompt
        assert "## Chat messages:" in text
        assert "[user]" in text and "Find my order" in text
        assert "[assistant]" in text and "Sure" in text
        assert "Polite." in text


class TestHistory:
    def test_empty_history(self):
        assert History([], []).prompt == "\n"

    def test_steps_are_numbered(self):
        text = History(["click('1')", "fill('2', 'x')"], ["t0", "t1"]).prompt
        assert "## Step 0\n### Thoughts:\nt0\n### Action:\nclick('1')" in text
        assert "## Step 1\n### Thoughts:\nt1\n### Action:\nfill('2', 'x')" in text
        assert text.index("Step 0") < text.index("Step 1")

    def test_mismatched_lengths_are_truncated(self):
        text = History(["a0", "a1", "a2"], ["t0"]).prompt
        assert "Step 0" in text
        assert "Step 1" not in text


class TestObservation:
    def _flags(self, **kw):
        flags = deepcopy(agent_configs.DEFAULT_OBS_FLAGS)
        for k, v in kw.items():
            setattr(flags, k, v)
        return flags

    def test_tabs_shown_when_enabled(self):
        text = Observation(_obs(), self._flags(use_tabs=True)).prompt
        assert "# Observation of current step:" in text
        assert "## Currently open tabs:" in text
        assert "Tab 1 (active tab):" in text and "Title: Page B" in text
        assert "Tab 0 (active tab)" not in text

    def test_tabs_hidden_when_disabled(self):
        text = Observation(_obs(), self._flags(use_tabs=False)).prompt
        assert "Currently open tabs" not in text

    @pytest.mark.parametrize(
        "error,use_logs,shown",
        [("Element not found", True, True), ("Element not found", False, False), ("", True, False)],
    )
    def test_error_visibility(self, error, use_logs, shown):
        obs = _obs(last_action_error=error)
        text = Observation(obs, self._flags(use_error_logs=use_logs, use_tabs=False)).prompt
        assert ("Element not found" in text) is shown

    def test_no_screenshot_when_disabled(self):
        msg = vap.HumanMessage("hi")
        out = Observation(_obs(), self._flags(use_screenshot=False)).add_screenshot(msg)
        assert out is msg
        assert _text(out) == "hi"

    @pytest.mark.parametrize(
        "use_som,key,caption",
        [
            (False, "screenshot", "Here is a screenshot of the page:"),
            (True, "screenshot_som", "annotated with bounding boxes"),
        ],
    )
    def test_screenshot_selection(self, monkeypatch, use_som, key, caption):
        seen = []

        def fake_url(img):
            seen.append(img)
            return "data:image/jpeg;base64,AAA"

        monkeypatch.setattr(vap, "image_to_jpg_base64_url", fake_url)
        obs = _obs()
        flags = self._flags(use_screenshot=True, use_som=use_som, openai_vision_detail="high")
        out = Observation(obs, flags).add_screenshot(vap.HumanMessage("hi"))
        assert seen[0] is obs[key]
        assert caption in _text(out)
        (img,) = _images(out)
        assert img["image_url"]["url"] == "data:image/jpeg;base64,AAA"
        assert img["image_url"]["detail"] == "high"


class TestMainPrompt:
    def _prompt(self, flags=None, actions=(), thoughts=(), obs=None):
        flags = flags or _flags()
        action_set = flags.action.action_set.make_action_set()
        return MainPrompt(action_set, obs or _obs(), list(actions), list(thoughts), flags)

    def test_prompt_structure_with_defaults(self):
        mp = self._prompt(actions=["mouse_click(1, 2)"], thoughts=["prev thought"])
        msg = mp.prompt
        text = _text(msg)
        assert "Buy the red shoes" in text
        assert "# Observation of current step:" in text
        assert "## Step 0" in text and "prev thought" in text
        assert "# Action space:" in text and "mouse_click" in text
        assert "<think>" in text
        assert "# Abstract Example" in text
        assert "# Concrete Example" not in text
        assert len(_images(msg)) == 1  # DEFAULT flags use the screenshot

    def test_concrete_example_and_no_thinking(self):
        flags = _flags(use_concrete_example=True, use_abstract_example=False, use_thinking=False)
        text = _text(self._prompt(flags).prompt)
        assert "# Concrete Example" in text
        assert "# Abstract Example" not in text
        assert "<think>" not in text

    def test_chat_mode_uses_chat_instructions(self):
        msgs = [{"role": "user", "message": "Hello agent", "timestamp": 0}]
        mp = self._prompt(_flags(enable_chat=True), obs=_obs(chat_messages=msgs))
        assert isinstance(mp.instructions, dp.ChatInstructions)
        assert "Hello agent" in _text(mp.prompt)

    def test_parse_answer_valid(self):
        ans = self._prompt()._parse_answer(VALID)
        assert ans["think"] == "I will click the button."
        assert ans["action"] == "mouse_click(10, 20)"

    def test_parse_answer_missing_think_is_tolerated(self):
        ans = self._prompt()._parse_answer("<action>mouse_click(1, 1)</action>")
        assert ans["action"] == "mouse_click(1, 1)"
        assert "parse_error" in ans

    @pytest.mark.parametrize(
        "answer",
        ["<think>hmm</think>", "<action>click('a1')</action>", "<action>not_python(</action>"],
    )
    def test_parse_answer_invalid_action_raises(self, answer):
        with pytest.raises(ParseError):
            self._prompt()._parse_answer(answer)


# ---------------------------------------------------------------- VisualAgentArgs


def test_args_post_init_builds_agent_name():
    args = VisualAgentArgs(chat_model_args=FakeModelArgs(), flags=_flags())
    assert args.agent_name == "VisualAgent-fake_vision-model"


def test_args_post_init_tolerates_missing_model_args():
    args = VisualAgentArgs()
    assert args.agent_name is None
    assert args.max_retry == 4


@pytest.mark.parametrize("multi_tab", [True, False])
def test_args_set_benchmark_sets_tabs(multi_tab):
    args = VisualAgentArgs(chat_model_args=FakeModelArgs(), flags=_flags())
    bench = MagicMock(is_multi_tab=multi_tab)
    args.set_benchmark(bench, demo_mode=False)
    assert args.flags.obs.use_tabs is multi_tab


def test_args_reproducibility_prepare_close():
    cma = MagicMock()
    cma.model_name = "m"
    cma.prepare_server.return_value = "p"
    cma.close_server.return_value = "c"
    args = VisualAgentArgs(chat_model_args=cma, flags=_flags())
    args.set_reproducibility_mode()
    assert cma.temperature == 0
    assert args.prepare() == "p"
    assert args.close() == "c"


def test_args_make_agent():
    cma = FakeModelArgs(responses=[VALID])
    args = VisualAgentArgs(chat_model_args=cma, flags=_flags(), max_retry=2)
    agent = args.make_agent()
    assert isinstance(agent, VisualAgent)
    assert agent.max_retry == 2
    assert agent.chat_llm is cma.chat
    assert agent.flags is args.flags
    assert agent.actions == [] and agent.thoughts == [] and agent.seed is None


# ---------------------------------------------------------------- VisualAgent


def _agent(responses, max_retry=4, flags=None):
    cma = FakeModelArgs(responses=responses)
    return VisualAgent(cma, flags or _flags(), max_retry=max_retry), cma


def test_obs_preprocessor_delegates(monkeypatch):
    seen = {}

    def fake_make(obs_flags):
        seen["flags"] = obs_flags
        return lambda obs: {**obs, "processed": True}

    monkeypatch.setattr(dp, "make_obs_preprocessor", fake_make)
    flags = _flags()
    agent, _ = _agent([], flags=flags)
    assert seen["flags"] is flags.obs
    assert agent.obs_preprocessor({"a": 1}) == {"a": 1, "processed": True}


def test_get_action_success():
    agent, cma = _agent([VALID])
    action, info = agent.get_action(_obs())

    assert action == "mouse_click(10, 20)"
    assert info.think == "I will click the button."
    assert info.stats["n_retry"] == 0
    assert info.stats["busted_retry"] == 0
    assert info.stats["n_calls"] == 1
    assert {"input_tokens", "output_tokens", "cost"} <= set(info.stats)
    assert [m["role"] for m in info.chat_messages] == ["system", "user", "assistant"]
    assert info.extra_info["chat_model_args"]["model_name"] == "fake/vision-model"
    assert agent.actions == ["mouse_click(10, 20)"]
    assert agent.thoughts == ["I will click the button."]


def test_get_action_history_flows_into_next_prompt():
    second = VALID.replace("mouse_click(10, 20)", "mouse_click(5, 5)")
    agent, _ = _agent([VALID, second])
    agent.get_action(_obs())
    action, info = agent.get_action(_obs())
    assert action == "mouse_click(5, 5)"
    user_text = _text(info.chat_messages[1])
    assert "## Step 0" in user_text and "mouse_click(10, 20)" in user_text
    assert agent.actions == ["mouse_click(10, 20)", "mouse_click(5, 5)"]


def test_get_action_counts_retries():
    agent, cma = _agent(["garbage", "<action>bogus(</action>", VALID])
    action, info = agent.get_action(_obs())
    assert action == "mouse_click(10, 20)"
    assert info.stats["n_retry"] == 2
    assert info.stats["busted_retry"] == 0
    assert cma.chat.calls == 3


def test_get_action_busted_retry():
    agent, cma = _agent(["garbage"] * 3, max_retry=3)
    action, info = agent.get_action(_obs())
    assert action is None
    assert info.think is None
    assert info.stats["n_retry"] == 4
    assert info.stats["busted_retry"] == 1
    assert cma.chat.calls == 3
    assert agent.actions == [None]
    assert agent.thoughts == [None]


def test_reset_clears_state():
    agent, _ = _agent([VALID])
    agent.get_action(_obs())
    agent.reset(seed=7)
    assert agent.seed == 7
    assert agent.actions == [] and agent.thoughts == []


# ---------------------------------------------------------------- configs


def test_visual_agent_configs():
    assert agent_configs.VISUAL_AGENT_4o.agent_name == "VisualAgent-gpt-4o-2024-05-13"
    assert agent_configs.VISUAL_AGENT_CLAUDE_3_5.agent_name.startswith("VisualAgent-")
    assert "/" not in agent_configs.VISUAL_AGENT_CLAUDE_3_5.agent_name
    flags = agent_configs.DEFAULT_PROMPT_FLAGS
    assert flags.obs.use_screenshot is True and flags.obs.use_som is False
    assert isinstance(flags.action.action_set, HighLevelActionSetArgs)
    assert tuple(flags.action.action_set.subsets) == ("coord",)


TMLR_LLM = "openai/gpt-4o-mini-2024-07-18"


def test_tmlr_base_agent():
    args = tmlr_config.get_base_agent(TMLR_LLM)
    assert isinstance(args, GenericAgentArgs)
    assert args.flags is tmlr_config.BASE_FLAGS
    assert args.agent_name == "GenericAgent-gpt-4o-mini-2024-07-18"
    assert args.flags.obs.use_screenshot is False


def test_tmlr_vision_agent_does_not_mutate_base_flags():
    args = tmlr_config.get_vision_agent(TMLR_LLM)
    assert args.agent_name == "GenericAgent-gpt-4o-mini-2024-07-18_vision"
    assert args.flags.obs.use_screenshot is True
    assert args.flags.obs.use_som is False
    assert tmlr_config.BASE_FLAGS.obs.use_screenshot is False


def test_tmlr_som_agent_does_not_mutate_base_flags():
    args = tmlr_config.get_som_agent(TMLR_LLM)
    assert args.agent_name == "GenericAgent-gpt-4o-mini-2024-07-18"
    assert args.flags.obs.use_screenshot is True
    assert args.flags.obs.use_som is True
    assert tmlr_config.BASE_FLAGS.obs.use_som is False
    assert tmlr_config.BASE_FLAGS.obs.use_screenshot is False


def test_tmlr_unknown_llm_raises():
    with pytest.raises(KeyError):
        tmlr_config.get_base_agent("not/a-model")
