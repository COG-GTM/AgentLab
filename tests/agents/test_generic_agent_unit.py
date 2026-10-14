"""Unit tests for GenericAgent / GenericAgentArgs / generic_agent_prompt with a fake LLM."""

import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest import mock

import pytest
from bgym import HighLevelActionSetArgs

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.generic_agent import generic_agent as ga
from agentlab.agents.generic_agent.generic_agent import GenericAgent, GenericAgentArgs
from agentlab.agents.generic_agent.generic_agent_prompt import (
    Criticise,
    GenericPromptFlags,
    MainPrompt,
    Memory,
    Plan,
)
from agentlab.llm.base_api import BaseModelArgs
from agentlab.llm.chat_api import make_assistant_message
from agentlab.llm.llm_utils import HumanMessage, ParseError


class FakeChat:
    """Returns scripted assistant messages and records each prompt it sees."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, messages):
        self.calls.append([dict(m) for m in messages])
        content = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        return make_assistant_message(content)

    def get_stats(self):
        return {"n_calls": len(self.calls)}


@dataclass
class FakeModelArgs(BaseModelArgs):
    model_name: str = "fake/model"
    responses: tuple = ("<think>ok</think><action>click('12')</action>",)
    prepared: list = field(default_factory=list)

    def make_model(self):
        return FakeChat(self.responses)

    def prepare_server(self):
        self.prepared.append("prepare")
        return "prepared"

    def close_server(self):
        self.prepared.append("close")
        return "closed"


def make_flags(**overrides):
    obs_kwargs = overrides.pop("obs", {})
    action_kwargs = overrides.pop("action", {})
    obs = dict(
        use_html=True,
        use_ax_tree=True,
        use_history=True,
        use_action_history=True,
        use_think_history=True,
        use_screenshot=False,
    )
    obs.update(obs_kwargs)
    action = dict(action_set=HighLevelActionSetArgs(subsets=["bid"]))
    action.update(action_kwargs)
    kwargs = dict(
        obs=dp.ObsFlags(**obs),
        action=dp.ActionFlags(**action),
        use_thinking=True,
        max_prompt_tokens=None,
    )
    kwargs.update(overrides)
    return GenericPromptFlags(**kwargs)


def make_obs(step=0, **overrides):
    obs = {
        "goal_object": [{"type": "text", "text": "Click the submit button"}],
        "chat_messages": [{"role": "user", "message": "Click submit", "timestamp": 0}],
        "pruned_html": f"<button bid='12'>Submit {step}</button>",
        "axtree_txt": f"[12] button 'Submit {step}'",
        "last_action_error": "",
        "focused_element_bid": "",
        "open_pages_urls": ["https://example.com"],
        "open_pages_titles": ["Example"],
        "active_page_index": 0,
    }
    obs.update(overrides)
    return obs


def make_agent(responses=None, flags=None, max_retry=4, **model_kwargs):
    if responses is not None:
        model_kwargs["responses"] = tuple(responses)
    args = GenericAgentArgs(
        chat_model_args=FakeModelArgs(**model_kwargs),
        flags=flags or make_flags(),
        max_retry=max_retry,
    )
    return args.make_agent()


# --------------------------------------------------------------------------- GenericAgentArgs


class TestGenericAgentArgs:
    def test_agent_name_derived_from_model_and_slashes_replaced(self):
        args = GenericAgentArgs(chat_model_args=FakeModelArgs(model_name="org/m-1"))
        assert args.agent_name == "GenericAgent-org_m-1"

    def test_missing_model_args_keeps_default_name(self):
        assert GenericAgentArgs().agent_name is None

    def _benchmark(self, name="miniwob_tiny", multi_tab=True):
        return SimpleNamespace(
            name=name,
            is_multi_tab=multi_tab,
            high_level_action_set_args=HighLevelActionSetArgs(
                subsets=["miniwob_all"], multiaction=False, strict=False
            ),
        )

    def test_set_benchmark_miniwob_enables_html_and_copies_action_set(self):
        flags = make_flags(obs=dict(use_html=False))
        args = GenericAgentArgs(chat_model_args=FakeModelArgs(), flags=flags)
        bench = self._benchmark()
        args.set_benchmark(bench, demo_mode=False)
        assert flags.obs.use_html is True
        assert flags.obs.use_tabs is True
        assert flags.action.action_set == bench.high_level_action_set_args
        assert flags.action.action_set is not bench.high_level_action_set_args
        assert flags.action.action_set.demo_mode is None

    def test_set_benchmark_non_miniwob_keeps_html_flag(self):
        flags = make_flags(obs=dict(use_html=False))
        args = GenericAgentArgs(chat_model_args=FakeModelArgs(), flags=flags)
        args.set_benchmark(self._benchmark(name="workarena_l1", multi_tab=False), demo_mode=False)
        assert flags.obs.use_html is False
        assert flags.obs.use_tabs is False

    def test_set_benchmark_legacy_flags_and_demo_mode(self):
        flags = make_flags(action=dict(multi_actions=True, is_strict=True))
        args = GenericAgentArgs(chat_model_args=FakeModelArgs(), flags=flags)
        bench = self._benchmark()
        args.set_benchmark(bench, demo_mode=True)
        action_set = flags.action.action_set
        assert action_set.multiaction is True
        assert action_set.strict is True
        assert action_set.demo_mode == "all_blue"
        # benchmark's own args are untouched
        assert bench.high_level_action_set_args.multiaction is False
        assert bench.high_level_action_set_args.demo_mode is None

    def test_set_reproducibility_mode_zeroes_temperature(self):
        args = GenericAgentArgs(chat_model_args=FakeModelArgs(temperature=0.7))
        args.set_reproducibility_mode()
        assert args.chat_model_args.temperature == 0

    def test_prepare_and_close_delegate_to_model_args(self):
        model_args = FakeModelArgs()
        args = GenericAgentArgs(chat_model_args=model_args)
        assert args.prepare() == "prepared"
        assert args.close() == "closed"
        assert model_args.prepared == ["prepare", "close"]

    def test_make_agent(self):
        flags = make_flags()
        args = GenericAgentArgs(chat_model_args=FakeModelArgs(), flags=flags, max_retry=2)
        agent = args.make_agent()
        assert isinstance(agent, GenericAgent)
        assert agent.flags is flags
        assert agent.max_retry == 2
        assert isinstance(agent.chat_llm, FakeChat)


# --------------------------------------------------------------------------- GenericAgent


class TestGenericAgentInit:
    def test_initial_state(self):
        agent = make_agent()
        assert agent.plan == "No plan yet"
        assert agent.plan_step == -1
        assert agent.actions == agent.memories == agent.thoughts == agent.obs_history == []
        assert agent.seed is None
        assert "click" in agent.action_set.action_set

    def test_som_without_screenshot_is_disabled(self):
        flags = make_flags(obs=dict(use_som=True, use_screenshot=False))
        with pytest.warns(UserWarning, match="use_som=True requires use_screenshot=True"):
            make_agent(flags=flags)
        assert flags.obs.use_som is False

    def test_screenshot_without_vision_is_disabled(self):
        flags = make_flags(obs=dict(use_screenshot=True))
        with pytest.warns(UserWarning, match="does not support vision"):
            make_agent(flags=flags, vision_support=False)
        assert flags.obs.use_screenshot is False

    def test_screenshot_with_vision_is_kept(self, recwarn):
        flags = make_flags(obs=dict(use_screenshot=True, use_som=True))
        make_agent(flags=flags, vision_support=True)
        assert flags.obs.use_screenshot is True and flags.obs.use_som is True
        assert not [w for w in recwarn if "Disabling" in str(w.message)]

    def test_obs_preprocessor_uses_flags(self, monkeypatch):
        seen = {}

        def fake_make(obs_flags):
            seen["flags"] = obs_flags
            return lambda obs: {**obs, "processed": True}

        monkeypatch.setattr(dp, "make_obs_preprocessor", fake_make)
        flags = make_flags()
        agent = make_agent(flags=flags)
        assert seen["flags"] is flags.obs
        assert agent.obs_preprocessor({"a": 1}) == {"a": 1, "processed": True}

    def test_reset_clears_state(self):
        agent = make_agent()
        agent.get_action(make_obs())
        agent.reset(seed=7)
        assert agent.seed == 7
        assert agent.plan == "No plan yet" and agent.plan_step == -1
        assert agent.actions == agent.memories == agent.thoughts == agent.obs_history == []


class TestGetMaxes:
    @pytest.mark.parametrize(
        "flag_max, total, inp, trunc, expected",
        [
            (None, None, None, 20, (None, 20)),
            (1000, None, None, 5, (1000, 5)),
            (1000, 800, 900, 20, (800, 20)),
            (None, 4096, 2048, 0, (2048, 20)),
            (None, None, 3000, None, (3000, 20)),
        ],
    )
    def test_get_maxes(self, flag_max, total, inp, trunc, expected):
        flags = make_flags(max_prompt_tokens=flag_max, max_trunc_itr=trunc)
        agent = make_agent(flags=flags, max_total_tokens=total, max_input_tokens=inp)
        assert agent._get_maxes() == expected


class TestGetAction:
    def test_happy_path(self):
        agent = make_agent()
        action, info = agent.get_action(make_obs())
        assert action == "click('12')"
        assert info.think == "ok"
        assert info.stats["n_retry"] == 0
        assert info.stats["busted_retry"] == 0
        assert info.stats["n_calls"] == 1
        # cost tracker stats are merged in by the decorator
        assert {"input_tokens", "output_tokens", "cost"} <= set(info.stats)
        assert info.extra_info["chat_model_args"]["model_name"] == "fake/model"
        roles = [m["role"] for m in info.chat_messages]
        assert roles == ["system", "user", "assistant"]
        assert agent.actions == ["click('12')"]
        assert agent.thoughts == ["ok"]
        assert agent.memories == [None]
        assert len(agent.obs_history) == 1

    def test_prompt_contains_goal_observation_and_history(self):
        agent = make_agent(
            responses=[
                "<think>first</think><action>click('12')</action>",
                "<think>second</think><action>click('13')</action>",
            ]
        )
        agent.get_action(make_obs(0))
        action, info = agent.get_action(make_obs(1))
        assert action == "click('13')"
        user_prompt = str(info.chat_messages[1])
        assert "Click the submit button" in user_prompt
        assert "[12] button 'Submit 1'" in user_prompt
        assert "# History of interaction with the task" in user_prompt
        assert "<action>\nclick('12')\n</action>" in user_prompt
        assert "<think>\nfirst\n</think>" in user_prompt
        assert info.chat_messages[0]["content"].startswith("You are an agent")

    def test_plan_and_memory_are_tracked(self):
        flags = make_flags(use_plan=True, use_memory=True)
        agent = make_agent(
            flags=flags,
            responses=[
                "<think>t</think><plan>1. click</plan><step>1</step>"
                "<memory>remember 12</memory><action>click('12')</action>"
            ],
        )
        agent.get_action(make_obs())
        assert agent.plan == "1. click"
        assert agent.plan_step == "1"
        assert agent.memories == ["remember 12"]

    def test_plan_unchanged_when_not_returned(self):
        agent = make_agent()
        agent.get_action(make_obs())
        assert agent.plan == "No plan yet" and agent.plan_step == -1

    def test_retry_after_parse_error(self):
        agent = make_agent(responses=["no tags here", "<action>click('12')</action>"])
        action, info = agent.get_action(make_obs())
        assert action == "click('12')"
        assert info.stats["n_retry"] == 1
        assert info.stats["busted_retry"] == 0
        assert info.stats["n_calls"] == 2
        roles = [m["role"] for m in info.chat_messages]
        assert roles == ["system", "user", "assistant", "user", "assistant"]
        assert "Missing the key <action>" in info.chat_messages[3]["content"]
        # think falls back to the raw text when its tag is missing
        assert info.think == "<action>click('12')</action>"

    def test_busted_retry_returns_none_action(self):
        agent = make_agent(responses=["garbage"], max_retry=2)
        action, info = agent.get_action(make_obs())
        assert action is None
        assert info.stats["busted_retry"] == 1
        assert info.stats["n_retry"] == 3
        assert info.stats["n_calls"] == 2
        assert agent.actions == [None]
        assert agent.thoughts == [None]

    def test_fit_tokens_called_with_maxes_and_system_prompt(self, monkeypatch):
        real_fit = dp.fit_tokens
        spy = mock.Mock(side_effect=real_fit)
        monkeypatch.setattr(dp, "fit_tokens", spy)
        flags = make_flags(max_prompt_tokens=None, max_trunc_itr=7)
        monkeypatch.setattr(dp, "count_tokens", lambda text, model=None: 1)
        agent = make_agent(flags=flags, max_input_tokens=5000)
        agent.get_action(make_obs())
        kwargs = spy.call_args.kwargs
        assert kwargs["max_prompt_tokens"] == 5000
        assert kwargs["max_iterations"] == 7
        assert kwargs["model_name"] == "fake/model"
        assert kwargs["additional_prompts"]["role"] == "system"
        assert isinstance(kwargs["shrinkable"], MainPrompt)

    def test_prompt_is_shrunk_to_fit(self, monkeypatch):
        monkeypatch.setattr(dp, "count_tokens", lambda text, model=None: len(text.split()))
        big_html = "\n".join(f"<p>line {i}</p>" for i in range(400))
        flags = make_flags(
            obs=dict(use_ax_tree=False), max_prompt_tokens=700, use_concrete_example=False
        )
        agent = make_agent(flags=flags)
        _, info = agent.get_action(make_obs(pruned_html=big_html))
        prompt = str(info.chat_messages[1])
        assert "lines to reduce prompt size" in prompt
        assert len(prompt.split()) <= 700


# --------------------------------------------------------------------------- generic_agent_prompt


def _main_prompt(flags, obs=None, **kwargs):
    action_set = flags.action.action_set.make_action_set()
    params = dict(
        action_set=action_set,
        obs_history=[obs or make_obs()],
        actions=[],
        memories=[],
        thoughts=[],
        previous_plan="No plan yet",
        step=-1,
        flags=flags,
    )
    params.update(kwargs)
    return MainPrompt(**params)


class TestMainPrompt:
    def test_goal_mode_warns_on_multiple_user_messages(self, caplog):
        msgs = [
            {"role": "user", "message": "a", "timestamp": 0},
            {"role": "user", "message": "b", "timestamp": 1},
        ]
        with caplog.at_level(logging.WARNING):
            mp = _main_prompt(make_flags(), obs=make_obs(chat_messages=msgs))
        assert "Consider switching to `enable_chat=True`" in caplog.text
        assert isinstance(mp.instructions, dp.GoalInstructions)

    def test_goal_mode_single_message_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            _main_prompt(make_flags())
        assert "enable_chat" not in caplog.text

    def test_chat_mode_uses_chat_instructions(self):
        flags = make_flags(enable_chat=True, extra_instructions="be brief")
        mp = _main_prompt(flags)
        assert isinstance(mp.instructions, dp.ChatInstructions)
        text = str(mp.prompt)
        assert "## Chat messages:" in text and "Click submit" in text
        assert "be brief" in text

    def test_abstract_and_concrete_examples(self):
        flags = make_flags(
            use_abstract_example=True,
            use_concrete_example=True,
            use_plan=True,
            use_memory=True,
            use_criticise=True,
        )
        text = str(_main_prompt(flags).prompt)
        assert "# Abstract Example" in text
        assert "# Concrete Example" in text
        for tag in ("<think>", "<plan>", "<memory>", "<action_draft>", "<criticise>", "<action>"):
            assert text.count(tag) >= 2, tag
        assert "You just executed step -1 of the previously proposed plan" in text

    def test_examples_disabled(self):
        flags = make_flags(use_abstract_example=False, use_concrete_example=False)
        text = str(_main_prompt(flags).prompt)
        assert "Abstract Example" not in text and "Concrete Example" not in text

    @pytest.mark.parametrize(
        "be_cautious, multiaction, visible",
        [(True, True, True), (True, False, False), (False, True, False)],
    )
    def test_be_cautious_only_in_multiaction(self, be_cautious, multiaction, visible):
        flags = make_flags(
            be_cautious=be_cautious,
            action=dict(
                action_set=HighLevelActionSetArgs(subsets=["bid"], multiaction=multiaction)
            ),
        )
        assert ("Be very cautious" in str(_main_prompt(flags).prompt)) is visible

    def test_hints_flag(self):
        assert "Note:\n* Some tasks" in str(_main_prompt(make_flags(use_hints=True)).prompt)
        assert "Some tasks may be game" not in str(_main_prompt(make_flags()).prompt)

    def test_screenshot_is_appended_when_enabled(self, monkeypatch):
        monkeypatch.setattr(dp, "image_to_jpg_base64_url", lambda img: "data:image/jpeg;base64,Q")
        flags = make_flags(obs=dict(use_screenshot=True))
        prompt = _main_prompt(flags, obs=make_obs(screenshot="IMG")).prompt
        assert isinstance(prompt, HumanMessage)
        assert prompt["content"][-1]["image_url"]["url"] == "data:image/jpeg;base64,Q"

    def test_shrink_delegates_to_history_and_obs(self):
        mp = _main_prompt(make_flags())
        with mock.patch.object(mp.history, "shrink") as h, mock.patch.object(mp.obs, "shrink") as o:
            mp.shrink()
        h.assert_called_once_with()
        o.assert_called_once_with()

    def test_parse_answer_combines_all_elements(self):
        flags = make_flags(use_plan=True, use_memory=True, use_criticise=True)
        mp = _main_prompt(flags)
        ans = mp._parse_answer(
            "<think>T</think><plan>P</plan><step>3</step><memory>M</memory>"
            "<action_draft>D</action_draft><criticise>C</criticise><action>click('12')</action>"
        )
        assert ans == {
            "think": "T",
            "plan": "P",
            "step": "3",
            "memory": "M",
            "action_draft": "D",
            "criticise": "C",
            "action": "click('12')",
        }

    def test_parse_answer_hidden_elements_ignored(self):
        flags = make_flags(use_thinking=False)
        ans = _main_prompt(flags)._parse_answer(
            "<think>T</think><plan>P</plan><action>click('12')</action>"
        )
        assert ans == {"action": "click('12')"}

    def test_parse_answer_raises_on_invalid_action(self):
        with pytest.raises(ParseError):
            _main_prompt(make_flags())._parse_answer("<action>not_an_action</action>")


class TestAnswerElements:
    def test_memory_optional_and_merged(self):
        assert Memory().parse_answer("nothing") == {}
        assert Memory().parse_answer("<memory>a</memory><memory>b</memory>") == {"memory": "a\nb"}

    def test_plan_parse_and_prompt(self):
        plan = Plan("1. do", 2)
        assert "You just executed step 2" in plan.prompt
        assert "1. do" in plan.prompt
        assert plan.parse_answer("<plan>x</plan><step>4</step>") == {"plan": "x", "step": "4"}
        assert plan.parse_answer("no plan") == {}
        assert Plan("p", 0, visible=False).parse_answer("<plan>x</plan>") == {}

    def test_plan_rejects_duplicate_tags(self):
        with pytest.raises(ParseError, match="multiple instances of the key plan"):
            Plan("p", 0).parse_answer("<plan>a</plan><plan>b</plan>")

    def test_criticise_parse(self):
        crit = Criticise()
        assert crit.prompt == ""
        assert crit.parse_answer("<action_draft>d</action_draft><criticise>c</criticise>") == {
            "action_draft": "d",
            "criticise": "c",
        }
        assert crit.parse_answer("nothing") == {}

    def test_flags_asdict_roundtrip(self):
        flags = make_flags(use_plan=True, extra_instructions="x")
        d = flags.asdict()
        assert d["use_plan"] is True
        assert d["obs"]["use_html"] is True
        assert d["extra_instructions"] == "x"
        clone = flags.copy()
        clone.obs.use_html = False
        assert flags.obs.use_html is True
