"""Unit tests for agentlab.agents.generic_agent.reproducibility_agent (hermetic, no LLM)."""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import pytest
from bgym import HighLevelActionSetArgs
from browsergym.experiments.agent import AgentInfo

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.generic_agent import reproducibility_agent as ra
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from agentlab.experiments.loop import ExpResult, StepInfo
from agentlab.llm.base_api import BaseModelArgs
from agentlab.llm.llm_utils import Discussion, HumanMessage, SystemMessage

TEST_STUDY = Path(__file__).parent.parent / "data" / "test_study"
EXP_WITH_SUMMARY = "2024-08-01_10-20-52_GenericAgent_on_miniwob.ascending-numbers_64_e6d2d5"


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    sleep = mock.Mock()
    monkeypatch.setattr(ra.time, "sleep", sleep)
    return sleep


@dataclass
class NoLLMArgs(BaseModelArgs):
    model_name: str = "fake/repro"

    def make_model(self):
        def _fail(messages):
            raise AssertionError("ReproAgent must not call the original model")

        return _fail


def make_flags():
    return GenericPromptFlags(
        obs=dp.ObsFlags(use_html=True, use_ax_tree=True, use_screenshot=False),
        action=dp.ActionFlags(action_set=HighLevelActionSetArgs(subsets=["bid"])),
        use_thinking=True,
        use_concrete_example=False,
    )


def make_obs():
    return {
        "goal_object": [{"type": "text", "text": "Click the submit button"}],
        "chat_messages": [],
        "pruned_html": "<button bid='12'>Submit</button>",
        "axtree_txt": "[12] button 'Submit'",
        "last_action_error": "",
        "focused_element_bid": "",
        "open_pages_urls": ["https://example.com"],
        "open_pages_titles": ["Example"],
        "active_page_index": 0,
    }


def write_step(exp_dir, step, chat_messages=None, action=None, with_chat=True):
    agent_info = {"chat_messages": chat_messages} if with_chat else {}
    StepInfo(step=step, obs=None, action=action, agent_info=agent_info).save_step_info(
        exp_dir, save_screenshot=False
    )


def old_messages(answer="<think>old</think><action>click('12')</action>"):
    return [
        {"role": "system", "content": "old system prompt"},
        {"role": "user", "content": "old user prompt"},
        {"role": "assistant", "content": answer},
    ]


def make_repro_agent(exp_dir):
    args = ra.ReproAgentArgs(
        chat_model_args=NoLLMArgs(), flags=make_flags(), max_retry=2, _repro_dir=exp_dir
    )
    return args.make_agent()


# --------------------------------------------------------------------------- ReproChatModel


class TestReproChatModel:
    def test_replays_next_message_and_sleeps(self, no_sleep):
        msgs = old_messages()
        model = ra.ReproChatModel(msgs, delay=3)
        out = model([msgs[0], msgs[1]])
        assert out is msgs[2]
        no_sleep.assert_called_once_with(3)
        assert model.new_messages == [msgs[0], msgs[1], msgs[2]]
        assert model.get_stats() == {}

    def test_input_list_is_not_mutated(self):
        msgs = old_messages()
        sent = [msgs[0], msgs[1]]
        ra.ReproChatModel(msgs, delay=0)(sent)
        assert len(sent) == 2

    def test_missing_response_returns_none_action_without_sleeping(self, no_sleep):
        msgs = old_messages()[:2]
        model = ra.ReproChatModel(msgs)
        out = model(list(msgs))
        assert out == {"role": "assistant", "content": "<action>None</action>"}
        no_sleep.assert_not_called()


# --------------------------------------------------------------------------- ReproAgentArgs


class TestReproAgentArgs:
    def test_name_prefixed(self):
        args = ra.ReproAgentArgs(chat_model_args=NoLLMArgs(model_name="a/b"))
        assert args.agent_name == "Repro_GenericAgent-a_b"

    def test_missing_model_args_does_not_raise(self):
        args = ra.ReproAgentArgs()
        assert args.agent_name == "Repro_None"

    def test_make_agent_binds_exp_dir(self, tmp_path):
        agent = make_repro_agent(tmp_path)
        assert isinstance(agent, ra.ReproAgent)
        assert isinstance(agent.exp_result, ExpResult)
        assert agent.exp_result.exp_dir == tmp_path
        assert agent.max_retry == 2


# --------------------------------------------------------------------------- ReproAgent.get_action


class TestReproAgentGetAction:
    def test_replays_recorded_answer_and_diffs(self, tmp_path):
        write_step(tmp_path, 0, chat_messages=old_messages())
        agent = make_repro_agent(tmp_path)
        action, info = agent.get_action(make_obs())
        assert action == "click('12')"
        assert info.think == "old"
        assert isinstance(agent.chat_llm, ra.ReproChatModel)
        assert "Old Version" in info.html_page and "New Version" in info.html_page
        assert "table.diff" in info.html_page
        # the prompts differ (old one was a placeholder), so some lines changed
        assert info.stats["lines_added"] > 0
        assert info.stats["lines_removed"] > 0
        assert 0 < info.stats["difference_ratio"] <= 1
        assert agent.actions == ["click('12')"]

    def test_uses_step_index_from_action_count(self, tmp_path):
        write_step(tmp_path, 0, chat_messages=old_messages())
        write_step(
            tmp_path, 1, chat_messages=old_messages("<think>2nd</think><action>click('7')</action>")
        )
        agent = make_repro_agent(tmp_path)
        agent.get_action(make_obs())
        action, info = agent.get_action(make_obs())
        assert action == "click('7')"
        assert info.think == "2nd"

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: ReproChatModel shallow-copies the Discussion, so the replayed answer is "
        "appended twice (n_retry=0.5 and a duplicated assistant message)",
    )
    def test_replay_does_not_duplicate_answer(self, tmp_path):
        write_step(tmp_path, 0, chat_messages=old_messages())
        agent = make_repro_agent(tmp_path)
        _, info = agent.get_action(make_obs())
        assert info.stats["n_retry"] == 0
        assert [m["role"] for m in info.chat_messages] == ["system", "user", "assistant"]

    def test_no_chat_messages_returns_error_page(self, tmp_path):
        write_step(tmp_path, 0, with_chat=False)
        (tmp_path / "summary_info.json").write_text(json.dumps({"err_msg": "env crashed"}))
        agent = make_repro_agent(tmp_path)
        action, info = agent.get_action(make_obs())
        assert action is None
        assert isinstance(info, AgentInfo)
        assert "Agent had no chat messages" in info.markdown_page
        assert "env crashed" in info.markdown_page
        assert agent.actions == []

    def test_missing_response_rebuilt_from_recorded_action(self, tmp_path):
        write_step(tmp_path, 0, chat_messages=old_messages()[:2], action="click('99')")
        agent = make_repro_agent(tmp_path)
        action, _ = agent.get_action(make_obs())
        assert action == "click('99')"
        assert agent.chat_llm.old_messages[2] == {
            "role": "assistant",
            "content": "<action>click('99')</action>",
        }

    def test_missing_response_without_recorded_action_yields_none(self, tmp_path):
        write_step(tmp_path, 0, chat_messages=old_messages()[:2], action=None)
        agent = make_repro_agent(tmp_path)
        action, info = agent.get_action(make_obs())
        assert action is None
        assert info.stats["busted_retry"] == 0


# --------------------------------------------------------------------------- helpers


class TestMakeAgentStats:
    def test_converts_dict_agent_info_and_adds_diff(self):
        old = [{"role": "user", "content": "a\nb"}]
        new = [{"role": "user", "content": "a\nc"}]
        action, info = ra._make_agent_stats("noop()", {"stats": {"k": 1}}, None, old, new)
        assert action == "noop()"
        assert isinstance(info, AgentInfo)
        assert info.stats["k"] == 1
        assert info.stats["lines_added"] == 1 and info.stats["lines_removed"] == 1
        assert "<table" in info.html_page


class TestFormatMessages:
    def test_list_of_dicts(self):
        out = ra._format_messages(
            [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
        )
        assert out == "system message:\nS\n\nuser message:\nU\n"

    def test_strings_get_unknown_role(self):
        assert ra._format_messages(["hi"]) == "<unknown role> message:\nhi\n"

    def test_discussion_uses_to_string(self):
        disc = Discussion([SystemMessage("S"), HumanMessage("U")])
        assert ra._format_messages(disc) == "S\nU"

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown message type"):
            ra._format_messages([42])


class TestMakeBackwardCompatible:
    def _args(self, action_set, multi_actions=None):
        flags = make_flags()
        flags.action.action_set = action_set
        flags.action.multi_actions = multi_actions
        return GenericAgentArgs(chat_model_args=NoLLMArgs(), flags=flags)

    @pytest.mark.parametrize(
        "legacy, subsets",
        [("bid", ["bid"]), ("bid+nav", ["bid", "nav"]), (["bid", "tab"], ["bid", "tab"])],
    )
    def test_legacy_action_set_converted(self, legacy, subsets):
        args = ra._make_backward_compatible(self._args(legacy, multi_actions=True))
        action_set = args.flags.action.action_set
        assert isinstance(action_set, HighLevelActionSetArgs)
        assert list(action_set.subsets) == subsets
        assert action_set.multiaction is True

    def test_modern_action_set_untouched(self):
        modern = HighLevelActionSetArgs(subsets=["bid"])
        args = ra._make_backward_compatible(self._args(modern))
        assert args.flags.action.action_set is modern


class TestDiffHelpers:
    def test_diff_stats_identical(self):
        assert ra._diff_stats("a\nb", "a\nb") == dict(
            lines_added=0, lines_removed=0, difference_ratio=0.0
        )

    def test_diff_stats_counts(self):
        stats = ra._diff_stats("a\nb\nc", "a\nx\nc\nd")
        assert stats == dict(lines_added=2, lines_removed=1, difference_ratio=3 / 8)

    @pytest.mark.xfail(
        strict=True, reason="BUG: _diff_stats divides by zero when both strings are empty"
    )
    def test_diff_stats_empty_strings(self):
        assert ra._diff_stats("", "")["difference_ratio"] == 0

    def test_make_diff_strips_nowrap_and_injects_style(self):
        page = ra._make_diff("same\noldline", "same\nnewline")
        assert 'nowrap="nowrap"' not in page
        assert "Old Version" in page and "New Version" in page
        assert ".diff_add {" in page
        assert "oldline" in page and "newline" in page

    def test_set_style_appends_by_default(self):
        html = "<html><head><style>body{}</style></head><body></body></html>"
        out = ra._set_style(html, "p{}")
        assert "<style>body{}\np{}</style>" in out

    def test_set_style_prepend(self):
        html = "<html><head><style>body{}</style></head><body></body></html>"
        out = ra._set_style(html, "p{}", prepend_previous_style=True)
        assert "<style>p{}\nbody{}</style>" in out

    def test_set_style_creates_tag_when_missing(self):
        out = ra._set_style("<html><head></head><body></body></html>", "p{}")
        assert "<head><style>\np{}</style></head>" in out


# --------------------------------------------------------------------------- study-level


class TestMakeReproAgent:
    def test_from_fixture_study(self):
        exp_dir = TEST_STUDY / EXP_WITH_SUMMARY
        agent_args = ExpResult(exp_dir).exp_args.agent_args
        repro = ra.make_repro_agent(agent_args, exp_dir=str(exp_dir))
        assert isinstance(repro, ra.ReproAgentArgs)
        assert repro._repro_dir == exp_dir
        assert repro.chat_model_args is agent_args.chat_model_args
        assert repro.flags is agent_args.flags
        assert repro.max_retry == agent_args.max_retry
        assert repro.agent_name == f"Repro_GenericAgent-{agent_args.chat_model_args.model_name}"

    def test_rejects_non_generic_agent_args(self, tmp_path):
        with pytest.raises(AssertionError):
            ra.make_repro_agent(object(), exp_dir=tmp_path)

    def test_rejects_missing_dir(self, tmp_path):
        args = GenericAgentArgs(chat_model_args=NoLLMArgs(), flags=make_flags())
        with pytest.raises(AssertionError):
            ra.make_repro_agent(args, exp_dir=tmp_path / "missing")


class TestReproduceStudy:
    def test_builds_one_repro_exp_per_original_exp(self, tmp_path, monkeypatch):
        study_dir = tmp_path / "study"
        shutil.copytree(TEST_STUDY, study_dir)

        created = {}

        class FakeStudy:
            def __init__(self, benchmark, agent_args):
                created["benchmark"] = benchmark
                created["agent_args"] = agent_args
                self.exp_args_list = None

        monkeypatch.setattr(ra, "Study", FakeStudy)
        study = ra.reproduce_study(study_dir, log_level=10)

        assert isinstance(study, FakeStudy)
        assert created["benchmark"] == "miniwob"
        assert len(study.exp_args_list) == 2
        exp_dirs = set()
        for exp_args in study.exp_args_list:
            agent_args = exp_args.agent_args
            assert isinstance(agent_args, ra.ReproAgentArgs)
            assert agent_args.agent_name.startswith("Repro_GenericAgent-")
            # legacy "bid" string action set was upgraded
            assert isinstance(agent_args.flags.action.action_set, HighLevelActionSetArgs)
            assert exp_args.logging_level == 10
            assert exp_args.env_args.task_name == "miniwob.ascending-numbers"
            exp_dirs.add(Path(agent_args._repro_dir).name)
        assert exp_dirs == {p.name for p in TEST_STUDY.iterdir() if p.is_dir()}
        assert created["agent_args"][0] is study.exp_args_list[-1].agent_args
