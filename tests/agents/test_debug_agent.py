from dataclasses import dataclass
from unittest.mock import MagicMock

import bgym
import numpy as np
import pytest
from bgym import HighLevelActionSetArgs

from agentlab.agents import debug_agent as debug_mod
from agentlab.agents.agent_args import AgentArgs
from agentlab.agents.debug_agent import DEBUG_AGENT, DebugAgent, DebugAgentArgs


def _benchmark(name, action_set_args=None):
    bench = MagicMock()
    bench.name = name
    bench.high_level_action_set_args = action_set_args or HighLevelActionSetArgs(subsets=["bid"])
    return bench


# ---------------------------------------------------------------- AgentArgs base


@dataclass
class _PlainArgs(AgentArgs):
    foo: int = 1

    def make_agent(self):
        return None


def test_agent_args_default_name_and_set_benchmark_is_noop():
    args = _PlainArgs()
    assert args.agent_name == "_PlainArgs"
    before = dict(vars(args))
    assert args.set_benchmark(_benchmark("webarena"), demo_mode=True) is None
    assert vars(args) == before


def test_agent_args_set_reproducibility_mode_not_implemented():
    with pytest.raises(NotImplementedError, match="_PlainArgs"):
        _PlainArgs().set_reproducibility_mode()


# ---------------------------------------------------------------- DebugAgentArgs


def test_post_init_defaults():
    args = DebugAgentArgs()
    assert args.agent_name == "debug"
    assert args.use_html is False
    expected = bgym.DEFAULT_BENCHMARKS["miniwob_tiny_test"]().high_level_action_set_args
    assert args.action_set_args == expected


def test_module_level_instance():
    assert isinstance(DEBUG_AGENT, DebugAgentArgs)
    assert DEBUG_AGENT.agent_name == "debug"


@pytest.mark.parametrize(
    "name,expected_html",
    [("miniwob", True), ("miniwob_tiny_test", True), ("webarena", False), ("workarena_l1", False)],
)
def test_set_benchmark(name, expected_html):
    args = DebugAgentArgs()
    asa = HighLevelActionSetArgs(subsets=["bid", "nav"])
    args.set_benchmark(_benchmark(name, asa), demo_mode=False)
    assert args.use_html is expected_html
    assert args.action_set_args is asa


def test_set_reproducibility_mode_not_supported():
    with pytest.raises(NotImplementedError, match="DebugAgentArgs"):
        DebugAgentArgs().set_reproducibility_mode()


@pytest.mark.parametrize("use_html", [True, False])
def test_make_agent(use_html):
    args = DebugAgentArgs()
    args.action_set_args = HighLevelActionSetArgs(subsets=["bid"])
    args.use_html = use_html
    agent = args.make_agent()
    assert isinstance(agent, DebugAgent)
    assert agent.use_html is use_html
    assert isinstance(agent.action_set, bgym.HighLevelActionSet)
    assert "click" in agent.action_set.describe()


# ---------------------------------------------------------------- DebugAgent


@pytest.fixture
def agent():
    return DebugAgent(HighLevelActionSetArgs(subsets=["bid"]))


def test_obs_preprocessor_adds_text_views_without_mutating_input(agent, monkeypatch):
    dom = MagicMock(return_value="<dom>")
    ax = MagicMock(return_value="[ax]")
    prune = MagicMock(return_value="<pruned>")
    som = MagicMock(return_value="SOM")
    monkeypatch.setattr(debug_mod, "flatten_dom_to_str", dom)
    monkeypatch.setattr(debug_mod, "flatten_axtree_to_str", ax)
    monkeypatch.setattr(debug_mod, "prune_html", prune)
    monkeypatch.setattr(debug_mod, "overlay_som", som)

    screenshot = np.zeros((4, 4, 3), np.uint8)
    obs = {
        "dom_object": {"d": 1},
        "axtree_object": {"a": 1},
        "extra_element_properties": {"x": {}},
        "screenshot": screenshot,
    }
    out = agent.obs_preprocessor(obs)

    assert out["dom_txt"] == "<dom>"
    assert out["axtree_txt"] == "[ax]"
    assert out["pruned_html"] == "<pruned>"
    assert out["screenshot_som"] == "SOM"
    assert set(obs) == {"dom_object", "axtree_object", "extra_element_properties", "screenshot"}

    expected_kwargs = dict(
        extra_properties={"x": {}},
        with_visible=True,
        with_clickable=True,
        with_center_coords=True,
        with_bounding_box_coords=True,
        filter_visible_only=False,
        filter_with_bid_only=False,
        filter_som_only=False,
    )
    assert dom.call_args.args == ({"d": 1},)
    assert dom.call_args.kwargs == expected_kwargs
    assert ax.call_args.args == ({"a": 1},)
    assert ax.call_args.kwargs == expected_kwargs
    prune.assert_called_once_with("<dom>")
    assert som.call_args.kwargs == {"extra_properties": {"x": {}}}
    # input was deep-copied before processing
    assert dom.call_args.args[0] is not obs["dom_object"]


@pytest.mark.parametrize(
    "use_html,shown", [(True, "<html pruned/>"), (False, "RootWebArea 'axtree'")]
)
def test_get_action_reads_from_stdin(monkeypatch, capsys, use_html, shown):
    agent = DebugAgent(HighLevelActionSetArgs(subsets=["bid"]), use_html=use_html)
    prompts = []

    def fake_input(prompt):
        prompts.append(prompt)
        return "click('a1')"

    monkeypatch.setattr("builtins.input", fake_input)
    obs = {"pruned_html": "<html pruned/>", "axtree_txt": "RootWebArea 'axtree'"}
    action, info = agent.get_action(obs)

    assert action == "click('a1')"
    assert prompts == [shown + "\n"]
    assert info.think == "nope"
    assert info.chat_messages == []
    assert info.stats == {}
