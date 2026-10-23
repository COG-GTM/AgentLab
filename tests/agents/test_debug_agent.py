from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from agentlab.agents import debug_agent
from agentlab.agents.debug_agent import DEBUG_AGENT, DebugAgent, DebugAgentArgs


def test_debug_agent_args_defaults():
    args = DebugAgentArgs()
    assert args.agent_name == "debug"
    assert args.use_html is False
    assert DEBUG_AGENT.agent_name == "debug"


def test_set_benchmark_enables_html_for_miniwob():
    args = DebugAgentArgs()
    action_set_args = MagicMock()
    args.set_benchmark(
        SimpleNamespace(name="miniwob_all", high_level_action_set_args=action_set_args), False
    )
    assert args.use_html is True
    assert args.action_set_args is action_set_args

    args = DebugAgentArgs()
    args.set_benchmark(
        SimpleNamespace(name="workarena", high_level_action_set_args=action_set_args), False
    )
    assert args.use_html is False


def test_make_agent():
    agent = DebugAgentArgs().make_agent()
    assert isinstance(agent, DebugAgent)
    assert agent.action_set is not None


def test_obs_preprocessor(monkeypatch):
    monkeypatch.setattr(debug_agent, "flatten_dom_to_str", lambda dom, **kw: "<p>dom</p>")
    monkeypatch.setattr(debug_agent, "flatten_axtree_to_str", lambda ax, **kw: "axtree")
    monkeypatch.setattr(debug_agent, "prune_html", lambda html: "pruned:" + html)
    monkeypatch.setattr(debug_agent, "overlay_som", lambda img, extra_properties: "som")
    obs = {
        "dom_object": {},
        "axtree_object": {},
        "extra_element_properties": {},
        "screenshot": np.zeros((2, 2, 3)),
    }
    out = DebugAgentArgs().make_agent().obs_preprocessor(obs)
    assert out["dom_txt"] == "<p>dom</p>"
    assert out["axtree_txt"] == "axtree"
    assert out["pruned_html"] == "pruned:<p>dom</p>"
    assert out["screenshot_som"] == "som"
    assert "dom_txt" not in obs  # input is not mutated


def test_get_action_prompts_user(monkeypatch):
    prompts = []
    monkeypatch.setattr("builtins.input", lambda p: prompts.append(p) or "noop()")
    obs = {"pruned_html": "<html/>", "axtree_txt": "tree"}

    agent = DebugAgentArgs().make_agent()
    action, info = agent.get_action(obs)
    assert action == "noop()"
    assert info.think == "nope"
    assert prompts[-1].startswith("tree")

    agent.use_html = True
    agent.get_action(obs)
    assert prompts[-1].startswith("<html/>")
