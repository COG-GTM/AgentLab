from unittest.mock import patch

import bgym
from browsergym.experiments.agent import AgentInfo

from agentlab.agents.debug_agent import DebugAgent, DebugAgentArgs


def test_debug_agent_args_instantiation():
    """Test that DebugAgentArgs can be instantiated."""
    args = DebugAgentArgs()
    assert args.agent_name == "debug"
    assert args.action_set_args is not None


def test_debug_agent_args_make_agent():
    """Test that DebugAgentArgs.make_agent() returns a DebugAgent."""
    args = DebugAgentArgs()
    agent = args.make_agent()
    assert isinstance(agent, DebugAgent)


def test_debug_agent_get_action():
    """Test DebugAgent.get_action() with mocked input."""
    args = DebugAgentArgs()
    agent = args.make_agent()

    mock_obs = {
        "axtree_txt": "[1] Click me button",
        "pruned_html": "<button bid='1'>Click me</button>",
    }

    with patch("builtins.input", return_value="click('1')"):
        action, agent_info = agent.get_action(mock_obs)

    assert action == "click('1')"
    assert isinstance(agent_info, AgentInfo)


def test_debug_agent_set_benchmark():
    """Test that set_benchmark configures correctly for miniwob."""
    args = DebugAgentArgs()

    class FakeBenchmark:
        name = "miniwob_tiny_test"
        high_level_action_set_args = bgym.HighLevelActionSetArgs(subsets=["bid"])

    args.set_benchmark(FakeBenchmark(), demo_mode=False)
    assert args.use_html is True
