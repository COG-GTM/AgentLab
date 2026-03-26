"""Tests for MultiToolGym - skipped if tapeagents is not installed."""

import pytest

try:
    from tapeagents.core import Action, Observation, StopStep
    from tapeagents.tools.base import Tool

    TAPEAGENTS_AVAILABLE = True
except ImportError:
    TAPEAGENTS_AVAILABLE = False


@pytest.mark.skipif(not TAPEAGENTS_AVAILABLE, reason="tapeagents not installed")
def test_multitool_gym_instantiation():
    """Test MultiToolGym can be instantiated with empty tools."""
    from agentlab.benchmarks.multitool_gym import MultiToolGym

    gym = MultiToolGym(tools=[], max_turns=10)
    assert gym.max_turns == 10
    assert gym._turns == 0


@pytest.mark.skipif(not TAPEAGENTS_AVAILABLE, reason="tapeagents not installed")
def test_multitool_gym_reset():
    """Test MultiToolGym reset clears turns."""
    from agentlab.benchmarks.multitool_gym import MultiToolGym

    gym = MultiToolGym(tools=[], max_turns=10)
    gym._turns = 5
    gym.reset()
    assert gym._turns == 0


@pytest.mark.skipif(not TAPEAGENTS_AVAILABLE, reason="tapeagents not installed")
def test_multitool_gym_stop_step():
    """Test MultiToolGym handles StopStep action."""
    from agentlab.benchmarks.multitool_gym import MultiToolGym

    gym = MultiToolGym(tools=[], max_turns=10)
    gym.reset()
    obs, reward, terminated, truncated, env_info = gym.step(StopStep())
    assert terminated is True
    assert truncated is False
    assert "action_exec_start" in env_info
    assert "action_exec_stop" in env_info


@pytest.mark.skipif(not TAPEAGENTS_AVAILABLE, reason="tapeagents not installed")
def test_multitool_gym_calculate_reward():
    """Test default reward calculation returns 0."""
    from agentlab.benchmarks.multitool_gym import MultiToolGym

    gym = MultiToolGym(tools=[], max_turns=10)
    reward = gym.calculate_reward(StopStep())
    assert reward == 0.0
