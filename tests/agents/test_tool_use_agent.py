from dataclasses import asdict, dataclass

from agentlab.agents.tool_use_agent.tool_use_agent import (
    Block,
    Goal,
    MsgGroup,
    Obs,
    StructuredDiscussion,
)


def test_block_make_creates_copy():
    """Test that Block.make() creates a proper copy."""
    goal = Goal(goal_as_system_msg=True)
    cloned = goal.make()
    assert isinstance(cloned, Goal)
    assert cloned is not goal
    assert cloned.goal_as_system_msg == goal.goal_as_system_msg


def test_block_make_preserves_fields():
    """Test that Block.make() preserves dataclass fields."""
    obs = Obs(use_screenshot=False, use_axtree=True, use_dom=True)
    cloned = obs.make()
    assert cloned.use_screenshot is False
    assert cloned.use_axtree is True
    assert cloned.use_dom is True


def test_structured_discussion_new_group():
    """Test StructuredDiscussion group management."""
    sd = StructuredDiscussion()
    assert len(sd.groups) == 0

    sd.new_group("first")
    assert len(sd.groups) == 1
    assert sd.groups[0].name == "first"

    sd.new_group("second")
    assert len(sd.groups) == 2
    assert sd.groups[1].name == "second"


def test_structured_discussion_auto_group_name():
    """Test that StructuredDiscussion auto-generates group names."""
    sd = StructuredDiscussion()
    sd.new_group()
    assert sd.groups[0].name == "group_0"
    sd.new_group()
    assert sd.groups[1].name == "group_1"


def test_structured_discussion_is_goal_set():
    """Test is_goal_set check."""
    sd = StructuredDiscussion()
    assert sd.is_goal_set() is False
    sd.new_group("goal")
    assert sd.is_goal_set() is True


def test_structured_discussion_get_last_summary_empty():
    """Test get_last_summary returns None for empty discussion."""
    sd = StructuredDiscussion()
    assert sd.get_last_summary() is None


def test_msg_group_defaults():
    """Test MsgGroup default values."""
    group = MsgGroup()
    assert group.name is None
    assert group.messages == []
    assert group.summary is None


def test_msg_group_with_name():
    """Test MsgGroup with explicit name."""
    group = MsgGroup(name="test_group")
    assert group.name == "test_group"
