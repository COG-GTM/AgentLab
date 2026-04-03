from bgym import HighLevelActionSetArgs

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.visual_agent.visual_agent import VisualAgent, VisualAgentArgs
from agentlab.agents.visual_agent.visual_agent_prompts import PromptFlags
from agentlab.llm.chat_api import CheatMiniWoBLLMArgs


def _make_visual_agent_flags():
    return PromptFlags(
        obs=dp.ObsFlags(
            use_html=False,
            use_ax_tree=True,
            use_tabs=False,
            use_focused_element=False,
            use_error_logs=True,
            use_history=False,
            use_past_error_logs=False,
            use_action_history=False,
            use_think_history=False,
            use_diff=False,
            use_screenshot=False,
            use_som=False,
            extract_visible_tag=False,
            extract_clickable_tag=False,
            extract_coords=False,
            filter_visible_elements_only=False,
        ),
        action=dp.ActionFlags(
            action_set=HighLevelActionSetArgs(subsets=["bid"], multiaction=False),
            long_description=False,
            individual_examples=False,
        ),
        use_thinking=True,
        use_concrete_example=False,
        use_abstract_example=True,
        enable_chat=False,
    )


def test_visual_agent_args_instantiation():
    """Test VisualAgentArgs can be created with cheat LLM."""
    flags = _make_visual_agent_flags()
    args = VisualAgentArgs(
        chat_model_args=CheatMiniWoBLLMArgs(),
        flags=flags,
    )
    assert args.chat_model_args is not None
    assert args.flags is not None


def test_visual_agent_args_make_agent():
    """Test that make_agent returns a VisualAgent."""
    flags = _make_visual_agent_flags()
    args = VisualAgentArgs(
        chat_model_args=CheatMiniWoBLLMArgs(),
        flags=flags,
    )
    agent = args.make_agent()
    assert isinstance(agent, VisualAgent)


def test_visual_agent_reset():
    """Test that reset clears agent state."""
    flags = _make_visual_agent_flags()
    args = VisualAgentArgs(
        chat_model_args=CheatMiniWoBLLMArgs(),
        flags=flags,
    )
    agent = args.make_agent()
    agent.actions.append("some_action")
    agent.thoughts.append("some_thought")
    agent.reset(seed=42)
    assert agent.actions == []
    assert agent.thoughts == []
    assert agent.seed == 42
