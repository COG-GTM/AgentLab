from copy import deepcopy

import numpy as np
import bgym
import pytest
from bgym import HighLevelActionSet, HighLevelActionSetArgs

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.generic_agent.agent_configs import FLAGS_GPT_3_5
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags, MainPrompt
from agentlab.llm.llm_utils import count_tokens

html_template = """
<html>
<body>
<div>
Hello World.
Step {}.
</div>
</body>
some extra text to make the html longer
</html>
"""

base_obs = {
    "goal": "do this and that",
    "goal_object": [{"type": "text", "text": "do this and that"}],
    "chat_messages": [{"role": "user", "message": "do this and that"}],
    "axtree_txt": "[1] Click me",
    "focused_element_bid": "45-256",
    "open_pages_urls": ["https://example.com"],
    "open_pages_titles": ["Example"],
    "active_page_index": 0,
}

OBS_HISTORY = [
    base_obs
    | {
        "pruned_html": html_template.format(1),
        "last_action_error": "",
    },
    base_obs
    | {
        "pruned_html": html_template.format(2),
        "last_action_error": "Hey, this is an error in the past",
    },
    base_obs
    | {
        "pruned_html": html_template.format(3),
        "last_action_error": "Hey, there is an error now",
    },
]
ACTIONS = ["click('41')", "click('42')"]
MEMORIES = ["memory A", "memory B"]
THOUGHTS = ["thought A", "thought B"]

ALL_TRUE_FLAGS = GenericPromptFlags(
    obs=dp.ObsFlags(
        use_html=True,
        use_ax_tree=True,
        use_tabs=True,
        use_focused_element=True,
        use_error_logs=True,
        use_history=True,
        use_past_error_logs=True,
        use_action_history=True,
        use_think_history=True,
        use_diff=True,
        html_type="pruned_html",
        use_screenshot=False,  # TODO test this
        use_som=False,  # TODO test this
        extract_visible_tag=True,
        extract_clickable_tag=True,
        extract_coords=False,
        filter_visible_elements_only=True,
    ),
    action=dp.ActionFlags(
        action_set=HighLevelActionSetArgs(
            subsets=["bid"],
            multiaction=True,
        ),
        long_description=True,
        individual_examples=True,
    ),
    use_plan=True,
    use_criticise=True,
    use_thinking=True,
    use_memory=True,
    use_concrete_example=True,
    use_abstract_example=True,
    use_hints=True,
    enable_chat=False,  # TODO test this
    max_prompt_tokens=None,
    be_cautious=True,
    extra_instructions=None,
)


FLAG_EXPECTED_PROMPT = [
    (
        "obs.use_html",
        ("HTML:", "</html>", "Hello World.", "Step 3."),  # last obs will be in obs
    ),
    (
        "obs.use_ax_tree",
        ("AXTree:", "Click me"),
    ),
    (
        "obs.use_tabs",
        ("Currently open tabs:", "(active tab)"),
    ),
    (
        "obs.use_focused_element",
        ("Focused element:", "bid='45-256'"),
    ),
    (
        "obs.use_error_logs",
        ("Hey, there is an error now",),
    ),
    (
        "use_plan",
        ("You just executed step", "1- think\n2- do it"),
    ),
    (
        "use_criticise",
        (
            "Criticise action_draft",
            "<criticise>",
            "</criticise>",
            "<action_draft>",
        ),
    ),
    (
        "use_thinking",
        ("<think>", "</think>"),
    ),
    (
        "obs.use_past_error_logs",
        ("Hey, this is an error in the past",),
    ),
    (
        "obs.use_action_history",
        ("<action>", "click('41')", "click('42')"),
    ),
    (
        "use_memory",
        ("<memory>", "</memory>", "memory A", "memory B"),
    ),
    # (
    #     "obs.use_diff",
    #     ("diff:", "- Step 2", "Identical"),
    # ),
    (
        "use_concrete_example",
        ("# Concrete Example", "<action>\nclick('a324')"),
    ),
    (
        "use_abstract_example",
        ("# Abstract Example",),
    ),
    # (
    #     "action.action_set.multiaction",
    #     ("One or several actions, separated by new lines",),
    # ),
]


def test_shrinking_observation():
    flags = deepcopy(FLAGS_GPT_3_5)
    flags.obs.use_html = True

    prompt_maker = MainPrompt(
        action_set=HighLevelActionSet(),
        obs_history=OBS_HISTORY,
        actions=ACTIONS,
        memories=MEMORIES,
        thoughts=THOUGHTS,
        previous_plan="1- think\n2- do it",
        step=2,
        flags=flags,
    )

    prompt = str(prompt_maker.prompt)
    new_prompt = str(
        dp.fit_tokens(prompt_maker, max_prompt_tokens=count_tokens(prompt) - 1, max_iterations=7)
    )
    assert count_tokens(new_prompt) < count_tokens(prompt)
    assert "[1] Click me" in prompt
    assert "[1] Click me" in new_prompt
    assert "</html>" in prompt
    assert "</html>" not in new_prompt


@pytest.mark.parametrize("flag_name, expected_prompts", FLAG_EXPECTED_PROMPT)
def test_main_prompt_elements_gone_one_at_a_time(flag_name: str, expected_prompts):

    if flag_name in ["use_thinking", "obs.use_action_history"]:
        # These flags interact with history flags (use_think_history, use_action_history).
        # Disabling use_thinking alone doesn't remove <think> from history when
        # use_think_history is still True. Dedicated tests below cover these cases.
        return

    # Disable the flag
    flags = deepcopy(ALL_TRUE_FLAGS)
    if "." in flag_name:
        prefix, flag_name = flag_name.split(".")
        sub_flags = getattr(flags, prefix)
        setattr(sub_flags, flag_name, False)
    else:
        setattr(flags, flag_name, False)

    if flag_name == "use_memory":
        memories = None
    else:
        memories = MEMORIES

    # Initialize MainPrompt
    prompt = str(
        MainPrompt(
            action_set=flags.action.action_set.make_action_set(),
            obs_history=OBS_HISTORY,
            actions=ACTIONS,
            memories=memories,
            thoughts=THOUGHTS,
            previous_plan="1- think\n2- do it",
            step=2,
            flags=flags,
        ).prompt
    )

    # Verify all elements are not present
    for expected in expected_prompts:
        assert expected not in prompt


def test_main_prompt_elements_present():
    # Make sure the flag is enabled

    # Initialize MainPrompt
    prompt = str(
        MainPrompt(
            action_set=HighLevelActionSet(),
            obs_history=OBS_HISTORY,
            actions=ACTIONS,
            memories=MEMORIES,
            thoughts=THOUGHTS,
            previous_plan="1- think\n2- do it",
            step=2,
            flags=ALL_TRUE_FLAGS,
        ).prompt
    )
    # Verify all elements are not present
    for _, expected_prompts in FLAG_EXPECTED_PROMPT:
        for expected in expected_prompts:
            assert expected in prompt


def _make_screenshot_obs():
    """Return a copy of base_obs with a mock screenshot (numpy array)."""
    obs = base_obs.copy()
    obs["screenshot"] = np.zeros((64, 64, 3), dtype=np.uint8)
    obs["screenshot_som"] = np.zeros((64, 64, 3), dtype=np.uint8)
    return obs


def test_use_screenshot_adds_image():
    """When use_screenshot=True the prompt should contain a Screenshot section."""
    flags = deepcopy(ALL_TRUE_FLAGS)
    flags.obs.use_screenshot = True
    flags.obs.use_som = False

    obs_with_screenshot = _make_screenshot_obs()
    obs_history = [
        obs_with_screenshot | {"pruned_html": html_template.format(1), "last_action_error": ""},
        obs_with_screenshot | {"pruned_html": html_template.format(2), "last_action_error": ""},
        obs_with_screenshot | {"pruned_html": html_template.format(3), "last_action_error": ""},
    ]

    prompt_maker = MainPrompt(
        action_set=flags.action.action_set.make_action_set(),
        obs_history=obs_history,
        actions=ACTIONS,
        memories=MEMORIES,
        thoughts=THOUGHTS,
        previous_plan="1- think\n2- do it",
        step=2,
        flags=flags,
    )
    prompt = prompt_maker.prompt
    prompt_str = prompt.__str__(warn_if_image=False)
    assert "Screenshot" in prompt_str


def test_use_som_adds_annotated_screenshot():
    """When use_som=True the screenshot section should mention bounding boxes."""
    flags = deepcopy(ALL_TRUE_FLAGS)
    flags.obs.use_screenshot = True
    flags.obs.use_som = True

    obs_with_screenshot = _make_screenshot_obs()
    obs_history = [
        obs_with_screenshot | {"pruned_html": html_template.format(1), "last_action_error": ""},
        obs_with_screenshot | {"pruned_html": html_template.format(2), "last_action_error": ""},
        obs_with_screenshot | {"pruned_html": html_template.format(3), "last_action_error": ""},
    ]

    prompt_maker = MainPrompt(
        action_set=flags.action.action_set.make_action_set(),
        obs_history=obs_history,
        actions=ACTIONS,
        memories=MEMORIES,
        thoughts=THOUGHTS,
        previous_plan="1- think\n2- do it",
        step=2,
        flags=flags,
    )
    prompt = prompt_maker.prompt
    prompt_str = prompt.__str__(warn_if_image=False)
    assert "bounding boxes" in prompt_str


def test_enable_chat_uses_chat_instructions():
    """When enable_chat=True the prompt should use ChatInstructions (UI Assistant)."""
    import time as _time

    flags = deepcopy(ALL_TRUE_FLAGS)
    flags.enable_chat = True

    # ChatInstructions requires 'timestamp' in chat_messages
    chat_obs = deepcopy(OBS_HISTORY)
    for obs in chat_obs:
        for msg in obs["chat_messages"]:
            msg["timestamp"] = _time.time()

    prompt_maker = MainPrompt(
        action_set=flags.action.action_set.make_action_set(),
        obs_history=chat_obs,
        actions=ACTIONS,
        memories=MEMORIES,
        thoughts=THOUGHTS,
        previous_plan="1- think\n2- do it",
        step=2,
        flags=flags,
    )
    prompt_str = str(prompt_maker.prompt)
    assert "UI Assistant" in prompt_str


def test_use_thinking_disabled_removes_think_tags():
    """When use_thinking=False and use_think_history=False the prompt has no <think> tags."""
    flags = deepcopy(ALL_TRUE_FLAGS)
    flags.use_thinking = False
    flags.obs.use_think_history = False

    prompt_str = str(
        MainPrompt(
            action_set=flags.action.action_set.make_action_set(),
            obs_history=OBS_HISTORY,
            actions=ACTIONS,
            memories=MEMORIES,
            thoughts=THOUGHTS,
            previous_plan="1- think\n2- do it",
            step=2,
            flags=flags,
        ).prompt
    )
    assert "<think>" not in prompt_str
    assert "</think>" not in prompt_str


def test_use_action_history_disabled_removes_action_tags():
    """When obs.use_action_history=False the history should not contain <action> tags."""
    flags = deepcopy(ALL_TRUE_FLAGS)
    flags.obs.use_action_history = False
    # Also disable the action in concrete/abstract examples which may contain action tags
    flags.use_concrete_example = False
    flags.use_abstract_example = False

    prompt_str = str(
        MainPrompt(
            action_set=flags.action.action_set.make_action_set(),
            obs_history=OBS_HISTORY,
            actions=ACTIONS,
            memories=MEMORIES,
            thoughts=THOUGHTS,
            previous_plan="1- think\n2- do it",
            step=2,
            flags=flags,
        ).prompt
    )
    assert "<action>" not in prompt_str
    assert "click('41')" not in prompt_str


if __name__ == "__main__":
    # for debugging
    test_shrinking_observation()
    test_main_prompt_elements_present()
    # for flag, expected_prompts in FLAG_EXPECTED_PROMPT:
    #     test_main_prompt_elements_gone_one_at_a_time(flag, expected_prompts)
