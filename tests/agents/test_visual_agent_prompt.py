from copy import deepcopy

from agentlab.agents.visual_agent.agent_configs import DEFAULT_PROMPT_FLAGS
from agentlab.agents.visual_agent.visual_agent_prompts import MainPrompt

OBS = {
    "goal_object": [{"type": "text", "text": "do this and that"}],
    "chat_messages": [{"role": "user", "message": "do this and that"}],
    "last_action_error": "",
    "open_pages_urls": ["https://example.com"],
    "open_pages_titles": ["Example"],
    "active_page_index": 0,
}

ANSWER = """
<think>
I should click on the login button.
</think>
<action>
mouse_click(324, 512)
</action>
"""


def make_prompt(flags):
    return MainPrompt(
        action_set=flags.action.action_set.make_action_set(),
        obs=OBS,
        actions=["mouse_click(1, 2)"],
        thoughts=["thought A"],
        flags=flags,
    )


def test_parse_answer_merges_think_and_action():
    ans_dict = make_prompt(deepcopy(DEFAULT_PROMPT_FLAGS))._parse_answer(ANSWER)

    assert ans_dict["think"].strip() == "I should click on the login button."
    assert ans_dict["action"].strip() == "mouse_click(324, 512)"
    assert "parse_error" not in ans_dict


def test_parse_answer_without_thinking():
    flags = deepcopy(DEFAULT_PROMPT_FLAGS)
    flags.use_thinking = False

    ans_dict = make_prompt(flags)._parse_answer(ANSWER)

    assert "think" not in ans_dict
    assert ans_dict["action"].strip() == "mouse_click(324, 512)"


def test_parse_answer_reports_missing_think():
    ans_dict = make_prompt(deepcopy(DEFAULT_PROMPT_FLAGS))._parse_answer(
        "<action>\nmouse_click(1, 2)\n</action>"
    )

    assert "parse_error" in ans_dict
    assert ans_dict["action"].strip() == "mouse_click(1, 2)"
