import platform
import time
from copy import copy

from browsergym.core.action.base import AbstractActionSet
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, overlay_som, prune_html

from agentlab.llm.llm_utils import (
    ParseError,
    extract_code_blocks,
    image_to_jpg_base64_url,
    parse_html_tags_raise,
    BaseMessage,
)

from .base import PromptElement, Shrinkable, Trunkater
from .flags import ActionFlags, ObsFlags


class HTML(Trunkater):
    def __init__(self, html, visible_elements_only: bool, visible: bool = True, prefix="") -> None:
        super().__init__(visible=visible, start_trunkate_iteration=5)
        if visible_elements_only:
            visible_elements_note = """\
Note: only elements that are visible in the viewport are presented. You might need to scroll the page, or open tabs or menus to see more.

"""
        else:
            visible_elements_note = ""
        self._prompt = f"\n{prefix}HTML:\n{visible_elements_note}{html}\n"


class AXTree(Trunkater):
    def __init__(
        self,
        ax_tree,
        visible_elements_only: bool,
        visible: bool = True,
        coord_type=None,
        visible_tag=True,
        prefix="",
    ) -> None:
        super().__init__(visible=visible, start_trunkate_iteration=10)
        bid_info = """\
Note: [bid] is the unique alpha-numeric identifier at the beginning of lines for each element in the AXTree. Always use bid to refer to elements in your actions.

"""
        if coord_type == "center":
            coord_note = """\
Note: center coordinates are provided in parenthesis and are relative to the top left corner of the page.

"""
        elif coord_type == "box":
            coord_note = """\
Note: bounding box of each object are provided in parenthesis and are relative to the top left corner of the page.

"""
        else:
            coord_note = ""
        if visible_elements_only:
            visible_elements_note = """\
Note: only elements that are visible in the viewport are presented. You might need to scroll the page, or open tabs or menus to see more.

"""
        else:
            visible_elements_note = ""

        if visible_tag:
            vsible_tag_note = """\
Note: You can only interact with visible elements. If the "visible" tag is not
present, the element is not visible on the page.

"""
        else:
            vsible_tag_note = ""
        self._prompt = f"\n{prefix}AXTree:\n{bid_info}{coord_note}{visible_elements_note}{vsible_tag_note}{ax_tree}\n"


class Error(PromptElement):
    def __init__(self, error: str, visible: bool = True, prefix="", limit_logs=True) -> None:
        logs_separator = "Call log:"
        if limit_logs and logs_separator in error:
            error, logs = error.split(logs_separator)
            logs = "\n".join(logs.split("\n")[:10])
            error = error + f"\n{logs_separator}\n{logs}"

        super().__init__(visible=visible)
        self._prompt = f"\n{prefix}Error from previous action:\n{error}\n"


class FocusedElement(PromptElement):
    def __init__(self, bid, visible: bool = True, prefix="") -> None:
        super().__init__(visible=visible)
        self._prompt = f"""
{prefix}Focused element:
"""
        if bid:
            self._prompt += f"""\
bid={repr(bid)}
"""
        else:
            self._prompt += f"""\
None
"""


class Tabs(PromptElement):
    def __init__(self, obs, visible: bool = True, prefix="") -> None:
        super().__init__(visible=visible)
        self.obs = obs
        self.prefix = prefix

    @property
    def _prompt(self) -> str:
        # by implementing this as a property, it's only coputed if visible
        prompt_pieces = [f"\n{self.prefix}Currently open tabs:"]
        for page_index, (page_url, page_title) in enumerate(
            zip(self.obs["open_pages_urls"], self.obs["open_pages_titles"])
        ):
            active_or_not = " (active tab)" if page_index == self.obs["active_page_index"] else ""
            prompt_piece = f"""\
Tab {page_index}{active_or_not}:
    Title: {page_title}
    URL: {page_url}
"""
            prompt_pieces.append(prompt_piece)
        return "\n".join(prompt_pieces)


class Observation(Shrinkable):
    """Observation of the current step.

    Contains the html, the accessibility tree and the error logs.
    """

    def __init__(self, obs, flags: ObsFlags) -> None:
        super().__init__()
        self.flags = flags
        self.obs = obs

        self.tabs = Tabs(
            obs,
            visible=lambda: flags.use_tabs,
            prefix="## ",
        )

        self.html = HTML(
            obs[flags.html_type],
            visible_elements_only=flags.filter_visible_elements_only,
            visible=lambda: flags.use_html,
            prefix="## ",
        )
        self.ax_tree = AXTree(
            obs["axtree_txt"],
            visible_elements_only=flags.filter_visible_elements_only,
            visible=lambda: flags.use_ax_tree,
            coord_type=flags.extract_coords,
            visible_tag=flags.extract_visible_tag,
            prefix="## ",
        )
        self.error = Error(
            obs["last_action_error"],
            visible=lambda: flags.use_error_logs and obs["last_action_error"],
            prefix="## ",
        )
        self.focused_element = FocusedElement(
            obs["focused_element_bid"],
            visible=flags.use_focused_element,
            prefix="## ",
        )

    def shrink(self):
        self.ax_tree.shrink()
        self.html.shrink()

    @property
    def _prompt(self) -> str:
        return f"""
# Observation of current step:
{self.tabs.prompt}{self.html.prompt}{self.ax_tree.prompt}{self.focused_element.prompt}{self.error.prompt}

"""

    def add_screenshot(self, prompt: BaseMessage) -> BaseMessage:
        if self.flags.use_screenshot:
            if self.flags.use_som:
                screenshot = self.obs["screenshot_som"]
                prompt.add_text(
                    "\n## Screenshot:\nHere is a screenshot of the page, it is annotated with bounding boxes and corresponding bids:"
                )
            else:
                screenshot = self.obs["screenshot"]
                prompt.add_text("\n## Screenshot:\nHere is a screenshot of the page:")
            img_url = image_to_jpg_base64_url(screenshot)
            prompt.add_image(img_url, detail=self.flags.openai_vision_detail)
        return prompt


class MacNote(PromptElement):
    def __init__(self) -> None:
        super().__init__(visible=platform.system() == "Darwin")
        self._prompt = (
            "\nNote: you are on mac so you should use Meta instead of Control for Control+C etc.\n"
        )


class BeCautious(PromptElement):
    def __init__(self, visible: bool = True) -> None:
        super().__init__(visible=visible)
        self._prompt = f"""\
\nBe very cautious. Avoid submitting anything before verifying the effect of your
actions. Take the time to explore the effect of safe actions first. For example
you can fill a few elements of a form, but don't click submit before verifying
that everything was filled correctly.\n"""


class GoalInstructions(PromptElement):
    def __init__(self, goal_object, visible: bool = True, extra_instructions=None) -> None:
        super().__init__(visible)
        self._prompt = [
            dict(
                type="text",
                text=f"""\
# Instructions
Review the current state of the page and all other information to find the best
possible next action to accomplish your goal. Your answer will be interpreted
and executed by a program, make sure to follow the formatting instructions.

## Goal:
""",
            )
        ]

        self._prompt += goal_object

        if extra_instructions:
            self._prompt += [
                dict(
                    type="text",
                    text=f"""

## Extra instructions:

{extra_instructions}
""",
                )
            ]


class ChatInstructions(PromptElement):
    def __init__(self, chat_messages, visible: bool = True, extra_instructions=None) -> None:
        super().__init__(visible)
        self._prompt = f"""\
# Instructions

You are a UI Assistant, your goal is to help the user perform tasks using a web browser. You can
communicate with the user via a chat, in which the user gives you instructions and in which you
can send back messages. You have access to a web browser that both you and the user can see,
and with which only you can interact via specific commands.

Review the instructions from the user, the current state of the page and all other information
to find the best possible next action to accomplish your goal. Your answer will be interpreted
and executed by a program, make sure to follow the formatting instructions.

## Chat messages:

"""
        self._prompt += "\n".join(
            [
                f"""\
 - [{msg['role']}] UTC Time: {time.asctime(time.gmtime(msg['timestamp']))} - Local Time: {time.asctime(time.localtime(msg['timestamp']))} - {msg['message']}"""
                for msg in chat_messages
            ]
        )

        if extra_instructions:
            self._prompt += f"""

## Extra instructions:

{extra_instructions}
"""


class Hints(PromptElement):
    """Not super useful and stale."""

    # NOTE: are these hints still relevant?
    _prompt = """\
Note:
* Some tasks may be game like and may require to interact with the mouse position
in x, y coordinates.
* Some text field might have auto completion. To see it, you have to type a few
characters and wait until next step.
* If you have to cut and paste, don't forget to select the text first.
* Coordinate inside an SVG are relative to it's top left corner.
* Make sure to use bid to identify elements when using commands.
* Interacting with combobox, dropdowns and auto-complete fields can be tricky,
sometimes you need to use select_option, while other times you need to use fill
or click and wait for the reaction of the page.
"""


class SystemPrompt(PromptElement):
    _prompt = """\
You are an agent trying to solve a web task based on the content of the page and
user instructions. You can interact with the page and explore, and send messages to the user. Each time you
submit an action it will be sent to the browser and you will receive a new page."""


class ActionPrompt(PromptElement):

    _concrete_ex = """
<action>
click('a324')
</action>
"""

    def __init__(self, action_set: AbstractActionSet, action_flags: ActionFlags) -> None:
        super().__init__()
        self.action_set = action_set
        self.action_flags = action_flags
        action_set_generic_info = """\
Note: This action set allows you to interact with your environment. Most of them
are python function executing playwright code. The primary way of referring to
elements in the page is through bid which are specified in your observations.

"""
        action_description = action_set.describe(
            with_long_description=action_flags.long_description,
            with_examples=action_flags.individual_examples,
        )
        self._prompt = (
            f"# Action space:\n{action_set_generic_info}{action_description}{MacNote().prompt}\n"
        )
        self._abstract_ex = f"""
<action>
{self.action_set.example_action(abstract=True)}
</action>
"""

    def _parse_answer(self, text_answer):
        try:
            ans_dict = parse_html_tags_raise(text_answer, keys=["action"], merge_multiple=True)
        except ParseError as e:
            if self.action_flags.is_strict:
                raise e
            else:
                # try to extract code blocks
                blocks = extract_code_blocks(text_answer)
                if len(blocks) == 0:
                    raise e
                else:
                    code = "\n".join([block for _, block in blocks])
                    ans_dict = {"action": code, "parse_error": str(e)}

        try:
            if ans_dict["action"] == "None":
                # Used by reproducibility agent for backward compatibility of
                # traces missing LLM's response in chat messages.
                ans_dict["action"] = None
            else:
                # just check if action can be mapped to python code but keep action as is
                # the environment will be responsible for mapping it to python
                self.action_set.to_python_code(ans_dict["action"])
        except Exception as e:
            raise ParseError(
                f"Error while parsing action\n: {e}\n"
                "Make sure your answer is restricted to the allowed actions."
            )

        return ans_dict


class Think(PromptElement):
    _prompt = ""

    _abstract_ex = """
<think>
Think step by step. If you need to make calculations such as coordinates, write them here. Describe the effect
that your previous action had on the current content of the page.
</think>
"""
    _concrete_ex = """
<think>
From previous action I tried to set the value of year to "2022",
using select_option, but it doesn't appear to be in the form. It may be a
dynamic dropdown, I will try using click with the bid "a324" and look at the
response from the page.
</think>
"""

    def _parse_answer(self, text_answer):
        try:
            return parse_html_tags_raise(text_answer, keys=["think"], merge_multiple=True)
        except ParseError as e:
            return {"think": text_answer, "parse_error": str(e)}


class HistoryStep(Shrinkable):
    def __init__(
        self, previous_obs, current_obs, action, memory, thought, flags: ObsFlags, shrink_speed=1
    ) -> None:
        super().__init__()
        self.error = Error(
            current_obs["last_action_error"],
            visible=(
                lambda: flags.use_error_logs
                and current_obs["last_action_error"]
                and flags.use_past_error_logs
            ),
            prefix="### ",
        )
        self.shrink_speed = shrink_speed
        self.action = action
        self.memory = memory
        self.thought = thought
        self.flags = flags

    def shrink(self):
        super().shrink()

    @property
    def _prompt(self) -> str:
        prompt = ""

        if self.flags.use_think_history:
            prompt += f"\n<think>\n{self.thought}\n</think>\n"

        if self.flags.use_action_history:
            prompt += f"\n<action>\n{self.action}\n</action>\n"

        prompt += f"{self.error.prompt}"

        if self.memory is not None:
            prompt += f"\n<memory>\n{self.memory}\n</memory>\n"

        return prompt


class History(Shrinkable):
    def __init__(
        self, history_obs, actions, memories, thoughts, flags: ObsFlags, shrink_speed=1
    ) -> None:
        if memories is None:
            memories = [None] * len(actions)
        super().__init__(visible=lambda: flags.use_history)
        assert len(history_obs) == len(actions) + 1
        assert len(history_obs) == len(memories) + 1

        self.shrink_speed = shrink_speed
        self.history_steps: list[HistoryStep] = []

        for i in range(1, len(history_obs)):
            self.history_steps.append(
                HistoryStep(
                    history_obs[i - 1],
                    history_obs[i],
                    actions[i - 1],
                    memories[i - 1],
                    thoughts[i - 1],
                    flags,
                )
            )

    def shrink(self):
        """Shrink individual steps"""
        # TODO set the shrink speed of older steps to be higher
        super().shrink()
        for step in self.history_steps:
            step.shrink()

    @property
    def _prompt(self):
        prompts = ["# History of interaction with the task:\n"]
        for i, step in enumerate(self.history_steps):
            prompts.append(f"## step {i}")
            prompts.append(step.prompt)
        return "\n".join(prompts) + "\n"


def make_obs_preprocessor(flags: ObsFlags):
    def obs_mapping(obs: dict):
        obs = copy(obs)
        obs["dom_txt"] = flatten_dom_to_str(
            obs["dom_object"],
            extra_properties=obs["extra_element_properties"],
            with_visible=flags.extract_visible_tag,
            with_clickable=flags.extract_clickable_tag,
            with_center_coords=flags.extract_coords == "center",
            with_bounding_box_coords=flags.extract_coords == "box",
            filter_visible_only=flags.filter_visible_elements_only,
            filter_with_bid_only=flags.filter_with_bid_only,
            filter_som_only=flags.filter_som_only,
        )
        obs["axtree_txt"] = flatten_axtree_to_str(
            obs["axtree_object"],
            extra_properties=obs["extra_element_properties"],
            with_visible=flags.extract_visible_tag,
            with_clickable=flags.extract_clickable_tag,
            with_center_coords=flags.extract_coords == "center",
            with_bounding_box_coords=flags.extract_coords == "box",
            filter_visible_only=flags.filter_visible_elements_only,
            filter_with_bid_only=flags.filter_with_bid_only,
            filter_som_only=flags.filter_som_only,
        )
        obs["pruned_html"] = prune_html(obs["dom_txt"])
        obs["screenshot_som"] = overlay_som(
            obs["screenshot"], extra_properties=obs["extra_element_properties"]
        )

        return obs

    return obs_mapping
