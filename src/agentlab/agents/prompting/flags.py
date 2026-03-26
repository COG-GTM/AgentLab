from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Literal

from bgym import HighLevelActionSetArgs


class Flags:
    """Base class for flags. Mostly for backward compatibility."""

    def copy(self):
        return deepcopy(self)

    def asdict(self):
        """Helper for JSON serializable requirement."""
        return asdict(self)

    @classmethod
    def from_dict(self, flags_dict):
        """Helper for JSON serializable requirement."""
        if isinstance(flags_dict, ObsFlags):
            return flags_dict

        if not isinstance(flags_dict, dict):
            raise ValueError(f"Unregcognized type for flags_dict of type {type(flags_dict)}.")
        return ObsFlags(**flags_dict)


@dataclass
class ObsFlags(Flags):
    """
    A class to represent various flags used to control features in an application.

    Attributes:
        use_html (bool): Use the HTML in the prompt.
        use_ax_tree (bool): Use the accessibility tree in the prompt.
        use_focused_element (bool): Provide the ID of the focused element.
        use_error_logs (bool): Expose the previous error in the prompt.
        use_history (bool): Enable history of previous steps in the prompt.
        use_past_error_logs (bool): If use_history is True, expose all previous errors in the history.
        use_action_history (bool): If use_history is True, include the actions in the history.
        use_think_history (bool): If use_history is True, include all previous chains of thoughts in the history.
        use_diff (bool): Add a diff of the current and previous HTML to the prompt.
        html_type (str): Type of HTML to use in the prompt, may depend on preprocessing of observation.
        use_screenshot (bool): Add a screenshot of the page to the prompt, following OpenAI's API. This will be automatically disabled if the model does not have vision capabilities.
        use_som (bool): Add a set of marks to the screenshot.
        extract_visible_tag (bool): Add a "visible" tag to visible elements in the AXTree.
        extract_clickable_tag (bool): Add a "clickable" tag to clickable elements in the AXTree.
        extract_coords (Literal['False', 'center', 'box']): Add the coordinates of the elements.
        filter_visible_elements_only (bool): Only show visible elements in the AXTree.
    """

    use_html: bool = True
    use_ax_tree: bool = False
    use_tabs: bool = False
    use_focused_element: bool = False
    use_error_logs: bool = False
    use_history: bool = False
    use_past_error_logs: bool = False
    use_action_history: bool = False
    use_think_history: bool = False
    use_diff: bool = False  #
    html_type: str = "pruned_html"
    use_screenshot: bool = True
    use_som: bool = False
    extract_visible_tag: bool = False
    extract_clickable_tag: bool = False
    extract_coords: Literal["False", "center", "box"] = "False"
    filter_visible_elements_only: bool = False
    # low sets the token count of each image to 65 (85?)
    # high sets the token count of each image to 2*65 (2*85?) times the amount of 512x512px patches
    # auto chooses between low and high based on image size (openai default)
    openai_vision_detail: Literal["low", "high", "auto"] = "auto"
    filter_with_bid_only: bool = False
    filter_som_only: bool = False


@dataclass
class ActionFlags(Flags):
    action_set: HighLevelActionSetArgs = None  # should be set by the set_benchmark method
    long_description: bool = True
    individual_examples: bool = False

    # for backward compatibility
    multi_actions: bool = None
    is_strict: bool = None
