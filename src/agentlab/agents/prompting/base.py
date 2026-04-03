import abc
import logging
from textwrap import dedent
from warnings import warn

from agentlab.llm.llm_utils import BaseMessage, count_tokens


class PromptElement:
    """Base class for all prompt elements. Prompt elements can be hidden."""

    _prompt = ""
    _abstract_ex = ""
    _concrete_ex = ""

    def __init__(self, visible: bool = True) -> None:
        """Prompt element that can be hidden.

        Args:
            visible : bool, optional
                Whether the prompt element should be visible, by default True. Can
                be a callable that returns a bool. This is useful when a specific
                flag changes during a shrink iteration.
        """
        self._visible = visible

    @property
    def prompt(self) -> str | BaseMessage:
        """Avoid overriding this method. Override _prompt instead."""
        if self.is_visible:
            return self._prompt
        else:
            return ""

    @property
    def abstract_ex(self):
        """Useful when this prompt element is requesting an answer from the llm.
        Provide an abstract example of the answer here. See Memory for an
        example.

        Avoid overriding this method. Override _abstract_ex instead

        Returns:
            str: The abstract example
        """
        if self.is_visible:
            return self._abstract_ex
        else:
            return ""

    @property
    def concrete_ex(self):
        """Useful when this prompt element is requesting an answer from the llm.
        Provide a concrete example of the answer here. See Memory for an
        example.

        Avoid overriding this method. Override _concrete_ex instead

        Returns:
            str: The concrete example
        """
        if self.is_visible:
            return self._concrete_ex
        else:
            return ""

    @property
    def is_visible(self):
        """Handle the case where visible is a callable."""
        visible = self._visible
        if callable(visible):
            visible = visible()
        return visible

    def _parse_answer(self, text_answer):
        """Override to actually extract elements from the answer."""
        return {}

    def parse_answer(self, text_answer) -> dict:
        if self.is_visible:
            return self._parse_answer(text_answer)
        else:
            return {}


class Shrinkable(PromptElement, abc.ABC):
    @abc.abstractmethod
    def shrink(self) -> None:
        """Implement shrinking of this prompt element.

        You need to recursively call all shrinkable elements that are part of
        this prompt. You can also implement a shriking startegy for this prompt.
        Shrinking is can be called multiple times to progressively shrink the
        prompt until it fits max_tokens. Default max shrink iterations is 20.
        """
        pass


class Trunkater(Shrinkable):
    """Shrinkable element that truncates the prompt element from the bottom
    after a certain number of iterations."""

    def __init__(self, visible, shrink_speed=0.3, start_trunkate_iteration=10):
        super().__init__(visible=visible)
        self.shrink_speed = shrink_speed
        self.start_trunkate_iteration = start_trunkate_iteration
        self.shrink_calls = 0
        self.deleted_lines = 0

    def shrink(self) -> None:
        if self.is_visible and self.shrink_calls >= self.start_trunkate_iteration:
            # remove the fraction of _prompt
            lines = self._prompt.splitlines()
            new_line_count = int(len(lines) * (1 - self.shrink_speed))
            self.deleted_lines += len(lines) - new_line_count
            self._prompt = "\n".join(lines[:new_line_count])
            self._prompt += f"\n... Deleted {self.deleted_lines} lines to reduce prompt size."

        self.shrink_calls += 1


def fit_tokens(
    shrinkable: Shrinkable,
    max_prompt_tokens=None,
    max_iterations=20,
    model_name="openai/gpt-4",
    additional_prompts=[""],
):
    """Shrink a prompt element until it fits `max_prompt_tokens`.

    Args:
        shrinkable (Shrinkable): The prompt element to shrink.
        max_prompt_tokens (int): The maximum number of tokens allowed.
        max_iterations (int, optional): The maximum number of shrink iterations, by default 20.
        model_name (str, optional): The name of the model used when tokenizing.
        additional_prompts (str or List[str], optional): Additional prompts to account for when shrinking, by default [""].

    Returns:
        str: the prompt after shrinking.

    Raises:
        ValueError: Unrecognized type for prompt
    """

    if max_prompt_tokens is None:
        return shrinkable.prompt

    if isinstance(additional_prompts, str):
        additional_prompts = [additional_prompts]

    for prompt in additional_prompts:
        max_prompt_tokens -= count_tokens(prompt, model=model_name) + 1  # +1 because why not ?

    for _ in range(max_iterations):
        prompt = shrinkable.prompt
        if isinstance(prompt, str):
            prompt_str = prompt
        elif isinstance(prompt, list):
            # warn deprecated
            warn(
                "Using list of prompts is deprecated. Use a Discussion object instead.",
                DeprecationWarning,
            )
            prompt_str = "\n".join([p["text"] for p in prompt if p["type"] == "text"])
        elif isinstance(prompt, BaseMessage):
            prompt_str = prompt.__str__(warn_if_image=False)
        else:
            raise ValueError(f"Unrecognized type for prompt: {type(prompt)}")
        n_token = count_tokens(prompt_str, model=model_name)
        if n_token <= max_prompt_tokens:
            return prompt
        shrinkable.shrink()

    logging.info(
        dedent(
            f"""\
            After {max_iterations} shrink iterations, the prompt is still
            {count_tokens(prompt_str)} tokens (greater than {max_prompt_tokens}). Returning the prompt as is."""
        )
    )
    return prompt
