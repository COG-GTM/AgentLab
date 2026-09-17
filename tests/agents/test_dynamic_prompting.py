"""Unit tests for the parsing and prompt-shrinking logic of dynamic_prompting."""

import pytest

from agentlab.agents import dynamic_prompting as dp
from agentlab.llm.llm_utils import ParseError


class StubActionSet:
    """Minimal action set recording calls to to_python_code."""

    def __init__(self, raise_on_code=False):
        self.raise_on_code = raise_on_code
        self.calls = []

    def describe(self, with_long_description=True, with_examples=True):
        return "stub action space\n"

    def example_action(self, abstract=False):
        return "click('a324')"

    def to_python_code(self, action):
        self.calls.append(action)
        if self.raise_on_code:
            raise ValueError("invalid action")
        return action


def make_action_prompt(is_strict=False, raise_on_code=False):
    action_set = StubActionSet(raise_on_code=raise_on_code)
    action_flags = dp.ActionFlags(is_strict=is_strict)
    return dp.ActionPrompt(action_set=action_set, action_flags=action_flags), action_set


def test_parse_answer_extracts_action():
    prompt, action_set = make_action_prompt()
    ans_dict = prompt._parse_answer("blah\n<action>\nclick('a324')\n</action>\n")
    assert ans_dict["action"] == "click('a324')"
    assert action_set.calls == ["click('a324')"]


def test_parse_answer_merges_multiple_actions():
    prompt, _ = make_action_prompt()
    ans_dict = prompt._parse_answer("<action>click('1')</action><action>click('2')</action>")
    assert ans_dict["action"] == "click('1')\nclick('2')"


def test_parse_answer_strict_reraises_parse_error():
    prompt, action_set = make_action_prompt(is_strict=True)
    with pytest.raises(ParseError):
        prompt._parse_answer("```python\nclick('a324')\n```")
    assert action_set.calls == []


def test_parse_answer_falls_back_to_code_blocks():
    prompt, action_set = make_action_prompt(is_strict=False)
    ans_dict = prompt._parse_answer("```python\nclick('1')\n```\nand\n```\nclick('2')\n```")
    assert ans_dict["action"] == "click('1')\nclick('2')"
    assert "parse_error" in ans_dict
    assert action_set.calls == ["click('1')\nclick('2')"]


def test_parse_answer_raises_when_no_code_block():
    prompt, _ = make_action_prompt(is_strict=False)
    with pytest.raises(ParseError):
        prompt._parse_answer("I have no idea what to do.")


def test_parse_answer_none_sentinel():
    prompt, action_set = make_action_prompt()
    ans_dict = prompt._parse_answer("<action>None</action>")
    assert ans_dict["action"] is None
    assert action_set.calls == []


def test_parse_answer_raises_when_action_is_invalid():
    prompt, _ = make_action_prompt(raise_on_code=True)
    with pytest.raises(ParseError):
        prompt._parse_answer("<action>not_an_action()</action>")


class LineTrunkater(dp.Trunkater):
    def __init__(self, n_lines=10, visible=True, **kwargs):
        super().__init__(visible=visible, **kwargs)
        self._prompt = "\n".join(f"line {i}" for i in range(n_lines))


def test_trunkater_waits_for_start_iteration():
    trunkater = LineTrunkater(n_lines=10, shrink_speed=0.3, start_trunkate_iteration=2)
    original = trunkater.prompt

    trunkater.shrink()
    trunkater.shrink()
    assert trunkater.prompt == original
    assert trunkater.deleted_lines == 0

    trunkater.shrink()
    assert trunkater.deleted_lines == 3
    assert trunkater.prompt.splitlines()[:7] == original.splitlines()[:7]
    assert "Deleted 3 lines" in trunkater.prompt


def test_trunkater_accumulates_deleted_lines():
    trunkater = LineTrunkater(n_lines=10, shrink_speed=0.5, start_trunkate_iteration=0)
    trunkater.shrink()
    assert trunkater.deleted_lines == 5
    trunkater.shrink()
    # 5 content lines + the note line, half of which are deleted
    assert trunkater.deleted_lines == 8


def test_trunkater_does_not_shrink_when_hidden():
    trunkater = LineTrunkater(n_lines=10, visible=False, start_trunkate_iteration=0)
    trunkater.shrink()
    assert trunkater.deleted_lines == 0
    assert trunkater.prompt == ""
    assert trunkater.shrink_calls == 1


@pytest.fixture
def word_count_tokens(monkeypatch):
    """Replace the tokenizer by a word counter, to keep the test offline and readable."""

    def count(text, model=None):
        return len(text.split())

    monkeypatch.setattr(dp, "count_tokens", count)
    return count


def test_fit_tokens_without_max_tokens_does_not_shrink(word_count_tokens):
    trunkater = LineTrunkater(n_lines=10, start_trunkate_iteration=0)
    prompt = dp.fit_tokens(trunkater, max_prompt_tokens=None)
    assert prompt == trunkater.prompt
    assert trunkater.shrink_calls == 0


def test_fit_tokens_returns_early_when_prompt_fits(word_count_tokens):
    trunkater = LineTrunkater(n_lines=10, start_trunkate_iteration=0)
    prompt = dp.fit_tokens(
        trunkater,
        max_prompt_tokens=word_count_tokens(trunkater.prompt),
        additional_prompts=[],
    )
    assert prompt == trunkater.prompt
    assert trunkater.shrink_calls == 0


def test_fit_tokens_accounts_for_additional_prompts(word_count_tokens):
    trunkater = LineTrunkater(n_lines=10, start_trunkate_iteration=0)
    additional = "some system prompt"
    # one token short of fitting once the additional prompt (+1) is subtracted
    budget = word_count_tokens(trunkater.prompt) + word_count_tokens(additional)
    prompt = dp.fit_tokens(
        trunkater, max_prompt_tokens=budget, additional_prompts=additional, max_iterations=3
    )
    # the additional prompt eats into the budget, forcing at least one shrink
    assert trunkater.shrink_calls > 0
    assert word_count_tokens(prompt) <= budget - word_count_tokens(additional) - 1
