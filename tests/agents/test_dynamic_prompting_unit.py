"""Unit tests for agentlab.agents.dynamic_prompting (no browser, no tokenizer download)."""

import logging
from unittest import mock

import numpy as np
import pytest
from bgym import HighLevelActionSetArgs

from agentlab.agents import dynamic_prompting as dp
from agentlab.llm.llm_utils import HumanMessage, ParseError


def _word_count(text, model=None):
    return len(text.split())


@pytest.fixture
def fake_tokens(monkeypatch):
    """Replace tiktoken/HF token counting with a deterministic word count."""
    monkeypatch.setattr(dp, "count_tokens", _word_count)
    return _word_count


def _obs(**overrides):
    obs = {
        "pruned_html": "<div>hello</div>",
        "axtree_txt": "[12] button 'Submit'",
        "last_action_error": "",
        "focused_element_bid": "12",
        "open_pages_urls": ["https://a.example", "https://b.example"],
        "open_pages_titles": ["A", "B"],
        "active_page_index": 1,
        "screenshot": np.zeros((4, 4, 3), dtype=np.uint8),
        "screenshot_som": np.full((4, 4, 3), 255, dtype=np.uint8),
    }
    obs.update(overrides)
    return obs


def _action_set(multiaction=False, strict=False):
    return HighLevelActionSetArgs(
        subsets=["bid"], multiaction=multiaction, strict=strict
    ).make_action_set()


# --------------------------------------------------------------------------- Flags


class TestFlags:
    def test_copy_is_deep_and_independent(self):
        flags = dp.ObsFlags(use_html=False)
        clone = flags.copy()
        assert clone == flags and clone is not flags
        clone.use_html = True
        assert flags.use_html is False

    def test_asdict_roundtrip_through_from_dict(self):
        flags = dp.ObsFlags(use_ax_tree=True, html_type="dom_txt", extract_coords="box")
        as_dict = flags.asdict()
        assert as_dict["use_ax_tree"] is True
        assert as_dict["extract_coords"] == "box"
        assert dp.ObsFlags.from_dict(as_dict) == flags

    def test_from_dict_returns_obsflags_instance_unchanged(self):
        flags = dp.ObsFlags()
        assert dp.ObsFlags.from_dict(flags) is flags

    @pytest.mark.parametrize("bad", [None, 3, "use_html", ["use_html"]])
    def test_from_dict_rejects_non_dict(self, bad):
        with pytest.raises(ValueError, match="Unregcognized type"):
            dp.ObsFlags.from_dict(bad)

    def test_action_flags_defaults(self):
        flags = dp.ActionFlags()
        assert flags.action_set is None
        assert flags.long_description is True
        assert flags.individual_examples is False
        assert flags.multi_actions is None and flags.is_strict is None

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: Flags.from_dict ignores cls and always builds ObsFlags, so ActionFlags.from_dict fails",
    )
    def test_action_flags_from_dict_returns_action_flags(self):
        flags = dp.ActionFlags(long_description=False)
        restored = dp.ActionFlags.from_dict(flags.asdict())
        assert isinstance(restored, dp.ActionFlags)
        assert restored == flags


# --------------------------------------------------------------------------- PromptElement


class _Answering(dp.PromptElement):
    _prompt = "P"
    _abstract_ex = "A"
    _concrete_ex = "C"

    def _parse_answer(self, text_answer):
        return {"echo": text_answer}


class TestPromptElement:
    def test_visible_exposes_all_parts(self):
        el = _Answering()
        assert (el.prompt, el.abstract_ex, el.concrete_ex) == ("P", "A", "C")
        assert el.parse_answer("x") == {"echo": "x"}

    def test_hidden_returns_empty_everything(self):
        el = _Answering(visible=False)
        assert (el.prompt, el.abstract_ex, el.concrete_ex) == ("", "", "")
        assert el.parse_answer("x") == {}

    def test_callable_visibility_is_reevaluated(self):
        state = {"on": False}
        el = _Answering(visible=lambda: state["on"])
        assert el.prompt == ""
        state["on"] = True
        assert el.prompt == "P"

    def test_base_parse_answer_returns_empty_dict(self):
        assert dp.PromptElement().parse_answer("<action>x</action>") == {}


# --------------------------------------------------------------------------- Trunkater / fit_tokens


class _Lines(dp.Trunkater):
    def __init__(self, n_lines, visible=True, start=0, speed=0.5):
        super().__init__(visible=visible, shrink_speed=speed, start_trunkate_iteration=start)
        self._prompt = "\n".join(f"line {i}" for i in range(n_lines))


class TestTrunkater:
    def test_no_truncation_before_start_iteration(self):
        el = _Lines(10, start=2)
        original = el.prompt
        el.shrink()
        el.shrink()
        assert el.prompt == original
        assert el.shrink_calls == 2
        el.shrink()
        assert el.prompt != original

    def test_truncation_removes_fraction_and_notes_deleted_lines(self):
        el = _Lines(10, start=0, speed=0.5)
        el.shrink()
        lines = el.prompt.splitlines()
        assert lines[:5] == [f"line {i}" for i in range(5)]
        assert lines[-1] == "... Deleted 5 lines to reduce prompt size."
        assert el.deleted_lines == 5
        el.shrink()
        assert el.deleted_lines == 5 + (6 - 3)
        assert el.prompt.endswith(f"Deleted {el.deleted_lines} lines to reduce prompt size.")

    def test_hidden_trunkater_does_not_truncate_but_counts_calls(self):
        el = _Lines(10, visible=False, start=0)
        raw = el._prompt
        el.shrink()
        assert el._prompt == raw
        assert el.shrink_calls == 1


class _Shrinker(dp.Shrinkable):
    def __init__(self, prompts):
        super().__init__()
        self.prompts = list(prompts)
        self.shrinks = 0

    @property
    def _prompt(self):
        return self.prompts[min(self.shrinks, len(self.prompts) - 1)]

    def shrink(self):
        self.shrinks += 1


class TestFitTokens:
    def test_no_max_returns_prompt_without_counting(self, monkeypatch):
        monkeypatch.setattr(
            dp, "count_tokens", mock.Mock(side_effect=AssertionError("should not count"))
        )
        el = _Shrinker(["a b c"])
        assert dp.fit_tokens(el, max_prompt_tokens=None) == "a b c"
        assert el.shrinks == 0

    def test_shrinks_until_fits(self, fake_tokens):
        el = _Shrinker(["w " * 50, "w " * 20, "w " * 5])
        out = dp.fit_tokens(el, max_prompt_tokens=10, additional_prompts=[])
        assert out == "w " * 5
        assert el.shrinks == 2

    def test_additional_prompt_string_reduces_budget(self, fake_tokens):
        # budget 10 - (3 words + 1) = 6 -> "w "*5 fits, "w "*8 does not
        el = _Shrinker(["w " * 8, "w " * 5])
        out = dp.fit_tokens(el, max_prompt_tokens=10, additional_prompts="x y z")
        assert out == "w " * 5
        assert el.shrinks == 1

    def test_additional_prompt_list_reduces_budget(self, fake_tokens):
        el = _Shrinker(["w " * 8, "w " * 5])
        out = dp.fit_tokens(el, max_prompt_tokens=10, additional_prompts=["x", "y z"])
        # budget 10 - 2 - 3 = 5
        assert out == "w " * 5

    def test_returns_last_prompt_and_logs_after_max_iterations(self, fake_tokens, caplog):
        el = _Shrinker(["w " * 100])
        with caplog.at_level(logging.INFO):
            out = dp.fit_tokens(el, max_prompt_tokens=3, max_iterations=4, additional_prompts=[])
        assert out == "w " * 100
        assert el.shrinks == 4
        assert "After 4 shrink iterations" in caplog.text

    def test_list_prompt_is_deprecated_but_counted(self, fake_tokens):
        prompt = [
            {"type": "text", "text": "one two"},
            {"type": "image_url", "image_url": "data:xxx"},
        ]
        el = _Shrinker([prompt])
        with pytest.warns(DeprecationWarning, match="list of prompts is deprecated"):
            out = dp.fit_tokens(el, max_prompt_tokens=5, additional_prompts=[])
        assert out is prompt
        assert el.shrinks == 0

    def test_base_message_prompt_is_counted_without_images(self, fake_tokens):
        msg = HumanMessage("one two three")
        msg.add_image("data:image/png;base64,AAAA")
        el = _Shrinker([msg])
        assert dp.fit_tokens(el, max_prompt_tokens=4, additional_prompts=[]) is msg

    def test_unknown_prompt_type_raises(self, fake_tokens):
        el = _Shrinker([12345])
        with pytest.raises(ValueError, match="Unrecognized type for prompt"):
            dp.fit_tokens(el, max_prompt_tokens=10, additional_prompts=[])


# --------------------------------------------------------------------------- Observation parts


class TestHTMLAndAXTree:
    def test_html_with_and_without_visible_note(self):
        with_note = dp.HTML("<b/>", visible_elements_only=True, prefix="## ")
        assert with_note.prompt.startswith("\n## HTML:\nNote: only elements that are visible")
        assert with_note.prompt.endswith("<b/>\n")
        assert "Note:" not in dp.HTML("<b/>", visible_elements_only=False).prompt
        assert with_note.start_trunkate_iteration == 5

    @pytest.mark.parametrize(
        "coord_type, expected, absent",
        [
            ("center", "center coordinates are provided", "bounding box"),
            ("box", "bounding box of each object", "center coordinates"),
            (None, None, "coordinates are provided"),
            ("False", None, "bounding box"),
        ],
    )
    def test_axtree_coord_notes(self, coord_type, expected, absent):
        prompt = dp.AXTree("[1] x", visible_elements_only=False, coord_type=coord_type).prompt
        if expected:
            assert expected in prompt
        assert absent not in prompt
        assert "[bid] is the unique alpha-numeric identifier" in prompt

    def test_axtree_visible_tag_and_visible_only_notes(self):
        p = dp.AXTree("[1] x", visible_elements_only=True, visible_tag=True, prefix="## ").prompt
        assert p.startswith("\n## AXTree:")
        assert "only elements that are visible in the viewport" in p
        assert 'If the "visible" tag is not' in p
        p2 = dp.AXTree("[1] x", visible_elements_only=False, visible_tag=False).prompt
        assert "visible in the viewport" not in p2
        assert '"visible" tag' not in p2


class TestError:
    def test_plain_error(self):
        assert dp.Error("boom", prefix="## ").prompt == "\n## Error from previous action:\nboom\n"

    def test_call_log_truncated_to_10_lines(self):
        logs = "\n".join(f"log{i}" for i in range(30))
        prompt = dp.Error(f"Timeout\nCall log:{logs}").prompt
        assert "Call log:" in prompt
        assert "log9" in prompt
        assert "log10" not in prompt and "log29" not in prompt

    def test_call_log_kept_when_limit_disabled(self):
        logs = "\n".join(f"log{i}" for i in range(30))
        prompt = dp.Error(f"Timeout\nCall log:{logs}", limit_logs=False).prompt
        assert "log29" in prompt


class TestFocusedElementAndTabs:
    @pytest.mark.parametrize("bid, expected", [("a12", "bid='a12'"), ("", "None"), (None, "None")])
    def test_focused_element(self, bid, expected):
        prompt = dp.FocusedElement(bid, prefix="## ").prompt
        assert prompt.startswith("\n## Focused element:\n")
        assert prompt.strip().endswith(expected)

    def test_tabs_marks_active_tab(self):
        prompt = dp.Tabs(_obs(), prefix="## ").prompt
        assert "## Currently open tabs:" in prompt
        assert "Tab 0:\n    Title: A\n    URL: https://a.example" in prompt
        assert "Tab 1 (active tab):\n    Title: B" in prompt

    def test_hidden_tabs_do_not_touch_obs(self):
        assert dp.Tabs({}, visible=False).prompt == ""


class TestObservation:
    def test_prompt_respects_flags(self):
        flags = dp.ObsFlags(
            use_html=False,
            use_ax_tree=True,
            use_tabs=True,
            use_focused_element=True,
            use_error_logs=True,
        )
        prompt = dp.Observation(_obs(last_action_error="kaboom"), flags).prompt
        assert prompt.startswith("\n# Observation of current step:")
        assert "## AXTree:" in prompt
        assert "## HTML:" not in prompt
        assert "Currently open tabs" in prompt
        assert "bid='12'" in prompt
        assert "kaboom" in prompt

    def test_error_hidden_when_no_error(self):
        flags = dp.ObsFlags(use_error_logs=True)
        assert "Error from previous action" not in dp.Observation(_obs(), flags).prompt

    def test_shrink_truncates_html_after_its_start_iteration(self):
        html = "\n".join(f"<p>{i}</p>" for i in range(100))
        obs = dp.Observation(_obs(pruned_html=html), dp.ObsFlags(use_html=True))
        before = obs.prompt
        for _ in range(5):
            obs.shrink()
        assert obs.prompt == before
        obs.shrink()
        assert "Deleted" in obs.prompt and len(obs.prompt) < len(before)
        assert obs.ax_tree.shrink_calls == 6

    @pytest.mark.parametrize(
        "use_som, text, key",
        [
            (False, "screenshot of the page:", "screenshot"),
            (True, "bounding boxes", "screenshot_som"),
        ],
    )
    def test_add_screenshot(self, use_som, text, key, monkeypatch):
        seen = {}

        def fake_b64(img):
            seen["img"] = img
            return "data:image/jpeg;base64,XYZ"

        monkeypatch.setattr(dp, "image_to_jpg_base64_url", fake_b64)
        obs_dict = _obs()
        flags = dp.ObsFlags(use_screenshot=True, use_som=use_som, openai_vision_detail="low")
        msg = dp.Observation(obs_dict, flags).add_screenshot(HumanMessage("hi"))
        assert seen["img"] is obs_dict[key]
        assert text in msg["content"][1]["text"]
        assert msg["content"][2] == {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,XYZ", "detail": "low"},
        }

    def test_add_screenshot_disabled_is_noop(self):
        msg = HumanMessage("hi")
        out = dp.Observation(_obs(), dp.ObsFlags(use_screenshot=False)).add_screenshot(msg)
        assert out is msg and out["content"] == "hi"

    def test_add_screenshot_real_encoding(self):
        flags = dp.ObsFlags(use_screenshot=True, openai_vision_detail="auto")
        msg = dp.Observation(_obs(), flags).add_screenshot(HumanMessage("hi"))
        assert msg["content"][-1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


# --------------------------------------------------------------------------- Misc elements


class TestMiscElements:
    @pytest.mark.parametrize("system, visible", [("Darwin", True), ("Linux", False)])
    def test_mac_note(self, monkeypatch, system, visible):
        monkeypatch.setattr(dp.platform, "system", lambda: system)
        assert bool(dp.MacNote().prompt) is visible

    def test_be_cautious(self):
        assert "Be very cautious" in dp.BeCautious().prompt
        assert dp.BeCautious(visible=False).prompt == ""

    def test_goal_instructions_with_extra(self):
        goal = [{"type": "text", "text": "Buy milk"}]
        el = dp.GoalInstructions(goal, extra_instructions="Be fast")
        prompt = el.prompt
        assert prompt[0]["text"].startswith("# Instructions")
        assert prompt[1] == {"type": "text", "text": "Buy milk"}
        assert "## Extra instructions:" in prompt[2]["text"]
        assert "Be fast" in prompt[2]["text"]
        assert len(dp.GoalInstructions(goal).prompt) == 2

    def test_chat_instructions_lists_messages(self, monkeypatch):
        monkeypatch.setenv("TZ", "UTC")
        msgs = [
            {"role": "user", "timestamp": 0, "message": "hello"},
            {"role": "assistant", "timestamp": 60, "message": "hi there"},
        ]
        prompt = dp.ChatInstructions(msgs, extra_instructions="be nice").prompt
        assert "## Chat messages:" in prompt
        assert "- [user] UTC Time: Thu Jan  1 00:00:00 1970" in prompt
        assert "- [assistant] UTC Time: Thu Jan  1 00:01:00 1970" in prompt
        assert prompt.rstrip().endswith("be nice")
        assert "Extra instructions" not in dp.ChatInstructions(msgs).prompt

    def test_static_prompts(self):
        assert "bid to identify elements" in dp.Hints().prompt
        assert dp.SystemPrompt().prompt.startswith("You are an agent")


# --------------------------------------------------------------------------- ActionPrompt


class TestActionPrompt:
    def test_prompt_and_examples(self, monkeypatch):
        monkeypatch.setattr(dp.platform, "system", lambda: "Linux")
        ap = dp.ActionPrompt(_action_set(), dp.ActionFlags(long_description=False))
        assert ap.prompt.startswith("# Action space:\nNote: This action set allows")
        assert "click(bid" in ap.prompt
        assert "Description:" not in ap.prompt
        assert "One single action" in ap.abstract_ex
        assert "click('a324')" in ap.concrete_ex

    def test_parse_valid_action(self):
        ap = dp.ActionPrompt(_action_set(), dp.ActionFlags())
        assert ap.parse_answer("<action>click('12')</action>") == {"action": "click('12')"}

    def test_parse_none_action_maps_to_python_none(self):
        ap = dp.ActionPrompt(_action_set(), dp.ActionFlags())
        assert ap.parse_answer("<action>None</action>") == {"action": None}

    def test_missing_tag_strict_raises(self):
        ap = dp.ActionPrompt(_action_set(), dp.ActionFlags(is_strict=True))
        with pytest.raises(ParseError, match="Missing the key <action>"):
            ap.parse_answer("```\nclick('12')\n```")

    def test_missing_tag_non_strict_falls_back_to_code_blocks(self):
        ap = dp.ActionPrompt(_action_set(multiaction=True), dp.ActionFlags(is_strict=False))
        ans = ap.parse_answer("```python\nclick('1')\n```\ntext\n```\nclick('2')\n```")
        assert ans["action"] == "click('1')\nclick('2')"
        assert "Missing the key <action>" in ans["parse_error"]

    def test_missing_tag_non_strict_without_code_blocks_raises(self):
        ap = dp.ActionPrompt(_action_set(), dp.ActionFlags(is_strict=False))
        with pytest.raises(ParseError, match="Missing the key <action>"):
            ap.parse_answer("I would click the button")

    @pytest.mark.parametrize("bad", ["", "click('1')\nclick('2')"])
    def test_unmappable_action_raises_parse_error(self, bad):
        ap = dp.ActionPrompt(_action_set(multiaction=False), dp.ActionFlags())
        with pytest.raises(ParseError, match="Error while parsing action"):
            ap.parse_answer(f"<action>{bad}</action>")


class TestThink:
    def test_parses_think(self):
        assert dp.Think().parse_answer("<think>hmm</think><action>x</action>") == {"think": "hmm"}

    def test_missing_tag_returns_whole_text_with_error(self):
        ans = dp.Think().parse_answer("just text")
        assert ans["think"] == "just text"
        assert "Missing the key <think>" in ans["parse_error"]

    def test_examples(self):
        assert "<think>" in dp.Think().abstract_ex and "<think>" in dp.Think().concrete_ex
        assert dp.Think(visible=False).parse_answer("<think>x</think>") == {}


# --------------------------------------------------------------------------- History


class TestHistory:
    def _history(self, flags, memories=("m0", "m1")):
        obs = [
            _obs(last_action_error=""),
            _obs(last_action_error="err at 1"),
            _obs(last_action_error=""),
        ]
        return dp.History(
            obs,
            ["click('1')", "click('2')"],
            list(memories) if memories is not None else None,
            ["t0", "t1"],
            flags,
        )

    def test_full_history_prompt(self):
        flags = dp.ObsFlags(
            use_history=True,
            use_think_history=True,
            use_action_history=True,
            use_error_logs=True,
            use_past_error_logs=True,
        )
        prompt = self._history(flags).prompt
        assert prompt.startswith("# History of interaction with the task:\n")
        assert "## step 0" in prompt and "## step 1" in prompt
        assert "<think>\nt0\n</think>" in prompt
        assert "<action>\nclick('2')\n</action>" in prompt
        assert "<memory>\nm1\n</memory>" in prompt
        assert prompt.count("### Error from previous action") == 1
        assert "err at 1" in prompt

    def test_history_hidden_and_sparse_flags(self):
        assert self._history(dp.ObsFlags(use_history=False)).prompt == ""
        prompt = self._history(dp.ObsFlags(use_history=True), memories=None).prompt
        assert "<think>" not in prompt and "<action>" not in prompt
        assert "<memory>" not in prompt and "Error" not in prompt

    def test_past_errors_hidden_without_past_error_flag(self):
        flags = dp.ObsFlags(use_history=True, use_error_logs=True, use_past_error_logs=False)
        assert "err at 1" not in self._history(flags).prompt

    def test_length_mismatch_asserts(self):
        with pytest.raises(AssertionError):
            dp.History([_obs()], ["click('1')"], None, ["t"], dp.ObsFlags())

    def test_shrink_propagates_to_steps(self):
        h = self._history(dp.ObsFlags(use_history=True))
        with mock.patch.object(dp.HistoryStep, "shrink", autospec=True) as step_shrink:
            h.shrink()
        assert step_shrink.call_count == 2
        # real shrink on HistoryStep is a no-op that must not raise
        h.history_steps[0].shrink()


# --------------------------------------------------------------------------- obs preprocessor


def test_make_obs_preprocessor_passes_flags_and_builds_fields(monkeypatch):
    calls = {}

    def fake_flatten(name):
        def _f(obj, **kwargs):
            calls[name] = (obj, kwargs)
            return f"{name}:{obj}"

        return _f

    monkeypatch.setattr(dp, "flatten_dom_to_str", fake_flatten("dom"))
    monkeypatch.setattr(dp, "flatten_axtree_to_str", fake_flatten("ax"))
    monkeypatch.setattr(dp, "prune_html", lambda s: f"pruned({s})")
    monkeypatch.setattr(
        dp, "overlay_som", lambda img, extra_properties: ("som", img, extra_properties)
    )

    flags = dp.ObsFlags(
        extract_visible_tag=True,
        extract_clickable_tag=False,
        extract_coords="center",
        filter_visible_elements_only=True,
        filter_with_bid_only=True,
        filter_som_only=False,
    )
    raw = {
        "dom_object": "D",
        "axtree_object": "A",
        "extra_element_properties": {"x": 1},
        "screenshot": "IMG",
    }
    out = dp.make_obs_preprocessor(flags)(raw)

    assert "dom_txt" not in raw  # input is not mutated
    assert out["dom_txt"] == "dom:D"
    assert out["axtree_txt"] == "ax:A"
    assert out["pruned_html"] == "pruned(dom:D)"
    assert out["screenshot_som"] == ("som", "IMG", {"x": 1})
    for name in ("dom", "ax"):
        kwargs = calls[name][1]
        assert kwargs == dict(
            extra_properties={"x": 1},
            with_visible=True,
            with_clickable=False,
            with_center_coords=True,
            with_bounding_box_coords=False,
            filter_visible_only=True,
            filter_with_bid_only=True,
            filter_som_only=False,
        )
