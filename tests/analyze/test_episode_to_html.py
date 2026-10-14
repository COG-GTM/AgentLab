import base64
import re
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from agentlab.analyze import episode_to_html as e2h
from agentlab.experiments.loop import ExpResult, StepInfo
from agentlab.llm.llm_utils import BaseMessage
from agentlab.llm.response_api import MessageBuilder

XSS = "<script>alert('x')</script>"


def _img_array(color, size=(8, 6)):
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[:, :] = color
    return arr


def _make_exp_result(tmp_path, steps, task_name="miniwob.click-test", seed=7, som=True):
    """Write StepInfo pickles + screenshots to tmp_path and return an ExpResult on it."""
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    for i, (obs, action, agent_info) in enumerate(steps):
        StepInfo(step=i, obs=obs, action=action, agent_info=agent_info).save_step_info(
            exp_dir, save_som=som
        )
    exp_result = ExpResult(exp_dir)
    exp_result._exp_args = SimpleNamespace(
        env_args=SimpleNamespace(task_name=task_name, task_seed=seed)
    )
    return exp_result


def _obs(i, goal=None, with_screens=True, **extra):
    obs = dict(extra)
    if goal is not None:
        obs["goal_object"] = goal
    if with_screens:
        obs["screenshot"] = _img_array((10 * i, 0, 0))
        obs["screenshot_som"] = _img_array((0, 10 * i, 0))
    return obs


@pytest.fixture
def full_exp(tmp_path):
    goal = [{"type": "text", "text": f"Click the button {XSS}"}]
    steps = [
        (_obs(0, goal=goal), None, {}),
        (
            _obs(1, axtree_txt="[a1] button 'OK' <b>", dom_txt="<div id='x'>hi</div>"),
            "click('a1')",
            {
                "think": "I should <click> & go",
                "chat_messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": [{"type": "text", "text": XSS}]},
                ],
            },
        ),
        (_obs(2), "noop()", {}),
    ]
    return _make_exp_result(tmp_path, steps)


class TestExpResultToHtml:
    def test_full_episode_embedded(self, full_exp):
        html = e2h.exp_result_to_html(full_exp)

        assert '<div class="episode-title">miniwob.click-test</div>' in html
        assert '<div class="episode-meta">Seed: 7</div>' in html
        assert "Click the button &lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;" in html
        assert XSS not in html

        # reset step with no action skipped, others rendered
        assert "<summary>Step 0</summary>" not in html
        assert "<summary>Step 1</summary>" in html
        assert "<summary>Step 2</summary>" in html
        assert "click(&#x27;a1&#x27;)" in html
        assert "I should &lt;click&gt; &amp; go" in html

        # screenshots embedded for step 1 and 2 (main + som each)
        assert html.count('src="data:image/png;base64,') == 4
        assert "file://" not in html
        assert "<summary>Screenshot_som[1]</summary>" in html

        assert "<summary>AXTree</summary>" in html
        assert "[a1] button &#x27;OK&#x27; &lt;b&gt;" in html
        assert "<summary>HTML</summary>" in html
        assert "&lt;div id=&#x27;x&#x27;&gt;hi&lt;/div&gt;" in html
        assert "<summary>Prompt</summary>" in html
        assert '<div class="chat-role">SYSTEM</div>' in html
        assert '<div class="chat-role">USER</div>' in html

        # step 2 has no think/axtree/prompt -> sections only appear once
        assert html.count("<summary>AXTree</summary>") == 1
        assert html.count("<strong>Think:</strong>") == 1
        assert html.count("<summary>Prompt</summary>") == 1

    def test_embedded_image_roundtrips(self, full_exp):
        html = e2h.exp_result_to_html(full_exp)
        b64 = (
            re.search(r'alt="Screenshot 1"', html)
            and re.findall(r'src="data:image/png;base64,([^"]+)" alt="Screenshot 1"', html)[0]
        )
        img = Image.open(BytesIO(base64.b64decode(b64)))
        assert img.size == (8, 6)
        assert img.convert("RGB").getpixel((0, 0)) == (10, 0, 0)

    def test_paths_instead_of_embedding(self, full_exp):
        html = e2h.exp_result_to_html(full_exp, embed_images=False)
        assert "data:image/png" not in html
        main = full_exp.exp_dir.resolve() / "screenshot_step_1.png"
        som = full_exp.exp_dir.resolve() / "screenshot_som_step_1.png"
        assert f'src="file://{main}" alt="Screenshot 1"' in html
        assert f'src="file://{som}" alt="SOM Screenshot 1"' in html

    @pytest.mark.parametrize(
        "kwargs, n_open_nested, step_open",
        [
            ({}, 0, True),
            ({"steps_open": False}, 0, False),
            ({"som_open": True}, 2, True),
            ({"axtree_open": True, "html_open": True, "prompt_open": True}, 3, True),
            (
                {"som_open": True, "axtree_open": True, "html_open": True, "prompt_open": True},
                5,
                True,
            ),
        ],
    )
    def test_open_flags(self, full_exp, kwargs, n_open_nested, step_open):
        html = e2h.exp_result_to_html(full_exp, **kwargs)
        n_open = len(re.findall(r'<details class="nested-details" open>', html))
        assert n_open == n_open_nested
        assert html.count("<details open>") == (2 if step_open else 0)
        assert html.count("<details >") == (0 if step_open else 2)

    def test_missing_screenshots(self, tmp_path):
        steps = [
            (_obs(0, goal="g", with_screens=False), None, {}),
            (_obs(1, with_screens=False), "noop()", {}),
        ]
        exp = _make_exp_result(tmp_path, steps, som=False)
        for embed in (True, False):
            html = e2h.exp_result_to_html(exp, embed_images=embed)
            assert "<p>Screenshot not available</p>" in html
            assert "Screenshot_som" not in html

    def test_first_step_with_action_is_rendered(self, tmp_path):
        exp = _make_exp_result(tmp_path, [(_obs(0, goal="g"), "noop()", {})])
        html = e2h.exp_result_to_html(exp)
        assert "<summary>Step 0</summary>" in html
        assert "noop()" in html

    def test_no_steps(self):
        exp = MagicMock(spec=ExpResult)
        exp.exp_args.env_args = SimpleNamespace(task_name="t", task_seed=1)
        exp.steps_info = []
        html = e2h.exp_result_to_html(exp)
        assert "goal-section" not in html.split("</style>")[1]
        assert "<details" not in html
        assert html.rstrip().endswith("</div>")

    def test_goal_lookup_failure_is_tolerated(self):
        exp = MagicMock(spec=ExpResult)
        exp.exp_args.env_args = SimpleNamespace(task_name="t", task_seed=1)
        exp.steps_info = [StepInfo(step=0, obs=None, action=None)]
        html = e2h.exp_result_to_html(exp)
        assert '<div class="goal-section">' not in html

    def test_missing_goal_key_uses_default(self):
        exp = MagicMock(spec=ExpResult)
        exp.exp_args.env_args = SimpleNamespace(task_name="t", task_seed=1)
        exp.steps_info = [StepInfo(step=0, obs={}, action=None)]
        html = e2h.exp_result_to_html(exp)
        assert "<div class='code-block'>No goal specified</div>" in html

    @pytest.mark.xfail(strict=True, reason="BUG: task_name is interpolated into HTML unescaped")
    def test_task_name_is_escaped(self):
        exp = MagicMock(spec=ExpResult)
        exp.exp_args.env_args = SimpleNamespace(task_name=XSS, task_seed=1)
        exp.steps_info = []
        assert XSS not in e2h.exp_result_to_html(exp)


class TestScreenshotHelpers:
    def test_screenshot_index_error(self):
        exp = MagicMock()
        exp.get_screenshot.side_effect = IndexError
        exp.get_screenshot_path.side_effect = IndexError
        assert e2h._get_screenshot_html(exp, 3, True) == "<p>Screenshot not available</p>"
        assert e2h._get_screenshot_html(exp, 3, False) == "<p>Screenshot not available</p>"
        assert e2h._get_som_screenshot_html(exp, 3, True, True) == ""
        assert e2h._get_som_screenshot_html(exp, 3, True, False) == ""

    def test_som_none_image_returns_empty(self):
        exp = MagicMock()
        exp.get_screenshot.return_value = None
        assert e2h._get_som_screenshot_html(exp, 0, False, True) == ""

    def test_som_nonexistent_path_returns_empty(self, tmp_path):
        exp = MagicMock()
        exp.get_screenshot_path.return_value = tmp_path / "nope.png"
        assert e2h._get_som_screenshot_html(exp, 0, False, False) == ""
        exp.get_screenshot_path.return_value = None
        assert e2h._get_som_screenshot_html(exp, 0, False, False) == ""

    def test_som_embedded_open(self):
        exp = MagicMock()
        exp.get_screenshot.return_value = Image.new("RGB", (2, 2))
        html = e2h._get_som_screenshot_html(exp, 4, True, True)
        exp.get_screenshot.assert_called_once_with(4, som=True)
        assert '<details class="nested-details" open>' in html
        assert "<summary>Screenshot_som[4]</summary>" in html
        assert 'alt="SOM Screenshot 4"' in html

    def test_image_to_html(self):
        assert e2h._image_to_html(None, "Pic") == "<p>Pic not available</p>"
        img = Image.new("RGB", (3, 2), (1, 2, 3))
        out = e2h._image_to_html(img, "Pic")
        m = re.fullmatch(
            r'<img src="data:image/png;base64,(.+)" alt="Pic" class="screenshot">', out
        )
        assert m
        decoded = Image.open(BytesIO(base64.b64decode(m.group(1))))
        assert decoded.size == (3, 2) and decoded.getpixel((0, 0)) == (1, 2, 3)

    def test_path_to_html(self, tmp_path, monkeypatch):
        assert e2h._path_to_html(None, "P") == "<p>P not available</p>"
        assert e2h._path_to_html(tmp_path / "missing.png", "P") == "<p>P not available</p>"
        f = tmp_path / "a.png"
        f.write_bytes(b"x")
        monkeypatch.chdir(tmp_path)
        out = e2h._path_to_html(Path("a.png"), "P")
        assert out == f'<img src="file://{f.resolve()}" alt="P" class="screenshot">'


class TestFormatGoal:
    def test_none(self):
        assert e2h._format_goal(None) == "<div class='code-block'>No goal specified</div>"

    def test_string_escaped(self):
        assert e2h._format_goal("a < b") == "<div class='code-block'>a &lt; b</div>"

    def test_multipart_keeps_only_text(self):
        goal = [
            {"type": "text", "text": "line1"},
            {"type": "image_url", "image_url": {"url": "data:..."}},
            {"type": "text", "text": "line2"},
        ]
        assert e2h._format_goal(goal) == "<div class='code-block'>line1\nline2</div>"


class TestFormatChatMessages:
    def test_empty(self):
        assert e2h._format_chat_messages_like_xray([]) == (
            "<div class='code-block'>No chat messages</div>"
        )
        assert "No chat messages" in e2h._format_chat_messages_like_xray(None)

    def test_object_with_role_and_content(self):
        msg = SimpleNamespace(role="assistant", content="<hi>")
        out = e2h._format_chat_messages_like_xray([msg])
        assert out == (
            '<div class="chat-message"><div class="chat-role">ASSISTANT</div>'
            '<div class="code-block">&lt;hi&gt;</div></div>'
        )

    def test_message_builder(self):
        mb = MessageBuilder.user().add_text("hello")
        out = e2h._format_chat_messages_like_xray([mb])
        assert '<div class="chat-role">USER</div>' in out
        assert "hello" in out

    def test_agentlab_base_message_is_dict(self):
        out = e2h._format_chat_messages_like_xray([BaseMessage("user", "plain")])
        assert out == (
            '<div class="chat-message"><div class="chat-role">USER</div>'
            '<div class="code-block">plain</div></div>'
        )

    def test_dict_multipart(self):
        msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "a&b"},
                {"type": "image", "image": "data:png"},
                {"type": "tool_use", "name": "click", "input": {"bid": "a1"}, "call_id": "c1"},
                {"type": "other", "x": 1},
                "raw <part>",
            ],
        }
        out = e2h._format_chat_messages_like_xray([msg])
        blocks = re.findall(r'<div class="code-block">(.*?)</div>', out)
        assert blocks == [
            "a&amp;b",
            "[IMAGE]",
            "Tool Call: click `{&#x27;bid&#x27;: &#x27;a1&#x27;}` (call_id: c1)",
            "{&#x27;type&#x27;: &#x27;other&#x27;, &#x27;x&#x27;: 1}",
            "raw &lt;part&gt;",
        ]

    def test_dict_defaults(self):
        out = e2h._format_chat_messages_like_xray([{"foo": 1}])
        assert '<div class="chat-role">UNKNOWN</div>' in out
        assert "{&#x27;foo&#x27;: 1}" in out

    def test_other_types(self):
        out = e2h._format_chat_messages_like_xray(["<str msg>", 42])
        assert out.count('<div class="chat-message">') == 2
        assert "&lt;str msg&gt;" in out and ">42<" in out

    @pytest.mark.xfail(strict=True, reason="BUG: chat message role is not HTML-escaped")
    def test_role_is_escaped(self):
        out = e2h._format_chat_messages_like_xray([{"role": XSS, "content": "x"}])
        assert "<SCRIPT>" not in out


def test_format_tool_call_defaults():
    assert e2h._format_tool_call_like_xray({}) == "Tool Call: unknown `{}` (call_id: unknown)"
    assert (
        e2h._format_tool_call_like_xray({"name": "f", "input": {"a": 1}, "call_id": "z"})
        == "Tool Call: f `{'a': 1}` (call_id: z)"
    )


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", ""),
        ("plain", "plain"),
        ("<a href=\"x\">'&'</a>", "&lt;a href=&quot;x&quot;&gt;&#x27;&amp;&#x27;&lt;/a&gt;"),
        ("&lt;", "&amp;lt;"),
        (123, "123"),
        (None, "None"),
    ],
)
def test_escape_html(text, expected):
    assert e2h._escape_html(text) == expected
