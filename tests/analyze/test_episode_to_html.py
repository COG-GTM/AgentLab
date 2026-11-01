from types import SimpleNamespace

import pytest
from PIL import Image

from agentlab.analyze.episode_to_html import (
    _escape_html,
    _format_chat_messages_like_xray,
    _format_goal,
    _format_tool_call_like_xray,
    _image_to_html,
    _path_to_html,
    exp_result_to_html,
)
from agentlab.llm.llm_utils import HumanMessage


def _step(action, obs=None, agent_info=None):
    return SimpleNamespace(action=action, obs=obs or {}, agent_info=agent_info or {})


class FakeExpResult:
    def __init__(self, steps_info, screenshot_dir=None, som=True):
        self.exp_args = SimpleNamespace(
            env_args=SimpleNamespace(task_name="miniwob.click", task_seed=7)
        )
        self.steps_info = steps_info
        self.screenshot_dir = screenshot_dir
        self.som = som

    def get_screenshot(self, step, som=False):
        if som and not self.som:
            return None
        if step >= len(self.steps_info):
            raise IndexError
        return Image.new("RGB", (4, 4), (0, 0, 255) if som else (255, 0, 0))

    def get_screenshot_path(self, step, som=False):
        if self.screenshot_dir is None:
            raise FileNotFoundError
        return self.screenshot_dir / f"screenshot{'_som' if som else ''}_step_{step}.png"


@pytest.fixture
def steps():
    return [
        _step(None, obs={"goal_object": [{"type": "text", "text": "Click <OK>"}]}),
        _step(
            "click('a<1>')",
            obs={"axtree_txt": "[a1] button 'OK'", "dom_txt": "<button>OK</button>"},
            agent_info={
                "think": "I'll click & go",
                "chat_messages": [{"role": "user", "content": "hi"}],
            },
        ),
        _step(None),
    ]


def test_escape_html():
    assert _escape_html("<a href=\"x\">'&'</a>") == (
        "&lt;a href=&quot;x&quot;&gt;&#x27;&amp;&#x27;&lt;/a&gt;"
    )
    assert _escape_html(42) == "42"
    assert _escape_html("") == ""


def test_format_tool_call_like_xray():
    assert (
        _format_tool_call_like_xray({"name": "click", "input": {"bid": "a1"}, "call_id": "c1"})
        == "Tool Call: click `{'bid': 'a1'}` (call_id: c1)"
    )
    assert _format_tool_call_like_xray({}) == "Tool Call: unknown `{}` (call_id: unknown)"


def test_image_to_html():
    assert _image_to_html(None, "Shot") == "<p>Shot not available</p>"
    html = _image_to_html(Image.new("RGB", (2, 2)), "Shot")
    assert html.startswith('<img src="data:image/png;base64,')
    assert 'alt="Shot"' in html


def test_path_to_html(tmp_path):
    assert _path_to_html(None, "Shot") == "<p>Shot not available</p>"
    assert _path_to_html(tmp_path / "missing.png", "Shot") == "<p>Shot not available</p>"
    img = tmp_path / "img.png"
    Image.new("RGB", (2, 2)).save(img)
    assert (
        _path_to_html(img, "Shot")
        == f'<img src="file://{img.resolve()}" alt="Shot" class="screenshot">'
    )


def test_format_goal():
    assert "No goal specified" in _format_goal(None)
    assert _format_goal("Do <it>") == "<div class='code-block'>Do &lt;it&gt;</div>"
    html = _format_goal([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
    assert html == "<div class='code-block'>a\nb</div>"


def test_format_chat_messages_empty():
    assert _format_chat_messages_like_xray([]) == "<div class='code-block'>No chat messages</div>"
    assert "No chat messages" in _format_chat_messages_like_xray(None)


def test_format_chat_messages_variants():
    obj_msg = SimpleNamespace(role="assistant", content="<done>")
    messages = [
        {"role": "system", "content": "be <brief>"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image", "source": "..."},
                {"type": "tool_use", "name": "click", "input": {"bid": "1"}, "call_id": "x"},
                {"type": "other", "value": 1},
                "raw part",
            ],
        },
        {"content": "no role"},
        obj_msg,
        HumanMessage("dict subclass"),
        12345,
    ]
    html = _format_chat_messages_like_xray(messages)
    assert html.count('<div class="chat-message">') == len(messages)
    assert '<div class="chat-role">SYSTEM</div>' in html
    assert "be &lt;brief&gt;" in html
    assert "[IMAGE]" in html
    assert "Tool Call: click" in html
    assert "raw part" in html
    assert "&#x27;type&#x27;: &#x27;other&#x27;" in html
    assert '<div class="chat-role">UNKNOWN</div>' in html
    assert '<div class="chat-role">ASSISTANT</div>' in html and "&lt;done&gt;" in html
    assert '<div class="chat-role">USER</div>' in html and "dict subclass" in html
    assert "12345" in html


def test_exp_result_to_html_embedded(steps):
    html = exp_result_to_html(FakeExpResult(steps), som_open=True, axtree_open=True)
    assert '<div class="episode-title">miniwob.click</div>' in html
    assert "Seed: 7" in html
    assert "Click &lt;OK&gt;" in html
    # step 0 has no action and is skipped; step 2 is rendered without action
    assert "<summary>Step 0</summary>" not in html
    assert "<summary>Step 1</summary>" in html and "<summary>Step 2</summary>" in html
    assert "click(&#x27;a&lt;1&gt;&#x27;)" in html
    assert "I&#x27;ll click &amp; go" in html
    assert "<summary>AXTree</summary>" in html
    assert "<summary>HTML</summary>" in html
    assert "<summary>Prompt</summary>" in html
    assert "<summary>Screenshot_som[1]</summary>" in html
    assert html.count("data:image/png;base64,") == 4  # main + som for steps 1 and 2
    assert html.count("<details open>") == 2
    assert html.endswith("</div>")


def test_exp_result_to_html_collapsed_without_som(steps):
    html = exp_result_to_html(FakeExpResult(steps, som=False), steps_open=False)
    assert "<details open>" not in html
    assert "<details >" in html
    assert "Screenshot_som" not in html


def test_exp_result_to_html_paths(tmp_path, steps):
    Image.new("RGB", (2, 2)).save(tmp_path / "screenshot_step_1.png")
    Image.new("RGB", (2, 2)).save(tmp_path / "screenshot_som_step_1.png")
    html = exp_result_to_html(FakeExpResult(steps, screenshot_dir=tmp_path), embed_images=False)
    assert f"file://{(tmp_path / 'screenshot_step_1.png').resolve()}" in html
    assert "<summary>Screenshot_som[1]</summary>" in html
    assert "Screenshot 2 not available" in html
    assert "Screenshot_som[2]" not in html


def test_exp_result_to_html_missing_screenshots(steps):
    html = exp_result_to_html(FakeExpResult(steps), embed_images=False)
    assert html.count("<p>Screenshot not available</p>") == 2


def test_exp_result_to_html_no_steps():
    html = exp_result_to_html(FakeExpResult([]))
    assert "Goal:" not in html
    assert "<summary>Step" not in html


def test_exp_result_to_html_goal_default():
    html = exp_result_to_html(FakeExpResult([_step("noop()")]))
    assert "No goal specified" in html
    assert "<summary>Step 0</summary>" in html
