from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from agentlab.analyze import episode_to_html as e2h


class _FakeExpResult:
    def __init__(self, steps_info, screenshots=None, paths=None, som=None):
        self.exp_args = SimpleNamespace(
            env_args=SimpleNamespace(task_name="miniwob.click", task_seed=7)
        )
        self.steps_info = steps_info
        self._screenshots = screenshots or {}
        self._paths = paths or {}
        self._som = som or {}

    def get_screenshot(self, step, som=False):
        store = self._som if som else self._screenshots
        if step not in store:
            raise FileNotFoundError
        return store[step]

    def get_screenshot_path(self, step, som=False):
        if step not in self._paths:
            raise FileNotFoundError
        return self._paths[step]


def _step(action=None, obs=None, agent_info=None):
    return SimpleNamespace(action=action, obs=obs or {}, agent_info=agent_info or {})


def test_exp_result_to_html_full_episode():
    img = Image.new("RGB", (4, 4))
    steps = [
        _step(obs={"goal_object": [{"type": "text", "text": "Click <b>"}]}),
        _step(
            action="click('a')",
            obs={"axtree_txt": "[1] button", "dom_txt": "<div>x</div>"},
            agent_info={
                "think": "I should click",
                "chat_messages": [{"role": "user", "content": "hi"}],
            },
        ),
    ]
    exp_result = _FakeExpResult(steps, screenshots={1: img}, som={1: img})

    html = e2h.exp_result_to_html(exp_result, steps_open=False, axtree_open=True)

    assert "miniwob.click" in html and "Seed: 7" in html
    assert "Click &lt;b&gt;" in html
    assert "Step 0" not in html  # initial reset step without action is skipped
    assert "<summary>Step 1</summary>" in html
    assert "<details >" in html  # steps closed
    assert "click(&#x27;a&#x27;)" in html
    assert "I should click" in html
    assert "Screenshot_som[1]" in html
    assert "AXTree" in html and "&lt;div&gt;x&lt;/div&gt;" in html
    assert "Prompt" in html and "USER" in html
    assert html.count("data:image/png;base64,") == 2


def test_exp_result_to_html_without_images_and_goal():
    steps = [_step(action="noop()")]
    html = e2h.exp_result_to_html(_FakeExpResult(steps))
    assert "Goal:" in html  # default "No goal specified"
    assert "Screenshot not available" in html
    assert "Screenshot_som" not in html


def test_exp_result_to_html_empty_episode():
    html = e2h.exp_result_to_html(_FakeExpResult([]))
    assert "Goal:" not in html
    assert "<summary>" not in html


def test_exp_result_to_html_with_file_paths(tmp_path):
    path = tmp_path / "screenshot_step_1.png"
    Image.new("RGB", (2, 2)).save(path)
    steps = [_step(action="noop()"), _step(action="noop()")]
    exp_result = _FakeExpResult(steps, paths={0: path, 1: tmp_path / "missing.png"})
    html = e2h.exp_result_to_html(exp_result, embed_images=False)
    assert f"file://{path.resolve()}" in html
    assert "Screenshot 1 not available" in html
    assert "Screenshot_som[0]" in html


def test_image_and_path_helpers_handle_none():
    assert e2h._image_to_html(None, "alt") == "<p>alt not available</p>"
    assert e2h._path_to_html(None, "alt") == "<p>alt not available</p>"
    assert e2h._path_to_html(Path("/does/not/exist.png"), "alt") == "<p>alt not available</p>"


def test_format_goal():
    assert "No goal specified" in e2h._format_goal(None)
    assert "buy milk" in e2h._format_goal("buy milk")


def test_format_chat_messages_like_xray():
    assert "No chat messages" in e2h._format_chat_messages_like_xray([])

    msg_obj = SimpleNamespace(role="system", content="be nice")
    messages = [
        msg_obj,
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "<hello>"},
                {"type": "image"},
                {"type": "tool_use", "name": "click", "input": {"bid": "1"}, "call_id": "c1"},
                {"type": "other", "x": 1},
                "raw part",
            ],
        },
        {"content": "no role"},
        42,
    ]
    html = e2h._format_chat_messages_like_xray(messages)
    assert html.count('class="chat-message"') == 4
    assert "SYSTEM" in html and "ASSISTANT" in html and "UNKNOWN" in html
    assert "&lt;hello&gt;" in html
    assert "[IMAGE]" in html
    assert "Tool Call: click" in html and "call_id: c1" in html
    assert "raw part" in html and "42" in html


def test_format_tool_call_defaults():
    assert e2h._format_tool_call_like_xray({}) == "Tool Call: unknown `{}` (call_id: unknown)"


def test_escape_html():
    assert e2h._escape_html("<a href=\"x\">'&'</a>") == (
        "&lt;a href=&quot;x&quot;&gt;&#x27;&amp;&#x27;&lt;/a&gt;"
    )
    assert e2h._escape_html(5) == "5"
