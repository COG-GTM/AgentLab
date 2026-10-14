import base64
import json
import math
import os
import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from openai.types.responses import ResponseFunctionToolCall  # noqa: E402
from PIL import Image  # noqa: E402

from agentlab.analyze import agent_xray as ax  # noqa: E402
from agentlab.experiments.loop import ExpResult, StepInfo, StepTimestamps  # noqa: E402
from agentlab.llm.llm_utils import Discussion  # noqa: E402
from agentlab.llm.response_api import MessageBuilder, ToolCall, ToolCalls  # noqa: E402

GR_UPDATE = {"__type__": "update"}


@pytest.fixture(autouse=True)
def fresh_info(monkeypatch):
    """Give every test its own module-global `info`; monkeypatch restores the original."""
    new_info = ax.Info()
    monkeypatch.setattr(ax, "info", new_info)
    yield new_info
    plt.close("all")


def _img_array(color, size=(8, 6)):
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[:, :] = color
    return arr


def _make_exp_result(tmp_path, steps, summary=None, logs=None, name="exp"):
    exp_dir = tmp_path / "study" / name
    exp_dir.mkdir(parents=True)
    for s in steps:
        s.save_step_info(exp_dir, save_som=True)
    if summary is not None:
        (exp_dir / "summary_info.json").write_text(json.dumps(summary))
    if logs is not None:
        (exp_dir / "experiment.log").write_text(logs)
    exp = ExpResult(exp_dir)
    exp._exp_args = SimpleNamespace(env_args=SimpleNamespace(task_name="task.a", task_seed=3))
    return exp


GOAL = [{"type": "text", "text": "Do it"}]


def _step(i, action=None, obs=None, agent_info=None, screens=True, goal=True, **kw):
    obs = dict(obs or {})
    if goal:
        obs.setdefault("goal_object", GOAL)
    if screens:
        obs["screenshot"] = _img_array((20 * i, 0, 0))
        obs["screenshot_som"] = _img_array((0, 20 * i, 0))
    return StepInfo(step=i, obs=obs, action=action, agent_info=agent_info or {}, **kw)


@pytest.fixture
def episode(tmp_path, fresh_info):
    steps = [
        _step(0, task_info={"k": 1}),
        _step(
            1,
            action="click('a1')",
            obs={
                "axtree_txt": "AX",
                "dom_txt": "<html/>",
                "pruned_html": "<p/>",
                "extra_element_properties": {"a1": {"bbox": (1, 1, 3, 3)}},
            },
            agent_info={
                "think": "thinking",
                "chat_messages": [
                    {"role": "system", "content": [{"type": "text", "text": "sys"}]},
                    {"role": "user", "content": [{"type": "text", "text": "user prompt"}]},
                ],
                "markdown_page": "# md",
                "html_page": "<b>hi</b>",
            },
            stats={"n_tokens": 5, "cost": 0.1},
        ),
        _step(2, action="noop()", obs={"last_action_error": "boom"}, terminated=True),
    ]
    exp = _make_exp_result(
        tmp_path,
        steps,
        summary={"cum_reward": 1.0, "err_msg": None, "stack_trace": "Traceback..."},
        logs="log line",
    )
    fresh_info.exp_result = exp
    fresh_info.step = 1
    return exp


# --------------------------------------------------------------------------- tables


def test_display_table_wraps_dotted_names_without_mutating():
    df = pd.DataFrame({"agent.name": [1], "b": [2]}).set_index("agent.name")
    df2 = pd.DataFrame({"env.task_name": [1]}, index=pd.Index([0], name="x.y"))
    out = ax.display_table(df2)
    assert list(out.columns) == ["env.\ntask_name"]
    assert list(out.index.names) == ["x.\ny"]
    assert list(df2.columns) == ["env.task_name"]
    assert list(ax.display_table(df).index.names) == ["agent.\nname"]


def test_remove_args_from_col_in_place():
    df = pd.DataFrame({"agent_args.x": [1], "env_args.task_name": [2]})
    df.index.name = "agent_args.name"
    out = ax.remove_args_from_col(df)
    assert out is df
    assert list(df.columns) == ["agent.x", "env.task_name"]
    assert list(df.index.names) == ["agent.name"]


@pytest.mark.parametrize(
    "cols, expected",
    [([], []), (["a"], ["a"]), (["a.b.c", "d_args"], ["a.\nb.\nc", "d_args"])],
)
def test_clean_column_names(cols, expected):
    assert ax.clean_column_names(cols) == expected


def test_click_mapper_maps_pixels_to_steps():
    fig, ax_ = plt.subplots()
    ax_.set_xlim(0, 10)
    fig.canvas.draw()
    mapper = ax.ClickMapper(ax_, step_times=[2.0, 5.0, 8.0])
    for t, expected_step in [(1.0, 0), (3.0, 1), (6.0, 2), (9.0, 3)]:
        x_pix, _ = ax_.transData.transform((t, 0))
        assert mapper.to_time(x_pix) == pytest.approx(t)
        assert mapper.to_step(x_pix) == expected_step


def test_dataclass_defaults():
    assert ax.EpisodeId() == ax.EpisodeId(None, None, None, None)
    assert ax.StepId().step is None
    info = ax.Info()
    assert info.active_tab == "Screenshot"
    assert info.exp_result is None


# --------------------------------------------------------------------------- Info


def _result_df(tmp_path):
    return pd.DataFrame(
        {
            ax.TASK_NAME_KEY: ["t1", "t1", "t2"],
            ax.TASK_SEED_KEY: [0, 1, 0],
            "agent.agent_name": ["A", "A", "B"],
            "cum_reward": [1.0, 0.0, 0.5],
            "err_msg": [None, "err", None],
            "n_steps": [3, 4, 5],
            "exp_dir": [tmp_path / "e0", tmp_path / "e1", tmp_path / "e2"],
        }
    )


class TestInfo:
    def test_get_agent_id(self):
        info = ax.Info(agent_id_keys=["agent.agent_name", "x"])
        row = pd.Series({"agent.agent_name": "A", "x": 2, "y": 3})
        assert info.get_agent_id(row) == [("agent.agent_name", "A"), ("x", 2)]

    def test_filter_agent_id_preserves_row_index(self, tmp_path):
        info = ax.Info(result_df=_result_df(tmp_path))
        info.filter_agent_id([("agent.\nagent_name", "A")])
        assert list(info.agent_df.index) == ["t1", "t1"]
        assert list(info.agent_df["_row_index"]) == [0, 1]
        info.filter_agent_id([("agent.agent_name", "B")])
        assert list(info.agent_df["_row_index"]) == [2]

    def test_update_exp_result(self, tmp_path):
        info = ax.Info(result_df=_result_df(tmp_path))
        info.filter_agent_id([("agent.agent_name", "A")])
        info.step = 5
        info.update_exp_result(ax.EpisodeId("A", "t1", 1, row_index=1))
        assert isinstance(info.exp_result, ExpResult)
        assert info.exp_result.exp_dir == tmp_path / "e1"
        assert info.step == 0

    def test_update_exp_result_unknown_row(self, tmp_path):
        info = ax.Info(result_df=_result_df(tmp_path))
        info.filter_agent_id([("agent.agent_name", "A")])
        with pytest.raises(ValueError, match="_row_index: 2"):
            info.update_exp_result(ax.EpisodeId("A", "t2", 0, row_index=2))
        assert info.exp_result is None

    def test_update_exp_result_duplicate_rows_warns(self, tmp_path, monkeypatch):
        info = ax.Info(result_df=_result_df(tmp_path))
        info.filter_agent_id([("agent.agent_name", "A")])
        info.agent_df["_row_index"] = 0
        warn = MagicMock()
        monkeypatch.setattr(ax, "warning", warn)
        info.update_exp_result(ax.EpisodeId("A", "t1", 0, row_index=0))
        assert "multiple rows" in warn.call_args[0][0]
        assert info.exp_result.exp_dir == tmp_path / "e0"

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: update_exp_result does not return after the result_df-is-None guard",
    )
    def test_update_exp_result_without_results_clears_selection(self):
        info = ax.Info()
        info.update_exp_result(ax.EpisodeId())
        assert info.exp_result is None


# --------------------------------------------------------------------------- markdown


class TestDictToMarkdown:
    def test_tool_calls_render_empty(self):
        assert ax.dict_to_markdown(ToolCalls()) == ""

    def test_non_dict_returns_repr_and_warns(self, monkeypatch):
        warn = MagicMock()
        monkeypatch.setattr(ax, "warning", warn)
        assert ax.dict_to_markdown([1, "a"]) == "[1, 'a']"
        warn.assert_called_once()

    def test_empty(self):
        assert ax.dict_to_markdown({}) == "No Data"

    def test_nested(self):
        d = {"a": 1, "b": {"c": "x"}, "l": [{"k": 2}, "s"], "e": {}}
        assert ax.dict_to_markdown(d) == (
            "- **a**: 1\n"
            "### b\n- **c**: x\n\n"
            "### l\n#### Item 0\n- **k**: 2\n\n- s\n"
            "### e\nNo Data\n"
        )


class TestDictMsgToMarkdown:
    def test_without_role_falls_back(self):
        assert ax.dict_msg_to_markdown({"x": 1}) == "- **x**: 1\n"

    def test_content_types(self):
        class PydanticLike:
            def dict(self):
                return {"type": "text", "text": "from dict()"}

        msg = {
            "role": "assistant",
            "content": [
                {"type": "image", "image": "data:img"},
                {"type": "text", "text": "hello"},
                {"type": "tool_use", "name": "click", "input": {"bid": "a"}, "id": "t1"},
                {"type": "weird", "v": 1},
                PydanticLike(),
            ],
        }
        out = ax.dict_msg_to_markdown(msg)
        assert out.startswith("### Assistant\n")
        assert "![Image](data:img)" in out
        assert "\n```\nhello\n```\n" in out
        assert "Tool Call: click  `{'bid': 'a'}` (call_id: t1)" in out
        assert "{'type': 'weird', 'v': 1}" in out
        assert "from dict()" in out

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: dict messages with plain-string content crash (iterates characters)",
    )
    def test_string_content(self):
        out = ax.dict_msg_to_markdown({"role": "user", "content": "hello"})
        assert "hello" in out


def test_format_tool_call():
    assert ax._format_tool_call("f", '{"a": 1}', "c9") == 'Tool Call: f  `{"a": 1}` (call_id: c9)'


class TestFormatChatMessage:
    def test_langchain_message(self, monkeypatch):
        class FakeLCMessage:
            def __init__(self, content):
                self.content = content

        monkeypatch.setattr(ax, "BaseMessage", FakeLCMessage)
        assert ax.format_chat_message(FakeLCMessage("lc content")) == "lc content"

    def test_message_builder(self):
        mb = MessageBuilder.assistant().add_text("yo")
        assert ax.format_chat_message(mb) == "### Assistant\n\n```\nyo\n```\n"

    def test_message_builder_with_tool_calls(self):
        tc = ToolCalls([ToolCall(name="click", arguments={"bid": "a1"}).response_text("ok")])
        md = ax.format_chat_message(MessageBuilder.add_responded_tool_calls(tc))
        assert "**Tool Call 1**: click(bid='a1')" in md
        assert "ok" in md

    def test_dict(self):
        out = ax.format_chat_message({"role": "user", "content": [{"type": "text", "text": "q"}]})
        assert out == "### User\n\n```\nq\n```\n"

    def test_response_function_tool_call(self):
        call = ResponseFunctionToolCall(
            name="fill", arguments='{"v": 1}', call_id="c1", type="function_call"
        )
        assert ax.format_chat_message(call) == (
            '### Tool Use\n```\nTool Call: fill  `{"v": 1}` (call_id: c1)\n```\n'
        )

    def test_other(self):
        assert ax.format_chat_message(12) == "12"
        assert ax.format_chat_message("No Chat Messages") == "No Chat Messages"


def test_page_to_iframe_roundtrip():
    page = "<h1>héllo</h1>"
    out = ax._page_to_iframe(page)
    b64 = re.search(r'src="data:text/html;base64,([^"]+)"', out).group(1)
    assert base64.b64decode(b64).decode("utf-8") == page
    assert out.strip().startswith("<iframe") and out.strip().endswith("</iframe>")


def test_code():
    assert ax.code("x") == "```\nx\n```"


# --------------------------------------------------------------------------- tabs


class TestIfActive:
    def test_runs_when_active(self, fresh_info):
        fresh_info.active_tab = "Logs"
        fn = MagicMock(return_value="ran")
        assert ax.if_active("Logs")(fn)(1, k=2) == "ran"
        fn.assert_called_once_with(1, k=2)

    @pytest.mark.parametrize("n_out, expected", [(1, GR_UPDATE), (3, (GR_UPDATE,) * 3)])
    def test_skips_when_inactive(self, fresh_info, n_out, expected):
        fresh_info.active_tab = "Screenshot"
        fn = MagicMock()
        assert ax.if_active("Logs", n_out)(fn)() == expected
        fn.assert_not_called()

    def test_tab_select_sets_active_tab(self, fresh_info):
        ax.tab_select(SimpleNamespace(value="AXTree"))
        assert fresh_info.active_tab == "AXTree"


class TestStepNavigation:
    def test_handle_key_event(self, fresh_info, episode):
        ep = ax.EpisodeId("A", "t", 0, 0)
        assert ax.handle_key_event("Cmd+Left", ax.StepId(ep, 1)) == ("", ax.StepId(ep, 0))
        assert ax.handle_key_event("Cmd+Left", ax.StepId(ep, 0))[1].step == 0
        assert ax.handle_key_event("Cmd+Right", ax.StepId(ep, 0))[1].step == 1
        # clamps to len(steps) - 2
        assert ax.handle_key_event("Cmd+Right", ax.StepId(ep, 1))[1].step == 1
        assert fresh_info.step == 1

    def test_handle_key_event_noop(self, fresh_info):
        sid = ax.StepId(None, 4)
        assert ax.handle_key_event("", sid) == ("", sid)
        assert ax.handle_key_event("Shift+Up", sid) == GR_UPDATE

    def test_gallery_step_change(self, fresh_info):
        ep = ax.EpisodeId("A")
        assert ax.gallery_step_change(SimpleNamespace(index=2), ep) == ax.StepId(ep, 2)
        assert fresh_info.step == 2

    def test_select_step_uses_click_mapper(self, fresh_info):
        fresh_info.click_mapper = MagicMock()
        fresh_info.click_mapper.to_step.return_value = 3
        out = ax.select_step(ax.EpisodeId("A"), SimpleNamespace(index=(120, 5)))
        fresh_info.click_mapper.to_step.assert_called_once_with(120)
        assert out.step == 3 and fresh_info.step == 3


# --------------------------------------------------------------------------- screenshots


class TestScreenshots:
    def test_get_screenshot_raw_and_som(self, fresh_info, episode):
        img, action = ax.get_screenshot(fresh_info, step=2)
        assert img.getpixel((0, 0)) == (40, 0, 0) and action is None
        img, _ = ax.get_screenshot(fresh_info, step=2, som_or_not="SOM Screenshots")
        assert img.getpixel((0, 0)) == (0, 40, 0)

    def test_get_screenshot_annotates_with_current_step(self, fresh_info, episode):
        img, action = ax.get_screenshot(fresh_info, annotate=True)
        assert "<span" in action and "'a1'" in action
        assert img.getpixel((1, 1)) != (20, 0, 0)  # bbox overlay drawn

    def test_annotation_failure_falls_back_to_raw_action(self, fresh_info, tmp_path, monkeypatch):
        fresh_info.exp_result = _make_exp_result(tmp_path, [_step(0, action="click('zz')")])
        warn = MagicMock()
        monkeypatch.setattr(ax, "warning", warn)
        _, action = ax.get_screenshot(fresh_info, step=0, annotate=True)
        assert action == "click('zz')"
        assert "Failed to annotate" in warn.call_args[0][0]

    def test_out_of_range_and_missing(self, fresh_info, episode, tmp_path):
        assert ax.get_screenshot(fresh_info, step=10) == (None, None)
        fresh_info.exp_result = _make_exp_result(
            tmp_path, [_step(0, screens=False)], name="noscreens"
        )
        assert ax.get_screenshot(fresh_info, step=0) == (None, None)

    def test_update_screenshot_and_pair(self, fresh_info, episode):
        assert ax.update_screenshot("Raw Screenshots").getpixel((0, 0)) == (20, 0, 0)
        s1, s2 = ax.update_screenshot_pair("SOM Screenshots")
        assert s1.getpixel((5, 5)) == (0, 20, 0)
        assert s2.getpixel((0, 0)) == (0, 40, 0)

    def test_update_screenshot_gallery(self, fresh_info, episode):
        gallery = ax.update_screenshot_gallery("Raw Screenshots")
        assert gallery.selected_index == 1
        assert gallery.columns == 2
        assert [item["caption"] for item in gallery.value] == ["Step 0", "Step 1", "Step 2"]


# --------------------------------------------------------------------------- tab contents


class TestTabUpdates:
    def test_update_episode_escapes_html_into_srcdoc(self, fresh_info, episode):
        out = ax.update_episode()
        assert out.startswith('<iframe srcdoc="')
        srcdoc = out.split('srcdoc="', 1)[1].split('"', 1)[0]
        assert "<" not in srcdoc and "&lt;div class=&quot;episode-container&quot;&gt;" in srcdoc

    def test_obs_based_tabs(self, fresh_info, episode):
        assert ax.update_html() == "<html/>"
        assert ax.update_pruned_html() == "<p/>"
        assert ax.update_axtree() == "```\nAX\n```"
        fresh_info.step = 2
        assert ax.update_html() == "No DOM HTML"
        assert ax.update_pruned_html() == "No Pruned HTML"
        assert ax.update_axtree() == "```\nNo AXTree\n```"

    def test_update_chat_messages_list(self, fresh_info, episode):
        out = ax.update_chat_messages()
        assert out == "### System\n\n```\nsys\n```\n\n\n### User\n\n```\nuser prompt\n```\n"

    def test_update_chat_messages_default_and_discussion(self, fresh_info, episode):
        fresh_info.step = 2
        assert ax.update_chat_messages() == "No Chat Messages"
        steps = fresh_info.exp_result.steps_info
        steps[2].agent_info["chat_messages"] = Discussion([ax.AgentLabBaseMessage("user", "d")])
        assert ax.update_chat_messages() == "Message 0\n\n```\nd\n```\n\n"

    def test_task_error_and_logs(self, fresh_info, episode, tmp_path):
        assert ax.update_task_error() == "```\nTraceback...\n```"
        assert ax.update_logs() == "log line"
        fresh_info.exp_result = _make_exp_result(tmp_path, [_step(0)], name="bare")
        assert ax.update_task_error() == "No Task Error"
        assert ax.update_logs() == "No Logs"

    def test_update_stats(self, fresh_info, episode):
        df = ax.update_stats()
        assert df.to_dict("records") == [
            {"name": "n_tokens", "value": 5},
            {"name": "cost", "value": 0.1},
        ]
        fresh_info.step = 9
        assert ax.update_stats() is None

    def test_update_agent_info_md(self, fresh_info, episode):
        assert ax.update_agent_info_md() == "# md"
        fresh_info.step = 2
        assert "Fill up markdown_page" in ax.update_agent_info_md()
        fresh_info.exp_result.steps_info[2].agent_info["markup_page"] = "legacy"
        assert ax.update_agent_info_md() == "legacy"
        fresh_info.step = 9
        assert ax.update_agent_info_md() is None

    def test_update_agent_info_html(self, fresh_info, episode):
        page, s1, s2 = ax.update_agent_info_html()
        b64 = re.search(r"base64,([^\"]+)", page).group(1)
        assert base64.b64decode(b64).decode() == "<b>hi</b>"
        assert s1.getpixel((0, 0)) == (20, 0, 0) and s2.getpixel((0, 0)) == (40, 0, 0)

        fresh_info.exp_result.steps_info[1].agent_info["html_page"] = None
        page, _, _ = ax.update_agent_info_html()
        assert "Fill up html_page" in page

        fresh_info.step = 9
        assert ax.update_agent_info_html() == (None, None, None)

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: missing html_page defaults to a list, which _page_to_iframe can't encode",
    )
    def test_update_agent_info_html_without_page(self, fresh_info, episode):
        fresh_info.step = 2
        page, _, _ = ax.update_agent_info_html()
        assert isinstance(page, str)

    def test_update_prompt_tests(self, fresh_info, episode):
        prompt = [{"type": "text", "text": "user prompt"}]
        assert ax.update_prompt_tests() == (prompt, prompt)
        fresh_info.exp_result.steps_info[1].agent_info["chat_messages"] = ["a", "plain"]
        assert ax.update_prompt_tests() == ("plain", "plain")


class TestSubmitAction:
    def _client(self, monkeypatch):
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="LLM says hi"))]
        )
        factory = MagicMock(return_value=client)
        monkeypatch.setattr(ax, "OpenAI", factory)
        return client

    def test_dict_messages(self, fresh_info, episode, monkeypatch):
        client = self._client(monkeypatch)
        original = fresh_info.exp_result.steps_info[1].agent_info["chat_messages"]
        assert ax.submit_action("new prompt") == "LLM says hi"
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "gpt-4o-mini"
        assert kwargs["messages"][1] == {"role": "user", "content": "new prompt"}
        assert len(kwargs["messages"]) == 2
        assert original[1]["content"] == [{"type": "text", "text": "user prompt"}]  # deep-copied

    def test_langchain_messages_converted(self, fresh_info, episode, monkeypatch):
        class FakeBase:
            def __init__(self, content):
                self.content = content

        class FakeHuman(FakeBase):
            pass

        monkeypatch.setattr(ax, "BaseMessage", FakeBase)
        monkeypatch.setattr(ax, "HumanMessage", FakeHuman)
        fresh_info.exp_result.steps_info[1].agent_info["chat_messages"] = [
            FakeBase("sys"),
            FakeHuman("usr"),
        ]
        client = self._client(monkeypatch)
        ax.submit_action("edited")
        msgs = client.chat.completions.create.call_args.kwargs["messages"]
        assert msgs[0] == {"role": "system", "content": "sys"}
        assert msgs[1]["role"] == "user" and msgs[1]["content"] == "edited"

    def test_rejects_non_user_second_message(self, fresh_info, episode, monkeypatch):
        self._client(monkeypatch)
        fresh_info.exp_result.steps_info[1].agent_info["chat_messages"] = [
            {"role": "system", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
        with pytest.raises(AssertionError, match="Second message should be user"):
            ax.submit_action("x")

    def test_rejects_unknown_message_types(self, fresh_info, episode, monkeypatch):
        self._client(monkeypatch)
        fresh_info.exp_result.steps_info[1].agent_info["chat_messages"] = ["a", "b"]
        with pytest.raises(ValueError, match="list of BaseMessage or dict"):
            ax.submit_action("x")


# --------------------------------------------------------------------------- step info panels


class TestStepInfoPanels:
    def test_update_step_info(self, fresh_info, episode):
        ep_info, action_info, state_error = ax.update_step_info()
        assert "### task.a (seed: 3)" in ep_info
        assert "### Step 1 / 2 (Reward: 1.0)" in ep_info
        assert "Do it" in ep_info
        assert "Terminated: None, Truncated: None" in ep_info
        assert "study/exp" in ep_info
        assert action_info.startswith("**Action:**")
        assert "<span" in action_info
        assert "**Think:**\n\n```\nthinking\n```" in action_info
        assert "```\nboom\n```" in state_error

    def test_get_episode_info_clamps_step_and_handles_missing_summary(self, fresh_info, tmp_path):
        fresh_info.exp_result = _make_exp_result(tmp_path, [_step(0, task_info=None)])
        fresh_info.step = 5
        out = ax.get_episode_info(fresh_info)
        assert fresh_info.step == 0
        assert "Step 0 / 0 (Reward: nan)" in out

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: get_episode_info errors when obs has no goal_object (str(BaseMessage(None)))",
    )
    def test_get_episode_info_no_goal_key(self, fresh_info, tmp_path):
        fresh_info.exp_result = _make_exp_result(
            tmp_path, [_step(0, goal=False)], summary={"cum_reward": 0.0}
        )
        fresh_info.step = 0
        assert "### task.a (seed: 3)" in ax.get_episode_info(fresh_info)

    def test_get_episode_info_error_is_reported(self, fresh_info):
        out = ax.get_episode_info(fresh_info)  # exp_result is None
        assert out.startswith("**Error while getting episode info**")
        assert "AttributeError" in out

    def test_get_action_info_without_think(self, fresh_info, episode):
        fresh_info.step = 2
        out = ax.get_action_info(fresh_info)
        assert "noop()" in out and "Think" not in out

    def test_get_action_info_out_of_bounds(self, fresh_info, episode):
        fresh_info.step = 7
        assert ax.get_action_info(fresh_info) == (
            "Step 7 is out of bounds. The episode has 3 steps."
        )

    def test_get_action_info_no_steps(self, fresh_info):
        fresh_info.exp_result = MagicMock(steps_info=[])
        fresh_info.step = 0
        assert ax.get_action_info(fresh_info) == "No steps were taken"

    @pytest.mark.parametrize("step", [0, 2, 10])
    def test_get_state_error_none(self, fresh_info, episode, step):
        fresh_info.step = step
        assert "```\nNo Error\n```" in ax.get_state_error(fresh_info)

    def test_get_state_error_without_exp_result(self, fresh_info):
        fresh_info.step = 0
        assert "No Error" in ax.get_state_error(fresh_info)


# --------------------------------------------------------------------------- seed/agent selection


class TestSelection:
    def test_get_seeds_df(self, tmp_path, fresh_info):
        fresh_info.result_df = _result_df(tmp_path)
        fresh_info.filter_agent_id([("agent.agent_name", "A")])
        seeds = ax.get_seeds_df(fresh_info.agent_df, "t1")
        assert list(seeds.columns) == ["seed", "reward", "err", "n_steps", "idx"]
        assert seeds.to_dict("records") == [
            {"seed": 0, "reward": 1.0, "err": False, "n_steps": 3, "idx": 0},
            {"seed": 1, "reward": 0.0, "err": True, "n_steps": 4, "idx": 1},
        ]

    def test_update_seeds_selects_first(self, tmp_path, fresh_info):
        fresh_info.result_df = _result_df(tmp_path)
        fresh_info.filter_agent_id([("agent.agent_name", "A")])
        seeds, ep = ax.update_seeds(("agentA", "t1"))
        assert len(seeds) == 2
        assert ep == ax.EpisodeId(agent_id="agentA", task_name="t1", seed=0, row_index=0)

    def test_on_select_agent(self, fresh_info):
        fresh_info.agent_id_keys = ["agent.agent_name"]
        df = pd.DataFrame({"agent.agent_name": ["A", "B"]})
        evt = SimpleNamespace(index=(1, 0))
        assert ax.on_select_agent(evt, df) == [("agent.agent_name", "B")]

    def test_on_select_task(self):
        df = pd.DataFrame({"x": [1], ax.TASK_NAME_KEY: ["t9"]})
        evt = SimpleNamespace(row_value=[1, "t9"])
        assert ax.on_select_task(evt, df, [("a", 1)]) == ([("a", 1)], "t9")

    def test_on_select_seed(self):
        df = pd.DataFrame(columns=["seed", "reward", "err", "n_steps", "idx"])
        evt = SimpleNamespace(row_value=[4, 1.0, False, 3, 12])
        ep = ax.on_select_seed(evt, df, ("ag", "t"))
        assert ep == ax.EpisodeId(agent_id="ag", task_name="t", seed=4, row_index=12)


# --------------------------------------------------------------------------- directory listing


def test_get_directory_contents(tmp_path, monkeypatch):
    (tmp_path / "file.txt").write_text("not a dir")
    (tmp_path / "2024-01-01_no_summary").mkdir()

    good = tmp_path / "2024-02-01_good"
    good.mkdir()
    pd.DataFrame(
        {"avg_reward": [0.25, 0.5], "n_completed": ["4/4", "3/4"], "n_err": [0, 1]}
    ).to_csv(good / "summary_df.csv", index=False)

    empty = tmp_path / "2024-03-01_empty"
    empty.mkdir()
    pd.DataFrame(columns=["avg_reward", "n_completed", "n_err"]).to_csv(
        empty / "summary_df.csv", index=False
    )

    broken = tmp_path / "2024-04-01_broken"
    broken.mkdir()
    pd.DataFrame({"other": [1]}).to_csv(broken / "summary_df.csv", index=False)

    out = ax.get_directory_contents(tmp_path)
    assert out == [
        ax.select_dir_instructions,
        "2024-04-01_broken",
        "2024-02-01_good - avg-reward: 50.0% - completed: 3/4 - errors: 1",
        "2024-01-01_no_summary",
    ]


def test_get_directory_contents_picks_most_recent_summary(tmp_path, monkeypatch):
    d = tmp_path / "study"
    d.mkdir()
    old, new = d / "summary_old.csv", d / "summary_new.csv"
    pd.DataFrame({"avg_reward": [0.1], "n_completed": ["1/2"], "n_err": [0]}).to_csv(
        old, index=False
    )
    pd.DataFrame({"avg_reward": [0.9], "n_completed": ["2/2"], "n_err": [3]}).to_csv(
        new, index=False
    )
    monkeypatch.setattr(os.path, "getctime", lambda p: 2.0 if "new" in str(p) else 1.0)
    out = ax.get_directory_contents(tmp_path)
    assert out[1] == "study - avg-reward: 90.0% - completed: 2/2 - errors: 3"


# --------------------------------------------------------------------------- profiling plot


def _prof(t0, **kw):
    p = StepTimestamps(env_start=t0, env_stop=t0 + 1, agent_start=t0 + 1, agent_stop=t0 + 2)
    p.action_exec_start, p.action_exec_stop = t0 + 0.2, t0 + 0.4
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def _passthrough_progress(it, desc=None):
    return it


class TestProfiling:
    def test_fig_to_pil(self):
        fig, _ = plt.subplots(figsize=(2, 1), dpi=50)
        img = ax.fig_to_pil(fig)
        assert isinstance(img, Image.Image) and img.size == (100, 50)
        assert not plt.fignum_exists(fig.number)

    @pytest.mark.parametrize("edge", [False, True])
    def test_add_patch(self, edge):
        _, ax_ = plt.subplots()
        ax.add_patch(ax_, 1, 4, "red", "lbl", edge=edge)
        (rect,) = ax_.patches
        assert (rect.get_x(), rect.get_width(), rect.get_height()) == (1, 3, 1)
        assert rect.get_label() == "lbl"
        assert rect.get_fill() is (not edge)

    def test_plot_profiling_empty(self, monkeypatch):
        warn = MagicMock()
        monkeypatch.setattr(ax, "warning", warn)
        _, ax_ = plt.subplots()
        assert ax.plot_profiling(ax_, [], {}, _passthrough_progress) is None
        warn.assert_called_once_with("No step info to plot")

    def test_plot_profiling_full(self):
        t0 = 1000.0
        steps = [
            StepInfo(step=0, obs={}, action="a0", profiling=_prof(t0)),
            StepInfo(
                step=1,
                obs={},
                action="a1",
                profiling=_prof(
                    t0 + 2,
                    wait_for_page_loading_start=t0 + 2.5,
                    wait_for_page_loading_stop=t0 + 2.6,
                    validation_start=t0 + 2.6,
                    validation_stop=t0 + 2.7,
                    get_observation_start=t0 + 2.7,
                    get_observation_stop=t0 + 2.8,
                ),
            ),
            StepInfo(
                step=2,
                obs={"last_action_error": "oops"},
                action=None,
                terminated=True,
                profiling=_prof(t0 + 4, wait_for_page_loading_start=None),
            ),
        ]
        _, ax_ = plt.subplots()
        times = ax.plot_profiling(
            ax_, steps, {"cum_reward": 1.0, "err_msg": None}, _passthrough_progress
        )
        assert times == pytest.approx([2.0, 4.0])
        labels = {p.get_label() for p in ax_.patches} - {"_nolegend_", None}
        assert {
            "reset",
            "env",
            "exec action",
            "wait for page",
            "validation",
            "get observation",
            "agent",
        } <= labels
        texts = [t.get_text() for t in ax_.texts]
        assert texts == ["1", "2", "R:1.0"]
        assert ax_.get_xlim() == pytest.approx((0, 7.0))

    @pytest.mark.parametrize(
        "kw, summary, expected_text",
        [
            ({"truncated": True}, {"cum_reward": 0.0, "err_msg": None}, "R:0.0"),
            ({"terminated": True}, {"cum_reward": 0.0, "err_msg": None}, "R:0.0"),
            ({"terminated": True}, {"err_msg": "crash"}, "Err"),
        ],
    )
    def test_plot_profiling_end_markers(self, kw, summary, expected_text):
        steps = [
            StepInfo(step=0, obs={}, action=None, profiling=_prof(0.0)),
            StepInfo(step=1, obs=None, action=None, profiling=_prof(2.0), **kw),
        ]
        _, ax_ = plt.subplots()
        ax.plot_profiling(ax_, steps, summary, _passthrough_progress)
        assert ax_.texts[-1].get_text() == expected_text
        # truncated / failed episodes are drawn with a black or red marker, not green
        assert ax_.lines[-1].get_color() in ("black",)

    def test_plot_profiling_rejects_bad_step_type(self):
        _, ax_ = plt.subplots()
        bad = SimpleNamespace(profiling=_prof(0.0), step=0)
        with pytest.raises(AssertionError, match="Expected StepInfo"):
            ax.plot_profiling(ax_, [bad], {}, _passthrough_progress)

    def test_generate_profiling(self, fresh_info, tmp_path):
        assert ax.generate_profiling(_passthrough_progress) is None

        steps = [
            _step(0, profiling=_prof(10.0)),
            _step(1, action="noop()", profiling=_prof(12.0), terminated=True),
        ]
        fresh_info.exp_result = _make_exp_result(tmp_path, steps)  # no summary_info.json
        with pytest.raises(KeyError, match="err_msg"):
            # summary falls back to {} and plot_profiling indexes summary_info["err_msg"]
            ax.generate_profiling(_passthrough_progress)

    def test_generate_profiling_with_summary(self, fresh_info, episode):
        steps = fresh_info.exp_result.steps_info
        for i, s in enumerate(steps):
            s.profiling = _prof(100.0 + 2 * i)
        img = ax.generate_profiling(_passthrough_progress)
        assert isinstance(img, Image.Image)
        assert isinstance(fresh_info.click_mapper, ax.ClickMapper)
        assert fresh_info.click_mapper.step_times == pytest.approx([4.0, 6.0])
        assert fresh_info.exp_result.progress_fn is None


def test_format_constant_and_variables(tmp_path, fresh_info):
    df = pd.DataFrame(
        {
            "const": ["c"] * 5,
            "var": ["a", "a", "b", "c", "d"],
            "stack_trace": ["s1", "s2", "s3", "s4", "s5"],
        }
    )
    fresh_info.result_df = df
    constants, variables = ax.format_constant_and_variables()
    assert constants.to_dict("records") == [{"name": "const", "value": "c"}]
    var_rows = variables[variables["Name"] == "var"]
    assert list(var_rows["i"]) == [0, 1, 2, "..."]
    assert var_rows.iloc[0]["count"] == "2/5" and var_rows.iloc[0]["value"] == "a"
    assert "stack_trace" not in set(variables["Name"])
    assert math.isnan(variables.iloc[-1]["n unique"])
