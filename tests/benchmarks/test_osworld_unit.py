"""Unit tests for agentlab.benchmarks.osworld that run without a real DesktopEnv/VM."""

import io
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from agentlab.benchmarks import osworld
from agentlab.benchmarks.abstract_env import AbstractEnv
from agentlab.benchmarks.osworld import (
    COMPUTER_13_ACTIONS_OAI_CHATCOMPLETION_TOOLS,
    OSWorldActionSet,
    OsworldBenchmark,
    OsworldEnvArgs,
    OsworldGym,
    format_chat_completion_tools_to_anthropic,
    format_chat_completion_tools_to_response_api,
)
from agentlab.benchmarks.osworld_axtree_preprocessing import (
    component_ns_ubuntu,
    state_ns_ubuntu,
)

COMPUTER_13_ACTION_NAMES = [
    "move_to",
    "click",
    "mouse_down",
    "mouse_up",
    "right_click",
    "double_click",
    "drag_to",
    "scroll",
    "typing",
    "press",
    "key_down",
    "key_up",
    "hotkey",
    "wait",
    "fail",
    "done",
]

AXTREE = (
    f'<desktop-frame xmlns:st="{state_ns_ubuntu}" xmlns:cp="{component_ns_ubuntu}">'
    '<push-button name="OK" st:showing="true" st:visible="true" st:enabled="true" '
    'cp:screencoord="(1, 2)" cp:size="(3, 4)"/>'
    "</desktop-frame>"
)


def _png(size=(8, 6), mode="RGB"):
    buf = io.BytesIO()
    Image.linear_gradient("L").resize(size).convert(mode).save(buf, format="PNG")
    return buf.getvalue()


def _raw_obs(**extra):
    obs = {"screenshot": _png(), "accessibility_tree": AXTREE, "instruction": "Do it"}
    obs.update(extra)
    return obs


def _task():
    return {"id": "task-1", "instruction": "Do it", "config": []}


GYM_KWARGS = dict(
    provider_name="docker",
    region="us-east-1",
    path_to_vm=None,
    snapshot_name="init_state",
    action_space="computer_13",
    cache_dir="cache",
    screen_size=(1920, 1080),
    headless=True,
    require_a11y_tree=True,
    require_terminal=False,
    os_type="Ubuntu",
    enable_proxy=False,
    max_steps=3,
)


@pytest.fixture
def desktop_env_cls(monkeypatch):
    cls = MagicMock(name="DesktopEnv")
    instance = cls.return_value
    instance.action_space = "computer_13"
    instance._get_obs.return_value = _raw_obs()
    instance.step.return_value = (_raw_obs(), 0.0, False, {})
    instance.evaluate.return_value = 1.0
    monkeypatch.setattr(osworld, "DesktopEnv", cls)
    monkeypatch.setattr(osworld.time, "sleep", MagicMock(name="sleep"))
    return cls


@pytest.fixture
def make_gym(desktop_env_cls, tmp_path):
    def _make(**overrides):
        kwargs = dict(GYM_KWARGS, task=_task(), exp_dir=tmp_path)
        kwargs.update(overrides)
        return OsworldGym(**kwargs)

    return _make


class TestModuleDefaults:
    def test_keyboard_keys_and_screen_bounds(self):
        assert {"ctrl", "alt", "enter", "a", "f12"} <= set(osworld.KEYBOARD_KEYS)
        assert osworld.X_MAX > 0 and osworld.Y_MAX > 0

    def test_computer_13_tool_catalog(self):
        names = [t["function"]["name"] for t in COMPUTER_13_ACTIONS_OAI_CHATCOMPLETION_TOOLS]
        assert names == COMPUTER_13_ACTION_NAMES
        for tool in COMPUTER_13_ACTIONS_OAI_CHATCOMPLETION_TOOLS:
            assert tool["type"] == "function"
            assert tool["function"]["parameters"]["type"] == "object"

    def test_coordinate_bounds_follow_screen_size(self):
        move_to = COMPUTER_13_ACTIONS_OAI_CHATCOMPLETION_TOOLS[0]["function"]["parameters"]
        assert move_to["properties"]["x"]["maximum"] == osworld.X_MAX
        assert move_to["properties"]["y"]["maximum"] == osworld.Y_MAX
        assert move_to["required"] == ["x", "y"]

    def test_key_enums_use_keyboard_keys(self):
        by_name = {
            t["function"]["name"]: t["function"]["parameters"]
            for t in COMPUTER_13_ACTIONS_OAI_CHATCOMPLETION_TOOLS
        }
        assert by_name["press"]["properties"]["key"]["enum"] == osworld.KEYBOARD_KEYS
        assert by_name["hotkey"]["properties"]["keys"]["items"]["enum"] == osworld.KEYBOARD_KEYS


class TestOsworldGymInit:
    def test_raises_import_error_without_desktop_env(self, monkeypatch, tmp_path):
        monkeypatch.setattr(osworld, "DesktopEnv", None)
        with pytest.raises(ImportError, match="desktop_env is not installed"):
            OsworldGym(task=_task(), exp_dir=tmp_path, **GYM_KWARGS)

    def test_builds_desktop_env_and_env_info(self, make_gym, desktop_env_cls, tmp_path):
        gym = make_gym()
        desktop_env_cls.assert_called_once_with(
            action_space="computer_13",
            provider_name="docker",
            region="us-east-1",
            path_to_vm=None,
            snapshot_name="init_state",
            cache_dir="cache",
            screen_size=(1920, 1080),
            headless=True,
            require_a11y_tree=True,
            require_terminal=False,
            os_type="Ubuntu",
        )
        assert isinstance(gym, AbstractEnv)
        assert gym.env is desktop_env_cls.return_value
        assert gym.env_info == {k: v for k, v in GYM_KWARGS.items() if k not in ("max_steps",)}
        assert gym._step_count == 0
        assert gym.max_steps == 3
        assert gym.exp_dir == tmp_path
        assert gym.record_video is True


class TestOsworldGymReset:
    def test_reset_waits_records_and_converts(self, make_gym):
        gym = make_gym()
        gym._step_count = 7
        obs, info = gym.reset(seed=42)

        gym.env.reset.assert_called_once_with(task_config=gym.task, seed=42)
        osworld.time.sleep.assert_called_once_with(60)
        gym.env.controller.start_recording.assert_called_once_with()
        assert info is gym.env_info
        assert gym._step_count == 0
        assert obs["screenshot"].shape == (6, 8, 3)
        assert "push-button\tOK" in obs["axtree_txt"]

    def test_reset_without_video_does_not_record(self, make_gym):
        gym = make_gym(record_video=False)
        gym.reset()
        gym.env.controller.start_recording.assert_not_called()


class TestOsworldGymStep:
    def test_regular_step_increments_and_adds_timing(self, make_gym):
        gym = make_gym()
        obs, reward, done, truncated, info = gym.step("move_to(x=1, y=2)")

        gym.env.step.assert_called_once_with(
            {"action_type": "MOVE_TO", "parameters": {"x": 1, "y": 2}}
        )
        gym.env.evaluate.assert_not_called()
        assert (reward, done, truncated) == (0.0, False, False)
        assert gym._step_count == 1
        assert {"action_exec_start", "action_exec_stop", "action_exec_timeout"} <= set(info)
        assert obs["goal_object"] == [{"type": "text", "text": "Do it"}]

    def test_done_triggers_evaluation(self, make_gym):
        gym = make_gym()
        gym.env.step.return_value = (_raw_obs(), 0.0, True, {})
        _, reward, done, truncated, _ = gym.step("done()")
        gym.env.step.assert_called_once_with("DONE")
        gym.env.evaluate.assert_called_once_with()
        assert (reward, done, truncated) == (1.0, True, False)

    def test_max_steps_truncates_and_evaluates(self, make_gym, caplog):
        gym = make_gym(max_steps=2)
        gym.step("wait()")
        with caplog.at_level(logging.WARNING, logger=osworld.logger.name):
            _, reward, done, truncated, _ = gym.step("wait()")
        assert (reward, done, truncated) == (1.0, False, True)
        assert gym.env.evaluate.call_count == 1
        assert "truncated after 2 steps" in caplog.text

    def test_fail_info_truncates(self, make_gym):
        gym = make_gym()
        gym.env.step.return_value = (_raw_obs(), 0.0, False, {"fail": True})
        _, _, done, truncated, info = gym.step("fail()")
        assert (done, truncated) == (False, True)
        assert info["fail"] is True
        gym.env.evaluate.assert_called_once()

    def test_evaluation_error_keeps_env_reward(self, make_gym, caplog):
        gym = make_gym()
        gym.env.step.return_value = (_raw_obs(), 0.25, True, {})
        gym.env.evaluate.side_effect = RuntimeError("boom")
        with caplog.at_level(logging.ERROR, logger=osworld.logger.name):
            _, reward, _, _, _ = gym.step("done()")
        assert reward == 0.25
        assert "Failed to evaluate" in caplog.text and "boom" in caplog.text


class TestActionConversion:
    def test_pyautogui_action_space_not_supported(self, make_gym):
        gym = make_gym()
        gym.env.action_space = "pyautogui"
        with pytest.raises(NotImplementedError, match="PyAutoGUI"):
            gym.agentlab_to_env_action("click()")

    def test_unknown_action_space_returns_none(self, make_gym):
        gym = make_gym()
        gym.env.action_space = "other"
        assert gym.agentlab_to_env_action("click()") is None

    @pytest.mark.parametrize(
        "action,expected", [("wait()", "WAIT"), ("done()", "DONE"), ("fail()", "FAIL")]
    )
    def test_terminal_actions_become_strings(self, make_gym, action, expected):
        assert make_gym().convert_agentlab_action_to_computer_13(action) == expected

    @pytest.mark.parametrize(
        "action,expected",
        [
            ("click(x=5, y=6, button='right')", {"x": 5, "y": 6, "button": "right"}),
            ("hotkey(keys=['ctrl', 'c'])", {"keys": ["ctrl", "c"]}),
            ("typing(text='hi')", {"text": "hi"}),
            ("mouse_up()", {}),
        ],
    )
    def test_kwargs_become_parameters(self, make_gym, action, expected):
        name = action.split("(")[0].upper()
        assert make_gym().convert_agentlab_action_to_computer_13(action) == {
            "action_type": name,
            "parameters": expected,
        }

    def test_positional_args_are_dropped_with_warning(self, make_gym, caplog):
        with caplog.at_level(logging.WARNING, logger=osworld.logger.name):
            result = make_gym().convert_agentlab_action_to_computer_13("move_to(10, 20, z=1)")
        assert result == {"action_type": "MOVE_TO", "parameters": {"z": 1}}
        assert "unexpected positional arguments: [10, 20]" in caplog.text

    def test_unparsable_action_is_passed_through_as_none_type(self, make_gym):
        # Current behavior: unparsable strings are not rejected but become a NONE action.
        assert make_gym().convert_agentlab_action_to_computer_13("not a call") == {
            "action_type": "NONE",
            "parameters": {},
        }

    @pytest.mark.parametrize(
        "action,expected",
        [
            ("move_to(x=1, y=2)", ("move_to", [], {"x": 1, "y": 2})),
            ("  scroll(3, dx=-1)  ", ("scroll", [3], {"dx": -1})),
            ("pyautogui.click(x=1)", ("pyautogui.click", [], {"x": 1})),
            ("press(key=None)", ("press", [], {"key": None})),
            ("f(t=(1, 2), d={'a': 1})", ("f", [], {"t": (1, 2), "d": {"a": 1}})),
        ],
    )
    def test_parse_valid_calls(self, action, expected):
        assert OsworldGym.parse_agentlab_action_str_to_func_args(action) == expected

    @pytest.mark.parametrize(
        "action", ["42", "click", "x = 1", "click(x=", "click(x=y)", "click(*args)", ""]
    )
    def test_parse_invalid_returns_nones(self, action, caplog):
        assert OsworldGym.parse_agentlab_action_str_to_func_args(action) == (None, None, None)

    def test_parse_logs_failures(self, caplog):
        with caplog.at_level(logging.WARNING, logger=osworld.logger.name):
            OsworldGym.parse_agentlab_action_str_to_func_args("click(x=")
        assert "Failed to parse" in caplog.text


class TestObservationConversion:
    def test_full_observation(self, make_gym):
        obs = make_gym().to_agentlab_observation(_raw_obs(terminal="$ ls"))
        assert isinstance(obs["screenshot"], np.ndarray)
        assert obs["screenshot"].dtype == np.uint8
        assert obs["axtree_txt"].splitlines()[1] == 'push-button\tOK\t""\t\t\t(1, 2)\t(3, 4)'
        assert obs["last_action_error"] == ""
        assert obs["focused_element_bid"] == ""
        assert obs["url"] == ""
        assert obs["open_pages_urls"] == []
        assert obs["open_pages_titles"] == []
        assert obs["active_page_index"] == 0
        assert obs["goal_object"] == [{"type": "text", "text": "Do it"}]
        assert obs["terminal_output"] == "$ ls"

    @pytest.mark.parametrize("terminal", [None, ""])
    def test_terminal_output_omitted_when_empty(self, make_gym, terminal):
        obs = make_gym().to_agentlab_observation(_raw_obs(terminal=terminal))
        assert "terminal_output" not in obs

    def test_missing_instruction_gives_empty_goal(self, make_gym):
        raw = _raw_obs()
        del raw["instruction"]
        obs = make_gym().to_agentlab_observation(raw)
        assert obs["goal_object"] == [{"type": "text", "text": ""}]

    @pytest.mark.parametrize("mode", ["RGBA", "L", "RGB"])
    def test_screenshot_is_always_rgb(self, make_gym, mode):
        arr = make_gym().convert_screenshot_to_numpy(_png((5, 4), mode=mode))
        assert arr.shape == (4, 5, 3)

    def test_som_screenshot(self, make_gym):
        converted = {}
        make_gym()._add_som_screenshot(
            converted, {"screenshot": _png((20, 20)), "accessibility_tree": AXTREE}
        )
        assert converted["som_screenshot"].shape == (20, 20, 3)


class TestClose:
    def test_close_stops_recording(self, make_gym, tmp_path):
        gym = make_gym()
        gym.env.close.return_value = "closed"
        assert gym.close() == "closed"
        gym.env.controller.end_recording.assert_called_once_with(str(tmp_path / "recording.mp4"))

    def test_close_without_video(self, make_gym):
        gym = make_gym(record_video=False)
        gym.close()
        gym.env.controller.end_recording.assert_not_called()
        gym.env.close.assert_called_once_with()


class TestOSWorldActionSet:
    def test_defaults_and_stub_methods(self):
        action_set = OSWorldActionSet()
        assert action_set.action_space == "computer_13"
        assert action_set.multiaction is False
        assert action_set.describe() is None
        assert action_set.example_action(abstract=True) is None
        assert action_set.to_python_code("click()") is None

    def test_json_roundtrip(self):
        action_set = OSWorldActionSet(action_space="pyautogui", multiaction=True)
        assert OSWorldActionSet.from_dict(action_set.to_dict()) == action_set

    def test_chatcompletion_returns_raw_tools(self):
        tools = OSWorldActionSet().to_tool_description(api="chatcompletion")
        assert tools is COMPUTER_13_ACTIONS_OAI_CHATCOMPLETION_TOOLS

    def test_default_api_is_openai_responses(self):
        tools = OSWorldActionSet().to_tool_description()
        assert [t["name"] for t in tools] == COMPUTER_13_ACTION_NAMES
        assert all(set(t) == {"type", "name", "description", "parameters"} for t in tools)

    def test_anthropic_format(self):
        tools = OSWorldActionSet().to_tool_description(api="anthropic")
        assert [t["name"] for t in tools] == COMPUTER_13_ACTION_NAMES
        assert all(set(t) == {"name", "description", "input_schema"} for t in tools)

    def test_unsupported_api(self):
        with pytest.raises(ValueError, match="Unsupported API type: gemini"):
            OSWorldActionSet().to_tool_description(api="gemini")

    def test_unsupported_action_space(self):
        with pytest.raises(ValueError, match="Only 'computer_13'"):
            OSWorldActionSet(action_space="pyautogui").to_tool_description()


class TestToolFormatters:
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "a",
                "description": "A",
                "parameters": {"type": "object", "properties": {}},
                "strict": True,
            },
        },
        {"type": "function", "function": {"name": "b", "description": "B", "parameters": {}}},
    ]

    def test_response_api_flattens_and_keeps_strict(self):
        assert format_chat_completion_tools_to_response_api(self.TOOLS) == [
            {
                "type": "function",
                "name": "a",
                "description": "A",
                "parameters": {"type": "object", "properties": {}},
                "strict": True,
            },
            {"type": "function", "name": "b", "description": "B", "parameters": {}},
        ]

    def test_anthropic_uses_input_schema_and_drops_strict(self):
        assert format_chat_completion_tools_to_anthropic(self.TOOLS) == [
            {"name": "a", "description": "A", "input_schema": {"type": "object", "properties": {}}},
            {"name": "b", "description": "B", "input_schema": {}},
        ]

    @pytest.mark.parametrize(
        "fn",
        [format_chat_completion_tools_to_anthropic, format_chat_completion_tools_to_response_api],
    )
    def test_empty_list(self, fn):
        assert fn([]) == []


class TestOsworldEnvArgs:
    def test_defaults(self):
        args = OsworldEnvArgs(task={})
        assert args.provider_name == "docker"
        assert args.action_space == "computer_13"
        assert args.screen_size == (1920, 1080)
        assert args.max_steps == 50
        assert args.task_name is None

    def test_make_env_forwards_all_fields(self, tmp_path):
        args = OsworldEnvArgs(task=_task(), task_name="t", headless=True, max_steps=7)
        with patch.object(osworld, "OsworldGym") as gym_cls:
            env = args.make_env(exp_dir=tmp_path)
        assert env is gym_cls.return_value
        gym_cls.assert_called_once_with(
            task=_task(),
            provider_name="docker",
            region="us-east-1",
            path_to_vm=None,
            snapshot_name="init_state",
            action_space="computer_13",
            cache_dir="cache",
            screen_size=(1920, 1080),
            headless=True,
            require_a11y_tree=True,
            require_terminal=False,
            os_type="Ubuntu",
            enable_proxy=False,
            max_steps=7,
            exp_dir=tmp_path,
        )

    def test_make_env_builds_real_gym_with_fake_desktop(self, desktop_env_cls, tmp_path):
        gym = OsworldEnvArgs(task=_task(), max_steps=4).make_env(exp_dir=tmp_path)
        assert isinstance(gym, OsworldGym)
        assert gym.max_steps == 4 and gym.exp_dir == tmp_path


def _write_test_set(root: Path, tasks_by_domain: dict, settings_file=None):
    (root / "test_all.json").write_text(json.dumps(tasks_by_domain))
    for domain, ids in tasks_by_domain.items():
        (root / "examples" / domain).mkdir(parents=True, exist_ok=True)
        for task_id in ids:
            config = [{"type": "launch", "parameters": {"command": ["app"]}}]
            if settings_file:
                config.append({"type": "chrome", "parameters": {"settings_file": settings_file}})
            config.append({"type": "noop"})
            task = {"id": task_id, "instruction": f"do {task_id}", "config": config}
            (root / "examples" / domain / f"{task_id}.json").write_text(json.dumps(task))


class TestOsworldBenchmark:
    def test_loads_all_domains(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OSWORLD_REPO", "/opt/OSWorld")
        _write_test_set(tmp_path, {"chrome": ["c1", "c2"], "gimp": ["g1"]}, "settings/x.json")
        bench = OsworldBenchmark(test_set_path=str(tmp_path))

        assert [a.task_name for a in bench.env_args_list] == [
            "osworld.c1",
            "osworld.c2",
            "osworld.g1",
        ]
        assert isinstance(bench.high_level_action_set_args, OSWorldActionSet)
        assert bench.high_level_action_set_args.action_space == "computer_13"
        first = bench.env_args_list[0]
        assert first.task["instruction"] == "do c1"
        assert (
            first.task["config"][1]["parameters"]["settings_file"] == "/opt/OSWorld/settings/x.json"
        )
        # every task gets its own copy of the base env args
        assert len({id(a) for a in bench.env_args_list}) == 3
        assert bench.env_args.task == {} and bench.env_args.task_name is None

    def test_domain_filter_and_custom_env_args(self, tmp_path):
        _write_test_set(tmp_path, {"chrome": ["c1"], "gimp": ["g1", "g2"]})
        base = OsworldEnvArgs(task={}, action_space="pyautogui", max_steps=9)
        bench = OsworldBenchmark(test_set_path=str(tmp_path), domain="gimp", env_args=base)
        assert [a.task_name for a in bench.env_args_list] == ["osworld.g1", "osworld.g2"]
        assert all(a.max_steps == 9 for a in bench.env_args_list)
        assert bench.high_level_action_set_args.action_space == "pyautogui"
        assert base.task == {}

    def test_custom_name_and_test_set_file(self, tmp_path):
        _write_test_set(tmp_path, {"os": ["o1"]})
        (tmp_path / "test_small.json").write_text(json.dumps({"os": ["o1"]}))
        bench = OsworldBenchmark(
            name="osw", test_set_path=str(tmp_path), test_set_name="test_small.json"
        )
        assert [a.task_name for a in bench.env_args_list] == ["osw.o1"]

    def test_unknown_domain_raises(self, tmp_path):
        _write_test_set(tmp_path, {"chrome": ["c1"]})
        with pytest.raises(KeyError):
            OsworldBenchmark(test_set_path=str(tmp_path), domain="vlc")

    def test_missing_test_set_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            OsworldBenchmark(test_set_path=str(tmp_path))

    def test_fix_settings_file_path_defaults_and_does_not_mutate(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OSWORLD_REPO", raising=False)
        _write_test_set(tmp_path, {})
        bench = OsworldBenchmark(test_set_path=str(tmp_path))
        task = {
            "config": [
                {"type": "a", "parameters": {"settings_file": "s/p.json"}},
                {"type": "b", "parameters": {"settings_file": ""}},
                {"type": "c", "parameters": {}},
                {"type": "d"},
            ]
        }
        fixed = bench.fix_settings_file_path_in_config(task)
        assert fixed["config"][0]["parameters"]["settings_file"] == "OSWorld/s/p.json"
        assert fixed["config"][1:] == task["config"][1:]
        assert task["config"][0]["parameters"]["settings_file"] == "s/p.json"
