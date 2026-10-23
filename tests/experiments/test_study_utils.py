from dataclasses import asdict

import bgym
import pytest

from agentlab.experiments import study as study_mod
from agentlab.experiments.loop import EnvArgs
from agentlab.experiments.study import (
    _convert_env_args,
    _make_study_name,
    get_most_recent_study,
    set_demo_mode,
)


def test_make_study_name_single_agent_and_benchmark():
    assert _make_study_name(["agent"], ["miniwob"], "v1") == "agent-on-miniwob-v1"


def test_make_study_name_multiple():
    name = _make_study_name(["a", "b", "a"], ["x", "y", "z"])
    assert name == "2-agents-on-3-benchmarks"


def test_get_most_recent_study(tmp_path):
    for name in [
        "2024-01-01_10-00-00_old",
        "2024-06-01_10-00-00_newer_foo",
        "2024-03-01_10-00-00_foo",
        "_2025-01-01_10-00-00_hidden",
        "not_a_date",
    ]:
        (tmp_path / name).mkdir()
    (tmp_path / "2026-01-01_10-00-00_file").write_text("x")

    assert get_most_recent_study(tmp_path).name == "2024-06-01_10-00-00_newer_foo"
    assert get_most_recent_study(tmp_path, contains="old").name == "2024-01-01_10-00-00_old"
    assert get_most_recent_study(tmp_path, contains="missing") is None


def test_get_most_recent_study_defaults_to_results_dir(tmp_path, monkeypatch):
    (tmp_path / "2024-01-01_10-00-00_x").mkdir()
    monkeypatch.setattr(study_mod, "RESULTS_DIR", tmp_path)
    assert get_most_recent_study().name == "2024-01-01_10-00-00_x"


def test_set_demo_mode():
    env_args = EnvArgs(task_name="miniwob.click-test")
    set_demo_mode([env_args])
    assert env_args.viewport == {"width": 1280, "height": 720}
    assert env_args.record_video is True
    assert env_args.wait_for_user_message is False
    assert env_args.slow_mo == 1000


def test_convert_env_args():
    new = EnvArgs(task_name="miniwob.a")
    old = bgym.EnvArgs(task_name="miniwob.b", task_seed=3)
    converted = _convert_env_args([new, old])
    assert converted[0] is new
    assert isinstance(converted[1], EnvArgs)
    assert asdict(converted[1])["task_seed"] == 3

    with pytest.raises(TypeError, match="Unexpected type"):
        _convert_env_args(["nope"])
