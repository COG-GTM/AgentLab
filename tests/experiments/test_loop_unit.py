import json
import tempfile
from pathlib import Path

import pytest

from agentlab.experiments.loop import (
    DataclassJSONEncoder,
    EnvArgs,
    ExpArgs,
    ExpResult,
    StepInfo,
    StepTimestamps,
    _flatten_dict,
    _get_env_name,
)


def test_env_args_instantiation():
    """Test EnvArgs can be created with required fields."""
    env_args = EnvArgs(task_name="miniwob.click-test")
    assert env_args.task_name == "miniwob.click-test"
    assert env_args.task_seed is None
    assert env_args.max_steps is None
    assert env_args.headless is True


def test_env_args_with_all_fields():
    """Test EnvArgs with all optional fields."""
    env_args = EnvArgs(
        task_name="miniwob.click-test",
        task_seed=42,
        max_steps=10,
        headless=False,
        record_video=True,
    )
    assert env_args.task_name == "miniwob.click-test"
    assert env_args.task_seed == 42
    assert env_args.max_steps == 10
    assert env_args.headless is False
    assert env_args.record_video is True


def test_env_args_json_serialization():
    """Test EnvArgs can be serialized and deserialized."""
    env_args = EnvArgs(task_name="miniwob.click-test", task_seed=42)
    json_str = env_args.to_json()
    restored = EnvArgs.from_json(json_str)
    assert restored.task_name == env_args.task_name
    assert restored.task_seed == env_args.task_seed


def test_step_info_defaults():
    """Test StepInfo default values."""
    step = StepInfo()
    assert step.step is None
    assert step.obs is None
    assert step.reward == 0
    assert step.terminated is None
    assert step.truncated is None
    assert step.action is None


def test_step_info_with_values():
    """Test StepInfo with explicit values."""
    step = StepInfo(step=0, reward=1.0, terminated=True, truncated=False)
    assert step.step == 0
    assert step.reward == 1.0
    assert step.terminated is True
    assert step.truncated is False


def test_step_info_is_done():
    """Test StepInfo.is_done property."""
    step = StepInfo(terminated=False, truncated=False)
    assert step.is_done is False

    step.terminated = True
    assert step.is_done is True

    step.terminated = False
    step.truncated = True
    assert step.is_done is True


def test_step_timestamps_defaults():
    """Test StepTimestamps default values."""
    ts = StepTimestamps()
    assert ts.env_start == 0
    assert ts.agent_start == 0
    assert ts.agent_stop == 0


def test_step_info_make_stats():
    """Test StepInfo.make_stats with string observations."""
    step = StepInfo(step=0)
    step.obs = {"axtree_txt": "some text content", "screenshot": b"binary"}
    step.agent_info = {"stats": {"custom_stat": 42}}
    step.profiling = StepTimestamps(env_start=1.0, env_stop=2.0, agent_start=1.5, agent_stop=1.8)
    step.make_stats()

    assert "n_token_axtree_txt" in step.stats
    assert step.stats["custom_stat"] == 42
    assert step.stats["step_elapsed"] == 1.0
    assert step.stats["agent_elapsed"] == pytest.approx(0.3)


def test_flatten_dict_simple():
    """Test _flatten_dict with a simple nested dict."""
    d = {"a": {"b": 1, "c": 2}, "d": 3}
    result = _flatten_dict(d)
    assert result == {"a.b": 1, "a.c": 2, "d": 3}


def test_flatten_dict_deeply_nested():
    """Test _flatten_dict with deeply nested dict."""
    d = {"a": {"b": {"c": 1}}}
    result = _flatten_dict(d)
    assert result == {"a.b.c": 1}


def test_flatten_dict_empty():
    """Test _flatten_dict with empty dict."""
    assert _flatten_dict({}) == {}


def test_get_env_name():
    """Test _get_env_name returns correct browsergym env name."""
    assert _get_env_name("miniwob.click-test") == "browsergym/miniwob.click-test"


def test_dataclass_json_encoder():
    """Test DataclassJSONEncoder handles special types."""
    import numpy as np

    encoder = DataclassJSONEncoder()
    assert encoder.default(np.int64(42)) == 42
    assert encoder.default(np.float64(3.14)) == pytest.approx(3.14)
    assert encoder.default(np.array([1, 2, 3])) == [1, 2, 3]


def test_exp_result_status_incomplete():
    """Test ExpResult status for non-existent directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        exp_result = ExpResult(tmp_dir)
        assert exp_result.status == "incomplete"
