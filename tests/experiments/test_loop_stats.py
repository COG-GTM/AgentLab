import json

import pytest

from agentlab.experiments.loop import (
    ExpResult,
    StepInfo,
    _aggregate_episode_stats,
    _extract_err_msg,
)


def test_aggregate_episode_stats_sum_and_max():
    episode_info = [
        StepInfo(stats={"n_token": 10, "elapsed": 1.5}),
        StepInfo(stats={"n_token": 5, "elapsed": 2.5}),
    ]

    stats = _aggregate_episode_stats(episode_info)

    assert stats["cum_steps"] == 2
    assert stats["cum_n_token"] == 15
    assert stats["max_n_token"] == 10
    assert stats["cum_elapsed"] == pytest.approx(4.0)
    assert stats["max_elapsed"] == pytest.approx(2.5)


def test_aggregate_episode_stats_converts_numpy_scalars_to_python():
    episode_info = [StepInfo(stats={"n_token": 3}), StepInfo(stats={"n_token": 4})]

    stats = _aggregate_episode_stats(episode_info)

    for key, val in stats.items():
        assert type(val) in (int, float), f"{key} is not a builtin scalar: {type(val)}"


def test_aggregate_episode_stats_ignores_none_values():
    episode_info = [
        StepInfo(stats={"n_token": None}),
        StepInfo(stats={"n_token": 7}),
        StepInfo(stats=None),
    ]

    stats = _aggregate_episode_stats(episode_info)

    assert stats["cum_steps"] == 3
    assert stats["cum_n_token"] == 7
    assert stats["max_n_token"] == 7


def test_aggregate_episode_stats_all_none_becomes_none():
    episode_info = [StepInfo(stats={"n_token": None}), StepInfo(stats={"n_token": None})]

    stats = _aggregate_episode_stats(episode_info)

    assert stats["cum_n_token"] == 0
    assert stats["max_n_token"] is None


def test_aggregate_episode_stats_empty_episode():
    assert _aggregate_episode_stats([]) == {"cum_steps": 0}


def test_extract_err_msg_returns_last_error():
    episode_info = [
        StepInfo(agent_info={"err_msg": "first", "stack_trace": "trace 1"}),
        StepInfo(agent_info={}),
        StepInfo(agent_info={"err_msg": "last", "stack_trace": "trace 2"}),
        StepInfo(agent_info=None),
    ]

    assert _extract_err_msg(episode_info) == ("last", "trace 2")


def test_extract_err_msg_without_stack_trace():
    assert _extract_err_msg([StepInfo(agent_info={"err_msg": "boom"})]) == ("boom", None)


def test_extract_err_msg_no_error():
    episode_info = [StepInfo(agent_info={}), StepInfo(agent_info=None)]

    assert _extract_err_msg(episode_info) == (None, None)


def _make_exp_result(tmp_path, summary_info: dict | None) -> ExpResult:
    if summary_info is not None:
        with open(tmp_path / "summary_info.json", "w") as f:
            json.dump(summary_info, f)
    return ExpResult(tmp_path)


def test_status_incomplete_when_no_summary_info(tmp_path):
    assert _make_exp_result(tmp_path, None).status == "incomplete"


def test_status_incomplete_when_summary_info_empty(tmp_path):
    (tmp_path / "summary_info.json").touch()

    assert ExpResult(tmp_path).status == "incomplete"


def test_status_error_when_err_msg(tmp_path):
    summary_info = {"err_msg": "boom", "terminated": True, "truncated": False}

    assert _make_exp_result(tmp_path, summary_info).status == "error"


@pytest.mark.parametrize("flag", ["terminated", "truncated"])
def test_status_done(tmp_path, flag):
    summary_info = {"err_msg": None, "terminated": False, "truncated": False}
    summary_info[flag] = True

    assert _make_exp_result(tmp_path, summary_info).status == "done"


def test_status_incomplete_when_not_finished(tmp_path):
    summary_info = {"err_msg": None, "terminated": False, "truncated": False}

    assert _make_exp_result(tmp_path, summary_info).status == "incomplete"
