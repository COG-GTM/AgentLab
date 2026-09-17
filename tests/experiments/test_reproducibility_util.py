import csv
import json

import pandas as pd
import pytest
from bgym import DEFAULT_BENCHMARKS

from agentlab.experiments import reproducibility_util


@pytest.mark.parametrize(
    "benchmark_name",
    ["miniwob", "workarena_l1", "webarena", "visualwebarena"],
)
def test_get_reproducibility_info(benchmark_name):

    benchmark = DEFAULT_BENCHMARKS[benchmark_name]()

    info = reproducibility_util.get_reproducibility_info(
        "test_agent", benchmark, "test_id", ignore_changes=True
    )

    print("reproducibility info:")
    print(json.dumps(info, indent=4))

    # assert keys in info
    assert "git_user" in info
    assert "benchmark" in info
    assert "benchmark_version" in info
    assert "agentlab_version" in info
    assert "agentlab_git_hash" in info
    assert "agentlab__local_modifications" in info
    assert "browsergym_version" in info
    assert "browsergym_git_hash" in info
    assert "browsergym__local_modifications" in info


def _make_info(agent_names=("agent_a",), **overrides):
    info = {
        "agent_names": list(agent_names),
        "benchmark": "miniwob",
        "benchmark_version": "0.1.0",
        "date": "2024-01-01_00-00-00",
        "avg_reward": None,
        "std_err": None,
        "n_err": None,
        "n_completed": None,
        "comment": None,
        "agentlab_version": "0.3.1",
        "agentlab_git_hash": "abc123",
        "agentlab__local_modifications": "",
    }
    info.update(overrides)
    return info


def _make_report_df(agent_names=("agent_a",), n_err=0, n_completed="3/3"):
    return pd.DataFrame(
        {
            "avg_reward": [0.5] * len(agent_names),
            "std_err": [0.1] * len(agent_names),
            "n_err": [n_err] * len(agent_names),
            "n_completed": [n_completed] * len(agent_names),
        },
        index=pd.Index(list(agent_names), name="agent.agent_name"),
    )


def test_assert_compatible_ignores_volatile_keys():
    info = _make_info()
    old_info = _make_info(
        date="2023-05-05_12-00-00", avg_reward=0.9, std_err=0.2, n_err=4, n_completed="2/3"
    )

    reproducibility_util.assert_compatible(info, old_info)


def test_assert_compatible_raises_on_changed_key():
    info = _make_info()
    old_info = _make_info(benchmark_version="0.2.0")

    with pytest.raises(ValueError, match="benchmark_version"):
        reproducibility_util.assert_compatible(info, old_info)


def test_assert_compatible_warns_when_not_strict(caplog):
    info = _make_info()
    old_info = _make_info(agentlab_git_hash="def456")

    reproducibility_util.assert_compatible(info, old_info, raise_if_incompatible=False)

    assert "agentlab_git_hash" in caplog.text


def test_verify_report_rejects_unknown_agent_names():
    report_df = _make_report_df(agent_names=("agent_a",))

    with pytest.raises(ValueError, match="do not match"):
        reproducibility_util._verify_report(report_df, ["agent_b"])


def test_verify_report_rejects_duplicate_agent_names():
    report_df = _make_report_df(agent_names=("agent_a", "agent_a"))

    with pytest.raises(ValueError, match="Duplicate agent names"):
        reproducibility_util._verify_report(report_df, ["agent_a", "agent_a"])


@pytest.mark.parametrize(
    "n_err,n_completed,match",
    [(2, "3/3", "2 errors"), (0, "2/3", "completed tasks")],
)
def test_verify_report_raises_on_incomplete_study(n_err, n_completed, match):
    report_df = _make_report_df(n_err=n_err, n_completed=n_completed)

    with pytest.raises(ValueError, match=match):
        reproducibility_util._verify_report(report_df, ["agent_a"])


def test_verify_report_warns_when_not_strict(caplog):
    report_df = _make_report_df(n_err=2, n_completed="2/3")

    verified_df = reproducibility_util._verify_report(
        report_df, ["agent_a"], strict_reproducibility=False
    )

    assert verified_df.index.name == "agent.agent_name"
    assert "2 errors" in caplog.text
    assert "completed tasks" in caplog.text


def test_append_to_journal_creates_then_appends(tmp_path):
    journal_path = tmp_path / "journal.csv"

    reproducibility_util.append_to_journal(
        _make_info(["agent_a"]), _make_report_df(["agent_a"]), journal_path=journal_path
    )

    rows = list(csv.reader(journal_path.read_text().splitlines()))
    headers = rows[0]
    assert "agent_name" in headers and "agent_names" not in headers
    assert len(rows) == 2
    first_row = dict(zip(headers, rows[1]))
    assert first_row["agent_name"] == "agent_a"
    assert first_row["avg_reward"] == "0.5"
    assert first_row["n_completed"] == "3/3"

    reproducibility_util.append_to_journal(
        _make_info(["agent_a", "agent_b"]),
        _make_report_df(["agent_a", "agent_b"]),
        journal_path=journal_path,
    )

    rows = list(csv.reader(journal_path.read_text().splitlines()))
    assert rows[0] == headers  # headers are reused, not rewritten
    assert len(rows) == 4
    assert [row[headers.index("agent_name")] for row in rows[1:]] == [
        "agent_a",
        "agent_a",
        "agent_b",
    ]


def test_append_to_journal_rejects_agent_count_mismatch(tmp_path):
    with pytest.raises(ValueError, match="Mismatch between the number of agents"):
        reproducibility_util.append_to_journal(
            _make_info(["agent_a", "agent_b"]),
            _make_report_df(["agent_a"]),
            journal_path=tmp_path / "journal.csv",
        )


def test_append_to_journal_propagates_verification_error(tmp_path):
    journal_path = tmp_path / "journal.csv"

    with pytest.raises(ValueError, match="1 errors"):
        reproducibility_util.append_to_journal(
            _make_info(["agent_a"]),
            _make_report_df(["agent_a"], n_err=1),
            journal_path=journal_path,
        )

    assert not journal_path.exists()

    reproducibility_util.append_to_journal(
        _make_info(["agent_a"]),
        _make_report_df(["agent_a"], n_err=1),
        journal_path=journal_path,
        strict_reproducibility=False,
    )

    assert len(journal_path.read_text().strip().splitlines()) == 2


if __name__ == "__main__":
    test_get_reproducibility_info("miniwob")
