import logging

import pandas as pd
import pytest

from agentlab.experiments import reproducibility_util as ru


def _report(rows):
    return pd.DataFrame(rows).set_index("agent.agent_name")


GOOD_ROWS = [
    {"agent.agent_name": "a1", "avg_reward": 0.5, "std_err": 0.1, "n_err": 0, "n_completed": "4/4"},
    {"agent.agent_name": "a2", "avg_reward": 0.7, "std_err": 0.2, "n_err": 0, "n_completed": "4/4"},
]


def _info():
    return {
        "git_user": "me",
        "agent_names": ["a1", "a2"],
        "benchmark": "miniwob",
        "date": "today",
    }


def test_assert_compatible_ignores_result_keys():
    info = {"benchmark": "b", "date": "1", "avg_reward": 0.1}
    old = {"benchmark": "b", "date": "2", "avg_reward": 0.9}
    ru.assert_compatible(info, old)


def test_assert_compatible_raises_or_warns(caplog):
    info, old = {"benchmark": "b1"}, {"benchmark": "b2"}
    with pytest.raises(ValueError, match="Key benchmark has changed"):
        ru.assert_compatible(info, old)
    with caplog.at_level(logging.WARNING):
        ru.assert_compatible(info, old, raise_if_incompatible=False)
    assert "Key benchmark has changed" in caplog.text


def test_verify_report_ok():
    df = ru._verify_report(_report(GOOD_ROWS), ["a1", "a2"])
    assert list(df.index) == ["a1", "a2"]


def test_verify_report_agent_name_mismatch():
    with pytest.raises(ValueError, match="do not match"):
        ru._verify_report(_report(GOOD_ROWS), ["a1"])


def test_verify_report_duplicate_agent_names():
    with pytest.raises(ValueError, match="Duplicate"):
        ru._verify_report(_report(GOOD_ROWS), ["a1", "a2", "a2"])


@pytest.mark.parametrize(
    "override, match",
    [({"n_err": 1}, "has 1 errors"), ({"n_completed": "3/4"}, "3 completed tasks out of 4")],
)
def test_verify_report_incomplete(override, match, caplog):
    rows = [dict(GOOD_ROWS[0], **override)]
    with pytest.raises(ValueError, match=match):
        ru._verify_report(_report(rows), ["a1"])
    with caplog.at_level(logging.WARNING):
        ru._verify_report(_report(rows), ["a1"], strict_reproducibility=False)
    assert "a1" in caplog.text


def test_get_csv_headers(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    assert ru._get_csv_headers(empty) is None
    f = tmp_path / "f.csv"
    f.write_text("a,b\n1,2\n")
    assert ru._get_csv_headers(f) == ["a", "b"]


def test_add_result_to_info():
    info = {"agent_name": "a2"}
    ru._add_result_to_info(info, _report(GOOD_ROWS))
    assert info == {
        "agent_name": "a2",
        "avg_reward": 0.7,
        "std_err": 0.2,
        "n_err": 0,
        "n_completed": "4/4",
    }


def test_append_to_journal_creates_and_appends(tmp_path):
    journal = tmp_path / "journal.csv"
    ru.append_to_journal(_info(), _report(GOOD_ROWS), journal_path=journal)
    ru.append_to_journal(_info(), _report(GOOD_ROWS), journal_path=journal)

    df = pd.read_csv(journal)
    assert list(df.columns) == ["git_user", "agent_name", "benchmark", "date"]
    assert list(df["agent_name"]) == ["a1", "a2", "a1", "a2"]


def test_append_to_journal_mismatched_agents(tmp_path):
    with pytest.raises(ValueError, match="Mismatch"):
        ru.append_to_journal(_info(), _report(GOOD_ROWS[:1]), journal_path=tmp_path / "j.csv")


def test_append_to_journal_default_path_without_git(tmp_path, monkeypatch):
    def no_repo(module):
        raise ru.InvalidGitRepositoryError()

    monkeypatch.setattr(ru, "_get_repo", no_repo)
    monkeypatch.setattr(ru, "RESULTS_DIR", tmp_path)
    ru.append_to_journal(_info(), _report(GOOD_ROWS))
    assert (tmp_path / "reproducibility_journal.csv").exists()
