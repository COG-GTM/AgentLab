"""Unit tests for agentlab.analyze.archive_studies on synthetic result dirs."""

import gzip
import json
import os
import pickle
import runpy
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from agentlab.analyze import archive_studies
from agentlab.analyze.archive_studies import StudyInfo, search_for_reasons_to_archive
from agentlab.experiments import exp_utils, loop
from agentlab.experiments.loop import AbstractAgentArgs, EnvArgs, ExpArgs


@dataclass
class FakeAgentArgs(AbstractAgentArgs):
    def make_agent(self):
        raise NotImplementedError


@pytest.fixture(autouse=True)
def isolated_exp_result_cache(monkeypatch):
    monkeypatch.setattr(loop, "EXP_RESULT_CACHE", {})


def write_summary(study_dir: Path, rows, name="summary.csv") -> Path:
    study_dir.mkdir(parents=True, exist_ok=True)
    path = study_dir / name
    pd.DataFrame(rows, columns=["agent", "avg_reward", "n_completed", "n_err"]).to_csv(
        path, index=False
    )
    return path


def write_exp(study_dir: Path, name: str, task_name: str, err_msg=None, reward=1.0):
    exp_dir = study_dir / name
    exp_dir.mkdir(parents=True)
    exp_args = ExpArgs(
        agent_args=FakeAgentArgs(agent_name="Fake"), env_args=EnvArgs(task_name=task_name)
    )
    with open(exp_dir / "exp_args.pkl", "wb") as f:
        pickle.dump(exp_args, f)
    summary = {
        "n_steps": 2,
        "cum_reward": 0.0 if err_msg else reward,
        "err_msg": err_msg,
        "stack_trace": None,
        "terminated": err_msg is None,
        "truncated": False,
    }
    (exp_dir / "summary_info.json").write_text(json.dumps(summary))


def by_name(infos):
    return {info.study_dir.name: info for info in infos}


def test_study_info_defaults(tmp_path):
    info = StudyInfo(study_dir=tmp_path, study=None, summary_df=pd.DataFrame())
    assert info.should_delete is False
    assert info.reason == ""


def test_healthy_study_is_kept_and_non_dirs_skipped(tmp_path):
    write_summary(tmp_path / "good", [["A", 0.5, "10/10", 1], ["B", 0.4, "8/10", 0]])
    (tmp_path / "stray_file.txt").write_text("not a study")
    infos = search_for_reasons_to_archive(tmp_path)
    assert [i.study_dir.name for i in infos] == ["good"]
    info = infos[0]
    assert info.should_delete is False
    assert info.reason == ""
    assert info.study is None  # no study.pkl.gz and no experiments to rebuild from
    assert len(info.summary_df) == 2


@pytest.mark.parametrize(
    "rows, name, min_size, expected_reasons",
    [
        ([["A", 0.0, "5/10", 0]], "2024_miniwob-tiny-test_x", 0, ["Miniwob tiny test"]),
        ([["A", 0.0, "0/10", 0]], "s", 0, ["No tasks completed", "Less than 50% tasks finished"]),
        (
            [["A", 0.0, "10/10", 6]],
            "s",
            0,
            ["Less than 50% tasks finished, n_err: 6, n_total: 10, n_finished: 4, n_completed: 10"],
        ),
        (
            [["A", 1.0, "3/3", 0], ["B", 1.0, "2/2", 0]],
            "s",
            5,
            ["Too few tasks. n_total (5) <= min_study_size (5)"],
        ),
    ],
)
def test_reasons_to_archive(tmp_path, rows, name, min_size, expected_reasons):
    write_summary(tmp_path / name, rows)
    (info,) = search_for_reasons_to_archive(tmp_path, min_study_size=min_size)
    assert info.should_delete is True
    for reason in expected_reasons:
        assert reason in info.reason
    assert info.reason.count("\n") == len(expected_reasons)


def test_exactly_half_finished_is_kept(tmp_path):
    write_summary(tmp_path / "s", [["A", 0.0, "10/10", 5]])
    (info,) = search_for_reasons_to_archive(tmp_path, min_study_size=9)
    assert info.should_delete is False


def test_uses_most_recent_summary_csv(tmp_path, monkeypatch):
    study = tmp_path / "s"
    old = write_summary(study, [["A", 0.0, "0/10", 0]], name="summary_old.csv")
    new = write_summary(study, [["A", 1.0, "10/10", 0]], name="summary_new.csv")
    ctimes = {str(old): 100.0, str(new): 200.0}
    monkeypatch.setattr(os.path, "getctime", lambda p: ctimes[str(p)])
    (info,) = search_for_reasons_to_archive(tmp_path)
    assert info.summary_df["n_completed"].tolist() == ["10/10"]
    assert info.should_delete is False


def test_loads_pickled_study(tmp_path):
    study_dir = tmp_path / "s"
    write_summary(study_dir, [["A", 1.0, "4/4", 0]])
    with gzip.open(study_dir / "study.pkl.gz", "wb") as f:
        pickle.dump(SimpleNamespace(name="my-study"), f)
    (info,) = search_for_reasons_to_archive(tmp_path)
    assert info.study.name == "my-study"
    assert info.study.dir == study_dir


def test_summary_computed_from_experiments_when_no_csv(tmp_path):
    study_dir = tmp_path / "s"
    write_exp(study_dir, "e0", "miniwob.a")
    write_exp(study_dir, "e1", "miniwob.b", err_msg="boom")
    write_exp(study_dir, "e2", "miniwob.c", reward=0.0)
    (info,) = search_for_reasons_to_archive(tmp_path)
    assert info.summary_df["n_completed"].tolist() == ["3/3"]
    assert info.summary_df["n_err"].tolist() == [1]
    assert info.should_delete is False


def test_study_without_results_is_skipped(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    assert search_for_reasons_to_archive(tmp_path) == []
    assert "Error processing" in capsys.readouterr().out


@pytest.mark.xfail(
    strict=True,
    raises=ZeroDivisionError,
    reason="BUG: empty summary (n_total == 0) raises ZeroDivisionError instead of being flagged",
)
def test_empty_summary_is_flagged(tmp_path):
    write_summary(tmp_path / "s", [])
    (info,) = search_for_reasons_to_archive(tmp_path)
    assert info.should_delete is True
    assert "Empty summary DataFrame" in info.reason
    assert "No tasks" in info.reason


def test_main_reports_without_moving(tmp_path, monkeypatch, capsys):
    results_dir = tmp_path / "agentlab_results"
    write_summary(results_dir / "bad_study", [["A", 0.0, "1/10", 1]])
    write_summary(results_dir / "good_study", [["A", 1.0, "10/10", 0]] * 2)
    monkeypatch.setattr(exp_utils, "RESULTS_DIR", results_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        runpy.run_module(archive_studies.__name__, run_name="__main__")
    out = capsys.readouterr().out
    assert "Study: bad_study" in out
    assert "good_study" not in out
    assert (tmp_path / "archived_agentlab_results").is_dir()
    assert (results_dir / "bad_study").is_dir()  # archive_dir is None -> nothing moved
