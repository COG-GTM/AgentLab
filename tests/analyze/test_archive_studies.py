import pandas as pd
import pytest

from agentlab.analyze.archive_studies import StudyInfo, search_for_reasons_to_archive


def _make_study(result_dir, name, rows):
    study_dir = result_dir / name
    study_dir.mkdir()
    pd.DataFrame(rows).to_csv(study_dir / "summary_df.csv", index=False)
    return study_dir


@pytest.fixture
def result_dir(tmp_path):
    _make_study(tmp_path, "good_study", [{"n_completed": "10/10", "n_err": 0}])
    _make_study(tmp_path, "miniwob-tiny-test_study", [{"n_completed": "10/10", "n_err": 0}])
    _make_study(tmp_path, "error_study", [{"n_completed": "10/10", "n_err": 8}])
    _make_study(tmp_path, "small_study", [{"n_completed": "2/2", "n_err": 0}])
    _make_study(tmp_path, "nothing_completed", [{"n_completed": "0/4", "n_err": 0}])
    _make_study(
        tmp_path,
        "multi_agent",
        [{"n_completed": "4/5", "n_err": 1}, {"n_completed": "5/5", "n_err": 0}],
    )
    (tmp_path / "not_a_dir.txt").write_text("ignored")
    (tmp_path / "no_summary_study").mkdir()
    return tmp_path


def test_search_for_reasons_to_archive(result_dir):
    infos = search_for_reasons_to_archive(result_dir, min_study_size=5)
    by_name = {info.study_dir.name: info for info in infos}

    # files and unreadable studies are skipped
    assert "not_a_dir.txt" not in by_name
    assert "no_summary_study" not in by_name
    assert len(by_name) == 6
    assert all(isinstance(info, StudyInfo) for info in infos)

    assert not by_name["good_study"].should_delete
    assert by_name["good_study"].reason == ""
    assert by_name["good_study"].study is None  # no study.pkl.gz in the folder

    assert by_name["miniwob-tiny-test_study"].should_delete
    assert "Miniwob tiny test" in by_name["miniwob-tiny-test_study"].reason

    assert by_name["error_study"].should_delete
    assert "Less than 50%" in by_name["error_study"].reason

    assert by_name["small_study"].should_delete
    assert "Too few tasks" in by_name["small_study"].reason

    assert by_name["nothing_completed"].should_delete
    assert "No tasks completed" in by_name["nothing_completed"].reason

    # rows are aggregated: 9/10 completed, 1 error -> kept
    assert not by_name["multi_agent"].should_delete


def test_search_uses_most_recent_summary(tmp_path):
    import time

    study_dir = _make_study(tmp_path, "study", [{"n_completed": "0/10", "n_err": 0}])
    newer = study_dir / "summary_new.csv"
    time.sleep(0.05)
    pd.DataFrame([{"n_completed": "10/10", "n_err": 0}]).to_csv(newer, index=False)

    (info,) = search_for_reasons_to_archive(tmp_path)
    assert info.summary_df["n_completed"].tolist() == ["10/10"]
    assert not info.should_delete


def test_min_study_size_default_keeps_small_study(tmp_path):
    _make_study(tmp_path, "small", [{"n_completed": "2/2", "n_err": 0}])
    (info,) = search_for_reasons_to_archive(tmp_path)
    assert not info.should_delete
