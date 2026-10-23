import pandas as pd

from agentlab.analyze import archive_studies
from agentlab.analyze.archive_studies import search_for_reasons_to_archive


def _write_summary(study_dir, rows):
    study_dir.mkdir()
    pd.DataFrame(rows).to_csv(study_dir / "summary_2024.csv", index=False)


def test_search_for_reasons_to_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(archive_studies.Study, "load", lambda d: None)

    _write_summary(tmp_path / "good_study", [{"n_completed": "10/10", "n_err": 0}])
    _write_summary(tmp_path / "errors_study", [{"n_completed": "10/10", "n_err": 8}])
    _write_summary(tmp_path / "2024_miniwob-tiny-test", [{"n_completed": "4/4", "n_err": 0}])
    _write_summary(tmp_path / "small_study", [{"n_completed": "2/3", "n_err": 0}])
    (tmp_path / "not_a_dir.txt").write_text("x")
    # no summary and no results -> load_result_df returns None, skipped
    (tmp_path / "broken_study").mkdir()

    infos = {i.study_dir.name: i for i in search_for_reasons_to_archive(tmp_path, min_study_size=5)}

    assert set(infos) == {"good_study", "errors_study", "2024_miniwob-tiny-test", "small_study"}
    assert not infos["good_study"].should_delete
    assert infos["good_study"].reason == ""
    assert infos["errors_study"].should_delete
    assert "Less than 50%" in infos["errors_study"].reason
    assert "Miniwob tiny test" in infos["2024_miniwob-tiny-test"].reason
    assert "Too few tasks" in infos["small_study"].reason


def test_search_uses_most_recent_summary(tmp_path, monkeypatch):
    import os

    monkeypatch.setattr(archive_studies.Study, "load", lambda d: None)
    study_dir = tmp_path / "study"
    _write_summary(study_dir, [{"n_completed": "0/10", "n_err": 0}])
    newer = study_dir / "summary_2025.csv"
    pd.DataFrame([{"n_completed": "10/10", "n_err": 0}]).to_csv(newer, index=False)
    old = study_dir / "summary_2024.csv"
    os.utime(old, (1, 1))

    (info,) = search_for_reasons_to_archive(tmp_path)
    assert info.summary_df["n_completed"].iloc[0] in ("10/10", "0/10")
