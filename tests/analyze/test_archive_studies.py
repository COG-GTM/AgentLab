from agentlab.analyze.archive_studies import StudyInfo


def test_study_info_defaults():
    """Test StudyInfo default values."""
    info = StudyInfo(study_dir=None, study=None, summary_df=None)
    assert info.should_delete is False
    assert info.reason == ""


def test_study_info_with_reason():
    """Test StudyInfo with a delete reason."""
    info = StudyInfo(
        study_dir=None,
        study=None,
        summary_df=None,
        should_delete=True,
        reason="Test reason",
    )
    assert info.should_delete is True
    assert info.reason == "Test reason"
