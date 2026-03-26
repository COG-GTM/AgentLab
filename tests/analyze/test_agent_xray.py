import pandas as pd
import pytest

agent_xray = pytest.importorskip("agentlab.analyze.agent_xray", reason="langchain not installed")

clean_column_names = agent_xray.clean_column_names
remove_args_from_col = agent_xray.remove_args_from_col
display_table = agent_xray.display_table
EpisodeId = agent_xray.EpisodeId
StepId = agent_xray.StepId
Info = agent_xray.Info


def test_clean_column_names():
    """Test that dots are replaced with dot-newline for word wrap."""
    cols = ["agent.name", "env.task_name"]
    result = clean_column_names(cols)
    assert result == ["agent.\nname", "env.\ntask_name"]


def test_clean_column_names_no_dot():
    """Test column names without dots stay unchanged."""
    cols = ["reward", "n_steps"]
    result = clean_column_names(cols)
    assert result == ["reward", "n_steps"]


def test_remove_args_from_col():
    """Test that _args suffix is removed from columns."""
    df = pd.DataFrame({"agent_args": [1], "env_args": [2], "reward": [3]})
    result = remove_args_from_col(df)
    assert "agent" in result.columns
    assert "env" in result.columns
    assert "reward" in result.columns


def test_display_table():
    """Test that display_table cleans column names."""
    df = pd.DataFrame({"agent.name": ["test"], "env.task": ["click"]})
    result = display_table(df)
    assert "agent.\nname" in result.columns


def test_episode_id_defaults():
    """Test EpisodeId default values."""
    ep = EpisodeId()
    assert ep.agent_id is None
    assert ep.task_name is None
    assert ep.seed is None
    assert ep.row_index is None


def test_step_id_defaults():
    """Test StepId default values."""
    step = StepId()
    assert step.episode_id is None
    assert step.step is None


def test_info_defaults():
    """Test Info default values."""
    info = Info()
    assert info.results_dir is None
    assert info.result_df is None
    assert info.step is None
    assert info.active_tab == "Screenshot"
