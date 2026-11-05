import gzip
import os
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from browsergym.experiments.agent import AgentInfo

from agentlab.analyze import inspect_results
from agentlab.experiments import safe_pickle
from agentlab.experiments.loop import (
    CODE_SANDBOX_DIR_NAME,
    EnvArgs,
    ExpResult,
    StepInfo,
    StepTimestamps,
    yield_all_exp_results,
)
from agentlab.experiments.study import Study
from agentlab.llm.llm_utils import Discussion, HumanMessage, SystemMessage

TEST_STUDY = Path(__file__).parent.parent / "data" / "test_study"


class _WriteMarker:
    """Pickle payload that writes a marker file when unpickled."""

    def __init__(self, marker: Path):
        self.marker = str(marker)

    def __reduce__(self):
        return (exec, (f"open({self.marker!r}, 'w').write('pwned')",))


class _WriteMarkerViaPath:
    """Same, but through a function of an allowlisted module."""

    def __init__(self, marker: Path):
        self.marker = Path(marker)

    def __reduce__(self):
        return (Path.write_text, (self.marker, "pwned"))


def _plant_experiment(exp_dir: Path, marker: Path):
    exp_dir.mkdir(parents=True)
    payload = pickle.dumps(_WriteMarker(marker))
    (exp_dir / "exp_args.pkl").write_bytes(payload)
    with gzip.open(exp_dir / "step_0.pkl.gz", "wb") as f:
        f.write(payload)
    (exp_dir / "summary_info.json").write_text("{}")


@pytest.fixture
def study_dir(tmp_path):
    study_dir = tmp_path / "study"
    shutil.copytree(TEST_STUDY, study_dir)
    return study_dir


def test_sandbox_files_are_not_discovered(study_dir, tmp_path):
    marker = tmp_path / "marker"
    sandbox = study_dir / CODE_SANDBOX_DIR_NAME
    _plant_experiment(sandbox / "x", marker)
    _plant_experiment(sandbox / "a" / "b", marker)
    # experiment dirs symlink to the sandbox (see gaia.init_code_sandbox)
    exp_dirs = sorted(p for p in study_dir.iterdir() if p.name != CODE_SANDBOX_DIR_NAME)
    os.symlink(sandbox, exp_dirs[0] / "code")
    # a nested dir inside a real experiment is not an experiment either
    _plant_experiment(exp_dirs[1] / "nested", marker)

    found = list(yield_all_exp_results(study_dir, progress_fn=None, use_cache=False))
    assert sorted(r.exp_dir for r in found) == exp_dirs

    result_df = inspect_results.load_result_df(study_dir, progress_fn=None)
    assert sorted(Path(d) for d in result_df["exp_dir"]) == exp_dirs
    assert not any(CODE_SANDBOX_DIR_NAME in str(d) for d in result_df["exp_dir"])

    study = Study.load(study_dir)
    study.load_exp_args_list()
    assert len(study.exp_args_list) == len(exp_dirs)

    assert not marker.exists()


def test_symlinked_dirs_are_not_followed(study_dir, tmp_path):
    marker = tmp_path / "marker"
    outside = tmp_path / "outside"
    _plant_experiment(outside / "x", marker)
    os.symlink(outside, study_dir / "link")

    found = list(yield_all_exp_results(study_dir, progress_fn=None, use_cache=False))
    assert all("link" not in r.exp_dir.parts for r in found)
    assert len(found) == 2
    assert not marker.exists()


@pytest.mark.parametrize("payload_cls", [_WriteMarker, _WriteMarkerViaPath])
def test_malicious_result_files_are_rejected(tmp_path, payload_cls):
    marker = tmp_path / "marker"
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    payload = pickle.dumps(payload_cls(marker))
    (exp_dir / "exp_args.pkl").write_bytes(payload)
    with gzip.open(exp_dir / "step_0.pkl.gz", "wb") as f:
        f.write(payload)

    with pytest.raises(safe_pickle.UnsafePickleError):
        ExpResult(exp_dir).exp_args
    with pytest.raises(safe_pickle.UnsafePickleError):
        ExpResult(exp_dir).get_step_info(0)
    assert not marker.exists()


def test_malicious_goal_object_is_rejected(tmp_path):
    marker = tmp_path / "marker"
    step_info = StepInfo(step=0, obs={"goal": "x", "goal_object": [{"type": "text", "text": "x"}]})
    step_info.save_step_info(tmp_path)
    with gzip.open(tmp_path / "goal_object.pkl.gz", "wb") as f:
        pickle.dump(_WriteMarker(marker), f)

    with pytest.raises(safe_pickle.UnsafePickleError):
        ExpResult(tmp_path).get_step_info(0)
    assert not marker.exists()


def test_legit_results_round_trip(tmp_path):
    discussion = Discussion([SystemMessage("system"), HumanMessage("hello")])
    obs = {
        "goal": "click",
        "goal_object": [{"type": "text", "text": "click"}],
        "chat_messages": [{"role": "user", "message": "click"}],
        "axtree_object": {"nodes": [{"nodeId": "1"}]},
        "open_pages_urls": ("http://localhost",),
        "active_page_index": np.array([0]),
        "elapsed_time": np.array([1.5]),
        "screenshot": np.zeros((4, 4, 3), dtype=np.uint8),
        "score": np.float32(0.5),
    }
    step_info = StepInfo(
        step=0,
        obs=dict(obs),
        reward=1.0,
        action="click('1')",
        agent_info=AgentInfo(think="t", chat_messages=discussion, stats={"n": 1}),
        profiling=StepTimestamps(env_start=1.0),
    )
    step_info.save_step_info(tmp_path)

    loaded = ExpResult(tmp_path).get_step_info(0)
    assert loaded.action == "click('1')"
    assert loaded.obs["goal_object"] == obs["goal_object"]
    assert loaded.obs["elapsed_time"].tolist() == [1.5]
    assert loaded.obs["score"] == np.float32(0.5)
    assert loaded.obs["screenshot"].shape == (4, 4, 3)
    assert str(loaded.agent_info.chat_messages) == str(discussion)

    value = {"env_args": EnvArgs(task_name="miniwob.click-test"), "t": datetime.now()}
    with open(tmp_path / "data.pkl", "wb") as f:
        pickle.dump(value, f)
    with open(tmp_path / "data.pkl", "rb") as f:
        assert safe_pickle.load(f) == value

    for exp_dir in TEST_STUDY.iterdir():
        assert ExpResult(exp_dir).exp_args.env_args.task_name == "miniwob.ascending-numbers"


def test_trusted_modules_env_var(tmp_path, monkeypatch):
    path = tmp_path / "data.pkl"
    with open(path, "wb") as f:
        pickle.dump(_Custom(), f)

    with open(path, "rb") as f, pytest.raises(safe_pickle.UnsafePickleError):
        safe_pickle.load(f)

    monkeypatch.setenv(safe_pickle.TRUSTED_MODULES_ENV_VAR, _Custom.__module__.split(".")[0])
    with open(path, "rb") as f:
        assert isinstance(safe_pickle.load(f), _Custom)


class _Custom:
    pass
