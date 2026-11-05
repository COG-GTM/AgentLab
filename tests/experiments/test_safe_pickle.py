import gzip
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

from agentlab.agents.generic_agent.agent_configs import FLAGS_GPT_4o
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.analyze.archive_studies import search_for_reasons_to_archive
from agentlab.experiments import safe_pickle
from agentlab.experiments.loop import ExpResult
from agentlab.experiments.study import Study, make_study
from agentlab.llm.chat_api import CheatMiniWoBLLMArgs

TEST_STUDY_DIR = Path(__file__).parent.parent / "data" / "test_study"


class _MarkerPayload:
    """Pickles into a call that creates a marker file when unpickled."""

    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return (exec, (f"open({str(self.marker)!r}, 'w').write('pwned')",))


def _global_ref(module: str, name: str) -> bytes:
    return b"\x80\x04" + b"c" + module.encode() + b"\n" + name.encode() + b"\n" + b"."


def _write_study_pickle(study_dir: Path, obj) -> None:
    study_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(study_dir / "study.pkl.gz", "wb") as f:
        pickle.dump(obj, f)


def test_study_load_rejects_malicious_study_pickle(tmp_path):
    marker = tmp_path / "marker"
    study_dir = tmp_path / "results" / "2026-01-01_00-00-00_evil"
    _write_study_pickle(study_dir, _MarkerPayload(marker))

    with pytest.raises(safe_pickle.UnsafePickleError):
        Study.load(study_dir)
    with pytest.raises(safe_pickle.UnsafePickleError):
        Study.load_most_recent(root_dir=tmp_path / "results")
    assert not marker.exists()


def test_study_load_legacy_branch_rejects_malicious_exp_args(tmp_path):
    marker = tmp_path / "marker"
    exp_dir = tmp_path / "legacy_study" / "exp_0"
    exp_dir.mkdir(parents=True)
    with open(exp_dir / "exp_args.pkl", "wb") as f:
        pickle.dump(_MarkerPayload(marker), f)

    with pytest.raises(safe_pickle.UnsafePickleError):
        Study.load(tmp_path / "legacy_study")
    with pytest.raises(safe_pickle.UnsafePickleError):
        ExpResult(exp_dir).exp_args
    assert not marker.exists()


def test_archive_studies_does_not_execute_planted_pickle(tmp_path):
    marker = tmp_path / "marker"
    results = tmp_path / "results"
    _write_study_pickle(results / "evil_study", _MarkerPayload(marker))
    exp_dir = results / "evil_legacy_study" / "exp_0"
    exp_dir.mkdir(parents=True)
    with open(exp_dir / "exp_args.pkl", "wb") as f:
        pickle.dump(_MarkerPayload(marker), f)

    infos = search_for_reasons_to_archive(results)

    assert not marker.exists()
    assert all(info.study is None for info in infos)


@pytest.mark.parametrize(
    "module,name",
    [
        ("os", "system"),
        ("builtins", "eval"),
        ("builtins", "getattr"),
        ("subprocess", "Popen"),
        ("pickle", "loads"),
        # gadgets reachable through trusted packages
        ("agentlab.experiments.study", "os.system"),
        ("agentlab.experiments.study", "pickle"),
        ("agentlab.experiments.study", "make_study"),
        ("agentlab.experiments.loop", "ExpResult"),
        ("agentlab.experiments.loop", "np"),
        ("agentlab.experiments.study", "Path"),
        ("agentlab_evil", "Thing"),
    ],
)
def test_rejects_non_allowlisted_globals(module, name):
    with pytest.raises(safe_pickle.UnsafePickleError):
        safe_pickle.loads(_global_ref(module, name))


def test_study_round_trip(tmp_path):
    agent_args = GenericAgentArgs(chat_model_args=CheatMiniWoBLLMArgs(), flags=FLAGS_GPT_4o)
    study = make_study(agent_args, benchmark="miniwob_tiny_test", ignore_dependencies=True)
    study.save(exp_root=tmp_path)

    loaded = Study.load(study.dir)

    assert isinstance(loaded, Study)
    assert loaded.dir == study.dir
    assert loaded.uuid == study.uuid
    assert loaded.benchmark.name == study.benchmark.name
    assert loaded.agent_args == study.agent_args
    assert [e.env_args for e in loaded.exp_args_list] == [e.env_args for e in study.exp_args_list]


def test_legacy_study_dir_still_loads(tmp_path):
    study_dir = tmp_path / "test_study"
    shutil.copytree(TEST_STUDY_DIR, study_dir)

    study = Study.load(study_dir)

    assert study.benchmark.name == "miniwob"
    assert isinstance(study.agent_args, GenericAgentArgs)


@dataclass
class _UserAgentArgs:
    name: str = "mine"


def test_extra_trusted_packages_from_env(monkeypatch):
    data = pickle.dumps(_UserAgentArgs())
    with pytest.raises(safe_pickle.UnsafePickleError):
        safe_pickle.loads(data)

    monkeypatch.setenv(safe_pickle.TRUSTED_PACKAGES_ENV_VAR, f"other_pkg, {__name__}")
    assert safe_pickle.loads(data) == _UserAgentArgs()
