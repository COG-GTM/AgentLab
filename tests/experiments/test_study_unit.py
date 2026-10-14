import gzip
import logging
import pickle
import queue
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import bgym
import pandas as pd
import pytest

from agentlab.agents.generic_agent.agent_configs import FLAGS_GPT_4o
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.analyze import inspect_results
from agentlab.benchmarks.abstract_env import AbstractEnvArgs
from agentlab.experiments import study as study_mod
from agentlab.experiments.loop import EnvArgs, ExpArgs
from agentlab.experiments.multi_server import BaseServer
from agentlab.experiments.study import (
    AbstractStudy,
    ParallelStudies,
    ParallelStudies_alt,
    SequentialStudies,
    Study,
    _convert_env_args,
    _init_worker,
    _make_study_name,
    _run_study,
    get_most_recent_study,
    make_study,
    set_demo_mode,
)
from agentlab.llm.chat_api import CheatMiniWoBLLMArgs

BENCH = "miniwob_tiny_test"


def _agent(suffix=""):
    agent = GenericAgentArgs(chat_model_args=CheatMiniWoBLLMArgs(), flags=deepcopy(FLAGS_GPT_4o))
    agent.agent_name = agent.agent_name + suffix
    return agent


def _benchmark():
    return bgym.DEFAULT_BENCHMARKS[BENCH]()


@pytest.fixture
def study(tmp_path):
    s = Study([_agent()], _benchmark())
    s.dir = tmp_path / "study_dir"
    return s


# ---------------------------------------------------------------------------
# make_study
# ---------------------------------------------------------------------------


def test_make_study_single_agent_from_benchmark_name():
    s = make_study(_agent(), BENCH.upper(), suffix="sfx", comment="c", ignore_dependencies=True)
    assert isinstance(s, Study)
    assert s.benchmark.name == BENCH
    assert (s.suffix, s.comment, s.ignore_dependencies) == ("sfx", "c", True)
    assert len(s.exp_args_list) == 4


def test_make_study_multiple_agents_non_webarena_single_study():
    s = make_study([_agent("_0"), _agent("_1")], _benchmark())
    assert isinstance(s, Study)
    assert len(s.exp_args_list) == 8
    assert [e.order for e in s.exp_args_list] == list(range(8))


class _RecordingStudy:
    def __init__(self, agent_args, benchmark, **kwargs):
        self.agent_args = agent_args
        self.benchmark = benchmark
        self.kwargs = kwargs


@pytest.mark.parametrize(
    "parallel_servers, expected_cls",
    [(None, SequentialStudies), (2, ParallelStudies)],
)
def test_make_study_webarena_splits_agents(monkeypatch, parallel_servers, expected_cls):
    monkeypatch.setattr(study_mod, "Study", _RecordingStudy)
    agents = ["agent_a", "agent_b"]
    bench = SimpleNamespace(name="webarena_fake")
    result = make_study(agents, bench, suffix="s", parallel_servers=parallel_servers)
    assert type(result) is expected_cls
    assert [st.agent_args for st in result.studies] == [["agent_a"], ["agent_b"]]
    assert all(st.kwargs["suffix"] == "s" for st in result.studies)
    if parallel_servers is not None:
        assert result.parallel_servers == 2


def test_make_study_parallel_servers_on_non_webarena(monkeypatch):
    monkeypatch.setattr(study_mod, "Study", _RecordingStudy)
    result = make_study(["a", "b"], SimpleNamespace(name="miniwob"), parallel_servers=[1, 2])
    assert isinstance(result, ParallelStudies)
    assert result.parallel_servers == [1, 2]


# ---------------------------------------------------------------------------
# Study construction / agents_on_benchmark
# ---------------------------------------------------------------------------


def test_study_post_init(tmp_path):
    s = Study([_agent()], BENCH, dir=str(tmp_path))
    assert s.dir == tmp_path
    assert s.uuid is not None
    assert all(isinstance(ea, EnvArgs) for ea in s.benchmark.env_args_list)
    assert all(isinstance(e, ExpArgs) for e in s.exp_args_list)
    assert {e.env_args.task_name for e in s.exp_args_list} == {
        ea.task_name for ea in s.benchmark.env_args_list
    }
    other = Study([_agent()], BENCH)
    assert other.uuid != s.uuid


def test_study_demo_mode_applies_to_env_and_agent():
    s = Study([_agent()], _benchmark(), demo_mode=True)
    for exp in s.exp_args_list:
        assert exp.env_args.record_video is True
        assert exp.env_args.slow_mo == 1000
    assert s.agent_args[0].flags.action.action_set.demo_mode == "all_blue"


def test_agents_on_benchmark_uses_dependencies(monkeypatch, study):
    calls = []

    def fake_add_dependencies(exp_args_list, graph):
        calls.append(graph)
        return exp_args_list[::-1]

    monkeypatch.setattr(study_mod, "add_dependencies", fake_add_dependencies)
    result = study.agents_on_benchmark(_agent(), _benchmark(), logging_level=logging.ERROR)
    assert len(calls) == 1
    assert set(calls[0]) == {"miniwob.click-checkboxes", "miniwob.click-dialog"}
    assert [e.order for e in result] == [3, 2, 1, 0]
    assert all(e.logging_level == logging.ERROR for e in result)


def test_agents_on_benchmark_ignore_dependencies_warns(monkeypatch, study, caplog):
    monkeypatch.setattr(study_mod, "add_dependencies", mock.Mock(side_effect=AssertionError))
    with caplog.at_level(logging.WARNING, logger=study_mod.logger.name):
        result = study.agents_on_benchmark([_agent()], _benchmark(), ignore_dependencies=True)
    assert len(result) == 4
    assert "Ignoring dependencies" in caplog.text


@pytest.mark.parametrize("name", ["webarena_tiny", "visualwebarena"])
def test_agents_on_benchmark_rejects_multiple_agents_on_webarena(study, name):
    with pytest.raises(ValueError, match="Only one agent"):
        study.agents_on_benchmark([_agent("_0"), _agent("_1")], SimpleNamespace(name=name))


def test_set_demo_mode():
    envs = [SimpleNamespace(), SimpleNamespace()]
    set_demo_mode(envs)
    for env in envs:
        assert env.viewport == {"width": 1280, "height": 720}
        assert env.record_video is True
        assert env.wait_for_user_message is False
        assert env.slow_mo == 1000


@dataclass
class _CustomEnvArgs(AbstractEnvArgs):
    task_name: str = "custom"

    def make_env(self, action_mapping, exp_dir, exp_task_kwargs):
        return None


def test_convert_env_args():
    new = EnvArgs(task_name="t1")
    custom = _CustomEnvArgs()
    old = bgym.EnvArgs(task_name="t2", max_steps=7)
    result = _convert_env_args([new, custom, old])
    assert result[0] is new
    assert result[1] is custom
    assert type(result[2]) is EnvArgs
    assert (result[2].task_name, result[2].max_steps) == ("t2", 7)


def test_convert_env_args_rejects_unknown_type():
    with pytest.raises(TypeError, match="Unexpected type"):
        _convert_env_args([{"task_name": "x"}])


# ---------------------------------------------------------------------------
# naming
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "agents, benchmarks, suffix, expected",
    [
        (["A"], ["b"], None, "a-on-b"),
        (["A", "A"], ["b"], "v1", "a-on-b-v1"),
        (["A", "B"], ["b"], "", "2-agents-on-b"),
        (["A"], ["b1", "b2", "b2"], "x", "a-on-2-benchmarks-x"),
    ],
)
def test_make_study_name(agents, benchmarks, suffix, expected):
    assert _make_study_name(agents, benchmarks, suffix) == expected


def test_make_study_name_truncates_long_names():
    assert len(_make_study_name(["a" * 300], ["b"])) <= 200


def test_study_name_property(study):
    study.suffix = "trial"
    assert study.name == "genericagent-test-cheat-miniwob-click-test-on-miniwob-tiny-test-trial"


def test_override_max_steps(study):
    study.override_max_steps(3)
    assert {e.env_args.max_steps for e in study.exp_args_list} == {3}


def test_shuffle_exps_keeps_same_experiments(study):
    random.seed(1234)
    before = list(study.exp_args_list)
    study.shuffle_exps()
    assert len(study.exp_args_list) == len(before)
    assert {id(e) for e in study.exp_args_list} == {id(e) for e in before}


# ---------------------------------------------------------------------------
# dir / save / load
# ---------------------------------------------------------------------------


def test_make_dir_creates_dated_dir(tmp_path):
    s = Study([_agent()], _benchmark())
    s.make_dir(exp_root=tmp_path)
    assert s.dir.parent == tmp_path
    assert s.dir.is_dir()
    assert s.dir.name.endswith(s.name)
    first = s.dir
    s.make_dir(exp_root=tmp_path / "other")
    assert s.dir == first


def test_save_and_load_roundtrip(study, tmp_path):
    study.save()
    assert (study.dir / "study.pkl.gz").exists()
    moved = tmp_path / "moved"
    (study.dir / "study.pkl.gz").rename(moved.with_suffix(".tmp"))
    moved.mkdir()
    moved.with_suffix(".tmp").rename(moved / "study.pkl.gz")

    loaded = Study.load(str(moved))
    assert loaded.uuid == study.uuid
    assert loaded.dir == moved
    assert len(loaded.exp_args_list) == 4
    assert loaded.agent_args[0].agent_name == study.agent_args[0].agent_name


def test_load_backward_compat_without_pickle(tmp_path, monkeypatch):
    agent = _agent()
    fake_result = SimpleNamespace(
        exp_args=SimpleNamespace(
            env_args=SimpleNamespace(task_name=f"{BENCH}.click-dialog"), agent_args=agent
        )
    )
    seen = {}

    def fake_yield(savedir_base, progress_fn=None):
        seen["dir"] = savedir_base
        yield fake_result

    monkeypatch.setattr(inspect_results, "yield_all_exp_results", fake_yield)
    loaded = Study.load(tmp_path)
    assert seen["dir"] == tmp_path
    assert loaded.benchmark.name == BENCH
    assert loaded.dir == tmp_path
    assert len(loaded.exp_args_list) == 4


def test_get_most_recent_study(tmp_path, monkeypatch):
    for name in [
        "2024-01-01_10-00-00_alpha",
        "2024-06-01_10-00-00_beta",
        "_2030-01-01_10-00-00_hidden",
        "not_a_date",
    ]:
        (tmp_path / name).mkdir()
    (tmp_path / "2031-01-01_10-00-00_file.txt").write_text("")

    assert get_most_recent_study(tmp_path).name == "2024-06-01_10-00-00_beta"
    assert get_most_recent_study(tmp_path, contains="alpha").name == "2024-01-01_10-00-00_alpha"
    assert get_most_recent_study(tmp_path, contains="nothing") is None

    monkeypatch.setattr(study_mod, "RESULTS_DIR", tmp_path)
    assert get_most_recent_study().name == "2024-06-01_10-00-00_beta"


def test_load_most_recent(tmp_path):
    older = Study([_agent()], _benchmark(), dir=tmp_path / "2024-01-01_00-00-00_older")
    newer = Study([_agent()], _benchmark(), dir=tmp_path / "2024-02-01_00-00-00_newer")
    older.save()
    newer.save()
    assert Study.load_most_recent(tmp_path).uuid == newer.uuid
    assert Study.load_most_recent(tmp_path, contains="older").uuid == older.uuid


# ---------------------------------------------------------------------------
# results / incomplete / journal
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_inspect(monkeypatch):
    result_df = pd.DataFrame({"x": [1, 2]})
    summary_df = pd.DataFrame({"avg_reward": [0.5]})
    monkeypatch.setattr(inspect_results, "load_result_df", lambda d: result_df)
    monkeypatch.setattr(inspect_results, "error_report", lambda df, **kw: "ERR REPORT")
    monkeypatch.setattr(inspect_results, "summarize_study", lambda df: summary_df)
    return result_df, summary_df


@pytest.mark.parametrize("suffix, file_suffix", [("", ""), ("trial_1", "_trial_1")])
def test_get_results_saves_files(study, fake_inspect, suffix, file_suffix):
    study.make_dir()
    result_df, summary_df, report = study.get_results(suffix=suffix)
    assert result_df is fake_inspect[0]
    assert summary_df is fake_inspect[1]
    assert report == "ERR REPORT"
    assert (study.dir / f"result_df{file_suffix}.csv").exists()
    assert (study.dir / f"summary_df{file_suffix}.csv").exists()
    assert (study.dir / f"error_report{file_suffix}.md").read_text() == "ERR REPORT"


def test_get_results_without_saving(study, fake_inspect):
    study.make_dir()
    study.get_results(also_save=False)
    assert list(study.dir.iterdir()) == []


def test_find_incomplete_counts_incomplete_and_errors(monkeypatch, study):
    exp_list = [
        SimpleNamespace(is_dummy=False, status="error"),
        SimpleNamespace(is_dummy=False),
        SimpleNamespace(is_dummy=True, status="done"),
    ]
    fake = mock.Mock(return_value=exp_list)
    monkeypatch.setattr(study_mod, "find_incomplete", fake)
    assert study.find_incomplete(include_errors=False) == (2, 1)
    fake.assert_called_once_with(study.dir, include_errors=False)
    assert study.exp_args_list is exp_list


def test_load_exp_args_list(monkeypatch, study):
    monkeypatch.setattr(
        inspect_results, "yield_all_exp_results", lambda savedir_base: iter(["e1", "e2"])
    )
    study.load_exp_args_list()
    assert study.exp_args_list == ["e1", "e2"]


def test_append_to_journal_delegates(monkeypatch, study):
    summary = pd.DataFrame({"a": [1]})
    monkeypatch.setattr(Study, "get_results", lambda self: (None, summary, None))
    fake = mock.Mock()
    monkeypatch.setattr(study_mod.repro, "append_to_journal", fake)
    study.reproducibility_info = {"k": "v"}
    study.append_to_journal(strict_reproducibility=False)
    fake.assert_called_once_with({"k": "v"}, summary, strict_reproducibility=False)


# ---------------------------------------------------------------------------
# reproducibility info
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_repro_info(monkeypatch):
    calls = []
    infos = []

    def fake_get_info(agent_names, benchmark, study_id, **kwargs):
        calls.append(
            dict(agent_names=agent_names, benchmark=benchmark, study_id=study_id, **kwargs)
        )
        return infos.pop(0) if infos else {"agent_names": agent_names, "v": 1}

    monkeypatch.setattr(study_mod.repro, "get_reproducibility_info", fake_get_info)
    return calls, infos


@pytest.mark.parametrize("strict", [True, False])
def test_set_reproducibility_info_arguments(study, fake_repro_info, strict):
    calls, _ = fake_repro_info
    study.set_reproducibility_info(strict_reproducibility=strict, comment="note")
    call = calls[0]
    assert call["agent_names"] == [study.agent_args[0].agent_name]
    assert call["benchmark"] is study.benchmark
    assert call["study_id"] == study.uuid
    assert call["ignore_changes"] is (not strict)
    assert call["allow_bypass_benchmark_version"] is (not strict)
    assert call["comment"] == "note"
    assert study.reproducibility_info["v"] == 1


def test_set_reproducibility_info_incompatible_strict_raises(study, fake_repro_info):
    _, infos = fake_repro_info
    study.reproducibility_info = {"v": 1, "date": "old"}
    infos.append({"v": 2, "date": "new"})
    with pytest.raises(ValueError, match="Key v has changed"):
        study.set_reproducibility_info(strict_reproducibility=True)
    assert study.reproducibility_info == {"v": 1, "date": "old"}


def test_set_reproducibility_info_incompatible_lenient_replaces(study, fake_repro_info, caplog):
    _, infos = fake_repro_info
    study.reproducibility_info = {"v": 1}
    infos.append({"v": 2})
    with caplog.at_level(logging.WARNING):
        study.set_reproducibility_info(strict_reproducibility=False)
    assert study.reproducibility_info == {"v": 2}
    assert "Key v has changed" in caplog.text


# ---------------------------------------------------------------------------
# run / _run
# ---------------------------------------------------------------------------


@pytest.fixture
def run_harness(monkeypatch, fake_repro_info):
    state = SimpleNamespace(run_calls=[], suffixes=[], incomplete=[], prepared=0)

    def fake_run_experiments(n_jobs, exp_args_list, study_dir, parallel_backend, avg_step_timeout):
        state.run_calls.append((n_jobs, len(exp_args_list), study_dir, parallel_backend))

    def fake_prepare(self):
        state.prepared += 1

    def fake_get_results(self, suffix="", also_save=True):
        state.suffixes.append(suffix)
        return None, "summary", "error report text"

    def fake_find_incomplete(self, include_errors=True):
        state.include_errors = include_errors
        return state.incomplete.pop(0)

    monkeypatch.setattr(study_mod, "run_experiments", fake_run_experiments)
    monkeypatch.setattr(bgym.Benchmark, "prepare_backends", fake_prepare)
    monkeypatch.setattr(Study, "get_results", fake_get_results)
    monkeypatch.setattr(Study, "find_incomplete", fake_find_incomplete)
    return state


@pytest.mark.parametrize(
    "incomplete, n_trials, log_msg",
    [
        ([(0, 0)], 1, "finished"),
        ([(2, 2)], 1, "More than 30%"),
        ([(2, 1), (1, 0), (1, 0)], 3, "did not finish after 3 trials"),
    ],
)
def test_run_relaunch_loop(study, run_harness, caplog, incomplete, n_trials, log_msg):
    run_harness.incomplete = list(incomplete)
    with caplog.at_level(logging.INFO, logger=study_mod.logger.name):
        study.run(n_jobs=2, parallel_backend="joblib", n_relaunch=3, relaunch_errors=False)
    assert len(run_harness.run_calls) == n_trials
    assert run_harness.run_calls[0] == (2, 4, study.dir, "joblib")
    assert run_harness.prepared == n_trials
    assert run_harness.suffixes == [f"trial_{i + 1}_of_3" for i in range(n_trials)]
    assert run_harness.include_errors is False
    assert (study.dir / "study.pkl.gz").exists()
    assert study.reproducibility_info is not None
    assert log_msg in caplog.text
    assert "error report text" in caplog.text


@pytest.mark.xfail(
    strict=True,
    reason="BUG: Study.run never updates last_error_count, so the no-progress stop never fires",
)
def test_run_stops_when_errors_do_not_decrease(study, run_harness):
    run_harness.incomplete = [(1, 1), (1, 1), (1, 1)]
    study.run(n_relaunch=3)
    assert len(run_harness.run_calls) == 2


def test_run_saved_study_is_loadable(study, run_harness):
    run_harness.incomplete = [(0, 0)]
    study.run(n_relaunch=1)
    with gzip.open(study.dir / "study.pkl.gz", "rb") as f:
        loaded = pickle.load(f)
    assert loaded.uuid == study.uuid
    assert loaded.reproducibility_info == study.reproducibility_info


def test_private_run_requires_exp_args_list(study):
    study.exp_args_list = None
    with pytest.raises(ValueError, match="exp_args_list is None"):
        study._run()


def test_private_run_passes_avg_step_timeout(study, monkeypatch):
    fake = mock.Mock()
    monkeypatch.setattr(study_mod, "run_experiments", fake)
    monkeypatch.setattr(bgym.Benchmark, "prepare_backends", lambda self: None)
    study.avg_step_timeout = 7
    study._run(n_jobs=3, parallel_backend="sequential")
    fake.assert_called_once_with(
        3, study.exp_args_list, study.dir, parallel_backend="sequential", avg_step_timeout=7
    )


# ---------------------------------------------------------------------------
# SequentialStudies / ParallelStudies
# ---------------------------------------------------------------------------


@pytest.fixture
def two_studies():
    return [Study([_agent("_0")], _benchmark()), Study([_agent("_1")], _benchmark())]


@pytest.fixture
def recorded_study_runs(monkeypatch):
    calls = []

    def fake_run(
        self, n_jobs=1, parallel_backend="ray", strict_reproducibility=False, n_relaunch=3
    ):
        calls.append(
            (
                self.agent_args[0].agent_name,
                n_jobs,
                parallel_backend,
                strict_reproducibility,
                n_relaunch,
            )
        )

    monkeypatch.setattr(Study, "run", fake_run)
    return calls


def test_sequential_studies_name(two_studies):
    seq = SequentialStudies(two_studies)
    assert seq.name == "2-agents-on-miniwob-tiny-test"
    seq.suffix = "abc"
    assert seq.name.endswith("-abc")


def test_sequential_studies_delegation(monkeypatch, two_studies):
    seq = SequentialStudies(two_studies)
    find_calls, journal_calls = [], []
    monkeypatch.setattr(
        Study,
        "find_incomplete",
        lambda self, include_errors=True: find_calls.append(include_errors),
    )
    monkeypatch.setattr(
        Study,
        "append_to_journal",
        lambda self, strict_reproducibility=True: journal_calls.append(strict_reproducibility),
    )
    seq.find_incomplete(include_errors=False)
    seq.append_to_journal(strict_reproducibility=False)
    seq.override_max_steps(5)
    assert find_calls == [False, False]
    assert journal_calls == [False, False]
    assert {e.env_args.max_steps for s in two_studies for e in s.exp_args_list} == {5}


def _patch_make_dir_default(monkeypatch, path):
    monkeypatch.setattr(AbstractStudy.make_dir, "__defaults__", (path,))


def test_sequential_studies_run(monkeypatch, tmp_path, two_studies, recorded_study_runs):
    _patch_make_dir_default(monkeypatch, tmp_path / "default_root")
    monkeypatch.setattr(SequentialStudies, "get_results", lambda self: (None, pd.DataFrame(), ""))
    seq = SequentialStudies(two_studies)
    seq.run(
        n_jobs=2,
        parallel_backend="joblib",
        strict_reproducibility=True,
        n_relaunch=1,
        exp_root=tmp_path / "root",
    )
    assert (seq.dir / "study.pkl.gz").exists()
    for st in two_studies:
        assert st.dir.parent == seq.dir
    assert [c[1:] for c in recorded_study_runs] == [(2, "joblib", True, 1)] * 2
    assert [c[0] for c in recorded_study_runs] == [
        two_studies[0].agent_args[0].agent_name,
        two_studies[1].agent_args[0].agent_name,
    ]


@pytest.mark.xfail(
    strict=True, reason="BUG: SequentialStudies.run ignores exp_root and uses RESULTS_DIR"
)
def test_sequential_studies_run_respects_exp_root(
    monkeypatch, tmp_path, two_studies, recorded_study_runs
):
    _patch_make_dir_default(monkeypatch, tmp_path / "default_root")
    monkeypatch.setattr(SequentialStudies, "get_results", lambda self: (None, pd.DataFrame(), ""))
    seq = SequentialStudies(two_studies)
    seq.run(exp_root=tmp_path / "root")
    assert seq.dir.parent == tmp_path / "root"


class _FakeFuture:
    def __init__(self, fn, args):
        self.exc = None
        try:
            self.value = fn(*args)
        except Exception as e:  # noqa: BLE001
            self.exc = e

    def result(self):
        if self.exc is not None:
            raise self.exc
        return self.value


class _FakeExecutor:
    instances = []

    def __init__(self, max_workers, initializer, initargs):
        self.max_workers = max_workers
        _FakeExecutor.instances.append(self)
        for _ in range(max_workers):
            initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args):
        return _FakeFuture(fn, args)


class _FakePool:
    instances = []

    def __init__(self, processes, initializer, initargs):
        self.processes = processes
        _FakePool.instances.append(self)
        for _ in range(processes):
            initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starmap(self, fn, iterable):
        return [fn(*args) for args in iterable]


@pytest.fixture
def fake_mp(monkeypatch):
    _FakeExecutor.instances = []
    _FakePool.instances = []
    monkeypatch.setattr(study_mod, "Manager", lambda: SimpleNamespace(Queue=queue.Queue))
    monkeypatch.setattr(study_mod, "ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(study_mod, "Pool", _FakePool)


class _Server(BaseServer):
    def __init__(self):
        self.init_count = 0

    def init(self):
        self.init_count += 1


@pytest.mark.parametrize(
    "cls, fake", [(ParallelStudies, _FakeExecutor), (ParallelStudies_alt, _FakePool)]
)
def test_parallel_studies_int_servers(fake_mp, two_studies, recorded_study_runs, cls, fake):
    ps = cls(two_studies, parallel_servers=3)
    ps._run(n_jobs=4, parallel_backend="ray", strict_reproducibility=False, n_relaunch=2)
    workers = fake.instances[0]
    assert getattr(workers, "max_workers", getattr(workers, "processes", None)) == 3
    assert sorted(c[1:] for c in recorded_study_runs) == [(4, "ray", False, 2)] * 2


@pytest.mark.parametrize("cls", [ParallelStudies, ParallelStudies_alt])
def test_parallel_studies_initializes_each_server(fake_mp, two_studies, recorded_study_runs, cls):
    servers = [_Server(), _Server()]
    cls(two_studies, parallel_servers=servers)._run()
    assert [s.init_count for s in servers] == [1, 1]
    assert len(recorded_study_runs) == 2


def test_parallel_studies_propagates_study_errors(fake_mp, monkeypatch, two_studies):
    def failing_run(self, *args, **kwargs):
        raise RuntimeError("study failed")

    monkeypatch.setattr(Study, "run", failing_run)
    with pytest.raises(RuntimeError, match="study failed"):
        ParallelStudies(two_studies, parallel_servers=1)._run()


def test_init_worker_pulls_and_inits_server(capsys):
    q = queue.Queue()
    server = _Server()
    q.put(server)
    _init_worker(q)
    assert server.init_count == 1
    assert q.empty()
    assert "initializing server instance" in capsys.readouterr().out


def test_run_study_forwards_arguments():
    fake_study = mock.Mock()
    _run_study(fake_study, 2, "joblib", True, 5)
    fake_study.run.assert_called_once_with(2, "joblib", True, 5)
