import dataclasses
import gzip
import logging
import pickle
from copy import deepcopy
from pathlib import Path

import pytest
from bgym import DEFAULT_BENCHMARKS

from agentlab.agents.generic_agent.agent_configs import FLAGS_GPT_4o
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.llm.chat_api import CheatMiniWoBLLMArgs
from agentlab.experiments.study import ParallelStudies, SequentialStudies, make_study, Study
from agentlab.experiments.multi_server import WebArenaInstanceVars

logging.getLogger().setLevel(logging.INFO)


def _make_agent_args_list():
    # CheatMiniWoB agents won't succeed on WebArena, this is just for testing parallelization
    agent_args_list = []
    for i in range(2):
        agent_args = GenericAgentArgs(
            chat_model_args=CheatMiniWoBLLMArgs(),
            flags=FLAGS_GPT_4o,
        )

        agent_args.agent_name = agent_args.agent_name + f"_{i}"
        agent_args_list.append(agent_args)
    return agent_args_list


@pytest.mark.skip(reason="This test requires WebArena instances to be running")
def manual_test_launch_parallel_study_webarena():
    agent_args_list = _make_agent_args_list()

    server_instance_1 = WebArenaInstanceVars.from_env_vars()
    server_instance_2 = server_instance_1.clone()
    server_instance_2.base_url = "http://webarena-slow.eastus.cloudapp.azure.com"
    parallel_servers = [server_instance_1, server_instance_2]
    # parallel_servers = [server_instance_2]

    for server in parallel_servers:
        print(server)

    study = make_study(
        agent_args_list,
        benchmark="webarena_tiny",
        parallel_servers=parallel_servers,
        ignore_dependencies=True,
    )
    study.override_max_steps(2)
    assert isinstance(study, ParallelStudies)

    study.run(n_jobs=4, parallel_backend="ray", n_relaunch=1)


@pytest.mark.skip(reason="This usecase isnt relevant atm")
def test_launch_parallel_study():
    agent_args_list = _make_agent_args_list()

    study = make_study(agent_args_list, benchmark="miniwob_tiny_test", parallel_servers=2)
    assert isinstance(study, ParallelStudies)

    study.run(n_jobs=4, parallel_backend="ray", n_relaunch=1)
    _, summary_df, _ = study.get_results()
    assert len(summary_df) == 2
    for n_completed in summary_df["n_completed"]:
        assert n_completed == "4/4"


def _make_cheat_agents(n):
    agents = []
    for i in range(n):
        agent_args = GenericAgentArgs(
            chat_model_args=CheatMiniWoBLLMArgs(),
            flags=deepcopy(FLAGS_GPT_4o),
        )
        agent_args.agent_name = f"{agent_args.agent_name}_{i}"
        agents.append(agent_args)
    return agents


def _tiny_benchmark(name=None):
    benchmark = DEFAULT_BENCHMARKS["miniwob_tiny_test"]()
    if name is not None:
        benchmark = dataclasses.replace(benchmark, name=name)
    return benchmark


def test_make_study_single_agent_returns_study():
    agent = _make_cheat_agents(1)[0]
    study = make_study(agent, benchmark="miniwob_tiny_test", ignore_dependencies=True)

    assert type(study) is Study
    assert study.agent_args == [agent]
    assert study.benchmark.name == "miniwob_tiny_test"
    assert study.ignore_dependencies is True


def test_make_study_multiple_agents_on_miniwob_returns_single_study():
    agents = _make_cheat_agents(2)
    study = make_study(agents, benchmark="miniwob_tiny_test", suffix="my_suffix")

    assert type(study) is Study
    assert study.agent_args == agents
    assert study.suffix == "my_suffix"
    assert len(study.exp_args_list) == 2 * len(study.benchmark.env_args_list)


def test_make_study_multiple_agents_on_webarena_returns_sequential_studies():
    agents = _make_cheat_agents(2)
    study = make_study(agents, benchmark=_tiny_benchmark("webarena_fake"), comment="hello")

    assert type(study) is SequentialStudies
    assert len(study.studies) == 2
    for sub_study, agent in zip(study.studies, agents):
        assert type(sub_study) is Study
        assert sub_study.agent_args == [agent]
        assert sub_study.comment == "hello"
        assert all(e.agent_args is agent for e in sub_study.exp_args_list)


def test_make_study_single_agent_on_webarena_returns_study():
    agent = _make_cheat_agents(1)[0]
    study = make_study(agent, benchmark=_tiny_benchmark("webarena_fake"))

    assert type(study) is Study


def test_make_study_parallel_servers_returns_parallel_studies():
    agents = _make_cheat_agents(3)
    study = make_study(agents, benchmark="miniwob_tiny_test", parallel_servers=2)

    assert type(study) is ParallelStudies
    assert study.parallel_servers == 2
    assert [s.agent_args for s in study.studies] == [[a] for a in agents]


def test_make_exp_args_list_is_agent_by_env_cross_product():
    agents = _make_cheat_agents(2)
    study = Study(
        agents,
        "miniwob_tiny_test",
        logging_level=logging.ERROR,
        logging_level_stdout=logging.CRITICAL,
    )
    env_keys = [(e.task_name, e.task_seed) for e in study.benchmark.env_args_list]
    assert len(env_keys) == 4

    exp_args_list = study.exp_args_list
    assert len(exp_args_list) == len(agents) * len(env_keys)

    combos = [
        (e.agent_args.agent_name, e.env_args.task_name, e.env_args.task_seed) for e in exp_args_list
    ]
    expected = [(a.agent_name, t, s) for a in agents for (t, s) in env_keys]
    assert combos == expected

    assert [e.order for e in exp_args_list] == list(range(len(exp_args_list)))
    for exp_args in exp_args_list:
        assert exp_args.logging_level == logging.ERROR
        assert exp_args.logging_level_stdout == logging.CRITICAL
        assert exp_args.depends_on == ()


def test_make_exp_args_list_regenerates_after_agent_change():
    study = Study(_make_cheat_agents(1), "miniwob_tiny_test")
    assert len(study.exp_args_list) == 4

    study.agent_args = _make_cheat_agents(3)
    study.make_exp_args_list()
    assert len(study.exp_args_list) == 12
    assert len({e.agent_args.agent_name for e in study.exp_args_list}) == 3


def test_multiple_agents_on_webarena_study_raises():
    with pytest.raises(ValueError, match="Only one agent"):
        Study(_make_cheat_agents(2), _tiny_benchmark("webarena_fake"))


def test_study_find_incomplete_counts():
    study = Study(_make_cheat_agents(1), "miniwob_tiny_test")
    study.dir = Path(__file__).parent.parent / "data" / "test_study"

    # the fixture holds one incomplete and one errored experiment
    assert study.find_incomplete(include_errors=False) == (1, 1)
    assert study.find_incomplete(include_errors=True) == (2, 1)
    assert len(study.exp_args_list) == 2


class _RunRecorder:
    """Patches the expensive parts of Study.run and scripts find_incomplete results."""

    def __init__(self, monkeypatch, incomplete_results):
        self.incomplete_results = list(incomplete_results)
        self.run_calls = []
        self.find_incomplete_calls = []
        self.get_results_suffixes = []
        self.repro_calls = []

        recorder = self

        def fake_set_reproducibility_info(study, strict_reproducibility=False, comment=None):
            recorder.repro_calls.append((strict_reproducibility, comment))
            study.reproducibility_info = {"fake": True}

        def fake_run(study, n_jobs=1, parallel_backend="joblib", strict_reproducibility=False):
            recorder.run_calls.append((n_jobs, parallel_backend, strict_reproducibility))

        def fake_get_results(study, suffix="", also_save=True):
            recorder.get_results_suffixes.append(suffix)
            return None, "summary", "error report"

        def fake_find_incomplete(study, include_errors=True):
            recorder.find_incomplete_calls.append(include_errors)
            return recorder.incomplete_results.pop(0)

        monkeypatch.setattr(Study, "set_reproducibility_info", fake_set_reproducibility_info)
        monkeypatch.setattr(Study, "_run", fake_run)
        monkeypatch.setattr(Study, "get_results", fake_get_results)
        monkeypatch.setattr(Study, "find_incomplete", fake_find_incomplete)


def _make_run_study():
    study = Study(_make_cheat_agents(2), "miniwob_tiny_test", comment="a comment")
    assert len(study.exp_args_list) == 8
    return study


def test_run_stops_early_when_all_complete(monkeypatch, tmp_path):
    recorder = _RunRecorder(monkeypatch, [(3, 0), (0, 0)])
    study = _make_run_study()

    study.run(n_jobs=3, parallel_backend="sequential", n_relaunch=5, exp_root=tmp_path)

    assert len(recorder.run_calls) == 2
    assert recorder.run_calls[0] == (3, "sequential", False)
    assert recorder.get_results_suffixes == ["trial_1_of_5", "trial_2_of_5"]
    assert recorder.find_incomplete_calls == [True, True]
    assert recorder.repro_calls == [(False, "a comment")]

    assert study.dir.parent == tmp_path
    with gzip.open(study.dir / "study.pkl.gz", "rb") as f:
        saved = pickle.load(f)
    assert len(saved.exp_args_list) == 8
    assert saved.reproducibility_info == {"fake": True}


def test_run_honours_n_relaunch_upper_bound(monkeypatch, tmp_path):
    recorder = _RunRecorder(monkeypatch, [(2, 0)] * 10)
    study = _make_run_study()

    study.run(n_relaunch=3, relaunch_errors=False, exp_root=tmp_path)

    assert len(recorder.run_calls) == 3
    assert recorder.get_results_suffixes == ["trial_1_of_3", "trial_2_of_3", "trial_3_of_3"]
    assert recorder.find_incomplete_calls == [False, False, False]


def test_run_stops_when_too_many_errors(monkeypatch, tmp_path):
    # 3 errors out of 8 experiments is above the 30% threshold
    recorder = _RunRecorder(monkeypatch, [(3, 3), (0, 0)])
    study = _make_run_study()

    study.run(n_relaunch=5, exp_root=tmp_path)

    assert len(recorder.run_calls) == 1


def test_run_continues_when_errors_below_threshold(monkeypatch, tmp_path):
    # 2 errors out of 8 experiments is below the 30% threshold
    recorder = _RunRecorder(monkeypatch, [(2, 2), (0, 0)])
    study = _make_run_study()

    study.run(n_relaunch=5, exp_root=tmp_path)

    assert len(recorder.run_calls) == 2


def test_run_passes_strict_reproducibility(monkeypatch, tmp_path):
    recorder = _RunRecorder(monkeypatch, [(0, 0)])
    study = _make_run_study()

    study.run(n_relaunch=3, strict_reproducibility=True, exp_root=tmp_path)

    assert recorder.repro_calls == [(True, "a comment")]
    assert recorder.run_calls == [(1, "ray", True)]


if __name__ == "__main__":
    # test_launch_parallel_study()
    manual_test_launch_parallel_study_webarena()
