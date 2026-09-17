import pytest
from agentlab.agents.generic_agent.agent_configs import FLAGS_GPT_4o
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.llm.chat_api import CheatMiniWoBLLMArgs
from agentlab.experiments.study import ParallelStudies, SequentialStudies, make_study, Study
from agentlab.experiments.multi_server import WebArenaInstanceVars
import logging


logging.getLogger().setLevel(logging.INFO)

BENCHMARK = "miniwob_tiny_test"


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


def test_make_study_dispatch_sequential():
    agent_args_list = _make_agent_args_list()

    study = make_study(agent_args_list[0], benchmark=BENCHMARK)
    assert isinstance(study, Study)
    assert study.agent_args == [agent_args_list[0]]

    study = make_study(agent_args_list, benchmark=BENCHMARK)
    assert isinstance(study, Study)
    assert study.agent_args == agent_args_list

    # a single agent never needs the multi-study machinery, even with parallel servers
    study = make_study(agent_args_list[0], benchmark=BENCHMARK, parallel_servers=2)
    assert isinstance(study, Study)


def test_make_study_dispatch_parallel():
    agent_args_list = _make_agent_args_list()

    study = make_study(agent_args_list, benchmark=BENCHMARK, parallel_servers=2)
    assert isinstance(study, ParallelStudies)
    assert study.parallel_servers == 2
    assert [s.agent_args for s in study.studies] == [[a] for a in agent_args_list]


def test_make_study_dispatch_webarena_is_sequential():
    agent_args_list = _make_agent_args_list()

    study = make_study(agent_args_list, benchmark="webarena_tiny", ignore_dependencies=True)
    assert isinstance(study, SequentialStudies)
    assert not isinstance(study, ParallelStudies)
    assert [s.agent_args for s in study.studies] == [[a] for a in agent_args_list]


def test_make_exp_args_list_cross_product():
    agent_args_list = _make_agent_args_list()
    study = Study(agent_args_list, benchmark=BENCHMARK)

    env_args_list = study.benchmark.env_args_list
    assert len(study.exp_args_list) == len(agent_args_list) * len(env_args_list)

    combinations = [
        (exp_args.agent_args.agent_name, exp_args.env_args.task_name, exp_args.env_args.task_seed)
        for exp_args in study.exp_args_list
    ]
    expected = [
        (agent.agent_name, env_args.task_name, env_args.task_seed)
        for agent in agent_args_list
        for env_args in env_args_list
    ]
    assert combinations == expected
    assert [exp_args.order for exp_args in study.exp_args_list] == list(range(len(expected)))


class _FakeRunStudy(Study):
    """Study that records calls to _run and replays a scripted find_incomplete sequence."""

    def set_reproducibility_info(self, strict_reproducibility=False, comment=None):
        self.reproducibility_info = {}

    def save(self, exp_root=None):
        pass

    def get_results(self, suffix="", also_save=True):
        return None, "summary", "error report"

    def _run(self, n_jobs=1, parallel_backend="joblib", strict_reproducibility=False):
        self.n_runs += 1

    def find_incomplete(self, include_errors=True):
        return self.incomplete_per_trial[min(self.n_runs - 1, len(self.incomplete_per_trial) - 1)]


def _make_fake_run_study(incomplete_per_trial):
    study = _FakeRunStudy(_make_agent_args_list(), benchmark=BENCHMARK)
    study.n_runs = 0
    study.incomplete_per_trial = incomplete_per_trial
    return study


def test_run_stops_when_all_experiments_are_complete():
    # (n_incomplete, n_error) per trial: still incomplete after the first trial, done after the second
    study = _make_fake_run_study([(2, 1), (0, 0)])
    study.run(n_relaunch=5)
    assert study.n_runs == 2


def test_run_is_bounded_by_n_relaunch():
    # errors keep decreasing and experiments stay incomplete, so only n_relaunch bounds the loop
    study = _make_fake_run_study([(3, 2), (2, 1), (1, 0)])
    study.run(n_relaunch=3)
    assert study.n_runs == 3


def test_run_stops_when_errors_do_not_decrease():
    n_exp = 8  # 2 agents x 4 tasks, keeps the error ratio below the 30% threshold
    study = _make_fake_run_study([(2, 1), (2, 1), (2, 1)])
    assert len(study.exp_args_list) == n_exp
    study.run(n_relaunch=5)
    assert study.n_runs == 2


def test_run_stops_when_too_many_errors():
    study = _make_fake_run_study([(4, 4)])
    study.run(n_relaunch=5)
    assert study.n_runs == 1


if __name__ == "__main__":
    # test_launch_parallel_study()
    manual_test_launch_parallel_study_webarena()
