import bgym
import pytest
import ray

from agentlab.experiments.exp_utils import MockedExpArgs, add_dependencies
from agentlab.experiments.graph_execution_ray import execute_task_graph

TASK_TIME = 3


@pytest.fixture
def ray_cluster():
    ray.init(num_cpus=4)
    try:
        yield
    finally:
        ray.shutdown()


def test_execute_task_graph(ray_cluster):
    # Define a list of ExpArgs with dependencies
    exp_args_list = [
        MockedExpArgs(exp_id="task1", depends_on=[]),
        MockedExpArgs(exp_id="task2", depends_on=["task1"]),
        MockedExpArgs(exp_id="task3", depends_on=["task1"]),
        MockedExpArgs(exp_id="task4", depends_on=["task2", "task3"]),
    ]

    results = execute_task_graph(exp_args_list)

    exp_args_list = [results[task_id] for task_id in ["task1", "task2", "task3", "task4"]]

    # A dependent task cannot start before its parent returned, so these comparisons are
    # causal (not timing-based) and hold regardless of machine load.
    t1, t2, t3, t4 = exp_args_list
    assert t1.end_time <= t2.start_time, "task2 started before its dependency task1 finished"
    assert t1.end_time <= t3.start_time, "task3 started before its dependency task1 finished"
    assert t2.end_time <= t4.start_time, "task4 started before its dependency task2 finished"
    assert t3.end_time <= t4.start_time, "task4 started before its dependency task3 finished"


def test_add_dependencies():
    # Prepare a simple list of ExpArgs

    def make_exp_args(task_name, exp_id):
        return bgym.ExpArgs(
            agent_args=None, env_args=bgym.EnvArgs(task_name=task_name), exp_id=exp_id
        )

    exp_args_list = [
        make_exp_args("task1", "1"),
        make_exp_args("task2", "2"),
        make_exp_args("task3", "3"),
    ]

    # Define simple task_dependencies
    task_dependencies = {"task1": ["task2"], "task2": [], "task3": ["task1"]}

    # Call the function
    modified_list = add_dependencies(exp_args_list, task_dependencies)

    # Verify dependencies
    assert modified_list[0].depends_on == ("2",)  # task1 depends on task2
    assert modified_list[1].depends_on == ()  # task2 has no dependencies
    assert modified_list[2].depends_on == ("1",)  # task3 depends on task1

    # assert raise if task_dependencies is wrong
    task_dependencies = {"task1": ["task2"], "task2": [], "task4": ["task3"]}
    with pytest.raises(ValueError):
        add_dependencies(exp_args_list, task_dependencies)
