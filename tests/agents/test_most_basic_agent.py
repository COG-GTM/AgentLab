import tempfile
from pathlib import Path

from agentlab.agents.most_basic_agent.most_basic_agent import MostBasicAgentArgs
from agentlab.analyze import inspect_results
from agentlab.experiments import launch_exp
from agentlab.experiments.loop import EnvArgs, ExpArgs
from agentlab.llm.chat_api import CheatMiniWoBLLMArgs


def test_most_basic_agent():
    exp_args = ExpArgs(
        agent_args=MostBasicAgentArgs(
            chat_model_args=CheatMiniWoBLLMArgs(),
        ),
        env_args=EnvArgs(task_name="miniwob.click-test", task_seed=42),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        launch_exp.run_experiments(
            1, [exp_args], Path(tmp_dir) / "most_basic_agent_test", parallel_backend="joblib"
        )

        result_record = inspect_results.load_result_df(tmp_dir, progress_fn=None)

        target = {
            "cum_reward": 1.0,
            "terminated": True,
            "truncated": False,
            "err_msg": None,
            "stack_trace": None,
        }

        for key, target_val in target.items():
            assert key in result_record
            assert result_record[key].iloc[0] == target_val
