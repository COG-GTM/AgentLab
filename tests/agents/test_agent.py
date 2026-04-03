import tempfile
from pathlib import Path

from agentlab.agents.generic_agent.agent_configs import FLAGS_GPT_3_5
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.analyze import inspect_results
from agentlab.experiments import launch_exp
from agentlab.experiments.loop import EnvArgs, ExpArgs
from agentlab.llm.chat_api import CheatMiniWoBLLMArgs

from tests.helpers.mock_llms import (
    CheatLLM_LLMError,
    CheatLLMArgs_LLMError,
    CheatMiniWoBLLM_ParseRetry,
    CheatMiniWoBLLMArgs_ParseRetry,
)


def test_generic_agent():
    exp_args = ExpArgs(
        agent_args=GenericAgentArgs(
            chat_model_args=CheatMiniWoBLLMArgs(),
            flags=FLAGS_GPT_3_5,
        ),
        env_args=EnvArgs(task_name="miniwob.click-test", task_seed=42),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        launch_exp.run_experiments(
            1, [exp_args], Path(tmp_dir) / "generic_agent_test", parallel_backend="joblib"
        )

        result_record = inspect_results.load_result_df(tmp_dir, progress_fn=None)

        target = {
            "n_steps": 1,
            "cum_reward": 1.0,
            "terminated": True,
            "truncated": False,
            "err_msg": None,
            "stack_trace": None,
            "agent.flags.obs.use_ax_tree": True,
        }

        for key, target_val in target.items():
            assert key in result_record
            assert result_record[key].iloc[0] == target_val


def test_generic_agent_parse_retry():
    exp_args = ExpArgs(
        agent_args=GenericAgentArgs(
            chat_model_args=CheatMiniWoBLLMArgs_ParseRetry(n_retry=2),
            flags=FLAGS_GPT_3_5,
        ),
        env_args=EnvArgs(task_name="miniwob.click-test", task_seed=42),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        # TODO why these tests don't work with ray backend?
        launch_exp.run_experiments(
            1, [exp_args], Path(tmp_dir) / "generic_agent_test", parallel_backend="joblib"
        )
        result_record = inspect_results.load_result_df(tmp_dir, progress_fn=None)
        print(result_record)
        target = {
            "stats.cum_n_retry": 2,
            "stats.cum_busted_retry": 0,
            "n_steps": 1,
            "cum_reward": 1.0,
        }

        for key, target_val in target.items():
            assert key in result_record
            assert result_record[key].iloc[0] == target_val


def test_bust_parse_retry():
    exp_args = ExpArgs(
        agent_args=GenericAgentArgs(
            chat_model_args=CheatMiniWoBLLMArgs_ParseRetry(n_retry=10),
            flags=FLAGS_GPT_3_5,
        ),
        env_args=EnvArgs(task_name="miniwob.click-test", task_seed=42),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        launch_exp.run_experiments(
            1, [exp_args], Path(tmp_dir) / "generic_agent_test", parallel_backend="joblib"
        )
        result_record = inspect_results.load_result_df(tmp_dir, progress_fn=None)

        target = {
            "stats.cum_n_retry": 5,
            "stats.cum_busted_retry": 1,
            "n_steps": 0,
            "cum_reward": 0,
            "err_msg": None,  # parsing error is considered an agent failure, not a code error
        }

        for key, target_val in target.items():
            assert key in result_record
            assert result_record[key].iloc[0] == target_val


def test_llm_error_success():
    exp_args = ExpArgs(
        agent_args=GenericAgentArgs(
            chat_model_args=CheatLLMArgs_LLMError(n_retry=3, success=True),
            flags=FLAGS_GPT_3_5,
        ),
        env_args=EnvArgs(task_name="miniwob.click-test", task_seed=42),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        launch_exp.run_experiments(
            1, [exp_args], Path(tmp_dir) / "generic_agent_test", parallel_backend="joblib"
        )
        result_record = inspect_results.load_result_df(tmp_dir, progress_fn=None)

        target = {
            "stats.cum_n_llm_retry": 3,
            "n_steps": 1,
            "cum_reward": 1.0,
            "err_msg": None,
        }

        for key, target_val in target.items():
            assert key in result_record
            assert result_record[key].iloc[0] == target_val


def test_llm_error_no_success():
    exp_args = ExpArgs(
        agent_args=GenericAgentArgs(
            chat_model_args=CheatLLMArgs_LLMError(n_retry=5, success=False),
            flags=FLAGS_GPT_3_5,
        ),
        env_args=EnvArgs(task_name="miniwob.click-test", task_seed=42),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        launch_exp.run_experiments(
            1, [exp_args], Path(tmp_dir) / "generic_agent_test", parallel_backend="joblib"
        )
        result_record = inspect_results.load_result_df(tmp_dir, progress_fn=None)

        target = {
            "n_steps": 0,
            "cum_reward": 0,
            "err_msg": "Exception uncaught by agent or environment in task miniwob.click-test.\nOpenAIError:\nLLM failed to respond",
        }

        for key, target_val in target.items():
            assert key in result_record
            assert result_record[key].iloc[0] == target_val


if __name__ == "__main__":
    # test_generic_agent()
    test_generic_agent_parse_retry()
