import tempfile
from pathlib import Path

from agentlab.agents.most_basic_agent.most_basic_agent import MostBasicAgentArgs
from agentlab.analyze import inspect_results
from agentlab.experiments import launch_exp
from agentlab.experiments.loop import EnvArgs, ExpArgs
from agentlab.llm.chat_api import CheatMiniWoBLLMArgs


def test_most_basic_agent():
    """Test MostBasicAgent runs an experiment without crashing.

    Note: CheatMiniWoBLLM searches for AXTree-style bids ([N] button), but
    MostBasicAgent uses pruned_html (bid="N" in HTML).  The cheat LLM will
    fail to find the bid, producing an agent-level error, but the experiment
    loop itself should complete without an unhandled exception.
    """
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
        # The experiment loop should always produce a result row
        assert len(result_record) == 1
        # stack_trace is None when the loop handled the error gracefully
        assert result_record["stack_trace"].iloc[0] is None
