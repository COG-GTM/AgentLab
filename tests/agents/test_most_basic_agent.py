import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from agentlab.agents.most_basic_agent.most_basic_agent import MostBasicAgent, MostBasicAgentArgs
from agentlab.analyze import inspect_results
from agentlab.experiments import launch_exp
from agentlab.experiments.loop import EnvArgs, ExpArgs
from agentlab.llm.chat_api import BaseModelArgs
from agentlab.llm.llm_utils import Discussion


class _CheatMiniWoBLLM_HTML:
    """Cheat LLM that finds bids in pruned HTML format (bid="X" attributes).

    MostBasicAgent passes ``obs["pruned_html"]`` to the LLM and expects the
    response wrapped in triple-tick code blocks.
    """

    def __call__(self, messages):
        if isinstance(messages, Discussion):
            prompt = messages.to_string()
        else:
            prompt = messages[1].get("content", "")

        # pruned_html encodes bids as bid="X" attributes
        match = re.search(r'bid="(\w+)"[^>]*>.*?button', prompt, re.IGNORECASE | re.DOTALL)
        if not match:
            match = re.search(r'bid="(\w+)"', prompt)

        if match:
            bid = match.group(1)
            action = f'click("{bid}")'
        else:
            raise Exception("Can't find the button's bid in HTML")

        return dict(role="assistant", content=f"I'll click the button.\n```\n{action}\n```\n")

    def get_stats(self):
        return {}


@dataclass
class _CheatMiniWoBLLMArgs_HTML(BaseModelArgs):
    model_name: str = "test/cheat_miniwob_html"

    def make_model(self):
        return _CheatMiniWoBLLM_HTML()


def test_most_basic_agent_instantiation():
    """Test MostBasicAgentArgs can be created and produces a MostBasicAgent."""
    args = MostBasicAgentArgs(chat_model_args=_CheatMiniWoBLLMArgs_HTML())
    agent = args.make_agent()
    assert isinstance(agent, MostBasicAgent)


def test_most_basic_agent():
    """End-to-end test with a cheat LLM that understands pruned HTML bids."""
    exp_args = ExpArgs(
        agent_args=MostBasicAgentArgs(
            chat_model_args=_CheatMiniWoBLLMArgs_HTML(),
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
