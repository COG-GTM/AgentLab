import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import bgym
import pytest

from agentlab.agents.visualwebarena.agent import VisualWebArenaAgentArgs
from agentlab.analyze import inspect_results
from agentlab.experiments import launch_exp
from agentlab.experiments.loop import EnvArgs, ExpArgs
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT


def _messages_to_text(messages) -> str:
    """Flatten the multi-modal messages of the VisualWebArena agent into plain text."""
    chunks = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            chunks.append(content)
        else:
            chunks.extend(part["text"] for part in content if part.get("type") == "text")
    return "\n".join(chunks)


@dataclass
class CheatVisualWebArenaLLM:
    """For unit-testing purposes only. It only works with the miniwob.click-test task."""

    def __call__(self, messages) -> dict:
        prompt = _messages_to_text(messages)
        # only look at the observation, few-shot examples also contain buttons
        prompt = prompt.rsplit("OBSERVATION:", 1)[-1]
        match = re.search(r"^\s*\[(\d+)\].*button", prompt, re.MULTILINE | re.IGNORECASE)
        if match is None:
            raise Exception("Can't find the button's bid")

        action = f'click("{match.group(1)}")'
        answer = f"""\
Let's think step-by-step. The objective is to click the button. In summary, the next action I \
will perform is ```{action}```"""
        return dict(role="assistant", content=answer)

    def get_stats(self):
        return {}


@dataclass
class CheatVisualWebArenaLLMArgs:
    model_name: str = "test/cheat_visualwebarena_click_test"
    max_total_tokens: int = 10240
    max_input_tokens: int = 8000
    max_new_tokens: int = 128

    def make_model(self):
        return CheatVisualWebArenaLLM()

    def prepare_server(self):
        pass

    def close_server(self):
        pass


@pytest.mark.parametrize("observation_type", ["axtree", "axtree_som"])
def test_visualwebarena_agent_on_miniwob(observation_type):
    agent_args = VisualWebArenaAgentArgs(
        temperature=0.0,
        chat_model_args=CheatVisualWebArenaLLMArgs(),
        observation_type=observation_type,
    )
    agent_args.set_benchmark(bgym.DEFAULT_BENCHMARKS["miniwob_tiny_test"](), demo_mode=False)

    exp_args = ExpArgs(
        agent_args=agent_args,
        env_args=EnvArgs(task_name="miniwob.click-test", task_seed=42, max_steps=5, headless=True),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        launch_exp.run_experiments(
            1, [exp_args], Path(tmp_dir) / "visualwebarena_agent_test", parallel_backend="joblib"
        )
        result_record = inspect_results.load_result_df(tmp_dir, progress_fn=None)

        target = {
            "n_steps": 1,
            "cum_reward": 1.0,
            "terminated": True,
            "truncated": False,
            "err_msg": None,
            "stack_trace": None,
        }

        for key, target_val in target.items():
            assert key in result_record
            assert result_record[key].iloc[0] == target_val


@pytest.mark.pricy
def test_agent():
    with tempfile.TemporaryDirectory() as exp_dir:
        env_args = EnvArgs(
            task_name="miniwob.click-button",
            task_seed=0,
            max_steps=10,
            headless=True,
        )

        chat_model_args = CHAT_MODEL_ARGS_DICT["openai/gpt-4o-mini-2024-07-18"]

        exp_args = [
            ExpArgs(
                agent_args=VisualWebArenaAgentArgs(
                    temperature=0.1,
                    chat_model_args=chat_model_args,
                ),
                env_args=env_args,
                logging_level=logging.INFO,
            ),
            ExpArgs(
                agent_args=VisualWebArenaAgentArgs(
                    temperature=0.0,
                    chat_model_args=chat_model_args,
                ),
                env_args=env_args,
                logging_level=logging.INFO,
            ),
        ]

        for exp_arg in exp_args:
            exp_arg.agent_args.prepare()
            exp_arg.prepare(exp_dir)

        for exp_arg in exp_args:
            exp_arg.run()
            exp_arg.agent_args.close()
