import tempfile
from pathlib import Path

import bgym
from bgym import DEFAULT_BENCHMARKS

from agentlab.agents.generic_agent.agent_configs import FLAGS_GPT_3_5
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.experiments import study
from agentlab.experiments.loop import EnvArgs, ExpArgs
from agentlab.experiments.study import Study, make_study
from agentlab.llm.chat_api import CheatMiniWoBLLMArgs


def test_make_study_returns_study():
    """Test that make_study returns a Study object for a simple benchmark."""
    agent_args = GenericAgentArgs(
        chat_model_args=CheatMiniWoBLLMArgs(),
        flags=FLAGS_GPT_3_5,
    )
    result = make_study(agent_args, benchmark="miniwob_tiny_test")
    assert isinstance(result, Study)


def test_make_study_with_list_of_agents():
    """Test make_study with multiple agent args."""
    agents = []
    for i in range(2):
        agent = GenericAgentArgs(
            chat_model_args=CheatMiniWoBLLMArgs(),
            flags=FLAGS_GPT_3_5,
        )
        agent.agent_name = f"test_agent_{i}"
        agents.append(agent)

    result = make_study(agents, benchmark="miniwob_tiny_test")
    assert isinstance(result, Study)
    assert len(result.exp_args_list) > 0


def test_study_exp_args_list_populated():
    """Test that Study populates exp_args_list from benchmark."""
    agent_args = GenericAgentArgs(
        chat_model_args=CheatMiniWoBLLMArgs(),
        flags=FLAGS_GPT_3_5,
    )
    result = make_study(agent_args, benchmark="miniwob_tiny_test")
    assert result.exp_args_list is not None
    assert len(result.exp_args_list) > 0
    for exp_arg in result.exp_args_list:
        assert isinstance(exp_arg, ExpArgs)


def test_study_name_includes_agent_and_benchmark():
    """Test that Study.name contains agent and benchmark info."""
    agent_args = GenericAgentArgs(
        chat_model_args=CheatMiniWoBLLMArgs(),
        flags=FLAGS_GPT_3_5,
    )
    result = make_study(agent_args, benchmark="miniwob_tiny_test")
    assert "miniwob" in result.name.lower()


def test_study_override_max_steps():
    """Test that override_max_steps updates all experiments."""
    agent_args = GenericAgentArgs(
        chat_model_args=CheatMiniWoBLLMArgs(),
        flags=FLAGS_GPT_3_5,
    )
    result = make_study(agent_args, benchmark="miniwob_tiny_test")
    result.override_max_steps(5)
    for exp_arg in result.exp_args_list:
        assert exp_arg.env_args.max_steps == 5


def test_study_save_and_load():
    """Test that Study can be saved and loaded."""
    agent_args = GenericAgentArgs(
        chat_model_args=CheatMiniWoBLLMArgs(),
        flags=FLAGS_GPT_3_5,
    )
    result = make_study(agent_args, benchmark="miniwob_tiny_test")

    with tempfile.TemporaryDirectory() as tmp_dir:
        result.save(exp_root=Path(tmp_dir))
        loaded = Study.load(result.dir)
        assert loaded.benchmark.name == result.benchmark.name
