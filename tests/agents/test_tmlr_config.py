from agentlab.agents.generic_agent import tmlr_config
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT

LLM = "openai/gpt-4o-mini-2024-07-18"


def test_base_agent():
    agent = tmlr_config.get_base_agent(LLM)
    assert agent.chat_model_args is CHAT_MODEL_ARGS_DICT[LLM]
    assert agent.flags.obs.use_screenshot is False
    assert agent.flags.obs.use_som is False


def test_vision_agent_does_not_mutate_base_flags():
    agent = tmlr_config.get_vision_agent(LLM)
    assert agent.flags.obs.use_screenshot is True
    assert agent.flags.obs.use_som is False
    assert agent.agent_name.endswith("_vision")
    assert tmlr_config.BASE_FLAGS.obs.use_screenshot is False


def test_som_agent():
    agent = tmlr_config.get_som_agent(LLM)
    assert agent.flags.obs.use_screenshot is True
    assert agent.flags.obs.use_som is True
    assert tmlr_config.BASE_FLAGS.obs.use_som is False
