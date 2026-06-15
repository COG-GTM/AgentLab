"""
GenericAgent implementation for AgentLab

This module defines a `GenericAgent` class and its associated arguments for use in the AgentLab framework. \
The `GenericAgent` class is designed to interact with a chat-based model to determine actions based on \
observations. It includes methods for preprocessing observations, generating actions, and managing internal \
state such as plans, memories, and thoughts. The `GenericAgentArgs` class provides configuration options for \
the agent, including model arguments and flags for various behaviors.
"""

from dataclasses import dataclass

import bgym
from bgym import Benchmark
from browsergym.experiments.agent import Agent

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.agent_args import ChatModelAgentArgs
from agentlab.agents.agent_utils import busted_retry_ans_dict, make_agent_info
from agentlab.llm.chat_api import BaseModelArgs
from agentlab.llm.llm_utils import Discussion, ParseError, SystemMessage, retry
from agentlab.llm.tracking import cost_tracker_decorator

from .visual_agent_prompts import MainPrompt, PromptFlags


@dataclass
class VisualAgentArgs(ChatModelAgentArgs):
    chat_model_args: BaseModelArgs = None
    flags: PromptFlags = None
    max_retry: int = 4

    def __post_init__(self):
        try:  # some attributes might be missing temporarily due to args.CrossProd for hyperparameter generation
            self.agent_name = f"VisualAgent-{self.chat_model_args.model_name}".replace("/", "_")
        except AttributeError:
            pass

    def set_benchmark(self, benchmark: Benchmark, demo_mode):
        """Override Some flags based on the benchmark."""
        self.flags.obs.use_tabs = benchmark.is_multi_tab

    def set_reproducibility_mode(self):
        self.chat_model_args.temperature = 0

    def make_agent(self):
        return VisualAgent(
            chat_model_args=self.chat_model_args, flags=self.flags, max_retry=self.max_retry
        )


class VisualAgent(Agent):

    def __init__(
        self,
        chat_model_args: BaseModelArgs,
        flags: PromptFlags,
        max_retry: int = 4,
    ):

        self.chat_llm = chat_model_args.make_model()
        self.chat_model_args = chat_model_args
        self.max_retry = max_retry

        self.flags = flags
        self.action_set = self.flags.action.action_set.make_action_set()
        self._obs_preprocessor = dp.make_obs_preprocessor(flags.obs)

        self.reset(seed=None)

    def obs_preprocessor(self, obs: dict) -> dict:
        return self._obs_preprocessor(obs)

    @cost_tracker_decorator
    def get_action(self, obs):

        main_prompt = MainPrompt(
            action_set=self.action_set,
            obs=obs,
            actions=self.actions,
            thoughts=self.thoughts,
            flags=self.flags,
        )

        system_prompt = SystemMessage(dp.SystemPrompt().prompt)
        try:
            # TODO, we would need to further shrink the prompt if the retry
            # cause it to be too long

            chat_messages = Discussion([system_prompt, main_prompt.prompt])
            ans_dict = retry(
                self.chat_llm,
                chat_messages,
                n_retry=self.max_retry,
                parser=main_prompt._parse_answer,
            )
            ans_dict["busted_retry"] = 0
            # inferring the number of retries, TODO: make this less hacky
            ans_dict["n_retry"] = (len(chat_messages) - 3) / 2
        except ParseError:
            ans_dict = busted_retry_ans_dict(self.max_retry)

        self.actions.append(ans_dict["action"])
        self.thoughts.append(ans_dict.get("think", None))

        agent_info = make_agent_info(self.chat_llm, ans_dict, chat_messages, self.chat_model_args)
        return ans_dict["action"], agent_info

    def reset(self, seed=None):
        self.seed = seed
        self.thoughts = []
        self.actions = []
