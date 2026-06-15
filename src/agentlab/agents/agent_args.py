from dataclasses import dataclass

import bgym
from bgym import AbstractAgentArgs, Benchmark


class AgentArgs(AbstractAgentArgs):
    """Base class for agent arguments for instantiating an agent.

    Define agent arguments as dataclass variables of this class. For example:

    class MyAgentArgs(AgentArgs):
        my_arg: str = "default_value"
        my_other_arg: int = 42

    Note: for working properly with AgentXRay, the arguments need to be serializable and hasable.
    """

    def set_benchmark(self, benchmark: Benchmark, demo_mode: bool):
        """Optional method to set benchmark specific flags.

        This allows the agent to have minor adjustments based on the benchmark.
        E.g. using a benchmark specific action space. Or letting the agent see
        HTML on MiniWoB since AXTree is not enough. Users should avoid making
        extensive benchmark specific prompt engineering.

        Args:
            benchmark: str
                Name of the benchmark.
            demo_mode: bool
                If True, the agent should adapt to demo mode. E.g. it can set
                the demo_mode flag in the browsergym action space.
        """
        pass

    def set_reproducibility_mode(self):
        """Optional method to set the agent in a reproducibility mode.

        This should adjust the agent configuration to make it as deterministic
        as possible e.g. setting the temperature of the model to 0.

        This is only called when reproducibility is requested.

        Raises:
            NotImplementedError: If the agent does not support reproducibility.
        """
        raise NotImplementedError(
            f"set_reproducibility_mode is not implemented for agent_args {self.__class__.__name__}"
        )


@dataclass
class ChatModelAgentArgs(AgentArgs):
    """Base class for agents whose LLM is configured via ``chat_model_args``.

    Provides the ``prepare``/``close`` implementations shared by every agent
    that holds a ``chat_model_args`` factory, starting and stopping the
    underlying LLM server. Subclasses are still responsible for declaring the
    ``chat_model_args`` field and implementing ``make_agent``.
    """

    def prepare(self):
        return self.chat_model_args.prepare_server()

    def close(self):
        return self.chat_model_args.close_server()
