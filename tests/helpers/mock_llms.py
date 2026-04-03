"""Shared mock LLM classes for testing.

These mock LLMs simulate various behaviors (parse retries, LLM errors) for
unit-testing agent behavior without making real API calls.
"""

import re
from dataclasses import dataclass

from openai import OpenAIError

from agentlab.llm.chat_api import BaseModelArgs
from agentlab.llm.llm_utils import Discussion


@dataclass
class CheatMiniWoBLLM_ParseRetry:
    """For unit-testing purposes only. It only work with miniwob.click-test task."""

    n_retry: int
    retry_count: int = 0

    def __call__(self, messages) -> str:
        if self.retry_count < self.n_retry:
            self.retry_count += 1
            return dict(role="assistant", content="I'm retrying")

        if isinstance(messages, Discussion):
            prompt = messages.to_string()
        else:
            prompt = messages[1].get("content", "")
        match = re.search(r"^\s*\[(\d+)\].*button", prompt, re.MULTILINE | re.IGNORECASE)

        if match:
            bid = match.group(1)
            action = f'click("{bid}")'
        else:
            raise Exception("Can't find the button's bid")

        answer = f"""I'm clicking the button as requested.
<action>
{action}
</action>
"""
        return dict(role="assistant", content=answer)

    def get_stats(self):
        return {}


@dataclass
class CheatMiniWoBLLMArgs_ParseRetry(BaseModelArgs):
    n_retry: int = 2
    model_name: str = "test/cheat_miniwob_click_test_parse_retry"

    def make_model(self):
        return CheatMiniWoBLLM_ParseRetry(n_retry=self.n_retry)


@dataclass
class CheatLLM_LLMError:
    """For unit-testing purposes only. Fails to call LLM"""

    n_retry: int = 0
    success: bool = False

    def __call__(self, messages) -> str:
        if self.success:
            if isinstance(messages, Discussion):
                prompt = messages.to_string()
            else:
                prompt = messages[1].get("content", "")
            match = re.search(r"^\s*\[(\d+)\].*button", prompt, re.MULTILINE | re.IGNORECASE)

            if match:
                bid = match.group(1)
                action = f'click("{bid}")'
            else:
                raise Exception("Can't find the button's bid")

            answer = f"""I'm clicking the button as requested.
    <action>
    {action}
    </action>
    """
            return dict(role="assistant", content=answer)
        raise OpenAIError("LLM failed to respond")

    def get_stats(self):
        return {"n_llm_retry": self.n_retry, "n_llm_busted_retry": int(not self.success)}


@dataclass
class CheatLLMArgs_LLMError(BaseModelArgs):
    n_retry: int = 2
    success: bool = False
    model_name: str = "test/cheat_miniwob_click_test_parse_retry"

    def make_model(self):
        return CheatLLM_LLMError(
            n_retry=self.n_retry,
            success=self.success,
        )
