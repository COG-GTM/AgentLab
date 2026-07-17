import pytest

from agentlab.llm.prompt_templates import (
    STARCHAT_PROMPT_TEMPLATE,
    PromptTemplate,
    get_prompt_template,
)


def test_format_message_roles():
    template = PromptTemplate(system="S:{input}\n", human="H:{input}\n", ai="A:{input}\n")
    assert template.format_message({"role": "system", "content": "x"}) == "S:x\n"
    assert template.format_message({"role": "user", "content": "y"}) == "H:y\n"
    assert template.format_message({"role": "assistant", "content": "z"}) == "A:z\n"


def test_format_message_unsupported_role():
    template = PromptTemplate(system="{input}", human="{input}", ai="{input}")
    with pytest.raises(ValueError):
        template.format_message({"role": "tool", "content": "x"})


def test_construct_prompt():
    template = PromptTemplate(
        system="S:{input}\n", human="H:{input}\n", ai="A:{input}\n", prompt_end="A:"
    )
    prompt = template.construct_prompt(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
    )
    assert prompt == "S:sys\nH:hi\nA:"


def test_construct_prompt_invalid_messages():
    template = PromptTemplate(system="{input}", human="{input}", ai="{input}")
    with pytest.raises(ValueError):
        template.construct_prompt(["not a dict"])


def test_get_prompt_template():
    assert get_prompt_template("starcoder-15b") is STARCHAT_PROMPT_TEMPLATE
    assert get_prompt_template("starchat-beta") is STARCHAT_PROMPT_TEMPLATE
    with pytest.raises(NotImplementedError):
        get_prompt_template("unknown-model")
