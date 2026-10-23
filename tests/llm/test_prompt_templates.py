import pytest

from agentlab.llm.prompt_templates import (
    STARCHAT_PROMPT_TEMPLATE,
    PromptTemplate,
    get_prompt_template,
)


def test_format_message_per_role():
    t = PromptTemplate(system="S:{input}|", human="H:{input}|", ai="A:{input}|")
    assert t.format_message({"role": "system", "content": "a"}) == "S:a|"
    assert t.format_message({"role": "user", "content": "b"}) == "H:b|"
    assert t.format_message({"role": "assistant", "content": "c"}) == "A:c|"


def test_format_message_unknown_role():
    with pytest.raises(ValueError, match="not supported"):
        STARCHAT_PROMPT_TEMPLATE.format_message({"role": "tool", "content": "x"})


def test_construct_prompt_starchat():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    prompt = STARCHAT_PROMPT_TEMPLATE.construct_prompt(messages)
    assert prompt == (
        "<|system|>\nsys<|end|>\n"
        "<|user|>\nhi<|end|>\n"
        "<|assistant|>\nhello<|end|>\n"
        "<|assistant|>"
    )


@pytest.mark.parametrize("bad", [["not a dict"], [{"role": "user"}], [{"content": "x"}]])
def test_construct_prompt_rejects_non_openai_messages(bad):
    with pytest.raises(ValueError, match="openai format"):
        STARCHAT_PROMPT_TEMPLATE.construct_prompt(bad)


@pytest.mark.parametrize("name", ["bigcode/starcoder", "HuggingFaceH4/starchat-beta"])
def test_get_prompt_template_known(name):
    assert get_prompt_template(name) is STARCHAT_PROMPT_TEMPLATE


def test_get_prompt_template_unknown():
    with pytest.raises(NotImplementedError):
        get_prompt_template("meta-llama/Llama-3")
