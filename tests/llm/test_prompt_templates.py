import pytest

from agentlab.llm.prompt_templates import (
    MODEL_PREFIX_TO_PROMPT_TEMPLATES,
    STARCHAT_PROMPT_TEMPLATE,
    PromptTemplate,
    get_prompt_template,
)


@pytest.fixture
def template():
    return PromptTemplate(
        system="S:{input}|", human="H:{input}|", ai="A:{input}|", prompt_end="END"
    )


def test_format_message_each_role(template):
    assert template.format_message({"role": "system", "content": "x"}) == "S:x|"
    assert template.format_message({"role": "user", "content": "y"}) == "H:y|"
    assert template.format_message({"role": "assistant", "content": "z"}) == "A:z|"


def test_format_message_unknown_role_raises(template):
    with pytest.raises(ValueError, match="tool"):
        template.format_message({"role": "tool", "content": "x"})


def test_construct_prompt_concatenates_and_appends_end(template):
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert template.construct_prompt(messages) == "S:sys|H:hi|A:hello|END"


def test_construct_prompt_empty_messages(template):
    assert template.construct_prompt([]) == "END"


@pytest.mark.parametrize(
    "messages",
    [
        ["not a dict"],
        [{"role": "user"}],
        [{"content": "missing role"}],
    ],
)
def test_construct_prompt_rejects_non_openai_format(template, messages):
    with pytest.raises(ValueError, match="openai format"):
        template.construct_prompt(messages)


def test_default_prompt_end_is_empty():
    t = PromptTemplate(system="{input}", human="{input}", ai="{input}")
    assert t.construct_prompt([{"role": "user", "content": "a"}]) == "a"


def test_starchat_template_format():
    prompt = STARCHAT_PROMPT_TEMPLATE.construct_prompt(
        [{"role": "system", "content": "be nice"}, {"role": "user", "content": "hi"}]
    )
    assert prompt == "<|system|>\nbe nice<|end|>\n<|user|>\nhi<|end|>\n<|assistant|>"


@pytest.mark.parametrize("model_name", ["bigcode/starcoder", "HuggingFaceH4/starchat-beta"])
def test_get_prompt_template_by_substring(model_name):
    assert get_prompt_template(model_name) is STARCHAT_PROMPT_TEMPLATE


def test_get_prompt_template_unknown_model():
    with pytest.raises(NotImplementedError, match="llama"):
        get_prompt_template("meta/llama")


def test_mapping_keys():
    assert set(MODEL_PREFIX_TO_PROMPT_TEMPLATES) == {"starcoder", "starchat"}
