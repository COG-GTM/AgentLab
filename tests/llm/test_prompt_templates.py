import pytest

from agentlab.llm import prompt_templates
from agentlab.llm.prompt_templates import (
    MODEL_PREFIX_TO_PROMPT_TEMPLATES,
    STARCHAT_PROMPT_TEMPLATE,
    PromptTemplate,
    get_prompt_template,
)


@pytest.fixture
def template():
    return PromptTemplate(
        system="S[{input}]", human="H[{input}]", ai="A[{input}]", prompt_end="END"
    )


@pytest.mark.parametrize(
    "role, expected",
    [("system", "S[hi]"), ("user", "H[hi]"), ("assistant", "A[hi]")],
)
def test_format_message_dispatches_on_role(template, role, expected):
    assert template.format_message({"role": role, "content": "hi"}) == expected


@pytest.mark.parametrize("role", ["tool", "function", "System", ""])
def test_format_message_unknown_role_raises(template, role):
    with pytest.raises(ValueError, match=f"Message role {role} not supported"):
        template.format_message({"role": role, "content": "hi"})


def test_construct_prompt_concatenates_in_order_and_appends_prompt_end(template):
    messages = [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    assert template.construct_prompt(messages) == "S[be nice]H[q1]A[a1]H[q2]END"


def test_construct_prompt_default_prompt_end_is_empty():
    t = PromptTemplate(system="<s>{input}", human="<h>{input}", ai="<a>{input}")
    assert t.prompt_end == ""
    assert t.construct_prompt([{"role": "user", "content": "x"}]) == "<h>x"


def test_construct_prompt_empty_list_returns_only_prompt_end(template):
    assert template.construct_prompt([]) == "END"


@pytest.mark.parametrize(
    "bad_messages",
    [
        ["not a dict"],
        [{"content": "missing role"}],
        [{"role": "user"}],
        [{"role": "user", "content": "ok"}, ("user", "tuple")],
    ],
)
def test_construct_prompt_rejects_non_openai_format(template, bad_messages):
    with pytest.raises(ValueError, match="openai format"):
        template.construct_prompt(bad_messages)


def test_construct_prompt_does_not_reformat_braces_in_content(template):
    # content is inserted via str.format, braces inside content must be kept verbatim
    assert template.construct_prompt([{"role": "user", "content": "{x} {}"}]) == "H[{x} {}]END"


def test_starchat_template_full_prompt():
    prompt = STARCHAT_PROMPT_TEMPLATE.construct_prompt(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
            {"role": "assistant", "content": "ast"},
        ]
    )
    assert prompt == (
        "<|system|>\nsys<|end|>\n"
        "<|user|>\nusr<|end|>\n"
        "<|assistant|>\nast<|end|>\n"
        "<|assistant|>"
    )


@pytest.mark.parametrize(
    "model_name",
    [
        "starcoder",
        "bigcode/starcoder2-15b",
        "HuggingFaceH4/starchat-beta",
        "my-starchat-finetune",
    ],
)
def test_get_prompt_template_matches_by_substring(model_name):
    assert get_prompt_template(model_name) is STARCHAT_PROMPT_TEMPLATE


@pytest.mark.parametrize("model_name", ["gpt-4", "meta-llama/Llama-3-8b", "", "StarCoder"])
def test_get_prompt_template_unknown_model_raises(model_name):
    with pytest.raises(NotImplementedError, match="has no supported chat template"):
        get_prompt_template(model_name)


def test_get_prompt_template_returns_first_matching_prefix(monkeypatch):
    first = PromptTemplate(system="1", human="1", ai="1")
    second = PromptTemplate(system="2", human="2", ai="2")
    monkeypatch.setattr(
        prompt_templates,
        "MODEL_PREFIX_TO_PROMPT_TEMPLATES",
        {"llama": first, "llama-3": second},
    )
    assert get_prompt_template("llama-3-70b") is first


def test_default_mapping_keys():
    assert set(MODEL_PREFIX_TO_PROMPT_TEMPLATES) == {"starcoder", "starchat"}
