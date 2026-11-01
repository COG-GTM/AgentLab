from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agentlab.llm import huggingface_utils
from agentlab.llm.huggingface_utils import HFBaseChatModel, _prepend_system_to_first_user
from agentlab.llm.llm_utils import AIMessage, Discussion, HumanMessage, SystemMessage
from agentlab.llm.prompt_templates import STARCHAT_PROMPT_TEMPLATE

MESSAGES = [
    {"role": "system", "content": "be nice"},
    {"role": "user", "content": "hi"},
]


class FakeGPT2Tokenizer:
    pass


def make_model(tokenizer=None, model_name="org/model", log_probs=False, n_retry=2):
    tokenizer = tokenizer if tokenizer is not None else MagicMock()
    with (
        patch.object(huggingface_utils, "AutoTokenizer") as auto,
        patch.object(huggingface_utils, "GPT2TokenizerFast", FakeGPT2Tokenizer),
    ):
        auto.from_pretrained.return_value = tokenizer
        model = HFBaseChatModel(model_name, None, n_retry_server=n_retry, log_probs=log_probs)
    model.temperature = 0.3
    model.llm = MagicMock(return_value="answer")
    return model, auto


def test_prepend_system_to_first_user():
    out = _prepend_system_to_first_user([dict(m) for m in MESSAGES])
    assert out == [{"role": "user", "content": "be nice\nhi"}]


def test_prepend_system_no_system_message():
    messages = [{"role": "user", "content": "hi"}]
    assert _prepend_system_to_first_user(messages) == [{"role": "user", "content": "hi"}]


def test_prepend_system_with_column_remap():
    messages = [{"type": "system", "text": "S"}, {"type": "human", "text": "U"}]
    out = _prepend_system_to_first_user(
        messages, column_remap={"HumanMessage": "human", "role": "type", "text": "text"}
    )
    assert out == [{"type": "human", "text": "S\nU"}]


def test_init_uses_base_model_name_for_tokenizer():
    with patch.object(huggingface_utils, "AutoTokenizer") as auto:
        auto.from_pretrained.return_value = MagicMock()
        HFBaseChatModel("org/finetuned", "org/base", n_retry_server=1, log_probs=False)
    auto.from_pretrained.assert_called_once_with("org/base")


def test_call_with_chat_template():
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "PROMPT"
    model, auto = make_model(tokenizer)
    auto.from_pretrained.assert_called_once_with("org/model")
    assert model.tokenizer is tokenizer and model._llm_type() == "huggingface"

    response = model(MESSAGES)
    assert isinstance(response, AIMessage)
    assert response["content"] == "answer"
    tokenizer.apply_chat_template.assert_called_once_with(MESSAGES, tokenize=False)
    model.llm.assert_called_once_with("PROMPT", temperature=0.3)


def test_call_merges_discussion_and_overrides_temperature():
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "PROMPT"
    model, _ = make_model(tokenizer)
    msg = HumanMessage("a")
    msg.add_text("b")
    model(Discussion([SystemMessage("s"), msg]), temperature=0.9)
    merged = tokenizer.apply_chat_template.call_args.args[0]
    assert merged[1]["content"] == "a\nb"
    model.llm.assert_called_once_with("PROMPT", temperature=0.9)


def test_call_retries_chat_template_without_system_role():
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.side_effect = [
        Exception("Conversation roles must alternate user/assistant"),
        "PROMPT",
    ]
    model, _ = make_model(tokenizer)
    model([dict(m) for m in MESSAGES])
    retried = tokenizer.apply_chat_template.call_args_list[1].args[0]
    assert retried == [{"role": "user", "content": "be nice\nhi"}]


def test_call_chat_template_other_error_raises():
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.side_effect = ValueError("boom")
    model, _ = make_model(tokenizer)
    with pytest.raises(ValueError, match="boom"):
        model(MESSAGES)


def test_gpt2_tokenizer_falls_back_to_prompt_template():
    model, _ = make_model(FakeGPT2Tokenizer(), model_name="bigcode/starcoder")
    assert model.tokenizer is None
    assert model.prompt_template is STARCHAT_PROMPT_TEMPLATE
    model(MESSAGES)
    prompt = model.llm.call_args.args[0]
    assert prompt == STARCHAT_PROMPT_TEMPLATE.construct_prompt(MESSAGES)


def test_multiple_samples():
    model, _ = make_model()
    model.llm.side_effect = ["a", "b", "c"]
    responses = model(MESSAGES, n_samples=3)
    assert [r["content"] for r in responses] == ["a", "b", "c"]


def test_server_retry_then_success():
    model, _ = make_model(n_retry=3)
    model.llm.side_effect = [RuntimeError("down"), RuntimeError("down"), "ok"]
    with patch.object(huggingface_utils.time, "sleep") as sleep:
        assert model(MESSAGES)["content"] == "ok"
    assert sleep.call_count == 2


def test_server_retry_exhausted():
    model, _ = make_model(n_retry=2)
    model.llm.side_effect = RuntimeError("down")
    with patch.object(huggingface_utils.time, "sleep"):
        with pytest.raises(RuntimeError, match="down"):
            model(MESSAGES)
    assert model.llm.call_count == 2


def test_log_probs():
    model, _ = make_model(log_probs=True)
    model.llm.return_value = SimpleNamespace(generated_text="txt", details={"lp": [-0.1]})
    response = model(MESSAGES)
    assert response["content"] == "txt"
    assert response["log_probs"] == {"lp": [-0.1]}
