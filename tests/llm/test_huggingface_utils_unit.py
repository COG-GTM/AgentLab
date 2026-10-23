"""Offline unit tests for HFBaseChatModel using a fake tokenizer and LLM."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from transformers import GPT2TokenizerFast

from agentlab.llm import huggingface_utils
from agentlab.llm.huggingface_utils import HFBaseChatModel, _prepend_system_to_first_user
from agentlab.llm.llm_utils import Discussion, HumanMessage, SystemMessage
from agentlab.llm.prompt_templates import STARCHAT_PROMPT_TEMPLATE


class _FakeTokenizer:
    def __init__(self, fail_with=None):
        self.fail_with = fail_with
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False):
        self.calls.append([dict(m) for m in messages])
        if self.fail_with and any(m["role"] == "system" for m in messages):
            raise Exception(self.fail_with)
        return "|".join(f"{m['role']}:{m['content']}" for m in messages)


def _make_model(monkeypatch, tokenizer, base_model_name=None, n_retry=3, log_probs=False):
    loaded = []

    def from_pretrained(name):
        loaded.append(name)
        return tokenizer

    monkeypatch.setattr(huggingface_utils.AutoTokenizer, "from_pretrained", from_pretrained)
    monkeypatch.setattr(huggingface_utils.time, "sleep", lambda s: None)
    model = HFBaseChatModel("model", base_model_name, n_retry, log_probs)
    model.temperature = 0.5
    model.llm = MagicMock(return_value="answer")
    return model, loaded


def _messages():
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]


def test_loads_tokenizer_from_base_model(monkeypatch):
    _, loaded = _make_model(monkeypatch, _FakeTokenizer(), base_model_name="base")
    assert loaded == ["base"]
    _, loaded = _make_model(monkeypatch, _FakeTokenizer())
    assert loaded == ["model"]


def test_call_with_chat_template(monkeypatch):
    model, _ = _make_model(monkeypatch, _FakeTokenizer())
    res = model(_messages())
    assert res["content"] == "answer"
    model.llm.assert_called_once_with("system:sys|user:hi", temperature=0.5)
    assert model._llm_type() == "huggingface"


def test_call_merges_discussion(monkeypatch):
    tokenizer = _FakeTokenizer()
    model, _ = _make_model(monkeypatch, tokenizer)
    msg = HumanMessage("a")
    msg.add_text("b")
    model(Discussion([SystemMessage("sys"), msg]), temperature=0.1)
    assert tokenizer.calls[0][1]["content"] == "a\nb"
    assert model.llm.call_args.kwargs["temperature"] == 0.1


def test_call_falls_back_when_system_role_unsupported(monkeypatch):
    tokenizer = _FakeTokenizer(fail_with="Conversation roles must alternate user/assistant")
    model, _ = _make_model(monkeypatch, tokenizer)
    model(_messages())
    model.llm.assert_called_once_with("user:sys\nhi", temperature=0.5)


def test_call_reraises_other_template_errors(monkeypatch):
    model, _ = _make_model(monkeypatch, _FakeTokenizer(fail_with="other"))
    with pytest.raises(Exception, match="other"):
        model(_messages())


def test_gpt2_tokenizer_uses_prompt_template(monkeypatch):
    gpt2 = MagicMock(spec=GPT2TokenizerFast)
    monkeypatch.setattr(huggingface_utils.AutoTokenizer, "from_pretrained", lambda n: gpt2)
    model = HFBaseChatModel("bigcode/starcoder", None, 1, False)
    model.temperature = 0.5
    model.llm = MagicMock(return_value="x")
    assert model.tokenizer is None
    assert model.prompt_template is STARCHAT_PROMPT_TEMPLATE
    model(_messages())
    assert model.llm.call_args.args[0].startswith("<|system|>")


def test_multiple_samples_and_retries(monkeypatch):
    model, _ = _make_model(monkeypatch, _FakeTokenizer(), n_retry=3)
    model.llm.side_effect = [RuntimeError("busy"), "a", "b"]
    res = model(_messages(), n_samples=2)
    assert [r["content"] for r in res] == ["a", "b"]
    assert model.llm.call_count == 3


def test_retries_exhausted(monkeypatch):
    model, _ = _make_model(monkeypatch, _FakeTokenizer(), n_retry=2)
    model.llm.side_effect = RuntimeError("down")
    with pytest.raises(RuntimeError, match="down"):
        model(_messages())
    assert model.llm.call_count == 2


def test_log_probs(monkeypatch):
    model, _ = _make_model(monkeypatch, _FakeTokenizer(), log_probs=True)
    model.llm.return_value = SimpleNamespace(generated_text="txt", details="details")
    res = model(_messages())
    assert res["content"] == "txt" and res["log_probs"] == "details"


def test_prepend_system_to_first_user():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "user", "content": "q2"},
    ]
    out = _prepend_system_to_first_user(messages)
    assert out == [{"role": "user", "content": "sys\nq1"}, {"role": "user", "content": "q2"}]


def test_prepend_system_without_system_or_user():
    no_system = [{"role": "user", "content": "q"}]
    assert _prepend_system_to_first_user(no_system) == [{"role": "user", "content": "q"}]
    only_system = [{"role": "system", "content": "s"}]
    assert _prepend_system_to_first_user(only_system) == [{"role": "system", "content": "s"}]


def test_prepend_system_with_column_remap():
    messages = [{"r": "system", "t": "sys"}, {"r": "human", "t": "q"}]
    out = _prepend_system_to_first_user(
        messages, column_remap={"HumanMessage": "human", "role": "r", "text": "t"}
    )
    assert out == [{"r": "human", "t": "sys\nq"}]
