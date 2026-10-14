from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from transformers import GPT2TokenizerFast

from agentlab.llm import huggingface_utils
from agentlab.llm.huggingface_utils import HFBaseChatModel, _prepend_system_to_first_user
from agentlab.llm.llm_utils import AIMessage, Discussion, HumanMessage, SystemMessage
from agentlab.llm.prompt_templates import STARCHAT_PROMPT_TEMPLATE

# ---------------------------------------------------------------------------
# _prepend_system_to_first_user
# ---------------------------------------------------------------------------


def test_prepend_system_merges_into_first_user_and_removes_system():
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "second"},
    ]
    out = _prepend_system_to_first_user(messages)
    assert out is messages  # modified in place
    assert out == [
        {"role": "user", "content": "SYS\nfirst"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "second"},
    ]


def test_prepend_system_only_uses_first_system_message():
    messages = [
        {"role": "system", "content": "S1"},
        {"role": "system", "content": "S2"},
        {"role": "user", "content": "u"},
    ]
    out = _prepend_system_to_first_user(messages)
    assert out == [{"role": "system", "content": "S2"}, {"role": "user", "content": "S1\nu"}]


def test_prepend_system_without_system_message_is_noop():
    messages = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
    expected = [dict(m) for m in messages]
    assert _prepend_system_to_first_user(messages) == expected


def test_prepend_system_without_user_message_keeps_system():
    messages = [{"role": "system", "content": "S"}, {"role": "assistant", "content": "a"}]
    expected = [dict(m) for m in messages]
    assert _prepend_system_to_first_user(messages) == expected


def test_prepend_system_system_after_user():
    messages = [{"role": "user", "content": "u"}, {"role": "system", "content": "S"}]
    assert _prepend_system_to_first_user(messages) == [{"role": "user", "content": "S\nu"}]


def test_prepend_system_with_column_remap():
    messages = [
        {"speaker": "system", "text": "S"},
        {"speaker": "human", "text": "hello"},
    ]
    remap = {"HumanMessage": "human", "role": "speaker", "text": "text"}
    out = _prepend_system_to_first_user(messages, column_remap=remap)
    assert out == [{"speaker": "human", "text": "S\nhello"}]


def test_prepend_system_default_column_remap_is_not_mutated():
    _prepend_system_to_first_user([{"role": "system", "content": "S"}])
    defaults = _prepend_system_to_first_user.__defaults__
    assert defaults == ({},)


# ---------------------------------------------------------------------------
# HFBaseChatModel
# ---------------------------------------------------------------------------


@pytest.fixture
def no_sleep(monkeypatch):
    sleep = Mock()
    monkeypatch.setattr(huggingface_utils.time, "sleep", sleep)
    return sleep


@pytest.fixture
def fake_auto_tokenizer(monkeypatch):
    auto = Mock()
    monkeypatch.setattr(huggingface_utils, "AutoTokenizer", auto)
    return auto


def _make_model(fake_auto_tokenizer, tokenizer=None, n_retry_server=3, log_probs=False, **kw):
    fake_auto_tokenizer.from_pretrained.return_value = tokenizer or Mock(name="tokenizer")
    model = HFBaseChatModel(
        model_name=kw.get("model_name", "org/model"),
        base_model_name=kw.get("base_model_name"),
        n_retry_server=n_retry_server,
        log_probs=log_probs,
    )
    model.temperature = kw.get("temperature", 0.5)
    return model


def test_init_loads_tokenizer_from_model_name_when_no_base(fake_auto_tokenizer):
    model = _make_model(fake_auto_tokenizer, model_name="org/model", base_model_name=None)
    fake_auto_tokenizer.from_pretrained.assert_called_once_with("org/model")
    assert model.tokenizer is fake_auto_tokenizer.from_pretrained.return_value
    assert model.n_retry_server == 3
    assert model.log_probs is False
    assert model._llm_type() == "huggingface"


def test_init_loads_tokenizer_from_base_model_name(fake_auto_tokenizer):
    _make_model(fake_auto_tokenizer, model_name="org/ft", base_model_name="org/base")
    fake_auto_tokenizer.from_pretrained.assert_called_once_with("org/base")


def test_init_gpt2_tokenizer_falls_back_to_prompt_template(fake_auto_tokenizer):
    gpt2_tok = Mock(spec=GPT2TokenizerFast)
    model = _make_model(fake_auto_tokenizer, tokenizer=gpt2_tok, model_name="bigcode/starcoder")
    assert model.tokenizer is None
    assert model.prompt_template is STARCHAT_PROMPT_TEMPLATE


def test_init_gpt2_tokenizer_unknown_model_raises(fake_auto_tokenizer):
    with pytest.raises(NotImplementedError):
        _make_model(fake_auto_tokenizer, tokenizer=Mock(spec=GPT2TokenizerFast), model_name="x/y")


def test_call_uses_chat_template_and_returns_ai_message(fake_auto_tokenizer, no_sleep):
    tok = Mock()
    tok.apply_chat_template.return_value = "PROMPT"
    model = _make_model(fake_auto_tokenizer, tokenizer=tok, temperature=0.3)
    model.llm = Mock(return_value="the answer")

    messages = [{"role": "user", "content": "hi"}]
    out = model(messages)

    tok.apply_chat_template.assert_called_once_with(messages, tokenize=False)
    model.llm.assert_called_once_with("PROMPT", temperature=0.3)
    assert isinstance(out, AIMessage)
    assert out["role"] == "assistant"
    assert out["content"] == "the answer"
    no_sleep.assert_not_called()


def test_call_explicit_temperature_overrides_default(fake_auto_tokenizer, no_sleep):
    tok = Mock(apply_chat_template=Mock(return_value="P"))
    model = _make_model(fake_auto_tokenizer, tokenizer=tok, temperature=0.3)
    model.llm = Mock(return_value="x")
    model([{"role": "user", "content": "hi"}], temperature=0.0)
    model.llm.assert_called_once_with("P", temperature=0.0)


def test_call_merges_discussion_before_templating(fake_auto_tokenizer, no_sleep):
    captured = {}

    def apply_chat_template(messages, tokenize):
        captured["content"] = messages[0]["content"]
        return "P"

    tok = Mock(apply_chat_template=Mock(side_effect=apply_chat_template))
    model = _make_model(fake_auto_tokenizer, tokenizer=tok)
    model.llm = Mock(return_value="x")

    msg = HumanMessage("part 1")
    msg.add_text("part 2")
    model(Discussion([msg]))
    assert captured["content"] == "part 1\npart 2"


def test_call_retries_chat_template_without_system_role(fake_auto_tokenizer, no_sleep):
    seen = []

    def apply_chat_template(messages, tokenize):
        seen.append([dict(m) for m in messages])
        if any(m["role"] == "system" for m in messages):
            raise ValueError("Conversation roles must alternate user/assistant/user/assistant")
        return "P2"

    tok = Mock(apply_chat_template=Mock(side_effect=apply_chat_template))
    model = _make_model(fake_auto_tokenizer, tokenizer=tok)
    model.llm = Mock(return_value="ok")

    out = model([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}])
    assert out["content"] == "ok"
    assert seen[-1] == [{"role": "user", "content": "S\nU"}]
    model.llm.assert_called_once_with("P2", temperature=0.5)


def test_call_reraises_other_chat_template_errors(fake_auto_tokenizer, no_sleep):
    tok = Mock(apply_chat_template=Mock(side_effect=RuntimeError("boom")))
    model = _make_model(fake_auto_tokenizer, tokenizer=tok)
    model.llm = Mock()
    with pytest.raises(RuntimeError, match="boom"):
        model([{"role": "user", "content": "U"}])
    model.llm.assert_not_called()


def test_call_uses_prompt_template_when_no_tokenizer(fake_auto_tokenizer, no_sleep):
    model = _make_model(
        fake_auto_tokenizer, tokenizer=Mock(spec=GPT2TokenizerFast), model_name="starchat"
    )
    model.llm = Mock(return_value="resp")
    model([{"role": "user", "content": "U"}])
    model.llm.assert_called_once_with("<|user|>\nU<|end|>\n<|assistant|>", temperature=0.5)


def test_call_n_samples_returns_list(fake_auto_tokenizer, no_sleep):
    tok = Mock(apply_chat_template=Mock(return_value="P"))
    model = _make_model(fake_auto_tokenizer, tokenizer=tok)
    model.llm = Mock(side_effect=["a", "b", "c"])
    out = model([{"role": "user", "content": "U"}], n_samples=3)
    assert isinstance(out, list)
    assert [m["content"] for m in out] == ["a", "b", "c"]
    assert model.llm.call_count == 3


def test_call_retries_server_errors_then_succeeds(fake_auto_tokenizer, no_sleep):
    tok = Mock(apply_chat_template=Mock(return_value="P"))
    model = _make_model(fake_auto_tokenizer, tokenizer=tok, n_retry_server=3)
    model.llm = Mock(side_effect=[ConnectionError("down"), TimeoutError("slow"), "finally"])
    out = model([{"role": "user", "content": "U"}])
    assert out["content"] == "finally"
    assert model.llm.call_count == 3
    assert no_sleep.call_count == 2
    no_sleep.assert_called_with(5)


def test_call_raises_last_error_after_n_retry_server(fake_auto_tokenizer, no_sleep):
    tok = Mock(apply_chat_template=Mock(return_value="P"))
    model = _make_model(fake_auto_tokenizer, tokenizer=tok, n_retry_server=2)
    model.llm = Mock(side_effect=[ConnectionError("1"), ConnectionError("2"), "never"])
    with pytest.raises(ConnectionError, match="2"):
        model([{"role": "user", "content": "U"}])
    assert model.llm.call_count == 2
    assert no_sleep.call_count == 1


def test_call_retry_counter_resets_per_sample(fake_auto_tokenizer, no_sleep):
    tok = Mock(apply_chat_template=Mock(return_value="P"))
    model = _make_model(fake_auto_tokenizer, tokenizer=tok, n_retry_server=2)
    model.llm = Mock(side_effect=[ConnectionError(), "a", ConnectionError(), "b"])
    out = model([{"role": "user", "content": "U"}], n_samples=2)
    assert [m["content"] for m in out] == ["a", "b"]


def test_call_log_probs_extracts_generated_text_and_details(fake_auto_tokenizer, no_sleep):
    tok = Mock(apply_chat_template=Mock(return_value="P"))
    model = _make_model(fake_auto_tokenizer, tokenizer=tok, log_probs=True)
    details = {"tokens": [{"text": "hi", "logprob": -0.1}]}
    model.llm = Mock(return_value=SimpleNamespace(generated_text="hi", details=details))
    out = model([{"role": "user", "content": "U"}])
    assert out["content"] == "hi"
    assert out["log_probs"] == details
