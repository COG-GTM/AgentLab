import numpy as np
import pandas as pd
import pytest
import requests

from agentlab.agents.tool_use_agent.tool_use_agent import TaskHint

HINT_DB = pd.DataFrame(
    {
        "task_name": ["task.a", "task.b", "task.c"],
        "semantic_keys": ["forms", "search", "navigation"],
        "hint": ["fill the form", "use the search bar", "use the back button"],
    }
)


def make_hint(**kwargs) -> TaskHint:
    hint_block = TaskHint(**kwargs)
    hint_block.hint_db = HINT_DB.copy()
    return hint_block


def test_choose_hints_dispatches_to_each_mode(monkeypatch):
    calls = {}

    def record(name, value):
        calls[name] = value
        return [name]

    for mode, expected_arg in [("llm", "the goal"), ("direct", "task.a"), ("emb", "the goal")]:
        hint_block = make_hint(hint_retrieval_mode=mode)
        monkeypatch.setattr(
            hint_block, "choose_hints_llm", lambda llm, goal: record("llm", goal), raising=False
        )
        monkeypatch.setattr(
            hint_block,
            "choose_hints_direct",
            lambda task_name: record("direct", task_name),
            raising=False,
        )
        monkeypatch.setattr(
            hint_block, "choose_hints_emb", lambda goal: record("emb", goal), raising=False
        )

        assert hint_block.choose_hints(None, "task.a", "the goal") == [mode]
        assert calls[mode] == expected_arg


def test_choose_hints_unknown_mode_raises():
    hint_block = make_hint()
    hint_block.hint_retrieval_mode = "unknown"

    with pytest.raises(ValueError, match="Unknown hint retrieval mode: unknown"):
        hint_block.choose_hints(None, "task.a", "the goal")


def test_choose_hints_emb_returns_top_n(monkeypatch):
    hint_block = make_hint(hint_retrieval_mode="emb", top_n=2)
    hint_block.uniq_hints = HINT_DB.copy()
    hint_block.hint_embeddings = np.eye(3)

    monkeypatch.setattr(hint_block, "_encode", lambda texts, prompt="": np.array([[1.0, 0.0, 0.0]]))
    monkeypatch.setattr(
        hint_block, "_similarity", lambda texts1, texts2: np.array([[0.1, 0.9, 0.5]])
    )

    # sorted by increasing similarity, so the most relevant hint comes last
    assert hint_block.choose_hints_emb("the goal") == ["use the back button", "use the search bar"]


def test_encode_retries_then_returns(monkeypatch):
    hint_block = make_hint(hint_retrieval_mode="emb")
    attempts = []

    class Response:
        def json(self):
            return {"embeddings": [[1.0, 2.0]]}

    def post(url, json, timeout):
        attempts.append(url)
        if len(attempts) < 3:
            raise requests.exceptions.Timeout("boom")
        return Response()

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr("agentlab.agents.tool_use_agent.tool_use_agent.time.sleep", lambda s: None)

    embeddings = hint_block._encode(["the goal"], prompt="task description")

    assert len(attempts) == 3
    np.testing.assert_allclose(embeddings, np.array([[1.0, 2.0]]))


def test_encode_reraises_after_max_retries(monkeypatch):
    hint_block = make_hint(hint_retrieval_mode="emb")
    attempts = []

    def post(url, json, timeout):
        attempts.append(url)
        raise requests.exceptions.ConnectionError("no server")

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr("agentlab.agents.tool_use_agent.tool_use_agent.time.sleep", lambda s: None)

    with pytest.raises(requests.exceptions.ConnectionError):
        hint_block._encode(["the goal"], max_retries=3)

    assert len(attempts) == 3
