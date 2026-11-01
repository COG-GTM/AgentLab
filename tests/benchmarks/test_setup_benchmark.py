import os
from unittest.mock import patch

import pytest

from agentlab.benchmarks import setup_benchmark
from agentlab.benchmarks.setup_benchmark import (
    _ensure_repo,
    _write_env_kv,
    ensure_benchmark,
    setup_miniwob,
)


def test_write_env_kv_creates_file(tmp_path):
    env = tmp_path / "sub" / ".env"
    _write_env_kv(env, "KEY", "value")
    assert env.read_text() == "KEY=value\n"


def test_write_env_kv_updates_and_dedupes(tmp_path):
    env = tmp_path / ".env"
    env.write_text("A=1\nKEY=old\nB=2\n  KEY=older\n")
    _write_env_kv(env, "KEY", "new")
    assert env.read_text() == "A=1\nKEY=new\nB=2\n"


def test_write_env_kv_appends_when_missing(tmp_path):
    env = tmp_path / ".env"
    env.write_text("A=1\n")
    _write_env_kv(env, "KEY", "v")
    _write_env_kv(env, "KEY", "v")  # idempotent
    assert env.read_text() == "A=1\nKEY=v\n"


def test_write_env_kv_does_not_match_prefix_keys(tmp_path):
    env = tmp_path / ".env"
    env.write_text("KEYS=1\n")
    _write_env_kv(env, "KEY", "2")
    assert env.read_text() == "KEYS=1\nKEY=2\n"


def test_ensure_repo_clones_and_checks_out(tmp_path):
    clone_dir = tmp_path / "repo"
    with patch.object(setup_benchmark.os, "system") as system:
        _ensure_repo("https://example.com/r.git", clone_dir, commit="abc")
    # clone_dir does not exist and no .git appears, so only the clone is attempted
    assert system.call_count == 1
    assert "git clone 'https://example.com/r.git'" in system.call_args.args[0]


def test_ensure_repo_existing_repo_checks_out_commit(tmp_path):
    clone_dir = tmp_path / "repo"
    (clone_dir / ".git").mkdir(parents=True)
    with patch.object(setup_benchmark.os, "system") as system:
        _ensure_repo("https://example.com/r.git", clone_dir, commit="abc")
    cmds = [c.args[0] for c in system.call_args_list]
    assert len(cmds) == 2
    assert "fetch --all --tags" in cmds[0]
    assert "checkout abc" in cmds[1]


def test_ensure_repo_existing_without_commit_does_nothing(tmp_path):
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    with patch.object(setup_benchmark.os, "system") as system:
        _ensure_repo("u", tmp_path / "repo")
    system.assert_not_called()


def test_setup_miniwob_writes_env(tmp_path, monkeypatch):
    monkeypatch.delenv("MINIWOB_URL", raising=False)
    with patch.object(setup_benchmark, "_ensure_repo") as ensure_repo:
        url = setup_miniwob(tmp_path)
    ensure_repo.assert_called_once()
    assert url.startswith("file://") and url.endswith("/miniwob/html/miniwob/")
    assert os.environ["MINIWOB_URL"] == url
    assert (tmp_path / ".env").read_text() == f"MINIWOB_URL={url}\n"


@pytest.mark.parametrize("name", ["miniwob", " MiniWob++ ", "miniwob-plusplus"])
def test_ensure_benchmark_runs_setup_when_url_missing(tmp_path, monkeypatch, name):
    monkeypatch.delenv("MINIWOB_URL", raising=False)
    with patch.object(setup_benchmark, "setup_miniwob", return_value="url") as setup:
        assert ensure_benchmark(name, tmp_path) == "url"
    setup.assert_called_once_with(tmp_path)


def test_ensure_benchmark_skips_when_url_set(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIWOB_URL", "http://localhost/miniwob/")
    with patch.object(setup_benchmark, "setup_miniwob") as setup:
        assert ensure_benchmark("miniwob", tmp_path) is None
    setup.assert_not_called()


def test_ensure_benchmark_other_benchmark_is_noop(tmp_path):
    with patch.object(setup_benchmark, "setup_miniwob") as setup:
        assert ensure_benchmark("webarena", tmp_path) is None
    setup.assert_not_called()
