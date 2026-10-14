import os
from unittest.mock import patch

import pytest

from agentlab.benchmarks import setup_benchmark as sb

PINNED_COMMIT = "7fd85d71a4b60325c6585396ec4f48377d049838"
MINIWOB_REPO = "https://github.com/Farama-Foundation/miniwob-plusplus.git"


@pytest.fixture
def isolated_miniwob_env(monkeypatch):
    """Guarantee MINIWOB_URL starts unset and is restored after the test.

    setup_miniwob writes os.environ directly, so register the key with monkeypatch first.
    """
    monkeypatch.setenv("MINIWOB_URL", "placeholder")
    monkeypatch.delenv("MINIWOB_URL")


class TestEnsureRepo:
    def test_clones_into_missing_dir_and_creates_parent(self, tmp_path):
        clone_dir = tmp_path / "deep" / "nested" / "repo"
        with patch.object(sb.os, "system", return_value=0) as system:
            sb._ensure_repo("https://example.com/r.git", clone_dir)
        assert clone_dir.parent.is_dir()
        system.assert_called_once_with(
            f"git clone 'https://example.com/r.git' '{clone_dir.resolve()}' >/dev/null 2>&1 || true"
        )

    def test_missing_dir_with_commit_but_failed_clone_skips_checkout(self, tmp_path):
        clone_dir = tmp_path / "repo"
        with patch.object(sb.os, "system", return_value=0) as system:
            sb._ensure_repo("u", clone_dir, commit="abc")
        assert system.call_count == 1
        assert "git clone" in system.call_args[0][0]

    def test_existing_git_repo_with_commit_fetches_and_checks_out(self, tmp_path):
        clone_dir = tmp_path / "repo"
        (clone_dir / ".git").mkdir(parents=True)
        with patch.object(sb.os, "system", return_value=0) as system:
            sb._ensure_repo("u", clone_dir, commit="deadbeef")
        resolved = clone_dir.resolve()
        assert [c.args[0] for c in system.call_args_list] == [
            f"git -C '{resolved}' fetch --all --tags >/dev/null 2>&1 || true",
            f"git -C '{resolved}' checkout deadbeef >/dev/null 2>&1 || true",
        ]

    def test_clone_then_checkout_when_clone_creates_git_dir(self, tmp_path):
        clone_dir = tmp_path / "repo"

        def fake_system(cmd):
            if cmd.startswith("git clone"):
                (clone_dir / ".git").mkdir(parents=True)
            return 0

        with patch.object(sb.os, "system", side_effect=fake_system) as system:
            sb._ensure_repo("u", clone_dir, commit="c0ffee")
        cmds = [c.args[0] for c in system.call_args_list]
        assert len(cmds) == 3
        assert cmds[0].startswith("git clone")
        assert "fetch --all --tags" in cmds[1]
        assert "checkout c0ffee" in cmds[2]

    @pytest.mark.parametrize("has_git,commit", [(True, None), (False, "abc"), (False, None)])
    def test_existing_dir_without_checkout_conditions_is_noop(self, tmp_path, has_git, commit):
        clone_dir = tmp_path / "repo"
        clone_dir.mkdir()
        if has_git:
            (clone_dir / ".git").mkdir()
        with patch.object(sb.os, "system") as system:
            sb._ensure_repo("u", clone_dir, commit=commit)
        system.assert_not_called()


class TestWriteEnvKv:
    def test_creates_file_and_parent_dirs(self, tmp_path):
        env = tmp_path / "sub" / ".env"
        sb._write_env_kv(env, "KEY", "value")
        assert env.read_text() == "KEY=value\n"

    def test_appends_when_key_absent_preserving_other_lines(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("A=1\n# comment\nB=2\n")
        sb._write_env_kv(env, "C", "3")
        assert env.read_text() == "A=1\n# comment\nB=2\nC=3\n"

    def test_updates_existing_key_in_place(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("A=1\nKEY=old\nB=2\n")
        sb._write_env_kv(env, "KEY", "new")
        assert env.read_text() == "A=1\nKEY=new\nB=2\n"

    def test_collapses_duplicate_keys_to_first_position(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("KEY=a\nA=1\n  KEY=b\nKEY=c\n")
        sb._write_env_kv(env, "KEY", "z")
        assert env.read_text() == "KEY=z\nA=1\n"

    @pytest.mark.parametrize("line", ["KEY_OTHER=1", "# KEY=commented", "MYKEY=2"])
    def test_does_not_touch_keys_that_only_share_a_prefix(self, tmp_path, line):
        env = tmp_path / ".env"
        env.write_text(line + "\n")
        sb._write_env_kv(env, "KEY", "v")
        assert env.read_text() == f"{line}\nKEY=v\n"

    def test_is_idempotent(self, tmp_path):
        env = tmp_path / ".env"
        sb._write_env_kv(env, "KEY", "v")
        first = env.read_text()
        sb._write_env_kv(env, "KEY", "v")
        assert env.read_text() == first == "KEY=v\n"

    def test_value_may_contain_equals(self, tmp_path):
        env = tmp_path / ".env"
        sb._write_env_kv(env, "URL", "http://h/?a=b")
        sb._write_env_kv(env, "URL", "http://h/?c=d")
        assert env.read_text() == "URL=http://h/?c=d\n"


class TestSetupMiniwob:
    def test_clones_pinned_commit_writes_env_and_sets_process_env(
        self, tmp_path, isolated_miniwob_env
    ):
        with patch.object(sb, "_ensure_repo") as ensure_repo:
            url = sb.setup_miniwob(tmp_path)

        ensure_repo.assert_called_once_with(
            repo_url=MINIWOB_REPO, clone_dir=tmp_path / "miniwob-plusplus", commit=PINNED_COMMIT
        )
        expected_dir = (tmp_path / "miniwob-plusplus" / "miniwob" / "html" / "miniwob").resolve()
        assert url == expected_dir.as_uri() + "/"
        assert url.startswith("file://") and url.endswith("/miniwob/")
        assert (tmp_path / ".env").read_text() == f"MINIWOB_URL={url}\n"
        assert os.environ["MINIWOB_URL"] == url

    def test_overwrites_previous_url_in_env_file(self, tmp_path, isolated_miniwob_env):
        (tmp_path / ".env").write_text("OTHER=1\nMINIWOB_URL=http://old/\n")
        with patch.object(sb.os, "system", return_value=0):
            url = sb.setup_miniwob(tmp_path)
        assert (tmp_path / ".env").read_text() == f"OTHER=1\nMINIWOB_URL={url}\n"


class TestEnsureBenchmark:
    @pytest.mark.parametrize("name", ["miniwob", "MiniWoB++", "  miniwob-plusplus  ", "MINIWOB"])
    def test_miniwob_aliases_trigger_setup_when_url_unset(
        self, tmp_path, isolated_miniwob_env, name
    ):
        with patch.object(sb, "setup_miniwob", return_value="file:///x/") as setup:
            assert sb.ensure_benchmark(name, tmp_path) == "file:///x/"
        setup.assert_called_once_with(tmp_path)

    def test_miniwob_skipped_when_url_already_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MINIWOB_URL", "http://localhost:8080/miniwob/")
        with patch.object(sb, "setup_miniwob") as setup:
            assert sb.ensure_benchmark("miniwob", tmp_path) is None
        setup.assert_not_called()

    def test_empty_url_counts_as_unset(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MINIWOB_URL", "")
        with patch.object(sb, "setup_miniwob", return_value="u") as setup:
            assert sb.ensure_benchmark("miniwob", tmp_path) == "u"
        setup.assert_called_once()

    @pytest.mark.parametrize("name", ["webarena", "workarena_l1", "osworld", "mini wob", ""])
    def test_other_benchmarks_are_noop(self, tmp_path, isolated_miniwob_env, name):
        with patch.object(sb, "setup_miniwob") as setup:
            assert sb.ensure_benchmark(name, tmp_path) is None
        setup.assert_not_called()
