import csv
import logging
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import pytest
from git import InvalidGitRepositoryError, Repo

from agentlab.experiments import reproducibility_util as repro

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _report(rows):
    df = pd.DataFrame(
        rows, columns=["agent.agent_name", "avg_reward", "std_err", "n_err", "n_completed"]
    )
    return df.set_index("agent.agent_name")


def _init_repo(path: Path, monkeypatch) -> Repo:
    for var in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(var, "Unit Tester")
    for var in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(var, "unit@test.invalid")
    repo = Repo.init(path)
    with repo.config_writer() as cw:
        cw.set_value("user", "name", "Repo Local User")
        cw.set_value("user", "email", "unit@test.invalid")
    pkg = path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("x = 1\n")
    (path / "README.md").write_text("readme\n")
    repo.index.add(["pkg/__init__.py", "README.md"])
    repo.index.commit("initial")
    return repo


def _fake_module(path: Path):
    return SimpleNamespace(__file__=str(path / "pkg" / "__init__.py"), __version__="9.9.9")


# ---------------------------------------------------------------------------
# _get_benchmark_version
# ---------------------------------------------------------------------------


def test_benchmark_version_prefers_get_version():
    bench = SimpleNamespace(name="anything", get_version=lambda: "1.2.3")
    assert repro._get_benchmark_version(bench) == "1.2.3"


@pytest.mark.parametrize(
    "name, dist",
    [
        ("miniwob_tiny_test", "browsergym.miniwob"),
        ("workarena_l1", "browsergym.workarena"),
        ("webarena_tiny", "browsergym.webarena"),
        ("visualwebarena", "browsergym.visualwebarena"),
        ("weblinx_test", "weblinx_browsergym"),
        ("assistantbench", "browsergym.assistantbench"),
    ],
)
def test_benchmark_version_from_distribution(monkeypatch, name, dist):
    calls = []

    def fake_distribution(d):
        calls.append(d)
        return SimpleNamespace(version=f"v-{d}")

    monkeypatch.setattr(repro.metadata, "distribution", fake_distribution)
    assert repro._get_benchmark_version(SimpleNamespace(name=name)) == f"v-{dist}"
    assert calls == [dist]


def test_benchmark_version_weblinx_missing_package_falls_back(monkeypatch):
    def raise_not_found(d):
        raise metadata.PackageNotFoundError(d)

    monkeypatch.setattr(repro.metadata, "distribution", raise_not_found)
    assert repro._get_benchmark_version(SimpleNamespace(name="weblinx")) == "0.0.1rc1"


def test_benchmark_version_unknown_raises_or_bypasses():
    bench = SimpleNamespace(name="my_custom_bench")
    with pytest.raises(ValueError, match="Unknown benchmark my_custom_bench"):
        repro._get_benchmark_version(bench)
    assert repro._get_benchmark_version(bench, allow_bypass_benchmark_version=True) == "bypassed"


# ---------------------------------------------------------------------------
# _get_git_username
# ---------------------------------------------------------------------------


def test_git_username_from_repo_config(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path, monkeypatch)
    assert repro._get_git_username(repo) == "Repo Local User"


def _bare_repo_mock(remote_url="https://gitlab.com/x/y.git"):
    repo = mock.MagicMock()
    repo.config_reader.return_value.get_value.return_value = None
    repo.remotes.origin.url = remote_url
    return repo


def test_git_username_from_github_api(monkeypatch):
    repo = _bare_repo_mock("git@github.com:some-owner/some-repo.git")
    response = mock.MagicMock()
    response.read.return_value = b'{"owner": {"login": "gh-owner"}}'
    response.__enter__.return_value = response
    with mock.patch("urllib.request.urlopen", return_value=response) as urlopen:
        assert repro._get_git_username(repo) == "gh-owner"
    urlopen.assert_called_once_with("https://api.github.com/repos/some-owner/some-repo")


@pytest.mark.parametrize("scope, expected", [("--global", "global-user"), ("--system", "sys-user")])
def test_git_username_from_global_then_system_config(monkeypatch, scope, expected):
    repo = _bare_repo_mock()
    repo.git.config.side_effect = lambda s, _list: s

    class FakeParser:
        def __init__(self, source, read_only):
            self.source = source

        def get_value(self, section, option, default):
            if self.source == "--global":
                return "global-user" if scope == "--global" else None
            return "sys-user"

    monkeypatch.setattr(repro, "GitConfigParser", FakeParser)
    assert repro._get_git_username(repo) == expected


@pytest.mark.parametrize(
    "author, committer, expected",
    [("auth", "comm", "auth"), (None, "comm", "comm"), (None, None, None)],
)
def test_git_username_env_fallback(monkeypatch, author, committer, expected):
    for var, value in (("GIT_AUTHOR_NAME", author), ("GIT_COMMITTER_NAME", committer)):
        if value is None:
            monkeypatch.delenv(var, raising=False)
        else:
            monkeypatch.setenv(var, value)
    # repo=None (no git repo) makes every lookup fail -> env vars
    assert repro._get_git_username(None) == expected


# ---------------------------------------------------------------------------
# _get_git_info
# ---------------------------------------------------------------------------


def test_git_info_clean_repo(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path, monkeypatch)
    git_hash, modified = repro._get_git_info(_fake_module(tmp_path))
    assert git_hash == repo.head.object.hexsha
    assert modified == []


def test_git_info_reports_unstaged_and_untracked(tmp_path, monkeypatch):
    _init_repo(tmp_path, monkeypatch)
    (tmp_path / "pkg" / "__init__.py").write_text("x = 2\n")
    (tmp_path / "new_untracked.txt").write_text("u")
    _, modified = repro._get_git_info(_fake_module(tmp_path))
    assert ("M", Path("pkg/__init__.py")) in modified
    assert ("??", Path("new_untracked.txt")) in modified
    assert len(modified) == 2


def test_git_info_staged_modification(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path, monkeypatch)
    (tmp_path / "README.md").write_text("changed\n")
    repo.index.add(["README.md"])
    _, modified = repro._get_git_info(_fake_module(tmp_path))
    assert modified == [("M", Path("README.md"))]


@pytest.mark.xfail(
    strict=True,
    reason="BUG: index.diff(head.commit) is reversed, so a staged new file is reported as 'D'",
)
def test_git_info_staged_new_file_reported_as_added(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path, monkeypatch)
    (tmp_path / "added.txt").write_text("a")
    repo.index.add(["added.txt"])
    _, modified = repro._get_git_info(_fake_module(tmp_path))
    assert modified == [("A", Path("added.txt"))]


def test_git_info_white_list_filters_matches(tmp_path, monkeypatch):
    _init_repo(tmp_path, monkeypatch)
    (tmp_path / "main.py").write_text("")
    (tmp_path / "reproducibility_journal.csv").write_text("")
    (tmp_path / "keep.py").write_text("")
    _, modified = repro._get_git_info(
        _fake_module(tmp_path), changes_white_list=("*main.py", "*reproducibility_journal.csv")
    )
    assert modified == [("??", Path("keep.py"))]


def test_git_info_not_a_repo(tmp_path):
    (tmp_path / "pkg").mkdir()
    with mock.patch.object(repro, "_get_repo", side_effect=InvalidGitRepositoryError("nope")):
        assert repro._get_git_info(_fake_module(tmp_path)) == (None, [])


def test_get_repo_finds_parent_repository(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path, monkeypatch)
    found = repro._get_repo(_fake_module(tmp_path))
    assert Path(found.working_tree_dir).resolve() == Path(repo.working_tree_dir).resolve()


# ---------------------------------------------------------------------------
# get_reproducibility_info
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_git(monkeypatch):
    git_infos = {}

    def fake_git_info(module, changes_white_list=()):
        return git_infos.get(module.__name__, ("deadbeef", []))

    monkeypatch.setattr(repro, "_get_repo", lambda module: "fake-repo")
    monkeypatch.setattr(repro, "_get_git_username", lambda repo: f"user-of-{repo}")
    monkeypatch.setattr(repro, "_get_git_info", fake_git_info)
    return git_infos


def test_reproducibility_info_contents(patched_git):
    bench = SimpleNamespace(name="bench_x", get_version=lambda: "4.5")
    info = repro.get_reproducibility_info("agent_a", bench, study_id="sid", comment="hello")
    assert info["agent_names"] == ["agent_a"]
    assert info["git_user"] == "user-of-fake-repo"
    assert info["benchmark"] == "bench_x"
    assert info["benchmark_version"] == "4.5"
    assert info["study_id"] == "sid"
    assert info["comment"] == "hello"
    assert info["playwright_version"] == metadata.distribution("playwright").version
    for module in ("agentlab", "browsergym"):
        assert info[f"{module}_git_hash"] == "deadbeef"
        assert info[f"{module}__local_modifications"] == ""
        assert f"{module}_version" in info


def test_reproducibility_info_without_git_repo(patched_git, monkeypatch):
    def raise_invalid(module):
        raise InvalidGitRepositoryError("no repo")

    monkeypatch.setattr(repro, "_get_repo", raise_invalid)
    bench = SimpleNamespace(name="unknown_bench")
    info = repro.get_reproducibility_info(["a", "b"], bench, allow_bypass_benchmark_version=True)
    assert info["git_user"] == "user-of-None"
    assert info["agent_names"] == ["a", "b"]
    assert info["benchmark_version"] == "bypassed"


def test_reproducibility_info_local_changes_raise(patched_git):
    patched_git["agentlab"] = ("abc", [("M", Path("src/x.py"))])
    bench = SimpleNamespace(name="b", get_version=lambda: "1")
    with pytest.raises(
        ValueError, match=r"(?s)Module agentlab has uncommitted changes.*M: src/x.py"
    ):
        repro.get_reproducibility_info("a", bench)


def test_reproducibility_info_local_changes_ignored(patched_git, caplog):
    patched_git["browsergym.core"] = ("abc", [("??", Path("new.py"))])
    bench = SimpleNamespace(name="b", get_version=lambda: "1")
    with caplog.at_level(logging.WARNING):
        info = repro.get_reproducibility_info("a", bench, ignore_changes=True)
    assert info["browsergym__local_modifications"] == "  ??: new.py"
    assert info["browsergym_git_hash"] == "abc"
    assert "Ignoring changes as requested" in caplog.text


# ---------------------------------------------------------------------------
# assert_compatible / _raise_or_warn
# ---------------------------------------------------------------------------


def test_assert_compatible_ignores_volatile_keys():
    info = {"a": 1, "date": "today", "avg_reward": 0.5, "n_err": 0}
    old = {"a": 1, "date": "yesterday", "avg_reward": 0.1, "n_err": 3}
    repro.assert_compatible(info, old)


def test_assert_compatible_raises_on_changed_key():
    with pytest.raises(ValueError, match="Key a has changed from 1 to 2"):
        repro.assert_compatible({"a": 2}, {"a": 1})


def test_assert_compatible_warns_when_not_strict(caplog):
    with caplog.at_level(logging.WARNING):
        repro.assert_compatible({"a": 2}, {"a": 1}, raise_if_incompatible=False)
    assert "Key a has changed" in caplog.text


def test_raise_or_warn(caplog):
    with pytest.raises(ValueError, match="boom"):
        repro._raise_or_warn("boom")
    with caplog.at_level(logging.WARNING):
        repro._raise_or_warn("soft", raise_error=False)
    assert "soft" in caplog.text


# ---------------------------------------------------------------------------
# _verify_report
# ---------------------------------------------------------------------------


def test_verify_report_ok_returns_indexed_df():
    df = repro._verify_report(_report([["a", 0.5, 0.1, 0, "4/4"]]), ["a"])
    assert list(df.index) == ["a"]
    assert df.loc["a", "avg_reward"] == 0.5


def test_verify_report_agent_mismatch():
    with pytest.raises(ValueError, match="do not match"):
        repro._verify_report(_report([["a", 0.5, 0.1, 0, "4/4"]]), ["b"])


def test_verify_report_duplicate_agent_names():
    with pytest.raises(ValueError, match="Duplicate agent names"):
        repro._verify_report(_report([["a", 0.5, 0.1, 0, "4/4"]]), ["a", "a"])


@pytest.mark.parametrize(
    "n_err, n_completed, match",
    [(2, "4/4", "has 2 errors"), (0, "3/4", "has 3 completed tasks out of 4")],
)
def test_verify_report_errors_and_incomplete(n_err, n_completed, match, caplog):
    report = _report([["a", 0.5, 0.1, n_err, n_completed]])
    with pytest.raises(ValueError, match=match):
        repro._verify_report(report, ["a"])
    with caplog.at_level(logging.WARNING):
        repro._verify_report(report, ["a"], strict_reproducibility=False)
    assert match in caplog.text


# ---------------------------------------------------------------------------
# csv helpers
# ---------------------------------------------------------------------------


def test_get_csv_headers(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    assert repro._get_csv_headers(empty) is None

    full = tmp_path / "full.csv"
    full.write_text("h1,h2\n1,2\n")
    assert repro._get_csv_headers(full) == ["h1", "h2"]


def test_add_result_to_info_unwraps_numpy_scalars():
    report = _report([["a", 0.75, 0.2, 1, "3/4"]])
    info = {"agent_name": "a"}
    repro._add_result_to_info(info, report)
    assert info == {
        "agent_name": "a",
        "avg_reward": 0.75,
        "std_err": 0.2,
        "n_err": 1,
        "n_completed": "3/4",
    }
    assert type(info["n_err"]) is int
    assert type(info["avg_reward"]) is float


# ---------------------------------------------------------------------------
# append_to_journal
# ---------------------------------------------------------------------------


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def test_append_to_journal_creates_header_and_appends(tmp_path):
    journal = tmp_path / "journal.csv"
    info = {"git_user": "me", "agent_names": ["a", "b"], "benchmark": "bench"}
    report = _report([["a", 0.5, 0.1, 0, "4/4"], ["b", 0.25, 0.1, 0, "4/4"]])

    repro.append_to_journal(info, report, journal_path=journal)
    rows = _read_csv(journal)
    assert rows[0] == ["git_user", "agent_name", "benchmark"]
    assert rows[1:] == [["me", "a", "bench"], ["me", "b", "bench"]]

    repro.append_to_journal(info, report, journal_path=journal)
    rows = _read_csv(journal)
    assert len(rows) == 5
    assert rows.count(["git_user", "agent_name", "benchmark"]) == 1
    assert info["agent_names"] == ["a", "b"]


def test_append_to_journal_writes_results_with_existing_headers(tmp_path):
    journal = tmp_path / "journal.csv"
    journal.write_text("agent_name,benchmark,avg_reward,std_err,n_err,n_completed\n")
    info = {"agent_names": ["a"], "benchmark": "bench"}
    report = _report([["a", 0.75, 0.217, 0, "4/4"]])
    repro.append_to_journal(info, report, journal_path=journal)
    assert _read_csv(journal)[1] == ["a", "bench", "0.75", "0.217", "0", "4/4"]


@pytest.mark.xfail(
    strict=True,
    reason="BUG: a newly created journal takes headers from info only, so results are dropped",
)
def test_append_to_journal_new_file_includes_results(tmp_path):
    journal = tmp_path / "journal.csv"
    info = {"agent_names": ["a"], "benchmark": "bench"}
    report = _report([["a", 0.75, 0.217, 0, "4/4"]])
    repro.append_to_journal(info, report, journal_path=journal)
    header = _read_csv(journal)[0]
    assert {"avg_reward", "std_err", "n_err", "n_completed"} <= set(header)


def test_append_to_journal_length_mismatch(tmp_path):
    info = {"agent_names": ["a", "b"]}
    with pytest.raises(ValueError, match="Mismatch between the number of agents"):
        repro.append_to_journal(
            info, _report([["a", 0.5, 0.1, 0, "4/4"]]), journal_path=tmp_path / "j.csv"
        )
    assert not (tmp_path / "j.csv").exists()


def test_append_to_journal_strict_incomplete_raises_before_writing(tmp_path):
    journal = tmp_path / "j.csv"
    info = {"agent_names": ["a"], "benchmark": "x"}
    report = _report([["a", 0.5, 0.1, 1, "3/4"]])
    with pytest.raises(ValueError):
        repro.append_to_journal(info, report, journal_path=journal)
    assert not journal.exists()
    repro.append_to_journal(info, report, journal_path=journal, strict_reproducibility=False)
    assert journal.exists()


def test_append_to_journal_default_path_in_git_clone(tmp_path, monkeypatch):
    fake_pkg_init = tmp_path / "root" / "src" / "agentlab" / "__init__.py"
    fake_pkg_init.parent.mkdir(parents=True)
    monkeypatch.setattr(repro, "agentlab", SimpleNamespace(__file__=str(fake_pkg_init)))
    monkeypatch.setattr(repro, "_get_repo", lambda module: "repo")
    info = {"agent_names": ["a"], "benchmark": "x"}
    repro.append_to_journal(info, _report([["a", 0.5, 0.1, 0, "4/4"]]))
    assert (tmp_path / "root" / "reproducibility_journal.csv").exists()


def test_append_to_journal_default_path_without_git(tmp_path, monkeypatch, caplog):
    def raise_invalid(module):
        raise InvalidGitRepositoryError("no")

    monkeypatch.setattr(repro, "_get_repo", raise_invalid)
    monkeypatch.setattr(repro, "RESULTS_DIR", tmp_path)
    info = {"agent_names": ["a"], "benchmark": "x"}
    with caplog.at_level(logging.WARNING):
        repro.append_to_journal(info, _report([["a", 0.5, 0.1, 0, "4/4"]]))
    assert (tmp_path / "reproducibility_journal.csv").exists()
    assert "Could not find a git repository" in caplog.text
