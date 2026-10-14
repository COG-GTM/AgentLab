import os
import sys
import types

import pytest

from agentlab.experiments import multi_server
from agentlab.experiments.multi_server import (
    BaseServer,
    WebArenaInstanceVars,
    _split_url,
    unimport_modules,
)

KEYS = [
    "SHOPPING",
    "SHOPPING_ADMIN",
    "REDDIT",
    "GITLAB",
    "WIKIPEDIA",
    "MAP",
    "HOMEPAGE",
    "FULL_RESET",
]


def _make_instance(**overrides):
    kwargs = dict(
        base_url="http://wa.example.com",
        shopping="8082/",
        shopping_admin="8083/admin",
        reddit="8080",
        gitlab="9001",
        wikipedia="8081/wiki",
        map="443",
        homepage="80",
        full_reset="7565",
        module_name="fake_wa_module_for_tests",
        prefix="TESTWA_",
    )
    kwargs.update(overrides)
    return WebArenaInstanceVars(**kwargs)


@pytest.fixture
def clean_prefix_env(monkeypatch):
    for prefix in ("TESTWA_", "WA_"):
        for key in list(os.environ):
            if key.startswith(prefix):
                monkeypatch.delenv(key)
        # pre-register every key so monkeypatch restores os.environ after init() writes to it
        for key in KEYS:
            monkeypatch.setenv(f"{prefix}{key}", "placeholder")
            monkeypatch.delenv(f"{prefix}{key}")
    return monkeypatch


def test_base_server_init_is_noop():
    assert BaseServer().init() is None


@pytest.mark.parametrize(
    "url, expected",
    [
        ("http://host:8080", ("http://host", "8080")),
        ("http://host:8083/admin", ("http://host", "8083/admin")),
        ("http://host", ("http://host", "")),
        ("https://h.com:1/a:b", ("https://h.com", "1/a:b")),
    ],
)
def test_split_url(url, expected):
    assert _split_url(url) == expected


def test_make_env_vars_uses_prefix_and_base_url():
    env = _make_instance().make_env_vars()
    assert set(env) == {f"TESTWA_{k}" for k in KEYS}
    assert env["TESTWA_SHOPPING_ADMIN"] == "http://wa.example.com:8083/admin"
    assert env["TESTWA_FULL_RESET"] == "http://wa.example.com:7565"
    assert env["TESTWA_WIKIPEDIA"] == "http://wa.example.com:8081/wiki"


def test_clone_is_independent_deep_copy():
    original = _make_instance()
    clone = original.clone()
    assert clone == original
    assert clone is not original
    clone.base_url = "http://other"
    assert original.base_url == "http://wa.example.com"


def test_unimport_modules_removes_only_matching_prefix(monkeypatch):
    monkeypatch.setitem(sys.modules, "zz_unimport_pkg", types.ModuleType("zz_unimport_pkg"))
    monkeypatch.setitem(sys.modules, "zz_unimport_pkg.sub", types.ModuleType("sub"))
    monkeypatch.setitem(sys.modules, "zz_other_pkg", types.ModuleType("zz_other_pkg"))
    unimport_modules("zz_unimport_pkg")
    assert "zz_unimport_pkg" not in sys.modules
    assert "zz_unimport_pkg.sub" not in sys.modules
    assert "zz_other_pkg" in sys.modules


def test_from_env_vars_roundtrip_default_prefix(clean_prefix_env):
    instance = _make_instance(prefix="WA_", module_name="webarena")
    for key, value in instance.make_env_vars().items():
        clean_prefix_env.setenv(key, value)
    assert WebArenaInstanceVars.from_env_vars() == instance


@pytest.mark.xfail(
    strict=True, reason="BUG: from_env_vars drops a custom prefix; instance keeps prefix='WA_'"
)
def test_from_env_vars_roundtrip_custom_prefix(clean_prefix_env):
    instance = _make_instance()
    for key, value in instance.make_env_vars().items():
        clean_prefix_env.setenv(key, value)
    loaded = WebArenaInstanceVars.from_env_vars(
        prefix="TESTWA_", module_name="fake_wa_module_for_tests"
    )
    assert loaded == instance


def test_from_env_vars_multiple_base_urls_raises(clean_prefix_env):
    clean_prefix_env.setenv("TESTWA_SHOPPING", "http://a.com:8082")
    clean_prefix_env.setenv("TESTWA_REDDIT", "http://b.com:8080")
    with pytest.raises(ValueError, match="Multiple base urls"):
        WebArenaInstanceVars.from_env_vars(prefix="TESTWA_")


class _FakeWebArenaInstance:
    """Mimics browsergym's WebArenaInstance by reading urls from the env at construction."""

    prefix = "TESTWA_"

    def __init__(self):
        self.urls = {"reddit": os.environ[f"{self.prefix}REDDIT"]}


def test_init_sets_env_vars_and_unimports_module(clean_prefix_env):
    clean_prefix_env.setattr(multi_server, "WebArenaInstance", _FakeWebArenaInstance)
    clean_prefix_env.setitem(
        sys.modules, "fake_wa_module_for_tests", types.ModuleType("fake_wa_module_for_tests")
    )
    instance = _make_instance()
    instance.init()
    assert "fake_wa_module_for_tests" not in sys.modules
    for key, value in instance.make_env_vars().items():
        assert os.environ[key] == value


def test_init_asserts_base_url_matches(clean_prefix_env):
    class WrongInstance:
        def __init__(self):
            self.urls = {"reddit": "http://someone-else.com:8080"}

    clean_prefix_env.setattr(multi_server, "WebArenaInstance", WrongInstance)
    with pytest.raises(AssertionError, match="Expected http://wa.example.com"):
        _make_instance().init()
