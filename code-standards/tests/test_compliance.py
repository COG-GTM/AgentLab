"""Code Standards Compliance Tests (Python port).

Verifies that the ruff / pyright configurations work correctly, that the
severity mapping is complete, and that the example files behave as documented.

Python port of the framework's ``tests/test_compliance.py``.

Requirements:
    - ruff (available directly or via ``uv run ruff``)
    - pytest
    - PyYAML

Run: pytest code-standards/tests/test_compliance.py -v
"""

import shutil
import subprocess
from pathlib import Path

import pytest

# =============================================================================
# Test Configuration
# =============================================================================

FRAMEWORK_ROOT = Path(__file__).parent.parent
EXAMPLES_DIR = FRAMEWORK_ROOT / "examples"
SCRIPTS_DIR = FRAMEWORK_ROOT / "scripts"
RUFF_CONFIG = FRAMEWORK_ROOT / "ruff.toml"


def ruff_cmd() -> list[str] | None:
    """Return an invocation for ruff, or None if it is unavailable."""
    if shutil.which("ruff"):
        return ["ruff"]
    if shutil.which("uv"):
        return ["uv", "run", "ruff"]
    return None


requires_ruff = pytest.mark.skipif(ruff_cmd() is None, reason="ruff not installed")


# =============================================================================
# Configuration File Tests
# =============================================================================


class TestConfigurationFiles:
    """Tests for configuration file existence and validity."""

    def test_ruff_config_exists(self):
        assert RUFF_CONFIG.exists(), "Missing ruff.toml configuration file"

    def test_pyright_config_exists(self):
        assert (FRAMEWORK_ROOT / "pyrightconfig.json").exists(), "Missing pyrightconfig.json"

    def test_severity_mapping_exists(self):
        assert (FRAMEWORK_ROOT / "rule-severity-mapping.yaml").exists(), (
            "Missing rule-severity-mapping.yaml"
        )

    def test_windsurf_rules_exist(self):
        rules = FRAMEWORK_ROOT / "windsurf" / "python-safety-critical-rules.md"
        assert rules.exists(), "Missing windsurf/python-safety-critical-rules.md"

    def test_validate_script_exists(self):
        assert (SCRIPTS_DIR / "validate.sh").exists(), "Missing scripts/validate.sh"

    def test_validate_script_structure(self):
        content = (SCRIPTS_DIR / "validate.sh").read_text()
        assert "#!/bin/bash" in content, "Script should have bash shebang"
        assert "ruff format" in content, "Script should use ruff format"
        assert "ruff check" in content or "check --config" in content, "Script should use ruff check"


# =============================================================================
# ruff Tests
# =============================================================================


@requires_ruff
class TestRuff:
    """Tests for the ruff configuration and example behavior."""

    def test_config_is_valid(self):
        """ruff should load the config and lint the compliant example cleanly."""
        result = subprocess.run(
            [*ruff_cmd(), "check", "--config", str(RUFF_CONFIG), str(EXAMPLES_DIR / "compliant.py")],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"compliant.py should pass ruff check:\n{result.stdout}\n{result.stderr}"
        )

    def test_compliant_example_is_formatted(self):
        result = subprocess.run(
            [
                *ruff_cmd(),
                "format",
                "--config",
                str(RUFF_CONFIG),
                "--check",
                str(EXAMPLES_DIR / "compliant.py"),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"compliant.py should be formatted:\n{result.stderr}"

    def test_violations_detected(self):
        """violations.py must trigger lint findings."""
        result = subprocess.run(
            [*ruff_cmd(), "check", "--config", str(RUFF_CONFIG), str(EXAMPLES_DIR / "violations.py")],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, "Expected violations.py to fail ruff check"

    def test_critical_violations_detected(self):
        """The critical gate must flag the critical examples in violations.py."""
        critical_codes = (
            "E722,BLE001,S102,S301,S302,S307,S311,S324,S506,"
            "S602,S604,S605,S608,S609,SIM115,B006,B008,F821,F822,F823"
        )
        result = subprocess.run(
            [
                *ruff_cmd(),
                "check",
                "--config",
                str(RUFF_CONFIG),
                "--select",
                critical_codes,
                str(EXAMPLES_DIR / "violations.py"),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, "Expected critical violations in violations.py"


# =============================================================================
# Severity Mapping Tests
# =============================================================================


class TestSeverityMapping:
    """Tests for rule severity classification."""

    @pytest.fixture
    def severity_config(self):
        yaml = pytest.importorskip("yaml")
        with open(FRAMEWORK_ROOT / "rule-severity-mapping.yaml") as handle:
            return yaml.safe_load(handle)

    def test_version_is_1_0(self, severity_config):
        assert str(severity_config.get("version")) == "1.0", "Framework version should be 1.0"

    def test_has_all_severity_levels(self, severity_config):
        levels = severity_config["severity_levels"]
        assert "critical" in levels
        assert "major" in levels
        assert "minor" in levels

    def test_critical_has_rules(self, severity_config):
        critical = severity_config["severity_levels"]["critical"]
        assert len(critical["rules"]) > 0

    def test_critical_rules_have_checks(self, severity_config):
        critical = severity_config["severity_levels"]["critical"]
        for rule in critical["rules"]:
            assert "rule_id" in rule
            assert "checks" in rule and len(rule["checks"]) > 0, (
                f"{rule.get('rule_id')} missing checks"
            )

    def test_rule_ids_are_unique(self, severity_config):
        ids = []
        for level in severity_config["severity_levels"].values():
            for rule in level.get("rules", []):
                ids.append(rule.get("rule_id"))
        ids = [r for r in ids if r]
        assert len(ids) == len(set(ids)), "Duplicate rule IDs found"

    def test_critical_rules_start_at_20(self, severity_config):
        critical = severity_config["severity_levels"]["critical"]
        ids = [r.get("rule_id", "") for r in critical.get("rules", [])]
        assert any("Rule 2" in r for r in ids), "Critical rules should start at Rule 20"


# =============================================================================
# Example / Documentation Tests
# =============================================================================


class TestExampleFiles:
    """Tests for example code files."""

    def test_compliant_example_exists(self):
        assert (EXAMPLES_DIR / "compliant.py").exists()

    def test_violations_example_exists(self):
        assert (EXAMPLES_DIR / "violations.py").exists()

    def test_compliant_has_main(self):
        content = (EXAMPLES_DIR / "compliant.py").read_text()
        assert "def main" in content

    def test_violations_documents_rules(self):
        content = (EXAMPLES_DIR / "violations.py").read_text()
        assert "Rule 20" in content or "VIOLATION" in content


class TestDocumentation:
    """Tests for documentation completeness."""

    def test_readme_exists(self):
        assert (FRAMEWORK_ROOT / "README.md").exists()

    def test_rule_reference_exists(self):
        assert (FRAMEWORK_ROOT / "docs" / "rule-reference.md").exists()

    def test_readme_has_severity_info(self):
        content = (FRAMEWORK_ROOT / "README.md").read_text()
        assert "Critical" in content and "Major" in content and "Minor" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
