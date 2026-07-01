#!/bin/bash
# =============================================================================
# Python Code Standards Validation Script
# =============================================================================
# Python port of the framework's scripts/validate.sh. Wraps `ruff format` and
# `ruff check` and preserves the framework's tiered exit-code contract.
#
# Usage: ./code-standards/scripts/validate.sh [directory] [--fix]
#
# Arguments:
#   directory   Target directory to validate (default: src/)
#   --fix       Apply automatic fixes (formatting + safe lint fixes)
#
# Exit codes:
#   0 - All checks passed
#   1 - Format violations found
#   2 - Critical lint violations found (block merge)
#   3 - Both format and critical violations found
#
# Examples:
#   ./code-standards/scripts/validate.sh              # Check src/
#   ./code-standards/scripts/validate.sh src/agentlab # Check a subdir
#   ./code-standards/scripts/validate.sh src/ --fix   # Auto-fix
# =============================================================================

set -uo pipefail

# Colors
RED='\033[0;31m'
YELLOW='\033[0;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG="$FRAMEWORK_ROOT/ruff.toml"

# Critical lint rule codes (the "block merge" gate). Mirrors the C framework's
# clang-tidy `WarningsAsErrors` list -- only these promote to a failing exit.
# A curated flake8-bandit subset is used (not bare `S`) so noisy, low-signal
# checks (S603/S607) are not re-enabled by the CLI --select override.
CRITICAL_CODES="E722,BLE001,S102,S301,S302,S307,S311,S324,S506,S602,S604,S605,S608,S609,SIM115,B006,B008,F821,F822,F823"

# Parse arguments
TARGET_DIR="src"
FIX_MODE=false
for arg in "$@"; do
    case $arg in
        --fix) FIX_MODE=true ;;
        *) [[ -d "$arg" ]] && TARGET_DIR="$arg" ;;
    esac
done

# Resolve the ruff invocation (prefer `uv run ruff`, fall back to `ruff`).
if command -v ruff &> /dev/null; then
    RUFF=(ruff)
elif command -v uv &> /dev/null; then
    RUFF=(uv run ruff)
else
    echo -e "${RED}ruff not found. Install with: pip install ruff${NC}"
    exit 4
fi

print_header() {
    echo ""
    echo -e "${BOLD}${BLUE}======================================================${NC}"
    echo -e "${BOLD}${BLUE}  $1${NC}"
    echo -e "${BOLD}${BLUE}======================================================${NC}"
    echo ""
}
print_section() { echo ""; echo -e "${BOLD}--- $1 ---${NC}"; echo ""; }
print_pass() { echo -e "${GREEN}✓ PASS${NC}: $1"; }
print_fail() { echo -e "${RED}✗ FAIL${NC}: $1"; }
print_warn() { echo -e "${YELLOW}⚠ WARN${NC}: $1"; }
print_info() { echo -e "${BLUE}ℹ INFO${NC}: $1"; }

print_header "Python Code Standards Validation"
echo "Configuration:"
echo "  Target directory: $TARGET_DIR"
echo "  Fix mode: $FIX_MODE"
echo "  Config: $CONFIG"

if [ ! -d "$TARGET_DIR" ]; then
    print_warn "Target directory '$TARGET_DIR' not found."
    exit 0
fi

FORMAT_ERRORS=0
CRITICAL_ERRORS=0
ADVISORY_WARNINGS=0

# =============================================================================
# STEP 1: Format check (Rule 40)
# =============================================================================
print_section "ruff format check"
if $FIX_MODE; then
    if "${RUFF[@]}" format --config "$CONFIG" "$TARGET_DIR"; then
        print_pass "Formatting applied"
    else
        print_fail "Formatting failed"
        FORMAT_ERRORS=1
    fi
else
    if "${RUFF[@]}" format --config "$CONFIG" --check "$TARGET_DIR"; then
        print_pass "All files properly formatted"
    else
        print_fail "Some files need formatting (run with --fix)"
        FORMAT_ERRORS=1
    fi
fi

# =============================================================================
# STEP 2a: Critical static analysis (block merge)
# =============================================================================
print_section "ruff check - Critical (block merge)"
if $FIX_MODE; then
    "${RUFF[@]}" check --config "$CONFIG" --select "$CRITICAL_CODES" --fix "$TARGET_DIR" || true
fi
if "${RUFF[@]}" check --config "$CONFIG" --select "$CRITICAL_CODES" "$TARGET_DIR"; then
    print_pass "No Critical violations"
else
    print_fail "Critical violations found (🔴 must fix before merge)"
    CRITICAL_ERRORS=1
fi

# =============================================================================
# STEP 2b: Major / Minor static analysis (advisory)
# =============================================================================
print_section "ruff check - Major / Minor (advisory)"
if "${RUFF[@]}" check --config "$CONFIG" "$TARGET_DIR" > /dev/null 2>&1; then
    print_pass "No Major/Minor violations"
else
    print_warn "Major/Minor violations found (🟡/🟢 review recommended)"
    "${RUFF[@]}" check --config "$CONFIG" --statistics "$TARGET_DIR" 2>/dev/null || true
    ADVISORY_WARNINGS=1
fi

# =============================================================================
# SUMMARY
# =============================================================================
print_header "Summary"
EXIT_CODE=0

if [ "$FORMAT_ERRORS" -gt 0 ]; then
    echo -e "${RED}Format:   needs formatting${NC}"
    EXIT_CODE=1
else
    echo -e "${GREEN}Format:   OK${NC}"
fi

if [ "$CRITICAL_ERRORS" -gt 0 ]; then
    echo -e "${RED}Analysis: Critical violations (block merge)${NC}"
    if [ "$EXIT_CODE" -eq 0 ]; then EXIT_CODE=2; else EXIT_CODE=3; fi
elif [ "$ADVISORY_WARNINGS" -gt 0 ]; then
    echo -e "${YELLOW}Analysis: Major/Minor warnings (advisory)${NC}"
else
    echo -e "${GREEN}Analysis: OK${NC}"
fi

echo ""
if [ "$EXIT_CODE" -eq 0 ]; then
    echo -e "${GREEN}${BOLD}✓ All checks passed!${NC}"
else
    echo -e "${RED}${BOLD}✗ Validation failed (exit code: $EXIT_CODE)${NC}"
fi
echo ""
exit $EXIT_CODE
