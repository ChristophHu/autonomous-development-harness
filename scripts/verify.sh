#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p data
# Record whether this host permits the same nested sandbox used by real tools.
# This is diagnostic only: never skip tests or mask pytest's exit status.
.venv/bin/harness isolation doctor --output data/isolation-capability.json || true
if .venv/bin/pytest --junitxml=data/verification-junit.xml; then
    :
else
    pytest_status=$?
    .venv/bin/harness verify-clusters \
        --junit data/verification-junit.xml \
        --output data/verification-failure-clusters.json || true
    exit "$pytest_status"
fi
.venv/bin/python -c 'import json; from harness.coverage_gate import assert_full_coverage; modules=assert_full_coverage(json.load(open("coverage.json"))); print("100% per module and function:", len(modules), "modules;", sum(module["functions"] for module in modules.values()), "functions")'
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/python -c 'from harness.verification import write_verification_report; write_verification_report("GAP_MATRIX.md", "coverage.json", "data/verification-junit.xml", "data/verification.json")'
