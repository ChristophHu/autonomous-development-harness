#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p data
.venv/bin/pytest --junitxml=data/verification-junit.xml
.venv/bin/python -c 'import json; from harness.coverage_gate import assert_full_coverage; modules=assert_full_coverage(json.load(open("coverage.json"))); print("100% per module and function:", len(modules), "modules;", sum(module["functions"] for module in modules.values()), "functions")'
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/python -c 'from harness.verification import write_verification_report; write_verification_report("GAP_MATRIX.md", "coverage.json", "data/verification-junit.xml", "data/verification.json")'
