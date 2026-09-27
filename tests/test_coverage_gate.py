import copy

import pytest

from harness.coverage_gate import assert_full_coverage


def report():
    summary = {
        "missing_lines": 0,
        "missing_branches": 0,
        "excluded_lines": 0,
        "num_statements": 2,
        "num_branches": 2,
    }
    return {
        "meta": {"branch_coverage": True},
        "files": {
            "module": {
                "summary": summary,
                "functions": {
                    "": {"summary": summary},
                    "function": {"summary": copy.copy(summary)},
                },
            }
        },
    }


def test_full_module_and_function_coverage():
    assert assert_full_coverage(report())["module"] == {
        "coverage": 100,
        "statements": 2,
        "branches": 2,
        "functions": 1,
    }


@pytest.mark.parametrize(
    "field", ["missing_lines", "missing_branches", "excluded_lines"]
)
def test_missing_or_excluded_coverage_fails(field):
    value = report()
    value["files"]["module"]["summary"][field] = 1
    with pytest.raises(ValueError, match="incomplete"):
        assert_full_coverage(value)


@pytest.mark.parametrize("field", ["missing_lines", "missing_branches"])
def test_missing_function_coverage_fails(field):
    value = report()
    value["files"]["module"]["functions"]["function"]["summary"][field] = 1
    with pytest.raises(ValueError, match="function"):
        assert_full_coverage(value)


def test_missing_measurements_fail():
    with pytest.raises(ValueError, match="branch"):
        assert_full_coverage({})
    value = report()
    value["meta"]["branch_coverage"] = False
    with pytest.raises(ValueError):
        assert_full_coverage(value)
    value = report()
    del value["files"]["module"]["functions"]
    with pytest.raises(ValueError, match="unavailable"):
        assert_full_coverage(value)


def test_provider_interface_fails_closed():
    from harness.agents import ModelProvider

    with pytest.raises(NotImplementedError):
        ModelProvider().complete("x")
