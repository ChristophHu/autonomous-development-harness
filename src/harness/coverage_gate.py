"""Verify exact statement, branch, line and function coverage per module."""


def assert_full_coverage(report):
    if not report.get("files") or not report.get("meta", {}).get("branch_coverage"):
        raise ValueError("branch-enabled coverage report with files is required")
    modules = {}
    for name, data in report["files"].items():
        summary = data["summary"]
        if (
            summary["missing_lines"]
            or summary["missing_branches"]
            or summary["excluded_lines"]
        ):
            raise ValueError(f"incomplete or excluded coverage: {name}")
        functions = data.get("functions")
        if functions is None:
            raise ValueError(f"function coverage is unavailable: {name}")
        actual = {function: value for function, value in functions.items() if function}
        for function, value in actual.items():
            if (
                value["summary"]["missing_lines"]
                or value["summary"]["missing_branches"]
            ):
                raise ValueError(f"incomplete function coverage: {name}:{function}")
        modules[name] = {
            "statements": summary["num_statements"],
            "branches": summary["num_branches"],
            "functions": len(actual),
            "coverage": 100,
        }
    return modules
