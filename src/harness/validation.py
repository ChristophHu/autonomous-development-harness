"""Independent validation based on actual process and filesystem observations."""

import json
import time

from .agents import ValidatorOutput


class EvidenceValidator:
    def __init__(self, tools, router):
        self.tools = tools
        self.router = router

    def path(self, name):
        path = (self.tools.workspace / name).resolve()
        if not path.is_relative_to(self.tools.workspace):
            raise PermissionError("validation path escapes workspace")
        return path

    def run_tests(self, task):
        reports = []
        commands = task.test_commands + task.lint_commands
        if task.coverage_command:
            commands = commands + [task.coverage_command]
        started = time.time()
        for command in commands:
            result = self.tools.shell(command)
            reports.append(
                {
                    "command": command,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            )
        coverage = None
        if task.coverage_command:
            path = self.path(task.coverage_report)
            if path.exists() and path.stat().st_mtime >= started:
                coverage = json.loads(path.read_text())
        return {"commands": reports, "coverage": coverage}

    def validate(self, task, outputs, tests):
        errors = [
            f"subtask {output.subtask_id} failed: {output.output}"
            for output in outputs
            if not output.success
        ]
        if not task.test_commands:
            errors.append("no test command specified")
        for report in tests["commands"]:
            if report["returncode"]:
                errors.append(
                    f"command failed: {report['command']}: {report['stdout']} {report['stderr']}"
                )
        coverage = tests["coverage"]
        if coverage is None:
            errors.append("fresh coverage report is required")
        else:
            totals = coverage["totals"]
            if totals.get("percent_covered", 0) < task.coverage_threshold:
                errors.append("coverage threshold not reached")
            if task.coverage_threshold == 100 and (
                totals.get("missing_lines", 0) or totals.get("missing_branches", 0)
            ):
                errors.append("uncovered statements or branches")
        observations = {}
        artifacts = {}
        for output in outputs:
            for name in output.changed_files:
                path = self.path(name)
                artifacts[name] = (
                    path.read_text()[:32000] if path.is_file() else "missing"
                )
        for criterion in task.acceptance_criteria:
            if criterion.kind == "command":
                result = self.tools.shell(criterion.command)
                observations[criterion.id] = {
                    "passed": result.returncode == 0,
                    "evidence": result.stdout + result.stderr,
                }
            elif criterion.kind.startswith("file_"):
                path = self.path(criterion.path)
                passed = path.is_file() and (
                    criterion.kind == "file_exists"
                    or criterion.contains in path.read_text()
                )
                observations[criterion.id] = {
                    "passed": passed,
                    "evidence": path.read_text() if path.is_file() else "missing",
                }
            if (
                criterion.id in observations
                and not observations[criterion.id]["passed"]
            ):
                errors.append(f"acceptance criterion failed: {criterion.id}")
        if not task.requirements or not task.acceptance_criteria:
            errors.append("requirements and acceptance criteria are required")
        review = json.loads(
            self.router.complete(
                "validator",
                "REVIEW: Independently verify every requirement and acceptance criterion against these observations. Return JSON with requirements (exact requirement strings mapped to booleans), criteria (criterion IDs mapped to booleans), and evidence (nonempty string).\n"
                + json.dumps(
                    {
                        "task": task.model_dump(mode="json"),
                        "observations": observations,
                        "tests": tests,
                        "artifacts": artifacts,
                    }
                ),
            )
        )
        if (
            not review.get("evidence")
            or any(
                review.get("requirements", {}).get(requirement) is not True
                for requirement in task.requirements
            )
            or any(
                review.get("criteria", {}).get(criterion.id) is not True
                for criterion in task.acceptance_criteria
            )
        ):
            errors.append(
                "independent review did not confirm all requirements and criteria"
            )
        return ValidatorOutput(
            valid=not errors,
            checks=[
                "actual tests",
                "fresh coverage",
                "acceptance criteria",
                "independent review",
            ],
            errors=errors,
            required_corrections=errors,
        )
