"""Independent validation based on actual process and filesystem observations."""

import hashlib
import json
import math
import os
import stat
import time
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr

from .agents import ProfileRegistry, ValidatorOutput
from .structured_output import parse_model_output


class IndependentReviewOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    requirements: dict[StrictStr, StrictBool]
    criteria: dict[StrictStr, StrictBool]
    evidence: StrictStr
    requirement_evidence: dict[StrictStr, list[StrictStr]]
    criterion_evidence: dict[StrictStr, list[StrictStr]]


class EvidenceValidator:
    def __init__(self, tools, router):
        self.tools = tools
        self.router = router
        profiles = ProfileRegistry(router.config)
        self.profile = profiles.for_role("independent-review")
        self.tester_profile = profiles.for_role("tester")
        self.validator_profile = profiles.for_role("validator")

    def path(self, name):
        path = (self.tools.workspace / name).resolve()
        if not path.is_relative_to(self.tools.workspace):
            raise PermissionError("validation path escapes workspace")
        return path

    def workspace_snapshot(self):
        """Return Git identity and a content manifest for the workspace."""
        root = str(self.tools.workspace)
        snapshot = {
            "applicable": False,
            "reason": "workspace is not a Git repository",
        }
        if not (Path(root) / ".git").exists():
            snapshot["filesystem"] = self._filesystem_snapshot()
            return snapshot
        root_result = self.tools.git(["rev-parse", "--show-toplevel"], cwd=root)
        if root_result.returncode:
            if "not a git repository" not in root_result.stderr.lower():
                raise RuntimeError("Git repository identity could not be verified")
        else:
            repository = Path(root_result.stdout.strip()).resolve()
            workspace = Path(root).resolve()
            if not repository.is_relative_to(workspace):
                raise PermissionError(
                    "Git repository root escapes configured workspace"
                )
            branch_result = self.tools.git(["branch", "--show-current"], cwd=root)
            head_result = self.tools.git(["rev-parse", "HEAD"], cwd=root)
            status_result = self.tools.git(
                ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
                cwd=root,
            )
            if any(
                result.returncode
                for result in (branch_result, head_result, status_result)
            ):
                raise RuntimeError("Git worktree state could not be verified")
            snapshot = {
                "applicable": True,
                "root": str(repository),
                "branch": branch_result.stdout.strip(),
                "head": head_result.stdout.strip(),
                "changed_paths": self._porcelain_paths(status_result.stdout),
            }
        snapshot["filesystem"] = self._filesystem_snapshot()
        return snapshot

    def _filesystem_snapshot(self):
        """Hash workspace entries without following symlinks or traversing .git."""
        root = self.tools.workspace
        manifest = {}
        excluded = set()
        config = getattr(getattr(self.tools, "permissions", None), "config", None)
        if config is not None:
            for key in ("database", "obsidian_vault", "logs"):
                try:
                    configured = config.path(key).resolve()
                    if configured.is_relative_to(root):
                        excluded.add(configured.relative_to(root).as_posix())
                except (KeyError, OSError, TypeError, ValueError):
                    continue

        def on_walk_error(error):
            raise error

        try:
            for current, directories, files in os.walk(
                root, followlinks=False, onerror=on_walk_error
            ):
                current_path = Path(current)
                directories.sort()
                files.sort()
                directories[:] = [name for name in directories if name != ".git"]
                for name in [*directories, *files]:
                    path = current_path / name
                    relative = path.relative_to(root).as_posix()
                    if any(
                        relative == excluded_path
                        or relative.startswith(excluded_path + "/")
                        for excluded_path in excluded
                    ):
                        continue
                    first = path.lstat()
                    mode = stat.S_IMODE(first.st_mode)
                    if stat.S_ISLNK(first.st_mode):
                        value = f"symlink:{mode:o}:{os.readlink(path)}"
                    elif stat.S_ISDIR(first.st_mode):
                        value = f"directory:{mode:o}"
                    elif stat.S_ISREG(first.st_mode):
                        digest = hashlib.sha256()
                        with path.open("rb") as source:
                            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                                digest.update(chunk)
                        second = path.lstat()
                        if (
                            first.st_ino != second.st_ino
                            or first.st_size != second.st_size
                            or first.st_mtime_ns != second.st_mtime_ns
                            or first.st_mode != second.st_mode
                        ):
                            raise OSError("workspace file changed during snapshot")
                        value = f"file:{mode:o}:{first.st_size}:{digest.hexdigest()}"
                    else:
                        raise OSError("unsupported filesystem entry")
                    manifest[relative] = value
        except (OSError, ValueError) as exc:
            raise RuntimeError("workspace filesystem could not be verified") from exc
        return manifest

    @staticmethod
    def _porcelain_paths(output):
        entries = output.split("\0")
        paths = []
        skip_source = False
        include_rename_source = False
        for entry in entries:
            if not entry:
                continue
            if skip_source:
                skip_source = False
                if include_rename_source:
                    paths.append(entry)
                continue
            if len(entry) < 4 or entry[2] != " ":
                raise ValueError("Git returned malformed porcelain status")
            status, name = entry[:2], entry[3:]
            paths.append(name)
            include_rename_source = "R" in status
            if include_rename_source or "C" in status:
                skip_source = True
        return sorted(set(paths))

    def _workspace_path(self, name):
        if not isinstance(name, str) or not name:
            raise ValueError("changed file path must be a nonempty string")
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise PermissionError("changed file path escapes workspace")
        path = (self.tools.workspace / Path(*relative.parts)).resolve()
        if not path.is_relative_to(self.tools.workspace):
            raise PermissionError("changed file path escapes workspace")
        return relative.as_posix()

    def _plan_findings(self, task, outputs):
        errors = []
        plan = task.plan
        steps = plan.get("subtasks") if isinstance(plan, dict) else None
        if not isinstance(steps, list) or not steps:
            return ["validated plan with subtasks is required"], {}
        planned = {}
        for step in steps:
            if not isinstance(step, dict) or not isinstance(step.get("id"), str):
                errors.append("plan contains an invalid step")
                continue
            step_id = step["id"]
            if step_id in planned:
                errors.append(f"plan contains duplicate step: {step_id}")
            planned[step_id] = step
        mapped_requirements = set()
        mapped_criteria = set()
        recovery_targets = set()
        for step in steps:
            if not isinstance(step, dict):
                continue
            requirement_ids = step.get("requirement_ids", [])
            criteria = step.get("acceptance_criteria", [])
            if not isinstance(requirement_ids, list) or not all(
                isinstance(item, str) for item in requirement_ids
            ):
                errors.append(f"plan requirement mapping is invalid: {step.get('id')}")
                requirement_ids = []
            if not isinstance(criteria, list) or not all(
                isinstance(item, str) for item in criteria
            ):
                errors.append(f"plan acceptance mapping is invalid: {step.get('id')}")
                criteria = []
            mapped_requirements.update(requirement_ids)
            mapped_criteria.update(criteria)
            targets = step.get("recovery_targets", [])
            if isinstance(targets, list) and all(
                isinstance(item, str) for item in targets
            ):
                recovery_targets.update(targets)
            elif targets:
                errors.append(f"plan recovery mapping is invalid: {step.get('id')}")
        scoped_recovery = bool(recovery_targets)
        all_criteria = {item.id for item in task.acceptance_criteria}
        required_requirements = (
            {
                target.removeprefix("requirement:")
                for target in recovery_targets
                if target.startswith("requirement:")
            }
            if scoped_recovery
            else set(task.requirements)
        )
        required_criteria = (
            {
                target.removeprefix("criterion:")
                for target in recovery_targets
                if target.startswith("criterion:")
            }
            if scoped_recovery
            else {item.id for item in task.acceptance_criteria}
        )
        for requirement in required_requirements:
            if requirement not in mapped_requirements:
                errors.append(f"requirement has no plan step: {requirement}")
                continue
            linked_steps = [
                step
                for step in steps
                if isinstance(step, dict)
                and requirement in step.get("requirement_ids", [])
            ]
            if not any(
                set(step.get("acceptance_criteria", [])) & all_criteria
                for step in linked_steps
            ):
                errors.append(
                    f"requirement has no linked acceptance criterion: {requirement}"
                )
        known_requirements = set(task.requirements)
        for requirement in sorted(mapped_requirements - known_requirements):
            errors.append(f"plan references unknown requirement: {requirement}")
        for criterion in sorted(required_criteria - mapped_criteria):
            errors.append(f"acceptance criterion has no plan step: {criterion}")
        for criterion in sorted(mapped_criteria - all_criteria):
            errors.append(f"plan references unknown acceptance criterion: {criterion}")
        actual_ids = [output.subtask_id for output in outputs]
        if len(actual_ids) != len(set(actual_ids)):
            errors.append("executor results contain duplicate plan steps")
        actual_set = set(actual_ids)
        planned_set = set(planned)
        for missing in sorted(planned_set - actual_set):
            errors.append(f"missing executor result for plan step: {missing}")
        for unexpected in sorted(actual_set - planned_set):
            errors.append(f"executor result has no plan step: {unexpected}")
        return errors, planned

    def _change_findings(
        self,
        outputs,
        planned,
        workspace_before,
        workspace_after,
        coverage_report="coverage.json",
    ):
        errors = []
        claimed = set()
        evidenced = set()
        evidence_by_output = []
        for output in outputs:
            step = planned.get(output.subtask_id, {})
            declared = step.get("write_paths", [])
            if not isinstance(declared, list):
                errors.append(f"plan write scope is invalid: {output.subtask_id}")
                declared = []
            try:
                declared_paths = {self._workspace_path(path) for path in declared}
            except (ValueError, PermissionError):
                errors.append(
                    f"plan write scope escapes workspace: {output.subtask_id}"
                )
                declared_paths = set()
            for name in output.changed_files:
                try:
                    normalized = self._workspace_path(name)
                except (ValueError, PermissionError):
                    errors.append(f"changed file path escapes workspace: {name}")
                    continue
                claimed.add(normalized)
                if declared_paths and normalized not in declared_paths:
                    errors.append(
                        f"changed file is outside planned write scope: {normalized}"
                    )
                if not declared_paths:
                    errors.append(
                        f"plan has no declared write scope: {output.subtask_id}"
                    )
            output_evidence_paths = set()
            for item in output.tool_evidence:
                if not isinstance(item, dict):
                    continue
                changed_paths = item.get("changed_paths") or [item.get("changed_path")]
                for changed_path in changed_paths:
                    if changed_path is None:
                        continue
                    try:
                        normalized_evidence = self._workspace_path(changed_path)
                        evidenced.add(normalized_evidence)
                        output_evidence_paths.add(normalized_evidence)
                    except (ValueError, PermissionError):
                        errors.append(
                            f"tool evidence path escapes workspace: {changed_path}"
                        )
            evidence_by_output.append((output, output_evidence_paths))
            if len(output.changed_files) != len(set(output.changed_files)):
                errors.append(
                    f"executor result contains duplicate changed paths: {output.subtask_id}"
                )
        observed = set(evidenced)
        before_filesystem = (
            workspace_before.get("filesystem")
            if isinstance(workspace_before, dict)
            else None
        )
        after_filesystem = (
            workspace_after.get("filesystem")
            if isinstance(workspace_after, dict)
            else None
        )
        if not isinstance(before_filesystem, dict) or not isinstance(
            after_filesystem, dict
        ):
            errors.append("complete before/after workspace snapshots are required")
        else:
            changed_entries = {
                name
                for name in before_filesystem.keys() | after_filesystem.keys()
                if before_filesystem.get(name) != after_filesystem.get(name)
                and not self._generated_test_output(name, coverage_report)
            }
            for name in changed_entries:
                entry = after_filesystem.get(name) or before_filesystem.get(name)
                if entry.startswith("directory:") and any(
                    child.startswith(name + "/") for child in changed_entries
                ):
                    continue
                try:
                    observed.add(self._workspace_path(name))
                except (ValueError, PermissionError):
                    errors.append(f"filesystem snapshot path escapes workspace: {name}")
        if (
            isinstance(workspace_before, dict)
            and workspace_before.get("applicable")
            and isinstance(workspace_after, dict)
            and workspace_after.get("applicable")
        ):
            for field, label in (
                ("root", "repository"),
                ("branch", "branch"),
                ("head", "HEAD"),
            ):
                if workspace_before.get(field) != workspace_after.get(field):
                    errors.append(f"Git {label} changed during execution")
            before = set(workspace_before.get("changed_paths", []))
            after = set(workspace_after.get("changed_paths", []))
            for name in after - before:
                try:
                    normalized = self._workspace_path(name)
                    observed.add(normalized)
                except (ValueError, PermissionError):
                    errors.append(f"Git diff path escapes workspace: {name}")
        for unexpected in sorted(observed - claimed):
            errors.append(
                f"workspace change is absent from executor results: {unexpected}"
            )
        for missing in sorted(claimed - observed):
            errors.append(f"claimed change was not observed: {missing}")
        for output, item_paths in evidence_by_output:
            for name in output.changed_files:
                try:
                    normalized = self._workspace_path(name)
                except (ValueError, PermissionError):
                    continue
                if normalized not in item_paths and normalized not in observed:
                    errors.append(f"changed file claim lacks tool evidence: {name}")
        return errors, sorted(observed)

    @staticmethod
    def _generated_test_output(path, coverage_report):
        return (
            path in {".coverage", coverage_report, ".pytest_cache"}
            or path.startswith((".coverage.", ".pytest_cache/"))
            or "/__pycache__/" in f"/{path}/"
            or path.endswith(".pyc")
        )

    @staticmethod
    def _test_side_effect_findings(
        workspace_after_executor, workspace_after_tests, coverage_report
    ):
        """Reject test-run mutations except known generated reports/caches."""
        if workspace_after_executor is None and workspace_after_tests is None:
            return []
        if not isinstance(workspace_after_executor, dict) or not isinstance(
            workspace_after_tests, dict
        ):
            return ["workspace snapshot around test execution is missing"]
        before = workspace_after_executor.get("filesystem")
        after = workspace_after_tests.get("filesystem")
        if not isinstance(before, dict) or not isinstance(after, dict):
            return ["workspace filesystem snapshot around tests is invalid"]
        identity_errors = [
            f"test execution changed workspace identity: {field}"
            for field in ("applicable", "root", "branch", "head")
            if field in workspace_after_executor or field in workspace_after_tests
            if workspace_after_executor.get(field) != workspace_after_tests.get(field)
        ]
        changed = sorted(
            path
            for path in set(before) | set(after)
            if before.get(path) != after.get(path)
            and not EvidenceValidator._generated_test_output(path, coverage_report)
        )
        return identity_errors + [
            f"test execution modified workspace outside generated outputs: {path}"
            for path in changed
        ]

    def run_tests(self, task):
        test_input = {
            "task_id": task.id,
            "test_commands": task.test_commands,
            "lint_commands": task.lint_commands,
            "coverage_command": task.coverage_command,
        }
        self.tester_profile.validate_input(test_input)
        reports = []
        commands = [(command, "test.run_tests") for command in task.test_commands]
        commands.extend((command, "quality.lint") for command in task.lint_commands)
        if task.coverage_command:
            commands.append((task.coverage_command, "test.run_coverage"))
        started = time.time_ns()
        for command, tool in commands:
            result = self.tools.execute(tool, {"command": command}, allow_nonzero=True)
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
            if path.exists() and path.stat().st_mtime_ns >= started:
                try:
                    coverage = json.loads(path.read_text())
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    coverage = None
        result = {"commands": reports, "coverage": coverage}
        self.tester_profile.validate_output(result)
        return result

    def validate(
        self,
        task,
        outputs,
        tests,
        open_required_questions=False,
        workspace_before=None,
        workspace_after=None,
        verify_workspace_changes=True,
        workspace_after_tests=None,
    ):
        self.validator_profile.validate_input(
            {
                "task": task.model_dump(mode="json"),
                "outputs": [item.model_dump(mode="json") for item in outputs],
                "tests": tests,
            }
        )
        failed_subtasks = [output for output in outputs if not output.success]
        errors = [
            f"subtask {output.subtask_id} failed: {output.output}"
            for output in failed_subtasks
        ]
        findings_by_rule = {}

        def add_findings(category, rule, messages, *, source="validator"):
            messages = [message for message in messages if message]
            if messages:
                findings_by_rule[(category, rule, source)] = {
                    "category": category,
                    "source": source,
                    "rule": rule,
                    "message": "; ".join(messages),
                    "evidence": {"messages": messages},
                }

        add_findings(
            "execution",
            "subtask.failed",
            [
                f"subtask {output.subtask_id} failed: {output.output}"
                for output in failed_subtasks
            ],
            source="executor",
        )
        add_findings(
            "task_contract",
            "test_command.missing",
            ["no test command specified" if not task.test_commands else ""],
        )
        add_findings(
            "tests",
            "test_command.failed",
            [
                f"command failed: {report['command']}: {report['stdout']} {report['stderr']}"
                for report in tests["commands"]
                if report["returncode"]
            ],
            source="test_runner",
        )
        add_findings(
            "human_input",
            "question.required_open",
            [
                "open required human questions block completion"
                if open_required_questions
                else ""
            ],
        )
        if not task.test_commands:
            errors.append("no test command specified")
        for report in tests["commands"]:
            if report["returncode"]:
                errors.append(
                    f"command failed: {report['command']}: {report['stdout']} {report['stderr']}"
                )
        if open_required_questions:
            errors.append("open required human questions block completion")
        plan_errors, planned = self._plan_findings(task, outputs)
        errors.extend(plan_errors)
        add_findings("plan", "plan.alignment", plan_errors, source="plan_validator")
        if verify_workspace_changes:
            change_errors, observed_changes = self._change_findings(
                outputs,
                planned,
                workspace_before,
                workspace_after,
                task.coverage_report,
            )
        else:
            change_errors, observed_changes = [], []
        errors.extend(change_errors)
        add_findings(
            "workspace",
            "workspace.integrity",
            change_errors,
            source="workspace_validator",
        )
        test_mutations = (
            self._test_side_effect_findings(
                workspace_after,
                workspace_after_tests,
                task.coverage_report,
            )
            if workspace_after_tests is not None
            else []
        )
        errors.extend(test_mutations)
        add_findings(
            "workspace",
            "tests.workspace_mutation",
            test_mutations,
            source="test_runner_validator",
        )
        coverage = tests.get("coverage") if isinstance(tests, dict) else None
        coverage_errors = []
        if coverage is None:
            coverage_errors.append("fresh coverage report is required")
        else:
            totals = coverage.get("totals") if isinstance(coverage, dict) else None
            percent = (
                totals.get("percent_covered") if isinstance(totals, dict) else None
            )
            missing_lines = (
                totals.get("missing_lines") if isinstance(totals, dict) else None
            )
            missing_branches = (
                totals.get("missing_branches") if isinstance(totals, dict) else None
            )
            valid_percent = (
                isinstance(percent, (int, float))
                and not isinstance(percent, bool)
                and math.isfinite(percent)
                and 0 <= percent <= 100
            )
            valid_missing = all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in (missing_lines, missing_branches)
            )
            if not valid_percent or not valid_missing:
                coverage_errors.append("coverage report is invalid")
            else:
                if percent < task.coverage_threshold:
                    coverage_errors.append("coverage threshold not reached")
                if task.coverage_threshold == 100 and (
                    missing_lines or missing_branches
                ):
                    coverage_errors.append("uncovered statements or branches")
        errors.extend(coverage_errors)
        add_findings(
            "coverage",
            "coverage.report",
            coverage_errors,
            source="coverage_validator",
        )
        observations = {}
        artifacts = {}
        acceptance_errors = []
        for output in outputs:
            for name in output.changed_files:
                path = self.path(name)
                artifacts[name] = (
                    path.read_text()[:32000] if path.is_file() else "missing"
                )
        for criterion in task.acceptance_criteria:
            if criterion.kind == "command":
                result = self.tools.execute(
                    "test.run_tests",
                    {"command": criterion.command},
                    allow_nonzero=True,
                )
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
                acceptance_errors.append(f"acceptance criterion failed: {criterion.id}")
        evidence_catalog = {
            *(f"plan-step:{step['id']}" for step in planned.values()),
            *(f"file:{name}" for name in artifacts),
            *(f"criterion:{name}" for name in observations),
            *(
                f"test:{report['command']}"
                for report in tests["commands"]
                if report["returncode"] == 0
            ),
        }
        errors.extend(acceptance_errors)
        add_findings(
            "acceptance",
            "acceptance.criterion_failed",
            acceptance_errors,
            source="acceptance_validator",
        )
        if not task.requirements or not task.acceptance_criteria:
            errors.append("requirements and acceptance criteria are required")
            add_findings(
                "task_contract",
                "requirements_or_criteria.missing",
                ["requirements and acceptance criteria are required"],
            )
        review_parse_failed = False
        try:
            review = parse_model_output(
                IndependentReviewOutput,
                self.router.complete(
                    self.profile.name,
                    "REVIEW: Independently compare every requirement and acceptance criterion with its mapped plan step, the actual observed workspace diff/artifact contents, and fresh test results. A plan claim alone is not proof: return true only when observed evidence supports the requirement. Return JSON with requirements (exact requirement strings mapped to booleans), criteria (criterion IDs mapped to booleans), requirement_evidence and criterion_evidence (each key mapped to one or more available IDs; cite at least one matching plan-step:<id> and at least one observed file:<path>, criterion:<id>, or test:<command> ID), and evidence (nonempty summary). Do not invent IDs or mark unsupported/unclear work as confirmed.\n"
                    + json.dumps(
                        {
                            "task": task.model_dump(mode="json"),
                            "observations": observations,
                            "tests": tests,
                            "artifacts": artifacts,
                            "plan_findings": plan_errors,
                            "workspace_findings": change_errors,
                            "workspace_before": workspace_before,
                            "workspace_after": workspace_after,
                            "observed_changes": observed_changes,
                            "available_evidence_ids": sorted(evidence_catalog),
                        }
                    ),
                    complexity=task.complexity,
                ),
                agent="independent_review",
            )
            self.profile.validate_output(review.model_dump(mode="json"))
            review = review.model_dump()
        except (TypeError, ValueError):
            review = None
            review_parse_failed = True
        if not self._review_confirms(review, task, evidence_catalog):
            review_error = (
                "independent review response is invalid"
                if review_parse_failed or not isinstance(review, dict)
                else "independent review did not confirm all requirements and criteria"
            )
            errors.append(review_error)
            add_findings(
                "review",
                "independent_review.unconfirmed",
                [review_error],
                source="independent_reviewer",
            )
        result = ValidatorOutput(
            valid=not errors,
            checks=[
                "actual tests",
                "fresh coverage",
                "plan alignment",
                "workspace and Git diff",
                "acceptance criteria",
                "independent review",
            ],
            errors=errors,
            required_corrections=errors,
            findings=[
                finding for finding in findings_by_rule.values() if finding is not None
            ],
        )
        self.validator_profile.validate_output(result.model_dump(mode="json"))
        return result

    @staticmethod
    def _review_confirms(review, task, available_evidence=None):
        if not isinstance(review, dict):
            return False
        evidence = review.get("evidence")
        requirements = review.get("requirements")
        criteria = review.get("criteria")
        requirement_evidence = review.get("requirement_evidence")
        criterion_evidence = review.get("criterion_evidence")
        available = set(available_evidence or ())

        def evidence_is_traceable(references, expected_steps):
            if not isinstance(references, list) or not references:
                return False
            if not all(
                isinstance(reference, str) and reference in available
                for reference in references
            ):
                return False
            cited_steps = {item for item in references if item.startswith("plan-step:")}
            observed = {
                item for item in references if not item.startswith("plan-step:")
            }
            return bool(cited_steps & expected_steps) and bool(observed)

        plan_steps = (
            task.plan.get("subtasks", []) if isinstance(task.plan, dict) else []
        )
        requirement_steps = {
            requirement: {
                f"plan-step:{step['id']}"
                for step in plan_steps
                if isinstance(step, dict)
                and isinstance(step.get("id"), str)
                and requirement in step.get("requirement_ids", [])
            }
            for requirement in task.requirements
        }
        criterion_steps = {
            criterion.id: {
                f"plan-step:{step['id']}"
                for step in plan_steps
                if isinstance(step, dict)
                and isinstance(step.get("id"), str)
                and criterion.id in step.get("acceptance_criteria", [])
            }
            for criterion in task.acceptance_criteria
        }
        return (
            isinstance(evidence, str)
            and bool(evidence.strip())
            and isinstance(requirements, dict)
            and isinstance(criteria, dict)
            and set(requirements) == set(task.requirements)
            and set(criteria) == {item.id for item in task.acceptance_criteria}
            and isinstance(requirement_evidence, dict)
            and set(requirement_evidence) == set(task.requirements)
            and isinstance(criterion_evidence, dict)
            and set(criterion_evidence)
            == {item.id for item in task.acceptance_criteria}
            and all(
                evidence_is_traceable(requirement_evidence[key], requirement_steps[key])
                for key in task.requirements
            )
            and all(
                evidence_is_traceable(criterion_evidence[key], criterion_steps[key])
                for key in criterion_steps
            )
            and all(requirements.get(value) is True for value in task.requirements)
            and all(
                criteria.get(value.id) is True for value in task.acceptance_criteria
            )
        )
