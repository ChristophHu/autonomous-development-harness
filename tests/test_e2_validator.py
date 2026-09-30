"""Plan/worktree evidence contracts for independent task validation."""

import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_evidence_workflow import runtime, specification

from harness.agents import ExecutorOutput


def _task():
    task = specification()
    task.plan = {
        "summary": "edit implementation",
        "subtasks": [
            {
                "id": "implement",
                "write_paths": ["addition.py"],
                "acceptance_criteria": ["sum"],
                "requirement_ids": ["add two integers"],
            }
        ],
    }
    return task


def _coverage():
    return {
        "commands": [],
        "coverage": {
            "totals": {
                "percent_covered": 100,
                "missing_lines": 0,
                "missing_branches": 0,
            }
        },
    }


def _output(changed_files=None):
    changed_files = ["addition.py"] if changed_files is None else changed_files
    return ExecutorOutput(
        subtask_id="implement",
        success=True,
        output="implemented",
        changed_files=changed_files,
        tool_evidence=[
            {
                "tool": "filesystem.write",
                "changed_path": path,
                "changed_paths": [path],
            }
            for path in changed_files
        ],
    )


def test_independent_review_must_cover_exactly_declared_contract_keys():
    from harness.validation import EvidenceValidator

    task = _task()
    task.requirements = ["add two integers"]
    valid = {
        "requirements": {"add two integers": True},
        "criteria": {"sum": True},
        "evidence": "checked output and tests",
    }
    extra = {
        **valid,
        "requirements": {**valid["requirements"], "invented": True},
    }
    missing = {**valid, "criteria": {}}

    assert EvidenceValidator._review_confirms(valid, task)
    assert not EvidenceValidator._review_confirms(extra, task)
    assert not EvidenceValidator._review_confirms(missing, task)


def test_plan_must_match_executor_outputs(tmp_path):
    _, orchestrator = runtime(tmp_path)
    task = _task()
    result = orchestrator.validator.validate(
        task, [], _coverage(), workspace_before=None
    )
    assert not result.valid
    assert any("missing executor result" in error for error in result.errors)
    plan_finding = next(
        item for item in result.findings if item.rule == "plan.alignment"
    )
    assert plan_finding.category == "plan"
    assert plan_finding.source == "plan_validator"
    assert "missing executor result" in plan_finding.message


def test_plan_requires_requirement_and_criterion_in_the_same_step(tmp_path):
    _, orchestrator = runtime(tmp_path)
    task = _task()
    task.plan["subtasks"][0]["acceptance_criteria"] = []

    errors, _ = orchestrator.validator._plan_findings(task, [_output()])

    assert "requirement has no linked acceptance criterion: add two integers" in errors


def test_missing_workspace_snapshots_block_mutation_validation(tmp_path):
    _, orchestrator = runtime(tmp_path)
    errors, observed = orchestrator.validator._change_findings(
        [_output([])],
        {"implement": {"write_paths": ["addition.py"]}},
        None,
        None,
    )

    assert observed == []
    assert "complete before/after workspace snapshots are required" in errors


def test_correction_finding_contract_does_not_parse_error_text(tmp_path):
    _, orchestrator = runtime(tmp_path)
    task = _task()
    result = orchestrator.validator.validate(
        task, [], _coverage(), open_required_questions=True
    )
    findings = {(item.category, item.rule): item for item in result.findings}
    assert findings[("human_input", "question.required_open")].source == "validator"
    assert findings[("plan", "plan.alignment")].source == "plan_validator"
    assert findings[("workspace", "workspace.integrity")].evidence["messages"]


def test_changed_file_claim_requires_observed_mutation_evidence(tmp_path):
    _, orchestrator = runtime(tmp_path)
    task = _task()
    output = _output()
    output.tool_evidence = []
    result = orchestrator.validator.validate(task, [output], _coverage())
    assert not result.valid
    assert any("changed file claim lacks tool evidence" in e for e in result.errors)


def test_changed_file_must_fit_planned_write_scope(tmp_path):
    _, orchestrator = runtime(tmp_path)
    task = _task()
    output = _output(["unplanned.py"])
    output.tool_evidence = [
        {"tool": "filesystem.write", "changed_path": "unplanned.py"}
    ]
    result = orchestrator.validator.validate(task, [output], _coverage())
    assert not result.valid
    assert any("outside planned write scope" in error for error in result.errors)


def test_git_head_or_branch_mutation_blocks_completion(tmp_path):
    _, orchestrator = runtime(tmp_path)
    task = _task()
    before = {
        "applicable": True,
        "root": str(tmp_path),
        "branch": "feature/task",
        "head": "a" * 40,
        "changed_paths": [],
    }
    after = before | {"branch": "main", "head": "b" * 40}

    result = orchestrator.validator.validate(
        task, [_output([])], _coverage(), workspace_before=before, workspace_after=after
    )

    assert not result.valid
    assert "Git branch changed during execution" in result.errors
    assert "Git HEAD changed during execution" in result.errors


def test_unreported_git_change_blocks_completion(tmp_path):
    _, orchestrator = runtime(tmp_path)
    task = _task()
    before = {"applicable": True, "changed_paths": []}
    after = {"applicable": True, "changed_paths": ["surprise.py"]}

    result = orchestrator.validator.validate(
        task, [_output([])], _coverage(), workspace_before=before, workspace_after=after
    )

    assert not result.valid
    assert (
        "workspace change is absent from executor results: surprise.py" in result.errors
    )


def test_git_snapshot_is_read_only_and_reports_repo_identity(tmp_path, monkeypatch):
    _, orchestrator = runtime(tmp_path)
    git = shutil.which("git", path="/opt/homebrew/bin:" + os.environ.get("PATH", ""))
    assert git
    monkeypatch.setenv("PATH", str(Path(git).parent) + ":" + os.environ.get("PATH", ""))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "addition.py").write_text("def add(a, b): return a+b\n")
    (tmp_path / ".gitignore").write_text("db.sqlite\n")
    subprocess.run(["git", "add", ".gitignore"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "addition.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=tmp_path, check=True)
    before = orchestrator.validator.workspace_snapshot()

    (tmp_path / "addition.py").write_text("def add(a, b): return a-b\n")
    after = orchestrator.validator.workspace_snapshot()

    assert before["applicable"] is True
    assert before["head"] == after["head"]
    assert before["branch"] == after["branch"]
    assert after["changed_paths"] == ["addition.py"]
    assert (tmp_path / "addition.py").read_text().endswith("a-b\n")


def test_non_git_workspace_is_explicitly_not_applicable(tmp_path):
    _, orchestrator = runtime(tmp_path)
    snapshot = orchestrator.validator.workspace_snapshot()
    assert snapshot["applicable"] is False
    assert snapshot["reason"] == "workspace is not a Git repository"
    assert isinstance(snapshot["filesystem"], dict)


def test_filesystem_snapshot_detects_mutation_of_preexisting_untracked_path(tmp_path):
    _, orchestrator = runtime(tmp_path)
    target = tmp_path / "addition.py"
    target.write_text("before\n")
    before = orchestrator.validator.workspace_snapshot()
    target.write_text("after\n")
    after = orchestrator.validator.workspace_snapshot()

    assert before["filesystem"]["addition.py"] != after["filesystem"]["addition.py"]
    errors, changed = orchestrator.validator._change_findings(
        [_output([])],
        {"implement": {"write_paths": ["addition.py"]}},
        before,
        after,
    )
    assert changed == ["addition.py"]
    assert any("absent from executor results" in error for error in errors)


def test_filesystem_snapshot_detects_second_change_to_preexisting_git_dirty_path(
    tmp_path,
):
    _, orchestrator = runtime(tmp_path)
    before = {
        "applicable": True,
        "root": str(tmp_path),
        "branch": "main",
        "head": "a" * 40,
        "changed_paths": ["addition.py"],
        "filesystem": {"addition.py": "file:644:6:before"},
    }
    after = before | {"filesystem": {"addition.py": "file:644:5:after"}}

    errors, changed = orchestrator.validator._change_findings(
        [_output([])],
        {"implement": {"write_paths": ["addition.py"]}},
        before,
        after,
    )

    assert changed == ["addition.py"]
    assert any("absent from executor results" in error for error in errors)


def test_filesystem_snapshot_detects_ignored_shell_created_file(tmp_path):
    _, orchestrator = runtime(tmp_path)
    (tmp_path / ".gitignore").write_text("*.sqlite\n")
    (tmp_path / ".git").mkdir()

    def git(args, **kwargs):
        result = SimpleNamespace(returncode=0, stderr="", stdout="")
        if args == ["rev-parse", "--show-toplevel"]:
            result.stdout = str(tmp_path)
        elif args == ["branch", "--show-current"]:
            result.stdout = "main"
        elif args == ["rev-parse", "HEAD"]:
            result.stdout = "a" * 40
        return result

    orchestrator.tools.git = git
    before = orchestrator.validator.workspace_snapshot()
    (tmp_path / "shell.sqlite").write_text("created by shell\n")
    after = orchestrator.validator.workspace_snapshot()

    assert before["changed_paths"] == after["changed_paths"] == []
    errors, changed = orchestrator.validator._change_findings(
        [_output([])],
        {"implement": {"write_paths": ["shell.sqlite"]}},
        before,
        after,
    )
    assert changed == ["shell.sqlite"]
    assert any("absent from executor results" in error for error in errors)


def test_filesystem_snapshot_detects_symlink_target_change_without_following_it(
    tmp_path,
):
    _, orchestrator = runtime(tmp_path)
    (tmp_path / "link").symlink_to("first-target")
    before = orchestrator.validator.workspace_snapshot()
    (tmp_path / "link").unlink()
    (tmp_path / "link").symlink_to("second-target")
    after = orchestrator.validator.workspace_snapshot()

    assert before["filesystem"]["link"] != after["filesystem"]["link"]


def test_filesystem_snapshot_fails_closed_on_unreadable_file(tmp_path, monkeypatch):
    _, orchestrator = runtime(tmp_path)
    target = tmp_path / "file.txt"
    target.write_text("data")
    original_open = Path.open

    def fail_target(path, *args, **kwargs):
        if path == target:
            raise OSError("read failed")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_target)
    with pytest.raises(
        RuntimeError, match="workspace filesystem could not be verified"
    ):
        orchestrator.validator.workspace_snapshot()


def test_filesystem_snapshot_handles_missing_permissions_config(tmp_path):
    _, orchestrator = runtime(tmp_path)
    orchestrator.tools.permissions = None

    assert isinstance(orchestrator.validator._filesystem_snapshot(), dict)


def test_filesystem_snapshot_ignores_unresolvable_runtime_exclusions(
    tmp_path, monkeypatch
):
    _, orchestrator = runtime(tmp_path)

    def invalid_path(_key):
        raise ValueError("invalid configured path")

    monkeypatch.setattr(orchestrator.config, "path", invalid_path)
    assert isinstance(orchestrator.validator._filesystem_snapshot(), dict)


def test_filesystem_snapshot_fails_closed_if_file_changes_mid_read(
    tmp_path, monkeypatch
):
    _, orchestrator = runtime(tmp_path)
    target = tmp_path / "racing.txt"
    target.write_text("stable bytes")
    original_lstat = Path.lstat
    calls = 0

    def racing_lstat(path):
        nonlocal calls
        first = original_lstat(path)
        if path == target:
            calls += 1
            if calls == 2:
                return SimpleNamespace(
                    st_ino=first.st_ino,
                    st_size=first.st_size,
                    st_mtime_ns=first.st_mtime_ns + 1,
                    st_mode=first.st_mode,
                )
        return first

    monkeypatch.setattr(Path, "lstat", racing_lstat)
    with pytest.raises(
        RuntimeError, match="workspace filesystem could not be verified"
    ):
        orchestrator.validator._filesystem_snapshot()


def test_filesystem_snapshot_fails_closed_on_special_file(tmp_path):
    _, orchestrator = runtime(tmp_path)
    os.mkfifo(tmp_path / "pipe")

    with pytest.raises(
        RuntimeError, match="workspace filesystem could not be verified"
    ):
        orchestrator.validator._filesystem_snapshot()


def test_filesystem_snapshot_fails_closed_when_walk_reports_error(
    tmp_path, monkeypatch
):
    _, orchestrator = runtime(tmp_path)

    def failing_walk(root, *, followlinks, onerror):
        onerror(OSError("directory could not be read"))
        return iter(())

    monkeypatch.setattr("harness.validation.os.walk", failing_walk)
    with pytest.raises(
        RuntimeError, match="workspace filesystem could not be verified"
    ):
        orchestrator.validator._filesystem_snapshot()


def test_filesystem_snapshot_path_escape_is_a_finding(tmp_path):
    _, orchestrator = runtime(tmp_path)
    before = {"filesystem": {"../escape.py": "before"}}
    after = {"filesystem": {"../escape.py": "after"}}

    errors, _ = orchestrator.validator._change_findings(
        [_output([])],
        {"implement": {"write_paths": ["addition.py"]}},
        before,
        after,
    )

    assert "filesystem snapshot path escapes workspace: ../escape.py" in errors


def test_created_parent_directory_is_attributed_to_changed_file(tmp_path):
    _, orchestrator = runtime(tmp_path)
    before = {"filesystem": {}}
    after = {
        "filesystem": {
            "newdir": "directory:755",
            "newdir/new.py": "file:644:5:hash",
        }
    }
    output = _output(["newdir/new.py"])
    output.tool_evidence = [{"tool": "shell.execute", "changed_path": "newdir/new.py"}]

    errors, observed = orchestrator.validator._change_findings(
        [output],
        {"implement": {"write_paths": ["newdir/new.py"]}},
        before,
        after,
    )

    assert observed == ["newdir/new.py"]
    assert not errors


def test_git_snapshot_fails_closed_for_invalid_repo_and_state(tmp_path, monkeypatch):
    _, orchestrator = runtime(tmp_path)
    (tmp_path / ".git").mkdir()
    failure = SimpleNamespace(
        returncode=1, stderr="fatal: not a git repository", stdout=""
    )
    monkeypatch.setattr(orchestrator.tools, "git", lambda *a, **k: failure)
    assert orchestrator.validator.workspace_snapshot() == {
        "applicable": False,
        "reason": "workspace is not a Git repository",
        "filesystem": {},
    }

    failure.stderr = "git unavailable"
    try:
        orchestrator.validator.workspace_snapshot()
    except RuntimeError as exc:
        assert "identity" in str(exc)
    else:
        raise AssertionError("Git identity failure must be fail-closed")

    outside = SimpleNamespace(returncode=0, stderr="", stdout=str(tmp_path.parent))
    monkeypatch.setattr(orchestrator.tools, "git", lambda *a, **k: outside)
    try:
        orchestrator.validator.workspace_snapshot()
    except PermissionError as exc:
        assert "escapes" in str(exc)
    else:
        raise AssertionError("repository outside workspace must be rejected")

    valid_root = SimpleNamespace(returncode=0, stderr="", stdout=str(tmp_path))
    sequence = iter(
        [
            valid_root,
            SimpleNamespace(returncode=0, stderr="", stdout="main"),
            failure,
            SimpleNamespace(returncode=0, stderr="", stdout=""),
        ]
    )
    monkeypatch.setattr(orchestrator.tools, "git", lambda *a, **k: next(sequence))
    try:
        orchestrator.validator.workspace_snapshot()
    except RuntimeError as exc:
        assert "worktree state" in str(exc)
    else:
        raise AssertionError("incomplete Git state must be rejected")


def test_porcelain_parser_handles_rename_and_rejects_malformed_records():
    from harness.validation import EvidenceValidator

    assert EvidenceValidator._porcelain_paths("R  new.py\0old.py\0?? fresh.py\0") == [
        "fresh.py",
        "new.py",
        "old.py",
    ]
    assert EvidenceValidator._porcelain_paths("C  copied.py\0source.py\0") == [
        "copied.py"
    ]
    try:
        EvidenceValidator._porcelain_paths("bad")
    except ValueError as exc:
        assert "malformed" in str(exc)
    else:
        raise AssertionError("malformed Git status must be rejected")


def test_workspace_path_rejects_traversal_and_symlink_escape(tmp_path):
    _, orchestrator = runtime(tmp_path)
    outside = tmp_path.parent / "outside-e2.txt"
    outside.write_text("outside")
    (tmp_path / "linked.txt").symlink_to(outside)
    for value in ("", "../outside-e2.txt", str(outside), "linked.txt"):
        try:
            orchestrator.validator._workspace_path(value)
        except (ValueError, PermissionError):
            pass
        else:
            raise AssertionError(f"unsafe workspace path accepted: {value}")


def test_plan_findings_reject_invalid_duplicate_and_extra_steps(tmp_path):
    _, orchestrator = runtime(tmp_path)
    task = _task()
    task.plan = {"subtasks": [None, {"id": "implement"}, {"id": "implement"}]}
    duplicate_outputs = [
        _output([]),
        _output([]),
        ExecutorOutput(subtask_id="rogue", success=True, output="unexpected"),
    ]

    errors, _ = orchestrator.validator._plan_findings(task, duplicate_outputs)

    assert "plan contains an invalid step" in errors
    assert "plan contains duplicate step: implement" in errors
    assert "executor results contain duplicate plan steps" in errors
    assert "executor result has no plan step: rogue" in errors


def test_change_findings_reject_invalid_scope_paths_and_duplicate_claims(tmp_path):
    _, orchestrator = runtime(tmp_path)
    output = _output(["../escape.py", "../escape.py"])
    output.tool_evidence = ["not a mapping", {"changed_paths": ["../escape.py", None]}]
    errors, _ = orchestrator.validator._change_findings(
        [output], {"implement": {"write_paths": "not-a-list"}}, None, None
    )
    scope_errors, _ = orchestrator.validator._change_findings(
        [_output([])], {"implement": {"write_paths": ["../bad.py"]}}, None, None
    )

    assert any("write scope is invalid" in error for error in errors)
    assert any("write scope escapes" in error for error in scope_errors)
    assert any("changed file path escapes" in error for error in errors)
    assert any("tool evidence path escapes" in error for error in errors)
    assert any("duplicate changed paths" in error for error in errors)


def test_git_diff_path_escape_and_missing_write_scope_are_findings(tmp_path):
    _, orchestrator = runtime(tmp_path)
    output = _output(["safe.py"])
    output.tool_evidence = [{"tool": "filesystem.write", "changed_path": "safe.py"}]
    errors, _ = orchestrator.validator._change_findings(
        [output],
        {"implement": {"write_paths": []}},
        {
            "applicable": True,
            "root": str(tmp_path),
            "branch": "main",
            "head": "a",
            "changed_paths": [],
        },
        {
            "applicable": True,
            "root": str(tmp_path),
            "branch": "main",
            "head": "a",
            "changed_paths": ["../escaped.py"],
        },
    )
    assert any("no declared write scope" in error for error in errors)
    assert any("Git diff path escapes workspace" in error for error in errors)


def test_git_repository_root_is_checked_after_workspace_resolution(tmp_path):
    _, orchestrator = runtime(tmp_path)
    (tmp_path / ".git").mkdir()
    sequence = iter(
        [
            SimpleNamespace(returncode=0, stderr="", stdout=str(tmp_path)),
            SimpleNamespace(returncode=0, stderr="", stdout="main"),
            SimpleNamespace(returncode=0, stderr="", stdout="a" * 40),
            SimpleNamespace(returncode=0, stderr="", stdout=""),
        ]
    )
    orchestrator.tools.git = lambda *args, **kwargs: next(sequence)
    snapshot = orchestrator.validator.workspace_snapshot()
    assert snapshot["applicable"] is True
