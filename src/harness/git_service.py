"""Opt-in task-bound local Git workflows; never push or delete autonomously."""

import re
from pathlib import Path

from .approvals import ApprovalService, GitApprovalTarget
from .domain import EventKind
from .workflows import Workflow


class GitWorkflowService:
    def __init__(self, store, workflow, tools):
        self.store = store
        self.workflow = workflow
        self.tools = tools
        self.settings = store.config.data.get("git", {})
        self.approvals = ApprovalService(store.questions, store)

    def save(self, task_id, state):
        task = self.store.get(task_id)
        task.git_state = state
        self.store.update(task)
        self.store.event(task_id, EventKind.GIT_WORKFLOW, state)

    def blocked(self, task_id, error):
        self.store.ask(
            task_id,
            "Git-Zustand manuell prüfen: " + str(error),
            "git:reconciliation",
            options=["retry"],
        )
        return False

    def request_repair(self, task_id, branch, error, paths):
        task = self.store.get(task_id)
        state = dict(task.git_state)
        workflow = Workflow(state["workflow"])
        main = self.settings.get("main", "main")
        dev = self.settings.get("dev", "dev")
        if workflow == Workflow.RELEASE and branch == main:
            return self.blocked(
                task_id,
                "Der bereits veröffentlichte Release-Branch benötigt einen neuen Release-Task.",
            )
        cwd = str(self.tools.workspace)
        try:
            self.workflow._validate_ref(branch)
            if (
                self.workflow._git(["branch", "--show-current"], cwd).stdout.strip()
                != branch
            ):
                raise RuntimeError("Fehlerhafter Zielbranch ist nicht ausgecheckt.")
            if self.workflow._git(["status", "--porcelain=v1"], cwd).stdout.strip():
                raise RuntimeError(
                    "Zielbranch ist nicht sauber und kann nicht repariert werden."
                )
            target_oid = self.oid("refs/heads/" + branch, cwd)
            repair_branch = self.workflow.branch_name(
                workflow, f"repair-{task_id}-{branch}-{target_oid[:8]}"
            )
            exact_paths = sorted(
                set(paths)
                | {item.path for item in task.acceptance_criteria if item.path}
            )
            exact_paths.extend(
                sorted(
                    {
                        token
                        for command in task.test_commands
                        for token in command
                        if not token.startswith("-")
                        and not Path(token).is_absolute()
                        and (self.tools.workspace / token).is_file()
                    }
                    - set(exact_paths)
                )
            )
            exact_paths = sorted(set(exact_paths))
            arguments = ["repair", branch, repair_branch, target_oid, *exact_paths]
            target = GitApprovalTarget(
                task_id, cwd, tuple(arguments), action="git.repair"
            )
            question_id = self.approvals.request(target, required=True)
            state["repair"] = {
                "branch": branch,
                "branch_oid": target_oid,
                "branch_name": repair_branch,
                "repository": cwd,
                "paths": exact_paths,
                "arguments": list(target.arguments),
                "target": target.__dict__,
                "question_id": question_id,
                "authorized": False,
                "validation_errors": str(error),
                "workflow": str(workflow),
                "primary_branch": main,
                "secondary_branch": dev,
            }
            self.save(task_id, state)
            self.store.event(
                task_id,
                EventKind.GIT_REPAIR_REQUESTED,
                {
                    "question_id": question_id,
                    "branch": branch,
                    "target_oid": target_oid,
                },
            )
            return False
        except RuntimeError as failure:
            return self.blocked(task_id, failure)

    def begin_repair(self, task):
        state = dict(task.git_state)
        repair = dict(state["repair"])
        cwd = str(self.tools.workspace)
        try:
            if repair["repository"] != cwd:
                raise RuntimeError(
                    "Repair-Repository stimmt nicht mit Workspace überein."
                )
            current = self.workflow._git(
                ["branch", "--show-current"], cwd
            ).stdout.strip()
            if self.oid("refs/heads/" + repair["branch"], cwd) != repair["branch_oid"]:
                raise RuntimeError(
                    "Freigegebener Zielbranch wurde zwischenzeitlich verändert."
                )
            question = self.store.questions.get(repair["question_id"])
            if not question or question["task_id"] != task.id:
                raise RuntimeError("Repair-Freigabe gehört nicht zu diesem Task.")
            target = GitApprovalTarget(**repair["target"])
            if (
                target.action != "git.repair"
                or list(target.arguments) != repair["arguments"]
            ):
                raise RuntimeError(
                    "Repair-Freigabe passt nicht zur gespeicherten Operation."
                )
            if question["reason"] != target.reason():
                raise RuntimeError("Repair-Freigabe ist nicht an das Ziel gebunden.")
            if question["answer"] == "deny":
                state["phase"] = "repair_declined"
                repair["declined"] = True
                state["repair"] = repair
                self.save(task.id, state)
                self.store.event(
                    task.id,
                    EventKind.GIT_REPAIR_DECLINED,
                    {"question_id": question["id"]},
                )
                return False
            if question["answer"] != "approve":
                raise RuntimeError(
                    "Repair benötigt eine explizite approve/deny-Antwort."
                )
            operation = ["switch", "-c", repair["branch_name"], repair["branch"]]
            if question["status"] == "executed":
                if current == repair["branch_name"]:
                    if (
                        self.oid("refs/heads/" + repair["branch_name"], cwd)
                        != repair["branch_oid"]
                    ):
                        raise RuntimeError(
                            "Reparaturbranch wurde vor dem Start verändert."
                        )
                elif current == repair["branch"]:
                    if self.workflow._git(
                        ["branch", "--list", repair["branch_name"]], cwd
                    ).stdout.strip():
                        raise RuntimeError(
                            "Reparaturbranch existiert mit unbekanntem Inhalt."
                        )
                    self.workflow._git(operation, cwd)
                else:
                    raise RuntimeError(
                        "Repository steht auf einem unerwarteten Branch."
                    )
            elif question["status"] in {"answered", "consumed"}:
                if (
                    current != repair["branch"]
                    or self.workflow._git(
                        ["status", "--porcelain=v1"], cwd
                    ).stdout.strip()
                ):
                    raise RuntimeError(
                        "Repair benötigt den sauberen freigegebenen Zielbranch."
                    )
                issue = (
                    self.approvals.resume_issue
                    if question["status"] == "consumed"
                    else self.approvals.issue
                )
                grant = issue(task.id, question["id"], target.action, target)
                if not grant.consume(target.action, target):
                    raise RuntimeError("Repair-Freigabe wurde bereits verbraucht.")
                self.workflow._git(operation, cwd)
            else:
                raise RuntimeError("Repair-Freigabe ist nicht ausführbar.")
            repair["original_branch"] = state["branch"]
            repair["authorized"] = True
            state.update(
                branch=repair["branch_name"], phase="branch_created", repair=repair
            )
            self.save(task.id, state)
            self.store.event(
                task.id,
                EventKind.GIT_REPAIR_AUTHORIZED,
                {"question_id": question["id"], "branch": repair["branch_name"]},
            )
            return True
        except RuntimeError as error:
            return self.blocked(task.id, error)

    def oid(self, ref, cwd):
        return self.workflow._git(
            ["rev-parse", "--verify", ref + "^{commit}"], cwd
        ).stdout.strip()

    def verify_source(self, state, cwd):
        if self.oid("refs/heads/" + state["branch"], cwd) != state.get("source_oid"):
            raise RuntimeError("Task-Branch wurde seit dem geprüften Commit verändert.")
        if self.workflow._git(["status", "--porcelain=v1"], cwd).stdout.strip():
            raise RuntimeError("Offene Änderungen/Konflikte zuerst manuell auflösen.")

    def resolve_commit_intent(self, state, cwd, task_id):
        branch_ref = "refs/heads/" + state["branch"]
        branch_head = self.oid(branch_ref, cwd)
        parent = state.get("commit_parent")
        if not parent:
            raise RuntimeError("Commit-Absicht besitzt keine Parent-ID.")
        if branch_head == parent:
            self.verify_intent_worktree(state, cwd)
            return False
        actual_parent = self.workflow._git(
            ["rev-parse", "--verify", branch_ref + "^"], cwd
        ).stdout.strip()
        subject = self.workflow._git(
            ["log", "-1", "--format=%s", branch_ref], cwd
        ).stdout.strip()
        changed = self.workflow._git(
            ["diff", "--name-only", parent, branch_head], cwd
        ).stdout.splitlines()
        allowed = set(state.get("commit_paths", []))
        if (
            actual_parent != parent
            or subject != state.get("commit_message")
            or not changed
            or not set(changed) <= allowed
        ):
            raise RuntimeError(
                "Branch-HEAD passt nicht zur gespeicherten Commit-Absicht."
            )
        state.update(phase="committed", source_oid=branch_head)
        self.save(task_id, state)
        return True

    def verify_intent_worktree(self, state, cwd):
        output = self.workflow._git(["status", "--porcelain=v1", "-z"], cwd).stdout
        paths = []
        for entry in output.split("\0"):
            if not entry:
                continue
            code, path = entry[:2], entry[3:]
            if "U" in code or "R" in code or "C" in code:
                raise RuntimeError(
                    "Commit-Absicht enthält ungelöste Konflikte/Umbenennungen."
                )
            paths.append(path)
        allowed = set(state.get("commit_paths", []))
        if not paths or not set(paths) <= allowed:
            raise RuntimeError(
                "Änderungen passen nicht zu den erlaubten Commit-Pfaden."
            )

    def merge_once(self, source, target, cwd):
        self.workflow._validate_ref(target)
        contained = self.workflow._git(
            ["branch", "--list", "--contains", source, target], cwd
        ).stdout.strip()
        self.workflow._git(["switch", target], cwd)
        if not contained:
            self.workflow.merge(source, cwd)
        return self.oid("refs/heads/" + target, cwd)

    def begin(self, task):
        decision = self.workflow.classify_task(task)
        workflow = decision.workflow
        self.store.event(
            task.id,
            EventKind.GIT_WORKFLOW_CLASSIFIED,
            {
                "workflow": str(workflow),
                "source": decision.source,
                "evidence": decision.evidence,
            },
        )
        if workflow == Workflow.OTHER:
            self.save(task.id, {"workflow": "other", "phase": "no_git"})
            return True
        cwd = str(self.tools.workspace)
        if task.git_state:
            repair = task.git_state.get("repair", {})
            if repair and not repair.get("authorized"):
                return self.begin_repair(task)
            if task.git_state.get("phase") in {
                "commit_pending",
                "committed",
                "merged_primary",
                "merged_secondary",
                "tagged",
                "merged",
                "awaiting_cleanup",
            }:
                try:
                    if task.git_state.get("repository") != cwd:
                        raise RuntimeError(
                            "Task-Repository stimmt nicht mit Workspace überein."
                        )
                    if task.git_state.get("phase") == "commit_pending":
                        if task.git_state.get("task_id") is None:
                            task.git_state["task_id"] = task.id
                        resolved = self.resolve_commit_intent(
                            task.git_state, cwd, task.id
                        )
                        if not resolved:
                            self.verify_intent_worktree(task.git_state, cwd)
                            self.workflow._git(
                                ["switch", task.git_state["branch"]], cwd
                            )
                            self.store.event(
                                task.id, EventKind.GIT_RECONCILED, task.git_state
                            )
                            return True
                    self.verify_source(task.git_state, cwd)
                    self.workflow._git(["switch", task.git_state["branch"]], cwd)
                    self.store.event(task.id, EventKind.GIT_RECONCILED, task.git_state)
                    return True
                except RuntimeError as error:
                    return self.blocked(task.id, error)
            current = self.workflow._git(
                ["branch", "--show-current"], cwd
            ).stdout.strip()
            if (
                task.git_state.get("phase") != "branch_created"
                or current != task.git_state.get("branch")
                or task.git_state.get("repository") != cwd
            ):
                return self.blocked(
                    task.id,
                    "Persistierter Workflow passt nicht zum aktuellen Git-Zustand.",
                )
            return True
        base = (
            self.settings.get("main", "main")
            if workflow == Workflow.HOTFIX
            else self.settings.get("dev", "dev")
        )
        self.workflow._validate_ref(base)
        name = f"task-{task.id}-{task.title}"
        if workflow == Workflow.RELEASE:
            if not re.fullmatch(
                r"v?\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?", task.release_version
            ):
                raise ValueError("release requires an explicit semantic version")
            name = task.release_version.removeprefix("v")
        try:
            if self.workflow._git(["status", "--porcelain=v1"], cwd).stdout.strip():
                return self.blocked(
                    task.id,
                    "Workspace ist nicht sauber; keine Änderungen werden gestasht/verworfen.",
                )
            self.workflow._git(["switch", base], cwd)
            remote = self.settings.get("remote")
            if remote:
                self.workflow._validate_ref(remote)
                self.workflow._git(["pull", "--ff-only", remote, base], cwd)
            self.workflow.create_branch(workflow, name, cwd, base=base)
            self.save(
                task.id,
                {
                    "workflow": str(workflow),
                    "repository": cwd,
                    "branch": self.workflow.branch_name(workflow, name),
                    "phase": "branch_created",
                },
            )
            return True
        except RuntimeError as error:
            return self.blocked(task.id, error)

    def finish(self, task_id, paths, validate_target=None):
        task = self.store.get(task_id)
        state = dict(task.git_state)
        if state.get("workflow") == "other":
            return True
        if (
            not task.validation_result
            or task.validation_result.get("valid") is not True
            or not task.test_result
            or not task.test_result.get("commands")
            or not task.test_result.get("coverage")
            or task.test_result["coverage"]["totals"].get("percent_covered", 0)
            < task.coverage_threshold
            or any(item["returncode"] for item in task.test_result["commands"])
        ):
            raise ValueError(
                "Git finish requires successful tests and independent validation"
            )
        if state.get("phase") not in {
            "commit_pending",
            "branch_created",
            "committed",
            "merged_primary",
            "merged_secondary",
            "tagged",
            "merged",
            "awaiting_cleanup",
            "repair_declined",
        }:
            return self.blocked(
                task_id, "Workflow-Merge muss vor Wiederholung reconciled werden."
            )
        cwd = str(self.tools.workspace)
        if state.get("repository") != cwd:
            return self.blocked(
                task_id, "Task-Repository stimmt nicht mit Workspace überein."
            )
        try:
            if (
                self.workflow._git(["branch", "--show-current"], cwd).stdout.strip()
                != state["branch"]
            ):
                return self.blocked(
                    task_id, "Aktueller Branch entspricht nicht dem Task."
                )
            candidates = set(paths) | {
                criterion.path
                for criterion in task.acceptance_criteria
                if criterion.path
            }
            # Historical paths are provenance only; Git determines what is actually changed.
            if not candidates:
                raise ValueError("Git commit requires explicit task artifact paths")
            if state["phase"] == "branch_created":
                paths = sorted(candidates)
                message = f"Task {task_id}: {task.title}"
                state.update(
                    phase="commit_pending",
                    task_id=task_id,
                    commit_parent=self.oid("HEAD", cwd),
                    commit_paths=paths,
                    commit_message=message,
                )
                self.save(task_id, state)
            if state["phase"] == "commit_pending":
                recovered = self.resolve_commit_intent(state, cwd, task_id)
                if not recovered:
                    self.workflow.commit(
                        state["commit_paths"], state["commit_message"], cwd
                    )
                    state.update(phase="committed", source_oid=self.oid("HEAD", cwd))
                    self.save(task_id, state)
            else:
                self.verify_source(state, cwd)
            workflow = Workflow(state["workflow"])
            repair = state.get("repair", {})
            authorized_repair = repair.get("authorized", False)
            target = (
                repair["branch"]
                if authorized_repair
                else self.settings.get("dev", "dev")
                if workflow in {Workflow.FEATURE, Workflow.BUGFIX}
                else self.settings.get("main", "main")
            )
            if (
                authorized_repair
                and self.oid("refs/heads/" + target, cwd) != repair["branch_oid"]
            ):
                raise RuntimeError("Repair-Zielbranch wurde nach Freigabe verändert.")
            if (
                not authorized_repair
                and state.get("primary_oid")
                and self.oid("refs/heads/" + target, cwd) != state["primary_oid"]
            ):
                raise RuntimeError(
                    "Primärer Zielbranch wurde nach dem Merge verändert."
                )
            primary_oid = self.merge_once(state["source_oid"], target, cwd)
            state["primary_oid"] = primary_oid
            state["phase"] = "merged_primary"
            self.save(task_id, state)
            if validate_target is not None and not validate_target(target):
                return self.request_repair(
                    task_id,
                    target,
                    "Tests/Validation des gemergten Zielbranches sind fehlgeschlagen.",
                    sorted(candidates),
                )
            if workflow == Workflow.RELEASE and not authorized_repair:
                tag = "v" + task.release_version.removeprefix("v")
                if self.workflow._git(["tag", "--list", tag], cwd).stdout.strip():
                    self.workflow._git(
                        ["rev-parse", "--verify", "refs/tags/" + tag + "^{tag}"], cwd
                    )
                    if self.oid("refs/tags/" + tag, cwd) != primary_oid:
                        raise RuntimeError(
                            "Release-Tag verweist auf einen anderen Commit."
                        )
                else:
                    self.workflow.tag_release(task.release_version, cwd)
                state["phase"] = "tagged"
                self.save(task_id, state)
            if workflow in {Workflow.HOTFIX, Workflow.RELEASE} and not (
                authorized_repair and target == repair["secondary_branch"]
            ):
                secondary = self.settings.get("dev", "dev")
                self.merge_once(primary_oid, secondary, cwd)
                state["phase"] = "merged_secondary"
                self.save(task_id, state)
                if validate_target is not None and not validate_target(secondary):
                    return self.request_repair(
                        task_id,
                        secondary,
                        "Tests/Validation des synchronisierten Zielbranches sind fehlgeschlagen.",
                        sorted(candidates),
                    )
            state["phase"] = (
                "awaiting_cleanup" if state.get("cleanup_question_id") else "merged"
            )
            self.save(task_id, state)
            if workflow != Workflow.RELEASE and not state.get("cleanup_question_id"):
                target = self.tools.git_target(
                    task_id, ["branch", "-d", state["branch"]], cwd
                )
                question_id = self.approvals.request(target)
                state.update(
                    cleanup_target=target.__dict__,
                    cleanup_question_id=question_id,
                    phase="awaiting_cleanup",
                )
                self.save(task_id, state)
            return True
        except RuntimeError as error:
            return self.blocked(task_id, error)

    def cleanup(self, task_id, question_id):
        task = self.store.get(task_id)
        state = dict(task.git_state)
        if state.get("cleanup_question_id") != question_id:
            raise ValueError("cleanup question does not match task workflow")
        question = self.store.questions.get(question_id)
        if question["answer"] == "deny":
            state["phase"] = "cleanup_declined"
        else:
            target = GitApprovalTarget(**state["cleanup_target"])
            grant = self.approvals.issue(task_id, question_id, target.action, target)
            self.workflow.execute(
                list(target.arguments), target.repository, grant, task_id=task_id
            )
            state["phase"] = "cleaned"
        self.save(task_id, state)
        return self.store.get(task_id)
