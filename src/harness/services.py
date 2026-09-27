"""Shared application boundary for task creation, editing and execution."""

import asyncio
import json
import uuid

from .domain import Status, Task


class TaskService:
    def __init__(self, store, orchestrator):
        self.store = store
        self.orchestrator = orchestrator

    def get(self, task_id):
        task = self.store.get(task_id)
        if task is None:
            raise ValueError("task not found")
        return task

    def create(self, task):
        if task.id is not None or task.status != Status.PENDING or task.git_state:
            raise ValueError(
                "new tasks must be pending without an ID or internal Git state"
            )
        for dependency in task.dependencies + (
            [task.parent_task_id] if task.parent_task_id else []
        ):
            self.get(dependency)
        return self.store.create(task)

    def patch(self, task_id, payload):
        self.get(task_id)
        protected = {
            "id",
            "status",
            "result",
            "created_at",
            "updated_at",
            "plan",
            "validation_result",
            "test_result",
            "git_state",
        }
        if not payload or protected.intersection(payload):
            raise ValueError("invalid editable task fields")
        with self.store.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT 1 FROM task_leases WHERE task_id=? AND expires_at>strftime('%s','now')",
                (task_id,),
            ).fetchone():
                raise ValueError("task is currently running")
            updated = Task.model_validate(self.get(task_id).model_dump() | payload)
            pending = updated.dependencies + (
                [updated.parent_task_id] if updated.parent_task_id else []
            )
            visited = set()
            while pending:
                dependency = pending.pop()
                if dependency == task_id:
                    raise ValueError("task dependency cycle")
                if dependency not in visited:
                    visited.add(dependency)
                    linked = self.get(dependency)
                    pending.extend(linked.dependencies)
                    if linked.parent_task_id:
                        pending.append(linked.parent_task_id)
            connection.execute(
                "UPDATE tasks SET title=?,description=?,metadata=?,updated_at=? WHERE id=?",
                (
                    updated.title,
                    updated.description,
                    json.dumps(updated.model_dump(mode="json")),
                    self.store.database.now(),
                    task_id,
                ),
            )
        return self.get(task_id)

    def list(self, status=None):
        return [self.get(row["id"]) for row in self.store.tasks.list(status)]

    def abort(self, task_id):
        self.get(task_id)
        self.store.tasks.transition(task_id, Status.CANCELLED)
        return self.get(task_id)

    def delete(self, task_id):
        self.get(task_id)
        with self.store.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT 1 FROM task_leases WHERE task_id=? AND expires_at>strftime('%s','now')",
                (task_id,),
            ).fetchone():
                raise ValueError("task is currently running")
            connection.execute(
                "DELETE FROM model_runs WHERE agent_run_id IN (SELECT id FROM agent_runs WHERE task_id=?)",
                (task_id,),
            )
            connection.execute("DELETE FROM agent_runs WHERE task_id=?", (task_id,))
            connection.execute("DELETE FROM tool_calls WHERE task_id=?", (task_id,))
            connection.execute("DELETE FROM tasks WHERE id=?", (task_id,))

    async def start(self, task_id):
        task = self.get(task_id)
        if any(
            self.get(dependency).status != Status.COMPLETED
            for dependency in task.dependencies
        ):
            raise ValueError("task dependencies are incomplete")
        owner = uuid.uuid4().hex
        if not self.store.tasks.claim(
            task_id, owner, exclusive=self.orchestrator.git_enabled
        ):
            raise ValueError("task is terminal or currently running")

        async def heartbeat():
            while True:
                await asyncio.sleep(30)
                if not self.store.tasks.renew(task_id, owner):
                    return

        renewal = asyncio.create_task(heartbeat())
        try:
            return await self.orchestrator._run(task_id, owner)
        finally:
            renewal.cancel()
            await asyncio.gather(renewal, return_exceptions=True)
            self.store.tasks.release(task_id, owner)

    async def answer(self, task_id, question_id, answer):
        task = self.get(task_id)
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("answer must not be blank")
        if not self.store.answer(question_id, answer, task_id):
            raise ValueError("open question not found or invalid answer")
        self.store.event(task_id, "question.answered", {"question_id": question_id})
        if task.git_state.get("cleanup_question_id") == question_id:
            return await asyncio.to_thread(
                self.orchestrator._invoke,
                task,
                "git-cleanup",
                "coding",
                self.orchestrator.git_service.cleanup,
                task_id,
                question_id,
            )
        if (
            task.status != Status.WAITING_HUMAN
            or self.store.questions.has_open_required(task_id)
        ):
            return self.get(task_id)
        return await self.start(task_id)
