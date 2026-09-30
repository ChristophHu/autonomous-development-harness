"""Shared application boundary for task creation, editing and execution."""

import asyncio
import threading
import uuid

from .domain import EventKind, Status, Task
from .process_control import RunControl, TaskCancelled, use_run_control


class TaskService:
    def __init__(self, store, orchestrator):
        self.store = store
        self.orchestrator = orchestrator
        self.active_controls = {}
        self._controls_lock = threading.Lock()

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
        if not self.store.update_task_if_idle(task_id, updated):
            raise ValueError("task is currently running")
        return self.get(task_id)

    def list(self, status=None):
        return [self.get(row["id"]) for row in self.store.tasks.list(status)]

    def status_counts(self):
        return {
            status.value: len(self.store.tasks.list(status.value)) for status in Status
        }

    def events(self, task_id, *filters):
        self.get(task_id)
        return self.store.list_events(task_id, *filters)

    def event_feed(self, *filters):
        return self.store.list_events(*filters)

    def events_after(self, cursor, task_id=None):
        if task_id is not None:
            self.get(task_id)
        return self.store.events.after(cursor, task_id)

    def questions(self, task_id):
        self.get(task_id)
        return self.store.list_questions(task_id)

    def plan(self, task_id):
        self.get(task_id)
        return self.store.latest_plan(task_id)

    def validation(self, task_id):
        self.get(task_id)
        return self.store.latest_validation(task_id)

    def ask_question(self, task_id, question, reason, options=None, required=True):
        self.get(task_id)
        return self.store.ask(task_id, question, reason, options, required)

    def model_usage(self, **filters):
        return self.store.model_usage.report(**filters)

    def abort(self, task_id):
        self.get(task_id)
        self.store.tasks.transition(task_id, Status.CANCELLED)
        with self._controls_lock:
            control = self.active_controls.get(task_id)
        if control is not None:
            control.request_stop("task_aborted")
        return self.get(task_id)

    def delete(self, task_id):
        self.get(task_id)
        if not self.store.tasks.delete_if_idle(task_id):
            raise ValueError("task is currently running")

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
        control = RunControl()
        with self._controls_lock:
            self.active_controls[task_id] = control

        async def heartbeat():
            while True:
                await asyncio.sleep(30)
                if not self.store.tasks.renew(task_id, owner):
                    control.request_stop("lease_lost")
                    return

        renewal = asyncio.create_task(heartbeat())
        try:
            with use_run_control(control):
                try:
                    return await self.orchestrator._run(task_id, owner)
                except TaskCancelled:
                    return self.get(task_id)
        finally:
            renewal.cancel()
            await asyncio.gather(renewal, return_exceptions=True)
            with self._controls_lock:
                if self.active_controls.get(task_id) is control:
                    del self.active_controls[task_id]
            self.store.tasks.release(task_id, owner)

    async def answer(self, task_id, question_id, answer):
        task = self.get(task_id)
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("answer must not be blank")
        if not self.store.answer(question_id, answer, task_id):
            raise ValueError("open question not found or invalid answer")
        self.store.event(
            task_id, EventKind.QUESTION_ANSWERED, {"question_id": question_id}
        )
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
