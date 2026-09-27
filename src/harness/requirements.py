"""Grounded requirement completion before planning or human clarification."""

import json

from .database import DecisionRepository
from .domain import Task

FIELDS = (
    "goal",
    "requirements",
    "acceptance_criteria",
    "test_commands",
    "coverage_command",
)


class RequirementCompleter:
    def __init__(self, store, router):
        self.store, self.router = store, router

    def complete(self, task, context):
        sources = [{"source": "task", "data": task.model_dump(mode="json")}]
        if task.parent_task_id:
            parent = self.store.get(task.parent_task_id)
            if parent:
                sources.append(
                    {"source": "parent", "data": parent.model_dump(mode="json")}
                )
        answers = [
            dict(row)
            for row in self.store.questions.list(task.id)
            if row["status"] != "open"
        ]
        sources.append({"source": "answers", "data": answers})
        sources.append(
            {
                "source": "decisions",
                "data": [
                    dict(row)
                    for row in DecisionRepository(self.store.database).list(task.id)
                ],
            }
        )
        retrieved = context() if callable(context) else context
        sources.append({"source": "memory_and_repository", "data": retrieved})
        for answer in answers:
            if answer["reason"] == "requirements:incomplete":
                try:
                    payload = json.loads(answer["answer"])
                    task = Task.model_validate(
                        task.model_dump()
                        | {name: payload[name] for name in FIELDS if name in payload}
                    )
                except (ValueError, TypeError):
                    pass
        missing = [name for name in FIELDS if not getattr(task, name)]
        if missing:
            try:
                payload = json.loads(
                    self.router.complete(
                        "planner",
                        "REQUIREMENTS: Derive only missing fields from these sources; do not invent requirements or commands. Return JSON with fields and rationale; leave unsupported fields empty.\n"
                        + json.dumps({"missing": missing, "sources": sources}),
                    )
                )
                fields = payload.get("fields", {})
                task = Task.model_validate(
                    task.model_dump()
                    | {name: fields[name] for name in missing if name in fields}
                )
                if payload.get("rationale"):
                    DecisionRepository(self.store.database).save(
                        task.id, "requirements completed", payload["rationale"]
                    )
            except (RuntimeError, ValueError, TypeError):
                pass
        self.store.tasks.update(task.id, metadata=task.model_dump(mode="json"))
        self.store.event(
            task.id,
            "requirements.inspected",
            {
                "sources": [source["source"] for source in sources],
                "missing": [name for name in FIELDS if not getattr(task, name)],
            },
        )
        return task, json.dumps(sources)
