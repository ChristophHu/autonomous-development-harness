"""RED-first contracts for correlated execution and truthful memory adapters."""

import asyncio
import json

import pytest
from test_evidence_workflow import ready_runtime

from harness.memory import ObsidianMemory, QdrantMemory


def test_tool_calls_are_persisted_with_task_and_agent(tmp_path):
    store, orchestrator, task = ready_runtime(tmp_path)
    created = store.create(task)

    def execute(step, context):
        orchestrator.tools.execute(
            "filesystem.write", {"path": "audit.txt", "content": "observed"}
        )
        from harness.agents import ExecutorOutput

        return ExecutorOutput(subtask_id=step.id, success=True, output="done")

    orchestrator.executor.execute = execute
    assert asyncio.run(orchestrator.run(created.id)).status == "completed"
    events = store.events.list(created.id, "TOOL_CALL_COMPLETED")
    assert len(events) == 1
    with store.database.connect() as connection:
        row = connection.execute(
            "SELECT * FROM tool_calls WHERE task_id=?", (created.id,)
        ).fetchone()
    assert row["agent_run_id"] and row["status"] == "completed"
    assert json.loads(row["input"]) == {"path": "audit.txt", "content": "observed"}


def test_obsidian_rejects_paths_outside_vault(tmp_path):
    vault = ObsidianMemory(tmp_path / "vault")
    with pytest.raises(PermissionError):
        vault.write("../outside", "must not escape")


def test_qdrant_never_uses_hashes_as_semantic_embeddings():
    with pytest.raises(ValueError, match="embedding"):
        QdrantMemory("http://fixture", "memory", dimension=2)._vector("meaning")
