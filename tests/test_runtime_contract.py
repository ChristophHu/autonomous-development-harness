"""Regression contracts for the three implementation steps (RED first)."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from harness.agents import ModelRegistry, ModelRouter, Planner, PlannerOutput
from harness.core import Config, Orchestrator, Store, Task
from harness.providers import OpenAICompatibleProvider


def test_unknown_provider_never_becomes_successful_local_response():
    registry = ModelRegistry(SimpleNamespace(data={}))
    with pytest.raises(ValueError, match="provider"):
        registry.get("unknown")


def test_invalid_planner_response_is_not_replaced_by_an_invented_plan():
    config = SimpleNamespace(
        data={"profiles": {"planner": {"model": {"primary": "fixture"}}}}
    )
    registry = ModelRegistry(config)
    registry.register("fixture", SimpleNamespace(complete=lambda *a, **kw: "not JSON"))
    with pytest.raises(ValueError):
        Planner(ModelRouter(registry, config)).plan(Task(title="fix parser"))


@pytest.mark.parametrize("dependencies", [["missing"], ["step"]])
def test_invalid_plan_dependencies_are_rejected(dependencies):
    with pytest.raises(ValueError):
        PlannerOutput.model_validate(
            {
                "summary": "x",
                "complexity": "simple",
                "subtasks": [
                    {
                        "id": "step",
                        "title": "x",
                        "description": "x",
                        "dependencies": dependencies,
                    }
                ],
            }
        )


def test_provider_preserves_tool_call_history_and_model_selection():
    observed = []

    def handler(request):
        observed.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "done"}}]})

    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "actual file"},
    ]
    provider = OpenAICompatibleProvider(
        "fixture",
        "http://fixture/v1",
        model="configured-model",
        transport=httpx.MockTransport(handler),
    )
    provider.complete(messages)
    assert observed[0]["messages"] == messages
    assert observed[0]["model"] == "configured-model"


def test_task_metadata_survives_sqlite_roundtrip(tmp_path):
    config = Config()
    config.data["paths"]["database"] = str(tmp_path / "task.db")
    store = Store(config)
    task = store.create(
        Task(
            title="implement",
            goal="working parser",
            requirements=["parse integers"],
            constraints=["stdlib only"],
            priority=5,
        )
    )
    loaded = store.get(task.id)
    assert loaded.goal == "working parser" and loaded.requirements == ["parse integers"]
    assert loaded.constraints == ["stdlib only"] and loaded.priority == 5


def test_required_question_blocks_even_a_failed_task(tmp_path):
    config = Config()
    config.data["paths"] = {
        "database": str(tmp_path / "blocked.db"),
        "workspace": str(tmp_path / "workspace"),
        "obsidian_vault": str(tmp_path / "vault"),
    }
    config.data["tools"]["mcp"]["servers"]["vault"]["enabled"] = False
    store = Store(config)
    task = store.create(Task(title="blocked"))
    store.ask(task.id, "Required?", "requirements", required=True)
    store.tasks.update(task.id, status="failed")
    with pytest.raises(ValueError, match="question"):
        asyncio.run(Orchestrator(store, config).run(task.id))
