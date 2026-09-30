import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from test_evidence_workflow import ready_runtime

from harness.api import app
from harness.core import (
    Config,
    ConfigurationService,
    Orchestrator,
    Permissions,
    Store,
    Task,
    ToolRegistry,
)
from harness.database import (
    Database,
    DecisionRepository,
    EventRepository,
    PlanRepository,
)
from harness.memory import ObsidianMemory
from harness.providers import CostCalculator, ModelUsage
from harness.security import SecretResolver
from harness.tools import ToolExecutor
from harness.workflows import GitWorkflow, Workflow


def test_task_lifecycle(tmp_path):
    c = Config()
    c.data["paths"]["database"] = str(tmp_path / "test.db")
    s = Store(c)
    t = s.create(Task(title="x"))
    assert s.get(t.id).title == "x"
    assert s.corrections is not None


def test_permissions():
    c = Config()
    c.data["tools"] = {"permissions": {"filesystem": "read"}}
    Permissions(c).require("filesystem")


def test_orchestrator_success(tmp_path):
    c = Config()
    c.data["paths"]["database"] = str(tmp_path / "run.db")
    c.data["profiles"] = {
        k: {"model": {"primary": "local"}}
        for k in ["planner", "software-architect", "coding", "validator"]
    }
    s = Store(c)
    t = s.create(Task(title="run"))
    result = asyncio.run(Orchestrator(s).run(t.id))
    assert result.status.value == "waiting_human"


def test_authorized_repair_plan_cannot_write_outside_approved_paths(tmp_path):
    from harness.agents import Subtask

    _store, orchestrator, task = ready_runtime(tmp_path)
    task.git_state = {"repair": {"authorized": True, "paths": ["allowed.py"]}}
    step = Subtask(
        id="repair", title="repair", description="repair", write_paths=["other.py"]
    )
    with pytest.raises(ValueError, match="approved artifact paths"):
        orchestrator._execute_step(task, step, "repair", None)


def test_permission_denied():
    c = Config()
    c.data["tools"] = {"permissions": {}}
    try:
        Permissions(c).require("shell")
    except PermissionError:
        pass
    else:
        assert False


def test_api_health_and_task(tmp_path, monkeypatch):
    from harness import api

    api.orchestrator.config.data["profiles"] = {
        k: {"model": {"primary": "local"}}
        for k in ["planner", "software-architect", "coding", "validator"]
    }
    store, orchestrator, _ = ready_runtime(tmp_path)
    monkeypatch.setattr(api, "store", store)
    monkeypatch.setattr(api, "orchestrator", orchestrator)
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    response = client.post("/tasks", json={"title": "api"})
    assert response.status_code == 200
    task_id = response.json()["id"]
    assert client.get(f"/tasks/{task_id}").status_code == 200
    assert client.post(f"/tasks/{task_id}/run").json()["status"] == "waiting_human"
    assert (
        client.get(f"/api/tasks/{task_id}/result").json()["status"] == "waiting_human"
    )
    assert client.get("/api/tasks").status_code == 200
    q = client.post(
        f"/api/tasks/{task_id}/questions",
        json={"question": "Approve?", "reason": "test"},
    ).json()
    assert client.get(f"/api/tasks/{task_id}/questions").json()[0]["status"] == "open"
    assert (
        client.post(
            f"/api/tasks/{task_id}/answers",
            json={"question_id": q["id"], "answer": "yes"},
        ).status_code
        == 200
    )
    assert (
        client.patch(f"/api/tasks/{task_id}", json={"description": "updated"}).json()[
            "description"
        ]
        == "updated"
    )
    assert (
        client.patch(f"/api/tasks/{task_id}", json={"status": "completed"}).status_code
        == 422
    )
    assert client.get(f"/api/tasks/{task_id}/plan").status_code == 200
    assert client.get(f"/api/tasks/{task_id}/validation").status_code == 200
    assert client.get(f"/api/tasks/{task_id}/events").status_code == 200
    assert client.get("/api/events?event_type=task.completed").status_code == 200
    store.tasks.update(task_id, status="pending")
    assert client.delete(f"/api/tasks/{task_id}").status_code == 204
    assert client.delete("/api/tasks/999999").status_code == 404
    assert (
        client.post(
            "/api/tasks/999999/questions",
            json={"question": "x", "reason": "not specified"},
        ).status_code
        == 404
    )
    assert client.get("/tasks/999999").status_code == 404


def test_tools(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("needle")
    c = Config()
    c.data["tools"] = {"permissions": {"filesystem": "read", "git": "read"}}
    tools = ToolRegistry(Permissions(c), workspace=tmp_path)
    assert tools.read_file(str(f)) == "needle"
    assert str(f) in tools.search(str(tmp_path), "needle")


def test_extended_infrastructure(tmp_path):
    db = Database(tmp_path / "db.sqlite")
    task_id = (
        __import__("harness.database", fromlist=["TaskRepository"])
        .TaskRepository(db)
        .create("repo")
    )
    PlanRepository(db).save(task_id, "summary", {"x": 1})
    DecisionRepository(db).save(task_id, "yes", "because")
    memory = ObsidianMemory(tmp_path / "vault")
    memory.write("decisions", "approved")
    memory.append("decisions", "next")
    assert memory.search("approved")
    usage = ModelUsage("local", "m", 100, 200)
    assert CostCalculator({"m": {"input": 1, "output": 2}}).calculate(usage) == 0.0005
    assert SecretResolver().redact("a secret", ["secret"]) == "a [REDACTED]"
    assert (
        GitWorkflow(None).branch_name(Workflow.FEATURE, "New Thing")
        == "feature/new-thing"
    )


def test_filesystem_workspace_boundary(tmp_path):
    c = Config()
    c.data["tools"] = {"permissions": {"filesystem": "write"}}
    tools = ToolExecutor(Permissions(c))
    root = tmp_path / "work"
    root.mkdir()
    tools.filesystem("write", "a.txt", workspace=str(root), content="hello")
    assert tools.filesystem("read", "a.txt", workspace=str(root)) == "hello"
    try:
        tools.filesystem("read", "../escape", workspace=str(root))
    except PermissionError:
        pass
    else:
        assert False


def test_question_repository(tmp_path):
    from harness.database import QuestionRepository

    db = Database(tmp_path / "questions.sqlite")
    repo_tasks = __import__(
        "harness.database", fromlist=["TaskRepository"]
    ).TaskRepository(db)
    task_id = repo_tasks.create("question task")
    questions = QuestionRepository(db)
    qid = questions.create(task_id, "Proceed?", "destructive", ["yes", "no"])
    assert questions.list(task_id)[0]["status"] == "open"
    assert questions.answer(qid, "maybe") is False
    assert questions.answer(qid, "yes") is True and questions.answer(qid, "no") is False


def test_completed_task_cannot_be_restarted(tmp_path):
    store, orchestrator, task = ready_runtime(tmp_path)
    created = store.create(task)
    completed = asyncio.run(orchestrator.run(created.id))
    assert completed.status == "completed"
    assert completed.complexity == "LOW"
    with pytest.raises(ValueError, match="terminal"):
        asyncio.run(orchestrator.run(created.id))


def test_git_deletion_requires_approval():
    workflow = GitWorkflow(type("T", (), {"git": lambda *args: "ok"})())
    try:
        workflow.execute(["branch", "-d", "feature/x"], ".")
    except PermissionError:
        pass
    else:
        assert False


def test_event_filters(tmp_path):
    db = Database(tmp_path / "events.db")
    repo = EventRepository(db)
    repo.append(None, "task.started", {})
    repo.append(None, "task.completed", {})
    assert [row["kind"] for row in repo.list(event_type="task.started")] == [
        "task.started"
    ]


def test_config_validates_port(tmp_path):
    import yaml

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"api": {"port": 70000}}))
    try:
        Config(config_path)
    except ValueError as exc:
        assert "api.port" in str(exc)
    else:
        assert False


def test_config_rejects_enabled_provider_without_url(tmp_path):
    import yaml

    path = tmp_path / "invalid-provider.yaml"
    path.write_text(
        yaml.safe_dump({"models": {"providers": {"broken": {"enabled": True}}}})
    )
    with pytest.raises(ValueError, match="base_url"):
        Config(path)


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({"api": []}, "api must be a mapping"),
        ({"api": {"port": True}}, "api.port must be an integer"),
        ({"api": {"host": "0.0.0.0"}}, "api.host must remain local"),
        ({"paths": {"workspace": 7}}, "paths.workspace must be a non-empty string"),
        ({"database": {"type": "postgres"}}, "database.type must be sqlite"),
        (
            {"memory": {"embeddings": {"dimensions": 0}}},
            "memory.embeddings.dimensions must be a positive integer",
        ),
        (
            {"models": {"providers": {"x": {"enabled": "yes"}}}},
            "models.providers.x.enabled must be a boolean",
        ),
        (
            {"models": {"providers": {"x": {"enabled": True, "base_url": "ftp://x"}}}},
            "models.providers.x.base_url must use http or https",
        ),
        (
            {"models": {"providers": {"x": {"enabled": False, "model": 3}}}},
            "models.providers.x.model must be a non-empty string",
        ),
        (
            {"models": {"providers": {"x": {"retry": []}}}},
            "models.providers.x.retry must be a mapping",
        ),
        (
            {"models": {"providers": {"x": {"retry": {"extra": 1}}}}},
            "models.providers.x.retry has unknown retry settings",
        ),
        (
            {"models": {"providers": {"x": {"retry": {"max_attempts": True}}}}},
            "models.providers.x.retry.max_attempts must be an integer",
        ),
        (
            {"models": {"providers": {"x": {"retry": {"max_attempts": 6}}}}},
            "models.providers.x.retry.max_attempts must be between 1 and 5",
        ),
        (
            {"models": {"providers": {"x": {"retry": {"base_delay": True}}}}},
            "models.providers.x.retry.base_delay must be between 0 and 60 seconds",
        ),
        (
            {"models": {"providers": {"x": {"retry": {"max_delay": 61}}}}},
            "models.providers.x.retry.max_delay must be between 0 and 60 seconds",
        ),
        (
            {
                "models": {
                    "providers": {"x": {"retry": {"base_delay": 2, "max_delay": 1}}}
                }
            },
            "models.providers.x.retry.base_delay cannot exceed max_delay",
        ),
        (
            {
                "models": {
                    "registry": {
                        "x": {"provider": "p", "model": "m", "capabilities": [3]}
                    }
                }
            },
            "models.registry.x.capabilities must be a list of strings",
        ),
        (
            {"profiles": {"coding": {"model": {"fallback": "x"}}}},
            "fallback must be a list",
        ),
        (
            {"profiles": {"coding": {"tools": [3]}}},
            "profiles.coding.tools must be a list of strings",
        ),
        ({"logging": {"level": "VERBOSE"}}, "logging.level must be one of"),
        ({"secrets": []}, "secrets must be a mapping"),
        ([], "configuration root must be a mapping"),
    ],
)
def test_config_rejects_invalid_known_schema(tmp_path, payload, error):
    import yaml

    path = tmp_path / "bad-schema.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match=error):
        Config(path)


def test_config_resolves_defaults_without_mutating_or_exposing_secrets(
    tmp_path, monkeypatch
):
    import yaml

    from harness import core

    monkeypatch.setattr(core, "ROOT", tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "harness": {"name": "fixture"},
                "api": {"port": 9000},
                "secrets": {
                    "OPENAI_API_KEY": "yaml-secret",
                    "nested": {"token": "inner"},
                },
            }
        )
    )
    monkeypatch.setattr(
        core, "SecretResolver", lambda: type("R", (), {"get": lambda *_: None})()
    )
    config = Config(path)
    resolved = config.resolved()
    assert resolved["harness"]["name"] == "fixture"
    assert resolved["api"] == {"host": "127.0.0.1", "port": 9000, "swagger": True}
    assert resolved["paths"]["database"] == "./data/harness.db"
    redacted = config.redacted()
    assert redacted["secrets"]["OPENAI_API_KEY"] == "********"
    assert redacted["secrets"]["nested"]["token"] == "********"
    assert config.data["secrets"]["OPENAI_API_KEY"] == "yaml-secret"


def test_config_accepts_disabled_provider_without_endpoint(tmp_path):
    import yaml

    path = tmp_path / "disabled-provider.yaml"
    path.write_text(
        yaml.safe_dump({"models": {"providers": {"local": {"enabled": False}}}})
    )
    config = Config(path)
    assert config.data["models"]["providers"]["local"]["enabled"] is False


def test_config_empty_yaml_uses_all_defaults(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    config = Config(path)
    assert config.resolved()["api"]["port"] == 8080


def test_config_rejects_non_mapping_root(tmp_path):
    path = tmp_path / "invalid-root.yaml"
    path.write_text("- not-a-mapping")
    with pytest.raises(ValueError, match="configuration root must be a mapping"):
        Config(path)


def test_qdrant_vector_and_failure():
    from harness.memory import QdrantMemory

    memory = QdrantMemory("http://127.0.0.1:9", "memory", dimension=8)
    assert memory.health() is False and memory.collection_exists() is False


def test_openai_compatible_provider_mock():
    import httpx

    from harness.providers import OpenAICompatibleProvider

    def handler(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "m1"}]})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3},
            },
        )

    provider = OpenAICompatibleProvider(
        "fake", "https://provider.test/v1", "secret", "m1", httpx.MockTransport(handler)
    )
    assert provider.health() and provider.models() == ["m1"]
    response, usage = provider.complete("hi")
    assert response == "hello" and usage.prompt_tokens == 2


def test_tool_registry_events(tmp_path):
    from harness.tools import ToolRegistry

    events = []
    c = Config()
    c.data["tools"] = {"permissions": {"filesystem": "read"}}
    registry = ToolRegistry(
        Permissions(c), lambda kind, payload: events.append(kind), tmp_path
    )
    assert (
        registry.execute(
            "filesystem.search", {"root": str(tmp_path), "query": "does-not-exist"}
        )
        == []
    )
    assert events == ["TOOL_CALL_STARTED", "TOOL_CALL_COMPLETED"]
    with pytest.raises(PermissionError, match="escapes"):
        registry.search("../outside", "x")


def test_orchestrator_audits_tool_events(tmp_path):
    config = Config()
    config.data["paths"] = {
        "database": str(tmp_path / "db.sqlite"),
        "workspace": str(tmp_path / "workspace"),
        "obsidian_vault": str(tmp_path / "vault"),
    }
    config.data["tools"]["mcp"]["servers"]["vault"]["enabled"] = False
    store = Store(config)
    orchestrator = Orchestrator(store, config)
    orchestrator.tools.execute(
        "filesystem.write", {"path": "audit.txt", "content": "real call"}
    )
    assert store.events.list()[-1]["kind"] == "TOOL_CALL_COMPLETED"


def test_orchestrator_replans_interrupted_work(tmp_path):
    store, orchestrator, task = ready_runtime(tmp_path)
    task.status = "executing"
    created = store.create(task)
    assert asyncio.run(orchestrator.run(created.id)).status == "completed"
    assert store.events.list(created.id, "recovery.inspected")
    assert store.plans.latest(created.id) is not None


def test_cancelled_task_cannot_resume(tmp_path):
    store, orchestrator, task = ready_runtime(tmp_path)
    task.status = "cancelled"
    created = store.create(task)
    with pytest.raises(ValueError, match="terminal"):
        asyncio.run(orchestrator.run(created.id))


def test_orchestrator_does_not_reuse_unverified_checkpoint(tmp_path):
    from harness.agents import PlannerOutput, Subtask

    store, orchestrator, task = ready_runtime(tmp_path)
    task.status = "failed"
    created = store.create(task)
    previous = PlannerOutput(
        summary="obsolete",
        complexity="simple",
        subtasks=[Subtask(id="old", title="old", description="old")],
    )
    store.plans.save(created.id, previous.summary, previous.model_dump(mode="json"))
    store.subtasks.save_plan(created.id, previous.subtasks)
    store.subtasks.update(created.id, "old", "completed", "{}")
    assert asyncio.run(orchestrator.run(created.id)).status == "completed"
    assert store.plans.latest(created.id)["summary"] != "obsolete"
    assert len(store.subtasks.list(created.id)) == 2


def test_recovery_does_not_duplicate_an_open_human_question(tmp_path):
    store, orchestrator, task = ready_runtime(tmp_path)
    created = store.create(task)
    store.ask(created.id, "Review?", "recovery")
    store.tasks.update(created.id, status="failed")
    with pytest.raises(ValueError, match="question"):
        asyncio.run(orchestrator.run(created.id))
    assert len(store.questions.list(created.id)) == 1


def test_failed_execution_exhausts_correction_and_persists_failure(tmp_path):
    from harness.agents import ExecutorOutput

    store, orchestrator, task = ready_runtime(tmp_path)
    orchestrator.config.data["harness"] = {"max_correction_attempts": 0}
    orchestrator.executor.execute = lambda step, context: ExecutorOutput(
        subtask_id=step.id, success=False, output="failed"
    )
    created = store.create(task)
    with pytest.raises(RuntimeError, match="subtask"):
        asyncio.run(orchestrator.run(created.id))
    assert store.get(created.id).status == "failed"
    assert store.get(created.id).validation_result["valid"] is False


def test_orchestrator_qdrant_optional(tmp_path):
    store, orchestrator, task = ready_runtime(tmp_path)
    orchestrator.qdrant_enabled = True
    seen = []
    orchestrator.qdrant.ensure_collection = lambda: True
    orchestrator.qdrant.upsert_many = lambda *args: seen.append(args)
    orchestrator.qdrant.scroll_source = lambda *args: []
    assert asyncio.run(orchestrator.run(store.create(task).id)).status == "completed"
    assert seen


def test_config_environment_keychain_and_paths(tmp_path, monkeypatch):
    from harness import core

    core.ROOT = tmp_path
    (tmp_path / ".env").write_text("LOCAL_API_KEY=dotenv\n# skipped\n\n")
    monkeypatch.setenv("LOCAL_API_KEY", "environment")
    monkeypatch.setattr(
        core,
        "SecretResolver",
        lambda: type(
            "R",
            (),
            {"get": lambda self, key: "keychain" if key == "OPENAI_API_KEY" else None},
        )(),
    )
    config = Config(tmp_path / "missing.yaml")
    assert (
        config.data["secrets"]["LOCAL_API_KEY"] == "environment"
        and config.data["secrets"]["OPENAI_API_KEY"] == "keychain"
    )
    assert config.path("database").is_absolute()


def test_configuration_service_layers_general_settings_and_secret_sources(
    tmp_path, monkeypatch
):
    import yaml

    from harness import core

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "api": {"port": 7000},
                "memory": {"qdrant": {"url": "http://yaml.test:6333"}},
                "secrets": {"OPENAI_API_KEY": "yaml-secret"},
            }
        )
    )
    (tmp_path / ".env").write_text(
        "API_PORT=7100\nQDRANT_URL=http://dotenv.test:6333\n"
        "OPENAI_API_KEY=dotenv-secret\n"
    )
    monkeypatch.setattr(core, "ROOT", tmp_path)
    monkeypatch.setenv("API_PORT", "7200")
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")
    monkeypatch.setattr(
        core,
        "SecretResolver",
        lambda: type(
            "Resolver",
            (),
            {
                "get": lambda _self, key: (
                    "keychain-secret" if key == "OPENAI_API_KEY" else None
                )
            },
        )(),
    )

    config = ConfigurationService(config_path)

    assert config.resolved()["api"]["port"] == 7200
    assert config.resolved()["memory"]["qdrant"]["url"] == "http://dotenv.test:6333"
    assert config.resolved()["secrets"]["OPENAI_API_KEY"] == "keychain-secret"
    assert config.redacted(resolved=True)["secrets"]["OPENAI_API_KEY"] == "********"


@pytest.mark.parametrize(
    ("dotenv", "expected"),
    [
        ('export API_HOST="127.0.0.1" # local', "127.0.0.1"),
        ("API_HOST=localhost # local", "localhost"),
        ("API_HOST='127.0.0.1'", "127.0.0.1"),
    ],
)
def test_configuration_service_parses_dotenv_assignments(
    tmp_path, monkeypatch, dotenv, expected
):
    from harness import core

    monkeypatch.setattr(core, "ROOT", tmp_path)
    (tmp_path / ".env").write_text(dotenv + "\n")

    assert (
        ConfigurationService(tmp_path / "absent.yaml").resolved()["api"]["host"]
        == expected
    )


def test_configuration_service_maps_documented_dotenv_names(tmp_path, monkeypatch):
    from harness import core

    monkeypatch.setattr(core, "ROOT", tmp_path)
    (tmp_path / ".env").write_text(
        "HARNESS_WORKSPACE=./work\nOBSIDIAN_VAULT_PATH=./notes\n"
        "QDRANT_URL=http://qdrant.test:6333\nQDRANT_COLLECTION=dev-memory\n"
        "QDRANT__SERVICE__API_KEY=qdrant-fixture-secret\n"
        "LLM_PROVIDER=openai\nLLM_MODEL=fixture-v1\n"
        "LLM_API_URL=https://api.example.test/v1\nGIT_REMOTE_URL=https://git.test/repo\n"
        "API_HOST=localhost\nAPI_PORT=8090\nOPENAI_MODEL=another-model\n"
    )
    monkeypatch.setattr(
        core, "SecretResolver", lambda: SimpleNamespace(get=lambda _key: None)
    )

    resolved = ConfigurationService(tmp_path / "missing.yaml").resolved()

    assert resolved["paths"]["workspace"] == "./work"
    assert resolved["paths"]["obsidian_vault"] == "./notes"
    assert resolved["memory"]["qdrant"] == {
        "enabled": False,
        "url": "http://qdrant.test:6333",
        "collection": "dev-memory",
    }
    assert resolved["secrets"]["QDRANT__SERVICE__API_KEY"] == ("qdrant-fixture-secret")
    assert (
        ConfigurationService(tmp_path / "missing.yaml").redacted(resolved=True)[
            "secrets"
        ]["QDRANT__SERVICE__API_KEY"]
        == "********"
    )
    assert resolved["models"]["defaults"] == {
        "provider": "openai",
        "model": "fixture-v1",
    }
    assert "llm" not in resolved["models"]["providers"]
    assert resolved["models"]["providers"]["openai"]["base_url"] == (
        "https://api.example.test/v1"
    )
    assert resolved["models"]["providers"]["openai"]["model"] == "another-model"
    assert resolved["git"]["remote"] == "https://git.test/repo"
    assert resolved["api"] == {"host": "localhost", "port": 8090, "swagger": True}


def test_env_example_is_loadable(tmp_path, monkeypatch):
    from pathlib import Path

    from harness import core

    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(core, "ROOT", tmp_path)
    (tmp_path / ".env").write_text((project_root / ".env.example").read_text())
    monkeypatch.setattr(
        core, "SecretResolver", lambda: SimpleNamespace(get=lambda _key: None)
    )

    config = ConfigurationService(tmp_path / "absent.yaml")

    assert config.data["paths"]["workspace"] == "./workspace"
    assert config.data["api"]["port"] == 8080


def test_llm_api_url_environment_override_uses_yaml_selected_provider(
    tmp_path, monkeypatch
):
    import yaml

    from harness import core

    monkeypatch.setattr(core, "ROOT", tmp_path)
    (tmp_path / ".env").write_text("LLM_API_URL=https://override.test/v1\n")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"models": {"defaults": {"provider": "openai"}}})
    )
    monkeypatch.setattr(
        core, "SecretResolver", lambda: SimpleNamespace(get=lambda _key: None)
    )

    providers = ConfigurationService(config_path).resolved()["models"]["providers"]

    assert providers["openai"]["base_url"] == "https://override.test/v1"
    assert "lmstudio" not in providers


def test_configuration_service_rejects_non_numeric_environment_port(
    tmp_path, monkeypatch
):
    from harness import core

    monkeypatch.setattr(core, "ROOT", tmp_path)
    monkeypatch.setenv("API_PORT", "not-a-port")
    with pytest.raises(ValueError, match="api.port must be an integer"):
        ConfigurationService(tmp_path / "absent.yaml")


def test_configuration_service_rejects_malformed_dotenv_without_echoing_values(
    tmp_path, monkeypatch
):
    from harness import core

    monkeypatch.setattr(core, "ROOT", tmp_path)
    (tmp_path / ".env").write_text("BROKEN_LINE=secret\nINVALID KEY=do-not-leak\n")

    with pytest.raises(
        ValueError, match=r"\.env line 2 has an invalid variable name"
    ) as error:
        ConfigurationService(tmp_path / "absent.yaml")

    assert "do-not-leak" not in str(error.value)


@pytest.mark.parametrize(
    ("dotenv", "error"),
    [
        ("BROKEN", r"\.env line 1 must be KEY=VALUE"),
        ('API_HOST="unterminated', r"\.env contains an unterminated quoted value"),
        (
            'API_HOST="localhost" trailing',
            r"\.env contains invalid text after a quoted value",
        ),
    ],
)
def test_configuration_service_rejects_malformed_dotenv(
    tmp_path, monkeypatch, dotenv, error
):
    from harness import core

    monkeypatch.setattr(core, "ROOT", tmp_path)
    (tmp_path / ".env").write_text(dotenv + "\n")

    with pytest.raises(ValueError, match=error):
        ConfigurationService(tmp_path / "absent.yaml")


def test_configuration_service_yaml_errors_do_not_echo_secret_lines(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("api: [\n  api_key: do-not-leak\n")

    with pytest.raises(ValueError, match="invalid YAML in broken.yaml") as error:
        ConfigurationService(path)

    assert "do-not-leak" not in str(error.value)


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({"docker": {"enabled": "yes"}}, "docker.enabled must be a boolean"),
        ({"testing": {"tdd": 1}}, "testing.tdd must be a boolean"),
        (
            {"testing": {"coverage": {"branches": 101}}},
            "testing.coverage.branches must be between 0 and 100",
        ),
        ({"git": {"workflow": []}}, "git.workflow must be a mapping"),
        (
            {"tools": {"http": {"allowed_hosts": "example.com"}}},
            "tools.http.allowed_hosts must be a list of strings",
        ),
        ({"api": {"enabled": 1}}, "api.enabled must be a boolean"),
        (
            {"docker": {"compose_preferred": "yes"}},
            "docker.compose_preferred must be a boolean",
        ),
        (
            {"profiles": {"coding": {"max_steps": 0}}},
            "profiles.coding.max_steps must be a positive integer",
        ),
        (
            {"tools": {"git": {"ssh": {"allowed_ports": [70000]}}}},
            "tools.git.ssh.allowed_ports must contain valid ports",
        ),
        ({"secrets": {"TOKEN": 3}}, "secrets.TOKEN must be a non-empty string"),
        ({"models": {"strategies": []}}, "models.strategies must be a mapping"),
        ({"models": []}, "models must be a mapping"),
        ({"models": {"defaults": []}}, "models.defaults must be a mapping"),
        ({"api": {"port": "invalid"}}, "api.port must be an integer"),
        (
            {"git": {"branches": {"main": 3}}},
            "git.branches.main must be a non-empty string",
        ),
        (
            {"docker": {"compose_file": 3}},
            "docker.compose_file must be a non-empty string",
        ),
        (
            {"models": {"strategies": {"fast": {"model": 3}}}},
            "models.strategies.fast.model must be a non-empty string",
        ),
        (
            {"tools": {"git": {"allowed_hosts": [3]}}},
            "tools.git.allowed_hosts must be a list of strings",
        ),
        (
            {"tools": {"git": {"ssh": {"allowed_hosts": [3]}}}},
            "tools.git.ssh.allowed_hosts must be a list of strings",
        ),
    ],
)
def test_config_validates_remaining_typed_sections(tmp_path, payload, error):
    import yaml

    path = tmp_path / "typed.yaml"
    path.write_text(yaml.safe_dump(payload))

    with pytest.raises(ValueError, match=error):
        Config(path)


def test_config_accepts_full_example_sections(tmp_path):
    import yaml

    path = tmp_path / "full-sections.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "api": {"enabled": True},
                "docker": {
                    "enabled": True,
                    "compose_preferred": True,
                    "compose_file": "./compose.yaml",
                },
                "git": {
                    "branches": {"main": "main", "development": "dev"},
                    "workflow": {"feature": "feature/"},
                },
                "models": {
                    "strategies": {
                        "fast": {"provider": "local", "model": "m", "profile": "coding"}
                    }
                },
                "profiles": {
                    "coding": {"instructions": "Implement safely", "max_steps": 3}
                },
                "testing": {"tdd": True, "coverage": {"branches": 100}},
                "tools": {
                    "git": {
                        "allowed_hosts": ["git.example.test"],
                        "ssh": {
                            "allowed_hosts": ["ssh.example.test"],
                            "allowed_ports": [22],
                        },
                    }
                },
            }
        )
    )
    config = Config(path)

    assert config.data["testing"]["coverage"]["branches"] == 100
    assert config.data["models"]["strategies"]["fast"]["model"] == "m"


def test_core_missing_config_branches_and_agent_usage(tmp_path):
    from harness.providers import ModelUsage

    c = Config()
    c.data["paths"]["database"] = str(tmp_path / "branches.db")
    c.data["tools"]["mcp"]["servers"]["vault"]["enabled"] = False
    store = Store(c)
    item = store.create(Task(title="optional question"))
    assert store.get(999) is None and store.ask(
        item.id, "optional", "reason", required=False
    )
    assert store.get(item.id).status.value == "pending"
    orch = Orchestrator(store)
    orch._record_model_usage(ModelUsage("local", "model", 2, 3, 0.01))
    try:
        asyncio.run(orch.run(999))
    except ValueError as exc:
        assert "not found" in str(exc)
    assert store.model_runs.record(ModelUsage("local", "model"))


def test_orchestrator_agent_and_qdrant_failures(tmp_path):
    store, orchestrator, task = ready_runtime(tmp_path)
    orchestrator.qdrant_enabled = True
    orchestrator.qdrant.ensure_collection = lambda: (_ for _ in ()).throw(
        OSError("down")
    )
    created = store.create(task)
    assert asyncio.run(orchestrator.run(created.id)).status == "completed"
    assert store.events.list(created.id, "memory.index_failed")
    other = store.create(task.model_copy(update={"id": None}))
    orchestrator.planner.plan = lambda *args: (_ for _ in ()).throw(
        ValueError("planner")
    )
    with pytest.raises(ValueError, match="planner"):
        asyncio.run(orchestrator.run(other.id))
    assert store.get(other.id).status == "failed"
