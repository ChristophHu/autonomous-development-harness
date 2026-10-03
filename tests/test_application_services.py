from datetime import UTC, datetime

import pytest
from test_evidence_workflow import ready_runtime

from harness.providers import ProviderHealth


def test_verification_evidence_service_audits_each_required_kind_independently(
    tmp_path,
):
    store, _orchestrator, _task = ready_runtime(tmp_path)
    from harness.services import VerificationEvidenceService

    service = VerificationEvidenceService(store.database)
    observed_at = datetime.now(UTC)
    service.record(
        {
            "kind": "ci",
            "source_id": "ci:service-boundary",
            "observed_at": observed_at,
            "subject_sha256": "a" * 64,
            "passed": True,
            "checks": {"tests": True},
        }
    )
    service.record(
        {
            "kind": "provider",
            "source_id": "provider:service-boundary",
            "observed_at": observed_at,
            "subject_sha256": "b" * 64,
            "passed": True,
            "checks": {"health": True},
        }
    )

    assert service.audit_kind("ci", subject_sha256="a" * 64)["healthy"] is True
    provider_audit = service.audit_kind("provider", subject_sha256="a" * 64)
    assert provider_audit["healthy"] is False
    assert provider_audit["items"][0]["reason"] == "subject_mismatch"


def test_task_service_shares_status_queries_and_sanitized_task_inspection(tmp_path):
    store, orchestrator, task = ready_runtime(tmp_path)
    created = store.create(task)
    service = orchestrator.service

    counts = service.status_counts()
    assert counts["pending"] == 1
    assert counts["completed"] == 0
    store.event(created.id, "task.created", {"api_key": "must-not-leak"})
    events = service.events(created.id)
    assert "[REDACTED]" in events[0]["payload"]
    assert service.event_feed(created.id)[0]["task_id"] == created.id
    assert service.events_after(0, created.id)[0]["kind"] == "task.created"
    assert service.questions(created.id) == []
    assert service.plan(created.id) is None
    assert service.validation(created.id) is None
    assert service.model_usage()["total"] == 0


def test_task_service_routes_question_creation_and_unscoped_event_cursor(tmp_path):
    store, orchestrator, task = ready_runtime(tmp_path)
    created = store.create(task)
    question_id = orchestrator.service.ask_question(
        created.id, "Review?", "Need owner decision", ["yes", "no"]
    )
    assert store.questions.get(question_id)["task_id"] == created.id
    store.event(created.id, "task.created", {})
    assert orchestrator.service.events_after(0) == store.events.after(0)


def test_model_operations_service_shares_health_and_inventory_contracts(tmp_path):
    from types import SimpleNamespace

    _store, orchestrator, _task = ready_runtime(tmp_path)
    registry = orchestrator.models
    registry.providers = {}
    registry.models = {
        "configured": {"provider": "local", "model": "loaded", "tier": "local"},
        "missing": {"provider": "offline", "model": "not-loaded"},
    }
    registry.register(
        "local",
        SimpleNamespace(
            health_report=lambda: ProviderHealth(
                "openai_compatible", True, True, ("loaded", "discovered")
            )
        ),
    )
    service = orchestrator.model_operations
    inventory = service.model_inventory()
    assert service.provider_health() == {"local": "available"}
    assert inventory == [
        {
            "provider": "local",
            "status": "available",
            "discovery_failed": False,
            "models": [
                {
                    "id": "loaded",
                    "alias": "configured",
                    "tier": "local",
                    "status": "available",
                },
                {
                    "id": "discovered",
                    "alias": "-",
                    "tier": "unknown",
                    "status": "available",
                },
            ],
        },
        {
            "provider": "offline",
            "status": "unavailable",
            "discovery_failed": False,
            "models": [
                {
                    "id": "not-loaded",
                    "alias": "missing",
                    "tier": "unknown",
                    "status": "unavailable",
                }
            ],
        },
    ]


def test_model_operations_service_handles_failed_discovery_without_details(tmp_path):
    from types import SimpleNamespace

    _store, orchestrator, _task = ready_runtime(tmp_path)

    def fail():
        raise OSError("secret provider detail")

    orchestrator.models.register(
        "offline", SimpleNamespace(health=lambda: True, models=fail)
    )
    orchestrator.models.providers = {
        "offline": orchestrator.models.providers["offline"]
    }
    orchestrator.models.models = {}
    inventory = orchestrator.model_operations.model_inventory()
    assert inventory == [
        {
            "provider": "offline",
            "status": "available",
            "discovery_failed": True,
            "models": [],
        }
    ]


def test_model_discovery_uses_last_good_inventory_marked_stale(tmp_path):
    from types import SimpleNamespace

    from harness.database import OperationalSnapshotRepository
    from harness.services import ModelOperationsService

    store, _orchestrator, _task = ready_runtime(tmp_path)
    snapshots = OperationalSnapshotRepository(store.database)
    provider = SimpleNamespace(
        health_report=lambda: ProviderHealth(
            "openai_compatible", True, True, ("cached-model",)
        )
    )
    registry = SimpleNamespace(providers={"local": provider}, models={}, discovered={})
    service = ModelOperationsService(registry, snapshots)
    assert service.model_inventory()[0]["models"][0]["status"] == "available"
    provider.health_report = lambda: ProviderHealth(
        "openai_compatible", True, False, ()
    )
    stale = service.model_inventory()[0]
    assert stale["discovery_failed"] is True
    assert stale["models"] == [
        {"id": "cached-model", "alias": "-", "tier": "unknown", "status": "stale"}
    ]
    assert snapshots.latest_model_discovery("local")["models"] == ["cached-model"]


def test_model_inventory_can_run_without_persistence(tmp_path):
    from types import SimpleNamespace

    from harness.services import ModelOperationsService

    provider = SimpleNamespace(
        health_report=lambda: ProviderHealth(
            "openai_compatible", True, True, ("model-a",)
        )
    )
    registry = SimpleNamespace(providers={"local": provider}, models={}, discovered={})
    result = ModelOperationsService(registry).model_inventory()
    assert result[0]["models"][0]["id"] == "model-a"


def test_model_operations_test_returns_only_safe_success_or_failure():
    from types import SimpleNamespace

    from harness.services import ModelOperationsService

    registry = SimpleNamespace(
        resolve=lambda name: (SimpleNamespace(complete=lambda *_a, **_k: "OK"), "id")
    )
    service = ModelOperationsService(registry)
    assert service.test_model("local") == {"model": "local", "status": "successful"}

    registry.resolve = lambda _name: (_ for _ in ()).throw(
        RuntimeError("secret provider response")
    )
    assert service.test_model("local") == {
        "model": "local",
        "status": "failed",
        "error_type": "RuntimeError",
    }


def test_task_knowledge_search_is_task_scoped_redacted_and_read_only(tmp_path):
    store, orchestrator, task = ready_runtime(tmp_path)
    created = store.create(task)
    store.audit.secrets["fixture"] = "PRIVATE-CANARY"
    vault = orchestrator.context.memory.vault
    vault.mkdir(parents=True)
    (vault / "Guide.md").write_text(
        "---\ntype: architecture\nreviewed_on: 2026-10-03\n---\n"
        "# Architecture\nThe fixture stores PRIVATE-CANARY safely.\n",
        encoding="utf-8",
    )

    result = orchestrator.service.knowledge_search(created.id, "fixture")
    assert result["task_id"] == created.id
    assert result["hits"][0]["source_ref"] == "context:vault/Guide.md#Architecture"
    assert "PRIVATE-CANARY" not in result["hits"][0]["excerpt"]
    assert result["hits"][0]["provenance"]["review_state"] == "current"
    assert (vault / "Guide.md").read_text(encoding="utf-8").find("PRIVATE-CANARY") >= 0

    with pytest.raises(ValueError, match="task not found"):
        orchestrator.service.knowledge_search(999999, "fixture")
    with pytest.raises(ValueError, match="limit"):
        orchestrator.service.knowledge_search(created.id, "fixture", limit=21)


@pytest.mark.parametrize(
    "method,args",
    [
        ("events", ()),
        ("questions", ()),
        ("plan", ()),
        ("validation", ()),
        ("ask_question", ("Q", "R")),
    ],
)
def test_task_service_inspection_and_question_methods_require_existing_task(
    tmp_path, method, args
):
    _store, orchestrator, _task = ready_runtime(tmp_path)
    with pytest.raises(ValueError, match="task not found"):
        getattr(orchestrator.service, method)(987654, *args)
