import asyncio
import hashlib
import json
import subprocess
from types import SimpleNamespace

import httpx
import pytest
from jsonschema import ValidationError
from test_evidence_workflow import ready_runtime, runtime, specification

from harness.agents import (
    Executor,
    PlannerOutput,
    Subtask,
)
from harness.claim_evidence import IndependentClaimVerifier
from harness.core import Event, TaskLifecycle
from harness.domain import AcceptanceCriterion, Task, may_transition
from harness.providers import (
    ModelUsage,
    OpenAICompatibleProvider,
    ProviderError,
    ToolCall,
)
from harness.requirements import RequirementCompleter
from harness.tools import ToolSpec


@pytest.mark.parametrize("value", ["", " "])
def test_blank_task_title(value):
    with pytest.raises(ValueError):
        Task(title=value)


@pytest.mark.parametrize(
    "payload",
    [
        {"id": "x", "description": "x", "kind": "file_exists"},
        {"id": "x", "description": "x", "kind": "file_contains", "path": "x"},
        {"id": "x", "description": "x", "kind": "command"},
    ],
)
def test_acceptance_contracts(payload):
    with pytest.raises(ValueError):
        AcceptanceCriterion(**payload)


def test_plan_must_map_each_requirement_and_criterion_to_steps():
    from harness.agents import PlannerOutput, Subtask

    task = Task(
        title="implement",
        requirements=["persist data"],
        acceptance_criteria=[
            AcceptanceCriterion(id="persisted", description="data persists")
        ],
    )
    step = Subtask(
        id="step-1",
        title="persist",
        description="store data",
        expected_result="data stored",
        acceptance_criteria=["persisted"],
        requirement_ids=["persist data"],
    )
    plan = PlannerOutput(summary="plan", complexity="low", subtasks=[step])
    assert plan.validate_task_coverage(task) == []
    step.requirement_ids = []
    assert "requirement has no plan step: persist data" in plan.validate_task_coverage(
        task
    )
    step.requirement_ids = ["persist data"]
    step.acceptance_criteria = []
    assert (
        "acceptance criterion has no plan step: persisted"
        in plan.validate_task_coverage(task)
    )
    assert (
        "requirement has no linked acceptance criterion: persist data"
        in plan.validate_task_coverage(task)
    )
    step.requirement_ids = ["unknown requirement"]
    step.acceptance_criteria = ["unknown criterion"]
    assert plan.validate_task_coverage(task) == [
        "requirement has no plan step: persist data",
        "acceptance criterion has no plan step: persisted",
        "plan references unknown requirement: unknown requirement",
        "plan references unknown acceptance criterion: unknown criterion",
    ]


def test_recovery_requirement_remains_linked_to_task_criterion(tmp_path):
    _store, orchestrator, task = ready_runtime(tmp_path)
    task.plan = {
        "subtasks": [
            {
                "id": "recover",
                "requirement_ids": ["add two integers"],
                "acceptance_criteria": ["sum"],
                "recovery_targets": ["requirement:add two integers", "verify"],
            }
        ]
    }

    errors, _planned = orchestrator.validator._plan_findings(task, [])

    assert (
        "requirement has no linked acceptance criterion: add two integers" not in errors
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_cost_budget", -0.01),
        ("model_cost_budget", float("inf")),
        ("model_token_budget", -1),
        ("model_token_budget", True),
    ],
)
def test_task_rejects_invalid_model_budgets(field, value):
    with pytest.raises(ValueError):
        Task(title="budget", **{field: value})


def test_validator_rejects_plan_with_unmapped_task_requirement(tmp_path):
    _store, orchestrator = runtime(tmp_path)
    task = specification()
    task.plan = {"subtasks": [{"id": "planned", "acceptance_criteria": ["sum"]}]}
    errors, _ = orchestrator.validator._plan_findings(task, [])
    assert "requirement has no plan step: add two integers" in errors


@pytest.mark.parametrize(
    "mapping,error",
    [
        ({"requirement_ids": "not-a-list"}, "plan requirement mapping is invalid"),
        ({"acceptance_criteria": [1]}, "plan acceptance mapping is invalid"),
        ({"recovery_targets": "bad"}, "plan recovery mapping is invalid"),
        ({"requirement_ids": ["unknown"]}, "plan references unknown requirement"),
        (
            {"acceptance_criteria": ["unknown"]},
            "plan references unknown acceptance criterion",
        ),
    ],
)
def test_validator_rejects_malformed_or_unknown_plan_mappings(tmp_path, mapping, error):
    _store, orchestrator = runtime(tmp_path)
    task = specification()
    task.plan = {"subtasks": [{"id": "planned", **mapping}]}
    errors, _ = orchestrator.validator._plan_findings(task, [])
    assert any(error in item for item in errors)


def test_validator_skips_nonmapping_plan_step_in_coverage_scan(tmp_path):
    _store, orchestrator = runtime(tmp_path)
    task = specification()
    task.plan = {"subtasks": [None]}
    errors, _ = orchestrator.validator._plan_findings(task, [])
    assert "plan contains an invalid step" in errors


def test_validator_treats_null_recovery_targets_as_absent(tmp_path):
    _store, orchestrator = runtime(tmp_path)
    task = specification()
    task.plan = {
        "subtasks": [
            {
                "id": "planned",
                "requirement_ids": ["add two integers"],
                "acceptance_criteria": ["sum"],
                "recovery_targets": None,
            }
        ]
    }
    errors, _ = orchestrator.validator._plan_findings(task, [])
    assert "plan recovery mapping is invalid: planned" not in errors


def test_duplicate_ids_and_terminal_transitions():
    criterion = AcceptanceCriterion(id="x", description="x")
    with pytest.raises(ValueError):
        Task(title="x", acceptance_criteria=[criterion, criterion])
    assert not may_transition("completed", "failed")
    assert not TaskLifecycle.can_start("failed", True)
    assert TaskLifecycle.can_start("waiting_human")
    assert TaskLifecycle.can_start("waiting_decision")
    assert TaskLifecycle.can_start("waiting_approval")
    assert not TaskLifecycle.can_start("executing")
    assert may_transition("executing", "waiting_decision")
    assert may_transition("executing", "waiting_approval")


@pytest.mark.parametrize(
    "source",
    [
        "analyzing",
        "planning",
        "ready",
        "executing",
        "testing",
        "validating",
        "correcting",
        "waiting_human",
        "waiting_decision",
        "waiting_approval",
        "failed",
        "blocked",
    ],
)
def test_nonterminal_interrupted_task_can_enter_recovering(source):
    assert may_transition(source, "recovering")


@pytest.mark.parametrize("source", ["pending", "completed", "cancelled"])
def test_recovery_entry_rejects_unstarted_or_terminal_task(source):
    assert not may_transition(source, "recovering")


def test_recovering_task_resumes_only_through_analysis():
    assert may_transition("recovering", "analyzing")
    assert not may_transition("recovering", "executing")
    assert TaskLifecycle.can_start("recovering")


def test_plan_cycles_duplicates_and_order():
    first = Subtask(id="first", title="x", description="x")
    second = Subtask(id="second", title="x", description="x", dependencies=["first"])
    assert [
        s.id
        for s in PlannerOutput(
            summary="x", complexity="simple", subtasks=[second, first]
        ).ordered_steps()
    ] == ["first", "second"]
    with pytest.raises(ValueError, match="duplicate"):
        PlannerOutput(summary="x", complexity="simple", subtasks=[first, first])
    first.dependencies = ["second"]
    with pytest.raises(ValueError, match="cyclic"):
        PlannerOutput(summary="x", complexity="simple", subtasks=[first, second])


def test_model_registry_capability_and_invalid_responses(tmp_path):
    _, orchestrator = runtime(tmp_path)
    registry = orchestrator.models
    with pytest.raises(ValueError, match="profile"):
        orchestrator.router.complete("unknown", "x")
    registry.models.update(
        {
            "alias": {
                "provider": "fixture",
                "model": "real",
                "capabilities": ["tools"],
            },
            "missing": {"provider": "fixture"},
            "text": {"provider": "fixture", "model": "real"},
        }
    )
    assert registry.resolve("alias", True)[1] == "real"
    with pytest.raises(ValueError, match="tools"):
        registry.resolve("text", True)
    with pytest.raises(ValueError, match="model ID"):
        registry.resolve("missing")
    registry.register("fixture", SimpleNamespace(complete=lambda *a, **kw: "ok"))
    orchestrator.config.data["profiles"]["coding"]["model"]["primary"] = "alias"
    assert orchestrator.router.complete("coding", "x") == "ok"
    for response in [
        ("ok", "not usage"),
        "",
        ("", ModelUsage("fixture", "model"), [ToolCall("1", "read", {})]),
    ]:
        registry.register(
            "fixture", SimpleNamespace(complete=lambda *a, _r=response, **kw: _r)
        )
        with pytest.raises(RuntimeError):
            orchestrator.router.complete("coding", "x")
    registry.register(
        "fixture",
        SimpleNamespace(
            complete=lambda *a, **kw: ("ok", ModelUsage("fixture", "m"), [])
        ),
    )
    assert orchestrator.router.complete("coding", "x", tools=[]).text == "ok"


def test_planner_rejects_incomplete_and_ungranted_steps(tmp_path):
    _, orchestrator = runtime(tmp_path)
    step = {"id": "x", "title": "x", "description": "x"}
    for extra in (
        {},
        {
            "expected_result": "x",
            "acceptance_criteria": ["x"],
            "required_tools": ["unknown"],
        },
    ):
        payload = {"summary": "x", "complexity": "simple", "subtasks": [step | extra]}
        orchestrator.models.register(
            "fixture",
            SimpleNamespace(complete=lambda *a, _p=payload, **kw: json.dumps(_p)),
        )
        with pytest.raises(ValueError):
            orchestrator.planner.plan(Task(title="x"))


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"choices": [{"message": {"content": ""}}]},
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"id": "", "function": {"name": "x", "arguments": "[]"}}
                        ]
                    }
                }
            ]
        },
    ],
)
def test_provider_malformed_response(response):
    provider = OpenAICompatibleProvider(
        "p",
        "http://fixture",
        model="m",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=response)),
    )
    with pytest.raises(ProviderError, match="invalid"):
        provider.complete("x")


def test_missing_model_is_rejected():
    with pytest.raises(ProviderError, match="model"):
        OpenAICompatibleProvider("p", "http://fixture").complete("x")


def test_repository_lease_and_transition_failures(tmp_path):
    store, orchestrator = runtime(tmp_path)
    task = store.create(Task(title="x"))
    with pytest.raises(ValueError):
        store.tasks.update(task.id, unexpected=True)
    assert store.tasks.claim(task.id, "owner")
    assert store.tasks.renew(task.id, "owner")
    assert not store.tasks.renew(task.id, "other")
    for task_id, target, owner in [
        (999, "analyzing", None),
        (task.id, "analyzing", "other"),
        (task.id, "completed", "owner"),
    ]:
        with pytest.raises(ValueError):
            store.tasks.transition(task_id, target, owner)
    store.tasks.transition(task.id, "analyzing", "owner")
    store.ask(task.id, "question", "reason")
    store.tasks.update(task.id, status="ready")
    with pytest.raises(ValueError, match="question"):
        store.tasks.transition(task.id, "executing", "owner")
    assert store.plans.latest(task.id) is None
    asyncio.run(orchestrator.emit(Event(task_id=task.id, kind="task.completed")))
    assert orchestrator.queue.qsize() == 1


def test_service_rejects_invalid_inputs_and_dependencies(tmp_path):
    _store, orchestrator = runtime(tmp_path)
    task = orchestrator.service.create(Task(title="parent"))
    with pytest.raises(ValueError):
        orchestrator.service.create(Task(title="invalid", id=3))
    child = orchestrator.service.create(
        Task(title="child", parent_task_id=task.id, dependencies=[task.id])
    )
    with pytest.raises(ValueError, match="dependencies"):
        asyncio.run(orchestrator.run(child.id))
    for payload in (
        {},
        {"status": "completed"},
        {"dependencies": [child.id]},
        {"dependencies": [999]},
    ):
        with pytest.raises(ValueError):
            orchestrator.service.patch(child.id, payload)
    assert orchestrator.service.patch(
        child.id, {"dependencies": [task.id]}
    ).dependencies == [task.id]
    with pytest.raises(ValueError, match="cycle"):
        orchestrator.service.patch(task.id, {"dependencies": [child.id]})
    assert (
        orchestrator.service.patch(child.id, {"parent_task_id": task.id}).parent_task_id
        == task.id
    )
    with pytest.raises(ValueError, match="blank"):
        asyncio.run(orchestrator.service.answer(task.id, 1, ""))


def test_requirement_sources_answers_and_derivation(tmp_path):
    store, orchestrator = runtime(tmp_path)
    parent = store.create(Task(title="parent"))
    task = store.create(Task(title="child", parent_task_id=parent.id))
    question = store.ask(task.id, "details", "requirements:incomplete")
    store.answer(question, specification().model_dump_json(), task.id)
    completed, context = RequirementCompleter(store, orchestrator.router).complete(
        task, "repo"
    )
    assert completed.goal and "parent" in context
    human_decision = store.decisions.list(task.id)[-1]
    assert human_decision["source"] == "human"
    assert human_decision["field_names"]
    assert human_decision["evidence"][0]["ref"] == f"answer:{question}"
    task = store.create(Task(title="invalid answer"))
    question = store.ask(task.id, "details", "requirements:incomplete")
    store.answer(question, "not JSON", task.id)

    def derive(prompt, **_kwargs):
        payload = json.loads(prompt.split("\n", 1)[1])
        assert (
            sum(
                item["source"] == "memory_and_repository" for item in payload["sources"]
            )
            == 1
        )
        assert not any(item["source"] == "context" for item in payload["sources"])
        context_ref = next(
            item["ref"]
            for item in payload["sources"]
            if item["source"] == "memory_and_repository"
        )
        return json.dumps(
            {
                "fields": {"goal": "derived"},
                "rationale": "source",
                "evidence": {"goal": [{"source": "context", "ref": context_ref}]},
            }
        )

    orchestrator.models.register("fixture", SimpleNamespace(complete=derive))
    completed, _ = RequirementCompleter(store, orchestrator.router).complete(
        task, "source"
    )
    assert completed.goal == "derived"
    agent_decision = store.decisions.list(task.id)[-1]
    assert agent_decision["source"] == "agent"
    assert agent_decision["field_names"] == ["goal"]
    assert agent_decision["evidence"] == [
        {"source": "context", "ref": agent_decision["evidence"][0]["ref"]}
    ]
    # A second pass receives the previous decision in its provenance catalog.
    RequirementCompleter(store, orchestrator.router).complete(task, "source")
    orphan = store.create(Task(title="orphan", parent_task_id=999))
    assert (
        RequirementCompleter(store, orchestrator.router).complete(orphan, "")[0].goal
        == "derived"
    )


def test_requirement_completion_rejects_uncited_agent_fields(tmp_path):
    store, orchestrator = runtime(tmp_path)
    task = store.create(Task(title="uncited"))
    orchestrator.models.register(
        "fixture",
        SimpleNamespace(
            complete=lambda *_args, **_kwargs: json.dumps(
                {
                    "fields": {"goal": "unsupported"},
                    "rationale": "No evidence was actually supplied.",
                    "evidence": {
                        "goal": [{"source": "context", "ref": "context:invented"}]
                    },
                }
            )
        ),
    )
    completed, _ = RequirementCompleter(store, orchestrator.router).complete(
        task, "actual source"
    )
    assert completed.goal == ""
    assert store.decisions.list(task.id) == []


def test_requirement_completion_rejects_context_exceeding_its_envelope_budget(tmp_path):
    store, orchestrator = runtime(tmp_path)
    task = store.create(Task(title="bounded evidence"))
    with pytest.raises(ValueError, match="declared byte budget"):
        RequirementCompleter(store, orchestrator.router).complete(
            task,
            {
                "fragments": [{"ref": "context:test/source", "text": "evidence"}],
                "budget_bytes": 1,
                "used_bytes": 1,
            },
        )


def test_conflicting_context_and_human_claims_block_requirement_completion(tmp_path):
    store, orchestrator = runtime(tmp_path)
    task = store.create(Task(title="conflicted"))
    for value in ("human alpha", "human beta"):
        question = store.ask(task.id, f"specify {value}", "requirements:incomplete")
        store.answer(question, json.dumps({"goal": value}), task.id)

    def derive(prompt, **_kwargs):
        payload = json.loads(prompt.split("\n", 1)[1])
        context_ref = next(
            item["ref"]
            for item in payload["sources"]
            if item["source"] == "memory_and_repository"
        )
        assert "goal" in payload["blocked_by_conflict"]
        return json.dumps(
            {
                "fields": {"goal": "model guess"},
                "rationale": "attempted conflict override",
                "evidence": {"goal": [{"source": "context", "ref": context_ref}]},
            }
        )

    orchestrator.models.register("fixture", SimpleNamespace(complete=derive))
    retrieved = {
        "text": "vault and vector disagree",
        "fragments": [
            {
                "kind": "vault",
                "ref": "context:vault/constraints.md#Goal",
                "text": "goal alpha",
            }
        ],
        "claims": {
            "goal": [
                {"ref": "context:vault/constraints.md#Goal", "value": "vault alpha"},
                {"ref": "context:qdrant/p1", "value": "vector beta"},
            ]
        },
        "conflicts": {
            "goal": {
                "sources": [
                    {
                        "ref": "context:vault/constraints.md#Goal",
                        "value": "vault alpha",
                    },
                    {"ref": "context:qdrant/p1", "value": "vector beta"},
                ]
            }
        },
        "budget_bytes": 1024,
        "used_bytes": 24,
    }
    completed, _ = RequirementCompleter(store, orchestrator.router).complete(
        task, retrieved
    )
    assert completed.goal == ""
    assert store.decisions.list(task.id) == []


def test_context_claim_with_unknown_reference_is_quarantined(tmp_path):
    store, orchestrator = runtime(tmp_path)
    task = store.create(Task(title="unlinked claim"))
    calls = []

    def derive(prompt, **_kwargs):
        payload = json.loads(prompt.split("\n", 1)[1])
        calls.append(payload)
        assert "goal" in payload["blocked_by_conflict"]
        return json.dumps(
            {
                "fields": {"goal": "untrusted"},
                "rationale": "Claim has no supplied source.",
                "evidence": {
                    "goal": [{"source": "context", "ref": "context:invented"}]
                },
            }
        )

    orchestrator.models.register("fixture", SimpleNamespace(complete=derive))
    completed, _ = RequirementCompleter(store, orchestrator.router).complete(
        task,
        {
            "fragments": [{"ref": "context:vault/actual.md#Goal", "text": "source"}],
            "claims": {
                "goal": [{"ref": "context:vault/fabricated.md#Goal", "value": "x"}]
            },
        },
    )
    assert calls
    assert completed.goal == ""
    assert store.list_questions(task.id) == []


def test_repository_claim_is_rejected_if_source_changes_after_context_build(tmp_path):
    store, orchestrator = runtime(tmp_path)
    task = store.create(Task(title="repository provenance"))
    source = tmp_path / "src" / "module.py"
    source.parent.mkdir()
    source.write_text("def original():\n    return True\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    source.write_text("def changed():\n    return False\n")
    reference = "context:repository/src/module.py#original"
    completed, context = RequirementCompleter(store, orchestrator.router).complete(
        task,
        {
            "fragments": [
                {
                    "kind": "repository_symbol",
                    "ref": reference,
                    "text": "original implementation",
                    "provenance": {"sha256": digest},
                }
            ],
            "claims": {"goal": [{"ref": reference, "value": "stale goal"}]},
        },
    )
    assert completed.goal != "stale goal"
    memory_context = next(
        item["data"]
        for item in json.loads(context)
        if item["source"] == "memory_and_repository"
    )
    assert memory_context["fragments"] == []
    assert memory_context["rejected_sources"] == [
        {"ref": reference, "status": "stale_or_unavailable"}
    ]
    assert "goal" in memory_context["claim_issues"]


def test_repository_claim_accepts_current_hash_bound_source(tmp_path):
    store, orchestrator = runtime(tmp_path)
    task = store.create(Task(title="repository provenance"))
    source = tmp_path / "src" / "module.py"
    source.parent.mkdir()
    source.write_text("def original():\n    return 'current goal'\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    reference = "context:repository/src/module.py#original"
    _, context = RequirementCompleter(store, orchestrator.router).complete(
        task,
        {
            "fragments": [
                {
                    "kind": "repository_symbol",
                    "ref": reference,
                    "text": "current goal",
                    "provenance": {"sha256": digest},
                }
            ],
            "claims": {"goal": [{"ref": reference, "value": "current goal"}]},
        },
    )
    memory_context = next(
        item["data"]
        for item in json.loads(context)
        if item["source"] == "memory_and_repository"
    )
    assert memory_context["fragments"][0]["ref"] == reference
    assert memory_context["claims"]["goal"] == [
        {
            "ref": reference,
            "value": "current goal",
            "evidence_quote": "current goal",
            "verification_method": "exact",
        }
    ]


def test_context_claim_without_source_quote_is_quarantined(tmp_path):
    store, orchestrator = runtime(tmp_path)
    task = store.create(Task(title="fabricated claim"))
    _, context = RequirementCompleter(store, orchestrator.router).complete(
        task,
        {
            "fragments": [
                {"ref": "context:vault/source.md#Goal", "text": "the actual alpha goal"}
            ],
            "claims": {
                "goal": [
                    {"ref": "context:vault/source.md#Goal", "value": "secret beta"}
                ]
            },
        },
    )
    memory_context = next(
        item["data"]
        for item in json.loads(context)
        if item["source"] == "memory_and_repository"
    )
    assert memory_context["claims"]["goal"] == []
    assert memory_context["claim_issues"]["goal"] == [
        "claim lacks exact support in its cited source"
    ]


def test_independent_claim_verifier_allows_cited_paraphrase(tmp_path):
    store, orchestrator = runtime(tmp_path)
    task = store.create(Task(title="semantic claim"))
    verifier = IndependentClaimVerifier(
        lambda _value, _text: {"status": "supported", "quote": "source passage"},
        identity="separate-reviewer",
        author_identity="retrieval-claim-writer",
    )
    _, context = RequirementCompleter(
        store, orchestrator.router, claim_verifier=verifier
    ).complete(
        task,
        {
            "fragments": [
                {"ref": "context:vault/source.md#Goal", "text": "A source passage."}
            ],
            "claims": {
                "goal": [
                    {"ref": "context:vault/source.md#Goal", "value": "a paraphrase"}
                ]
            },
        },
    )
    memory_context = next(
        item["data"]
        for item in json.loads(context)
        if item["source"] == "memory_and_repository"
    )
    claim = memory_context["claims"]["goal"][0]
    assert claim["verification_method"] == "independent"
    assert claim["evidence_quote"] == "source passage"
    assert claim["verifier_id"] == "separate-reviewer"


def test_conflict_resolution_pauses_for_selection_and_records_human_decision(
    tmp_path,
):
    store, orchestrator = runtime(tmp_path)
    task = store.create(Task(title="resolve sources"))
    retrieved = {
        "fragments": [
            {"ref": "context:vault/goal.md#Goal", "text": "goal alpha"},
            {"ref": "context:qdrant/point-2", "text": "goal beta"},
        ],
        "claims": {
            "goal": [
                {"ref": "context:vault/goal.md#Goal", "value": "alpha"},
                {"ref": "context:qdrant/point-2", "value": "beta"},
            ]
        },
    }
    completer = RequirementCompleter(store, orchestrator.router)
    task, _ = completer.complete(task, retrieved)
    questions = store.list_questions(task.id)
    assert len(questions) == 1
    question = questions[0]
    assert question["purpose"] == "decision"
    assert question["status"] == "open"
    selected = json.dumps(
        {"ref": "context:qdrant/point-2", "value": "beta"},
        separators=(",", ":"),
    )

    assert store.answer(question["id"], selected, task.id)
    task, _ = completer.complete(store.get(task.id), retrieved)

    assert task.goal == "beta"
    decisions = store.decisions.list(task.id)
    assert len(decisions) == 1
    assert decisions[0]["source"] == "human"
    assert decisions[0]["question_id"] == question["id"]


def test_orchestrator_keeps_conflict_decision_without_generic_input_question(
    tmp_path,
):
    store, orchestrator = runtime(tmp_path)
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "first.md").write_text(
        "---\nlast_reviewed: 2026-10-01\nclaims: {goal: alpha}\n---\n# First\nDecision conflict alpha"
    )
    (vault / "second.md").write_text(
        "---\nlast_reviewed: 2026-10-01\nclaims: {goal: beta}\n---\n# Second\nDecision conflict beta"
    )
    task = store.create(Task(title="Decision conflict"))

    result = asyncio.run(orchestrator.run(task.id))

    assert result.status == "waiting_decision"
    questions = store.list_questions(task.id)
    assert len(questions) == 1
    assert questions[0]["purpose"] == "decision"


def test_conflict_resolution_accepts_validated_human_alternative(tmp_path):
    store, orchestrator = runtime(tmp_path)
    task = store.create(Task(title="new resolution"))
    retrieved = {
        "fragments": [
            {"ref": "context:vault/a.md#Goal", "text": "alpha"},
            {"ref": "context:qdrant/b", "text": "beta"},
        ],
        "claims": {
            "goal": [
                {"ref": "context:vault/a.md#Goal", "value": "alpha"},
                {"ref": "context:qdrant/b", "value": "beta"},
            ]
        },
    }
    completer = RequirementCompleter(store, orchestrator.router)
    completer.complete(task, retrieved)
    question = store.list_questions(task.id)[0]
    answer = json.dumps(
        {"value": "gamma", "rationale": "The linked product brief is authoritative."}
    )
    assert store.answer(question["id"], answer, task.id)

    task, _ = completer.complete(store.get(task.id), retrieved)

    assert task.goal == "gamma"
    assert store.decisions.list(task.id)[0]["rationale"] == (
        "The linked product brief is authoritative."
    )


def test_conflict_resolution_rejects_selection_after_sources_change(tmp_path):
    store, orchestrator = runtime(tmp_path)
    task = store.create(Task(title="changed sources"))
    completer = RequirementCompleter(store, orchestrator.router)

    def context(second):
        return {
            "fragments": [
                {"ref": "context:vault/a.md#Goal", "text": "alpha"},
                {"ref": f"context:qdrant/{second}", "text": second},
            ],
            "claims": {
                "goal": [
                    {"ref": "context:vault/a.md#Goal", "value": "alpha"},
                    {"ref": f"context:qdrant/{second}", "value": second},
                ]
            },
        }

    completer.complete(task, context("beta"))
    old_question = store.list_questions(task.id)[0]
    assert store.answer(
        old_question["id"],
        json.dumps({"ref": "context:qdrant/beta", "value": "beta"}),
        task.id,
    )
    task, _ = completer.complete(store.get(task.id), context("gamma"))

    assert task.goal == ""
    assert store.decisions.list(task.id) == []
    open_decisions = [
        item
        for item in store.list_questions(task.id)
        if item["status"] == "open" and item["purpose"] == "decision"
    ]
    assert len(open_decisions) == 1


@pytest.mark.parametrize(
    ("fields", "rationale", "expected_goal", "expected_decisions"),
    [({"goal": "derived"}, "", "", 0), ({"goal": ""}, "grounded", "", 0)],
)
def test_requirement_completion_rejects_empty_or_unchanged_derivations(
    tmp_path, fields, rationale, expected_goal, expected_decisions
):
    store, orchestrator = runtime(tmp_path)
    task = store.create(Task(title="unchanged"))

    def derive(prompt, **_kwargs):
        payload = json.loads(prompt.split("\n", 1)[1])
        context_ref = next(
            item["ref"]
            for item in payload["sources"]
            if item["source"] == "memory_and_repository"
        )
        return json.dumps(
            {
                "fields": fields,
                "rationale": rationale,
                "evidence": {"goal": [{"source": "context", "ref": context_ref}]},
            }
        )

    orchestrator.models.register("fixture", SimpleNamespace(complete=derive))
    completed, _ = RequirementCompleter(store, orchestrator.router).complete(
        task, "source"
    )
    assert completed.goal == expected_goal
    assert store.decisions.list(task.id) == []


def test_requirement_completion_ignores_noop_human_answer_and_invalid_fields(tmp_path):
    store, orchestrator = runtime(tmp_path)
    task = store.create(Task(title="noop"))
    question = store.ask(task.id, "details", "requirements:incomplete")
    store.answer(question, json.dumps({"requirements": []}), task.id)
    completed, _ = RequirementCompleter(store, orchestrator.router).complete(
        task, "source"
    )
    assert completed.requirements == []
    assert store.decisions.list(task.id) == []


def test_tool_validation_schema_exit_and_output(tmp_path):
    _, orchestrator = runtime(tmp_path)
    tools = orchestrator.tools
    with pytest.raises(ValueError):
        tools.schemas(["missing"])
    with pytest.raises(ValueError):
        tools.executor.shell([])
    with pytest.raises(ValueError):
        tools.execute("filesystem.read", {"path": 1})
    tools.register(
        ToolSpec(
            "bad",
            "bad",
            {"type": "object"},
            "shell",
            "WRITE",
            lambda: subprocess.CompletedProcess([], 2),
        )
    )
    with pytest.raises(RuntimeError):
        tools.execute("bad", {})
    tools.register(
        ToolSpec(
            "output",
            "output",
            {"type": "object"},
            "shell",
            "WRITE",
            lambda: "ok",
            {"type": "string"},
        )
    )
    assert tools.execute("output", {}) == "ok"


def test_validator_failure_gates_and_executable_criteria(tmp_path):
    _store, orchestrator = runtime(tmp_path)
    validator = orchestrator.validator
    with pytest.raises(PermissionError):
        validator.path("../outside")
    task = specification()
    task.coverage_command = []
    task.test_commands = []
    task.acceptance_criteria = [
        AcceptanceCriterion(
            id="sum", description="command", kind="command", command=["/usr/bin/false"]
        )
    ]
    tests = validator.run_tests(task)
    assert not validator.validate(task, [], tests).valid
    task.acceptance_criteria = [
        AcceptanceCriterion(
            id="sum", description="file", kind="file_exists", path="missing"
        )
    ]
    assert not validator.validate(
        task,
        [],
        {
            "commands": [],
            "coverage": {"totals": {"percent_covered": 50, "missing_lines": 1}},
        },
    ).valid
    task.requirements = []
    task.acceptance_criteria = []
    orchestrator.models.register(
        "fixture", SimpleNamespace(complete=lambda *a, **kw: '{"evidence":""}')
    )
    assert not validator.validate(task, [], tests).valid
    task = specification()
    task.test_commands = []
    task.lint_commands = [["/usr/bin/true"]]
    task.coverage_command = ["/usr/bin/true"]
    assert validator.run_tests(task)["coverage"] is None


def test_tester_profile_contract_rejects_invalid_test_input_and_output(tmp_path):
    from harness.agents import AgentProfile

    _store, orchestrator = runtime(tmp_path)
    validator = orchestrator.validator
    validator.tester_profile = AgentProfile(
        name="test-engineer",
        instructions="test",
        model="fixture",
        input_schema={"type": "object", "required": ["trusted_test_context"]},
        output_schema={
            "type": "object",
            "required": ["commands", "coverage"],
            "properties": {"commands": {"type": "array", "minItems": 1}},
        },
    )
    task = specification()
    task.test_commands = []
    task.lint_commands = []
    task.coverage_command = []
    with pytest.raises(ValidationError):
        validator.run_tests(task)

    validator.tester_profile.input_schema = None
    with pytest.raises(ValidationError):
        validator.run_tests(task)


def test_validator_profile_contract_rejects_incomplete_runtime_payload(tmp_path):
    from harness.agents import AgentProfile

    _store, orchestrator = runtime(tmp_path)
    validator = orchestrator.validator
    validator.validator_profile = AgentProfile(
        name="validator",
        instructions="validate",
        model="fixture",
        input_schema={"type": "object", "required": ["trusted_observation"]},
    )
    with pytest.raises(ValidationError):
        validator.validate(specification(), [], {"commands": [], "coverage": None})

    validator.validator_profile.input_schema = None
    validator.validator_profile.output_schema = {
        "type": "object",
        "required": ["valid"],
        "properties": {"valid": {"const": True}},
    }
    task = specification()
    task.requirements = []
    task.acceptance_criteria = []
    task.test_commands = []
    task.coverage_command = []
    validator.router.complete = lambda *_args, **_kwargs: json.dumps(
        {
            "requirements": {},
            "criteria": {},
            "evidence": "reviewed",
            "requirement_evidence": {},
            "criterion_evidence": {},
        }
    )
    with pytest.raises(ValidationError):
        validator.validate(task, [], {"commands": [], "coverage": None})


def test_executor_observation_and_failure_paths(tmp_path):
    from harness.agents import ModelResponse

    _store, orchestrator = runtime(tmp_path)
    config = orchestrator.config
    config.data["profiles"]["coding"].update(
        {"tools": ["filesystem.write"], "permissions": ["filesystem"]}
    )
    step = Subtask(
        id="x", title="x", description="x", required_tools=["filesystem.write"]
    )
    sequence = iter(
        [
            ModelResponse(
                text="",
                tool_calls=[
                    ToolCall(
                        "call", "filesystem.write", {"path": "x", "content": "real"}
                    )
                ],
            ),
            ModelResponse(text='{"output":"done"}'),
        ]
    )
    observed = []

    def complete(profile, messages, **kwargs):
        observed.append(json.loads(json.dumps(messages)))
        return next(sequence)

    executor = Executor(
        SimpleNamespace(config=config, complete=complete), orchestrator.tools
    )
    result = executor.execute(step)
    assert result.success and result.changed_files == ["x"]
    assert observed[1][-1]["tool_call_id"] == "call"
    assert observed[1][-2]["tool_calls"][0]["id"] == "call"
    assert (tmp_path / "x").read_text() == "real"
    plain = Executor(
        SimpleNamespace(
            config=config,
            complete=lambda *a, **kw: ModelResponse(text="claims success"),
        ),
        orchestrator.tools,
    )
    invalid_report = plain.execute(step)
    assert not invalid_report.success
    assert "claims success" not in invalid_report.output
    config.data["profiles"]["coding"]["tools"] = []
    assert not executor.execute(step).success
    config.data["profiles"]["coding"]["tools"] = ["filesystem.write"]
    bad_tools = SimpleNamespace(
        schemas=lambda names: [],
        execute=lambda *a, **kw: subprocess.CompletedProcess([], 1),
    )
    failed = Executor(
        SimpleNamespace(
            config=config,
            complete=lambda *a, **kw: ModelResponse(
                text="",
                tool_calls=[
                    ToolCall("call", "filesystem.write", {"path": "x", "content": "x"})
                ],
            ),
        ),
        bad_tools,
    )
    assert not failed.execute(step).success
    config.data["profiles"]["coding"]["permissions"] = []
    with pytest.raises(PermissionError):
        orchestrator.tools.execute(
            "filesystem.write",
            {"path": "x", "content": "x"},
            profile=executor.profiles.get("coding"),
        )


def test_remaining_validator_paths(tmp_path):
    _, orchestrator = runtime(tmp_path)
    task = specification()
    task.plan = {
        "subtasks": [
            {
                "id": "planned",
                "requirement_ids": ["add two integers"],
                "acceptance_criteria": ["sum"],
            }
        ]
    }
    task.acceptance_criteria = [
        AcceptanceCriterion(
            id="sum",
            description="independent review",
            kind="command",
            command=["/usr/bin/true"],
        )
    ]
    report = {
        "commands": [],
        "coverage": {
            "totals": {
                "percent_covered": 100,
                "missing_lines": 0,
                "missing_branches": 0,
            }
        },
    }
    from harness.agents import ExecutorOutput

    snapshot = {
        "applicable": False,
        "reason": "workspace is not a Git repository",
        "filesystem": {},
    }
    assert orchestrator.validator.validate(
        task,
        [ExecutorOutput(subtask_id="planned", success=True, output="done")],
        report,
        workspace_before=snapshot,
        workspace_after=snapshot,
    ).valid


def test_manual_acceptance_criterion_has_no_automatic_observation(tmp_path):
    _, orchestrator = runtime(tmp_path)
    task = specification()
    task.acceptance_criteria = [
        AcceptanceCriterion(id="manual", description="requires human judgment")
    ]
    report = {
        "commands": [],
        "coverage": {
            "totals": {
                "percent_covered": 100,
                "missing_lines": 0,
                "missing_branches": 0,
            }
        },
    }

    result = orchestrator.validator.validate(
        task, [], report, verify_workspace_changes=False
    )

    assert not result.valid
    assert any("independent review" in error for error in result.errors)


def test_validator_fails_closed_on_malformed_coverage_and_review(tmp_path):
    _store, orchestrator = runtime(tmp_path)
    task = specification()
    report = {"commands": [], "coverage": {"totals": {"percent_covered": "100"}}}

    result = orchestrator.validator.validate(task, [], report)
    assert not result.valid
    assert "coverage report is invalid" in result.errors

    orchestrator.models.register(
        "fixture", SimpleNamespace(complete=lambda *a, **kw: "not-json")
    )
    report["coverage"] = {
        "totals": {
            "percent_covered": 100,
            "missing_lines": 0,
            "missing_branches": 0,
        }
    }
    result = orchestrator.validator.validate(task, [], report)
    assert not result.valid
    assert "independent review response is invalid" in result.errors


def test_validator_rejects_bad_fresh_coverage_file(tmp_path):
    _store, orchestrator = runtime(tmp_path)
    task = specification()
    task.coverage_command = ["coverage"]

    def write_invalid_coverage(command, cwd=None, **_kwargs):
        (tmp_path / task.coverage_report).write_text("{")
        return subprocess.CompletedProcess(command, 0, "", "")

    orchestrator.tools.executor.shell = write_invalid_coverage
    assert orchestrator.validator.run_tests(task)["coverage"] is None


@pytest.mark.parametrize(
    ("threshold", "percent", "missing_lines", "missing_branches", "expected"),
    [
        (50, 100, 0, 0, True),
        (100, 50, 0, 0, False),
        (100, 100, 0, 1, False),
    ],
)
def test_validator_coverage_threshold_and_branch_gate(
    tmp_path, threshold, percent, missing_lines, missing_branches, expected
):
    _store, orchestrator = runtime(tmp_path)
    task = specification()
    task.plan = {
        "subtasks": [
            {
                "id": "planned",
                "requirement_ids": ["add two integers"],
                "acceptance_criteria": ["sum"],
            }
        ]
    }
    from harness.agents import ExecutorOutput

    outputs = [ExecutorOutput(subtask_id="planned", success=True, output="done")]
    task.test_commands = [["true"]]
    task.acceptance_criteria = [
        AcceptanceCriterion(
            id="sum", description="review", kind="command", command=["/usr/bin/true"]
        )
    ]
    task.coverage_threshold = threshold
    report = {
        "commands": [],
        "coverage": {
            "totals": {
                "percent_covered": percent,
                "missing_lines": missing_lines,
                "missing_branches": missing_branches,
            }
        },
    }
    snapshot = {
        "applicable": False,
        "reason": "workspace is not a Git repository",
        "filesystem": {},
    }
    assert (
        orchestrator.validator.validate(
            task,
            outputs,
            report,
            workspace_before=snapshot,
            workspace_after=snapshot,
        ).valid
        is expected
    )


def test_validator_blocks_open_required_questions(tmp_path):
    store, orchestrator = runtime(tmp_path)
    task = store.create(specification())
    store.ask(task.id, "Resolve this ambiguity", "required")
    report = {
        "commands": [],
        "coverage": {
            "totals": {
                "percent_covered": 100,
                "missing_lines": 0,
                "missing_branches": 0,
            }
        },
    }

    result = orchestrator.validator.validate(
        task, [], report, open_required_questions=True
    )
    assert not result.valid
    assert "open required human questions block completion" in result.errors


def test_validator_shell_commands_use_audited_test_tools(tmp_path):
    _store, orchestrator = runtime(tmp_path)
    events = []
    orchestrator.tools.event_sink = lambda kind, payload: events.append((kind, payload))
    task = specification()
    task.plan = {
        "subtasks": [
            {
                "id": "planned",
                "requirement_ids": ["add two integers"],
                "acceptance_criteria": ["sum"],
            }
        ]
    }
    task.test_commands = [["/usr/bin/true"]]
    task.lint_commands = [["/usr/bin/true"]]
    task.coverage_command = []
    orchestrator.tools.executor.shell = lambda command, cwd=None, **_kwargs: (
        subprocess.CompletedProcess(command, 0, "ok", "")
    )

    result = orchestrator.validator.run_tests(task)

    assert result["commands"][0]["returncode"] == 0
    assert [
        event[1]["tool"] for event in events if event[0] == "TOOL_CALL_STARTED"
    ] == ["test.run_tests", "quality.lint"]
    task.acceptance_criteria = [
        AcceptanceCriterion(
            id="sum", description="command", kind="command", command=["/usr/bin/true"]
        )
    ]
    report = {
        "commands": [],
        "coverage": {
            "totals": {
                "percent_covered": 100,
                "missing_lines": 0,
                "missing_branches": 0,
            }
        },
    }
    from harness.agents import ExecutorOutput

    snapshot = {
        "applicable": False,
        "reason": "workspace is not a Git repository",
        "filesystem": {},
    }
    assert orchestrator.validator.validate(
        task,
        [ExecutorOutput(subtask_id="planned", success=True, output="done")],
        report,
        workspace_before=snapshot,
        workspace_after=snapshot,
    ).valid
    assert [
        event[1]["tool"] for event in events if event[0] == "TOOL_CALL_STARTED"
    ] == ["test.run_tests", "quality.lint", "test.run_tests"]


def test_dependency_failure_does_not_execute_dependent_step(tmp_path):
    from harness.agents import ExecutorOutput

    store, orchestrator, task = ready_runtime(tmp_path)
    first = Subtask(id="first", title="first", description="first")
    second = Subtask(
        id="second", title="second", description="second", dependencies=["first"]
    )
    orchestrator.planner.plan = lambda *a: PlannerOutput(
        summary="x", complexity="simple", subtasks=[second, first]
    )
    orchestrator.config.data["harness"] = {"max_correction_attempts": 0}
    calls = []

    def execute(step, context):
        calls.append(step.id)
        return ExecutorOutput(subtask_id=step.id, success=False, output="broken")

    orchestrator.executor.execute = execute
    with pytest.raises(RuntimeError):
        asyncio.run(orchestrator.run(store.create(task).id))
    assert calls == ["first"]


@pytest.mark.parametrize("state", ["cancelled", "waiting_human"])
def test_failure_does_not_overwrite_pause_or_abort(tmp_path, state):
    store, orchestrator, task = ready_runtime(tmp_path)
    created = store.create(task)

    def plan(*args):
        store.tasks.update(created.id, status=state)
        raise RuntimeError("interrupted")

    orchestrator.planner.plan = plan
    with pytest.raises(RuntimeError):
        asyncio.run(orchestrator.run(created.id))
    assert store.get(created.id).status == state


def test_heartbeat_renews_and_stops_on_lost_lease(tmp_path, monkeypatch):
    store, orchestrator = runtime(tmp_path)
    task = store.create(Task(title="heartbeat"))

    async def scenario():
        seen = asyncio.Event()

        async def sleep(delay):
            return

        def renew(*args):
            renew.calls += 1
            if renew.calls == 1:
                return True
            seen.set()
            return False

        renew.calls = 0

        async def run(*args):
            await seen.wait()
            return task

        monkeypatch.setattr("harness.services.asyncio.sleep", sleep)
        monkeypatch.setattr(store.tasks, "renew", renew)
        monkeypatch.setattr(orchestrator, "_run", run)
        assert (await orchestrator.run(task.id)).id == task.id

    asyncio.run(scenario())


def test_shared_abort_delete_list_and_context(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from harness import api, cli

    store, orchestrator = runtime(tmp_path)
    task = orchestrator.service.create(Task(title="metadata"))
    monkeypatch.setattr(api, "store", store)
    monkeypatch.setattr(api, "orchestrator", orchestrator)
    monkeypatch.setattr(
        cli, "build", lambda: (orchestrator.config, store, orchestrator)
    )
    assert (
        TestClient(api.app).post(f"/tasks/{task.id}/abort").json()["status"]
        == "cancelled"
    )
    other = store.create(Task(title="CLI"))
    cli.task_abort(other.id)
    assert len(orchestrator.service.list("cancelled")) == 2
    orchestrator.service.delete(other.id)
    (tmp_path / "README.md").write_text("Repository conventions")
    (tmp_path / "package.json").symlink_to(tmp_path.parent / "outside")
    orchestrator.context.memory.write("matching", "metadata decisions")
    context = orchestrator.context.build(task, str(tmp_path))
    assert "Repository conventions" in context and "metadata decisions" in context
    task.coverage_command = ["/usr/bin/true"]
    (tmp_path / "coverage.json").write_text("{}")
    import os

    os.utime(tmp_path / "coverage.json", (1, 1))
    assert orchestrator.validator.run_tests(task)["coverage"] is None


def test_question_creation_is_deduplicated_and_terminal_safe(tmp_path):
    store, orchestrator = runtime(tmp_path)
    task = store.create(Task(title="question"))
    task.description = "persisted description"
    store.update(task)
    assert store.get(task.id).description == "persisted description"
    first = store.ask(task.id, "Choose?", "reason")
    assert store.ask(task.id, "Choose?", "reason") == first
    assert len(store.questions.list(task.id)) == 1
    with pytest.raises(ValueError, match="blank"):
        store.ask(task.id, "", "reason")
    with pytest.raises(ValueError, match="not found"):
        store.ask(999, "Choose?", "reason")
    orchestrator.service.abort(task.id)
    with pytest.raises(ValueError, match="terminal"):
        store.ask(task.id, "Reopen?", "reason")
    assert store.ask(task.id, "Optional", "reason", required=False)
