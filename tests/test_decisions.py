import json
import sqlite3

import pytest
from test_evidence_workflow import runtime

from harness.decisions import DecisionInput, DecisionService
from harness.domain import EventKind, Task


def task_store(tmp_path):
    store, _ = runtime(tmp_path)
    task = store.create(Task(title="decision task"))
    return store, task


def answer_question(store, task_id):
    question_id = store.ask(task_id, "Choose", "requirements:incomplete")
    assert store.answer(question_id, '{"goal":"safe"}', task_id)
    return question_id


def test_decision_service_records_typed_provenance_and_event(tmp_path):
    store, task = task_store(tmp_path)
    question_id = answer_question(store, task.id)
    service = DecisionService(store)
    created = service.record(
        {
            "task_id": task.id,
            "category": "requirement_resolution",
            "source": "human",
            "question_id": question_id,
            "decision": "Set the task goal",
            "rationale": "The user supplied the missing goal.",
            "field_names": ["goal"],
            "evidence": [{"source": "answer", "ref": f"answer:{question_id}"}],
            "alternatives": [
                {"option": "SQLite", "consequences": ["Local"], "selected": True},
                {"option": "Remote SQL", "consequences": ["Network"]},
            ],
            "outcome": "SQLite selected",
            "tags": ["storage", "architecture"],
        }
    )
    assert created.id > 0
    assert created.question_id == question_id
    assert created.alternatives[0].selected
    assert created.outcome == "SQLite selected"
    assert created.tags == ["storage", "architecture"]
    listed = service.list(task.id)
    assert listed[-1] == created
    event = next(
        row
        for row in store.events.list(task.id)
        if row["kind"] == EventKind.DECISION_RECORDED.value
    )
    payload = json.loads(event["payload"])
    assert payload == {
        "decision_id": created.id,
        "category": "requirement_resolution",
        "source": "human",
        "question_id": question_id,
        "field_names": ["goal"],
        "alternatives": [
            {"option": "SQLite", "consequences": ["Local"], "selected": True},
            {"option": "Remote SQL", "consequences": ["Network"], "selected": False},
        ],
        "outcome": "SQLite selected",
        "tags": ["storage", "architecture"],
        "evidence_refs": [f"answer:{question_id}"],
    }
    assert "The user supplied" not in event["payload"]


def test_decision_query_filters_and_get(tmp_path):
    store, task = task_store(tmp_path)
    service = DecisionService(store)
    first = service.record(
        {
            "task_id": task.id,
            "category": "architecture",
            "source": "agent",
            "decision": "Use SQLite",
            "rationale": "Local persistence",
            "evidence": [{"source": "task", "ref": f"task:{task.id}"}],
            "tags": ["storage"],
        }
    )
    service.record(
        {
            "task_id": task.id,
            "category": "operational",
            "source": "task",
            "decision": "Run locally",
            "rationale": "Keep operations local",
            "evidence": [{"source": "task", "ref": f"task:{task.id}"}],
            "tags": ["runtime"],
        }
    )
    assert service.query(task.id, "architecture", "agent", "storage") == [first]
    assert service.query(category="missing") == []
    assert service.get(first.id) == first
    assert service.get(999) is None


def test_decisions_are_superseded_append_only_and_single_successor(tmp_path):
    store, task = task_store(tmp_path)
    service = DecisionService(store)
    original = service.record(
        {
            "task_id": task.id,
            "category": "architecture",
            "source": "agent",
            "decision": "Use the old provider",
            "rationale": "Initial evidence",
            "evidence": [{"source": "task", "ref": f"task:{task.id}"}],
        }
    )
    successor = service.record(
        {
            "task_id": task.id,
            "category": "architecture",
            "source": "agent",
            "decision": "Use the new provider",
            "rationale": "New evidence",
            "evidence": [{"source": "task", "ref": f"task:{task.id}"}],
            "supersedes_id": original.id,
        }
    )
    assert successor.supersedes_id == original.id
    assert service.get(original.id) == original
    assert [item.id for item in service.list(task.id)] == [original.id, successor.id]
    vault = store.config.path("obsidian_vault")
    from harness.memory_projection import DecisionProjection

    DecisionProjection(vault).sync(store.decisions.list_all())
    assert (
        f"supersedes_id: {original.id}"
        in (vault / "_harness" / "decisions" / f"{successor.id}.md").read_text()
    )
    other_task = store.create(Task(title="different scope"))
    for decision_task, ref, target in (
        (other_task.id, other_task.id, original.id),
        (task.id, task.id, 999),
    ):
        with pytest.raises(ValueError, match="same task"):
            service.record(
                {
                    "task_id": decision_task,
                    "category": "architecture",
                    "source": "agent",
                    "decision": "Invalid supersession",
                    "rationale": "Not permitted",
                    "evidence": [{"source": "task", "ref": f"task:{ref}"}],
                    "supersedes_id": target,
                }
            )
    with pytest.raises(ValueError, match="already superseded"):
        service.record(
            {
                "task_id": task.id,
                "category": "architecture",
                "source": "agent",
                "decision": "Competing successor",
                "rationale": "Must not fork",
                "evidence": [{"source": "task", "ref": f"task:{task.id}"}],
                "supersedes_id": original.id,
            }
        )


def test_decision_record_automatically_projects_when_memory_is_enabled(tmp_path):
    store, task = task_store(tmp_path)
    vault = store.config.path("obsidian_vault")
    store.config.data.setdefault("memory", {}).setdefault("obsidian", {})["enabled"] = (
        True
    )

    created = DecisionService(store).record(
        {
            "task_id": task.id,
            "category": "architecture",
            "source": "agent",
            "decision": "Use SQLite",
            "rationale": "Keep a canonical store",
            "evidence": [{"source": "task", "ref": f"task:{task.id}"}],
        }
    )

    note = vault / "_harness" / "decisions" / f"{created.id}.md"
    assert note.exists()
    assert "Use SQLite" in note.read_text()


def test_decision_projection_preserves_alternatives_outcome_and_tags(tmp_path):
    store, task = task_store(tmp_path)
    store.config.data.setdefault("memory", {}).setdefault("obsidian", {})["enabled"] = (
        True
    )
    decision = DecisionService(store).record(
        {
            "task_id": task.id,
            "category": "architecture",
            "source": "agent",
            "decision": "Use SQLite",
            "rationale": "Local-first persistence",
            "evidence": [{"source": "task", "ref": f"task:{task.id}"}],
            "alternatives": [
                {"option": "SQLite", "consequences": ["Local"], "selected": True},
                {"option": "Cloud DB", "consequences": ["Remote"]},
            ],
            "outcome": "SQLite chosen",
            "tags": ["architecture", "storage"],
        }
    )
    note = (
        store.config.path("obsidian_vault")
        / "_harness"
        / "decisions"
        / f"{decision.id}.md"
    )
    content = note.read_text()
    assert "tags:" in content and "architecture" in content
    assert "## Alternatives" in content and "SQLite (selected)" in content
    assert "## Outcome" in content and "SQLite chosen" in content


def test_decision_alternatives_and_outcome_are_redacted_before_persistence(tmp_path):
    store, task = task_store(tmp_path)
    store.config.data.setdefault("secrets", {})["API_TOKEN"] = "sensitive-value"
    created = DecisionService(store).record(
        {
            "task_id": task.id,
            "category": "architecture",
            "source": "agent",
            "decision": "Choose a secure backend",
            "rationale": "A secret is not required here.",
            "evidence": [{"source": "task", "ref": f"task:{task.id}"}],
            "alternatives": [
                {
                    "option": "backend with sensitive-value",
                    "selected": True,
                }
            ],
            "outcome": "token=sensitive-value",
        }
    )
    stored = json.dumps(store.decisions.get(created.id))
    event = store.events.list(task.id)[-1]["payload"]
    assert "sensitive-value" not in stored
    assert "sensitive-value" not in event


def test_decision_record_respects_disabled_obsidian_projection(tmp_path):
    store, task = task_store(tmp_path)
    vault = store.config.path("obsidian_vault")
    store.config.data.setdefault("memory", {}).setdefault("obsidian", {})["enabled"] = (
        False
    )

    DecisionService(store).record(
        {
            "task_id": task.id,
            "category": "architecture",
            "source": "agent",
            "decision": "Use SQLite",
            "rationale": "Keep a canonical store",
            "evidence": [{"source": "task", "ref": f"task:{task.id}"}],
        }
    )

    assert not (vault / "_harness" / "manifest.json").exists()


def test_decision_projection_failure_does_not_lose_canonical_decision(
    tmp_path, monkeypatch
):
    from harness.memory_projection import DecisionProjection

    store, task = task_store(tmp_path)
    store.config.data.setdefault("memory", {}).setdefault("obsidian", {})["enabled"] = (
        True
    )
    monkeypatch.setattr(
        DecisionProjection,
        "sync",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("sensitive path")),
    )

    created = DecisionService(store).record(
        {
            "task_id": task.id,
            "category": "architecture",
            "source": "agent",
            "decision": "Keep SQLite",
            "rationale": "Canonical source",
            "evidence": [{"source": "task", "ref": f"task:{task.id}"}],
        }
    )

    assert store.decisions.get(created.id)["decision"] == "Keep SQLite"
    failure = next(
        item
        for item in store.events.list(task.id)
        if item["kind"] == "memory.projection_failed"
    )
    assert "sensitive path" not in failure["payload"]
    assert json.loads(failure["payload"])["error_type"] == "OSError"


@pytest.mark.parametrize(
    "payload",
    [
        {"category": "unknown"},
        {"source": "system"},
        {"decision": " "},
        {"rationale": ""},
        {"evidence": []},
        {"field_names": ["goal", "goal"]},
        {"field_names": ["Invalid-field"]},
        {"field_names": []},
        {"evidence": [{"source": "task", "ref": "context:not-matching"}]},
        {"unexpected": True},
    ],
)
def test_decision_input_rejects_invalid_contracts(payload):
    base = {
        "task_id": 1,
        "category": "requirement_resolution",
        "source": "agent",
        "decision": "Resolve goal",
        "rationale": "Supported by a cited source.",
        "field_names": ["goal"],
        "evidence": [{"source": "context", "ref": "context:abc"}],
    }
    with pytest.raises(ValueError):
        DecisionInput.model_validate(base | payload)


def test_decision_input_requires_one_selected_unique_alternative_and_valid_tags():
    base = {
        "task_id": 1,
        "category": "architecture",
        "source": "agent",
        "decision": "Use SQLite",
        "rationale": "Local-first storage",
        "evidence": [{"source": "task", "ref": "task:1"}],
        "alternatives": [
            {"option": "SQLite", "selected": True},
            {"option": "Remote SQL"},
        ],
        "tags": ["storage", "local-first"],
    }
    assert DecisionInput.model_validate(base).alternatives[0].selected
    for invalid in (
        {**base, "alternatives": [{"option": "SQLite"}, {"option": "Remote SQL"}]},
        {
            **base,
            "alternatives": [
                {"option": "SQLite", "selected": True},
                {"option": "SQLite"},
            ],
        },
        {**base, "tags": ["Storage"]},
        {**base, "tags": ["storage", "storage"]},
        {
            **base,
            "alternatives": [
                {"option": "SQLite", "selected": True, "consequences": [""]}
            ],
        },
        {
            **base,
            "alternatives": [
                {"option": "SQLite", "selected": True, "consequences": ["x" * 1001]}
            ],
        },
    ):
        with pytest.raises(ValueError):
            DecisionInput.model_validate(invalid)


def test_decision_service_checks_tasks_questions_and_answer_evidence(tmp_path):
    store, task = task_store(tmp_path)
    other = store.create(Task(title="other task"))
    service = DecisionService(store)
    valid = {
        "task_id": task.id,
        "category": "requirement_resolution",
        "source": "human",
        "decision": "Resolve goal",
        "rationale": "Human answered the question.",
        "field_names": ["goal"],
        "evidence": [{"source": "answer", "ref": "answer:1"}],
    }
    with pytest.raises(ValueError, match="task not found"):
        service.record(valid | {"task_id": 999, "question_id": 1})
    with pytest.raises(ValueError, match="question reference"):
        DecisionInput.model_validate(valid)
    foreign_question = answer_question(store, other.id)
    with pytest.raises(ValueError, match="does not belong"):
        service.record(valid | {"question_id": foreign_question})
    own_question = store.ask(task.id, "Open", "clarify")
    own_answer = {"source": "answer", "ref": f"answer:{own_question}"}
    with pytest.raises(ValueError, match="not answered"):
        service.record(valid | {"question_id": own_question, "evidence": [own_answer]})
    assert store.answer(own_question, "provided", task.id)
    with pytest.raises(ValueError, match="answer evidence is unavailable"):
        service.record(
            valid
            | {
                "source": "agent",
                "evidence": [{"source": "answer", "ref": f"answer:{own_question + 1}"}],
            }
        )
    second_question = store.ask(task.id, "Another", "clarify")
    assert store.answer(second_question, "provided too", task.id)
    with pytest.raises(ValueError, match="evidence does not match"):
        service.record(
            valid
            | {
                "question_id": own_question,
                "evidence": [{"source": "answer", "ref": f"answer:{second_question}"}],
            }
        )
    with pytest.raises(ValueError, match="integer"):
        service.record(
            {
                "task_id": task.id,
                "category": "architecture",
                "source": "task",
                "decision": "Malformed ref",
                "rationale": "Invalid ID syntax.",
                "evidence": [{"source": "task", "ref": "task:not-an-id"}],
            }
        )
    with pytest.raises(ValueError, match="task evidence"):
        service.record(
            {
                "task_id": task.id,
                "category": "architecture",
                "source": "task",
                "decision": "Foreign task reference",
                "rationale": "Wrong task ID.",
                "evidence": [{"source": "task", "ref": "task:999"}],
            }
        )
    with pytest.raises(ValueError, match="task evidence"):
        service.record(
            {
                "task_id": task.id,
                "category": "architecture",
                "source": "task",
                "decision": "Mismatched task evidence",
                "rationale": "The task reference differs.",
                "evidence": [{"source": "task", "ref": f"task:{other.id}"}],
            }
        )


def test_decision_service_accepts_agent_evidence_and_global_records(tmp_path):
    store, task = task_store(tmp_path)
    service = DecisionService(store)
    record = service.record(
        {
            "task_id": task.id,
            "category": "requirement_resolution",
            "source": "agent",
            "decision": "Use the cited requirement",
            "rationale": "This field is supported.",
            "field_names": ["goal"],
            "evidence": [{"source": "task", "ref": f"task:{task.id}"}],
        }
    )
    global_record = service.record(
        {
            "task_id": None,
            "category": "architecture",
            "source": "repository",
            "decision": "Use a stable interface",
            "rationale": "A repository-level convention applies.",
            "evidence": [{"source": "context", "ref": "context:revision-a"}],
        }
    )
    global_reference = service.record(
        {
            "task_id": task.id,
            "category": "architecture",
            "source": "repository",
            "decision": "Reuse global convention",
            "rationale": "The global decision is applicable.",
            "evidence": [
                {"source": "decision", "ref": f"decision:{global_record.id}"},
                {"source": "decision", "ref": f"decision:{global_record.id}"},
                {"source": "context", "ref": "context:revision-b"},
            ],
        }
    )
    assert [item.id for item in service.list(task.id)] == [
        record.id,
        global_record.id,
        global_reference.id,
    ]
    assert [item.id for item in service.list(999)] == [global_record.id]
    assert store.decisions.get(999) is None


def test_decision_service_validates_parent_and_prior_decision_refs_and_redacts(
    tmp_path,
):
    store, parent = task_store(tmp_path)
    child = store.create(Task(title="child", parent_task_id=parent.id))
    store.audit.secrets = {"PRIVATE_KEY": "sensitive-value"}
    service = DecisionService(store)
    parent_decision = service.record(
        {
            "task_id": child.id,
            "category": "architecture",
            "source": "parent",
            "decision": "Follow parent constraint",
            "rationale": "The parent supplied sensitive-value.",
            "evidence": [{"source": "parent", "ref": f"parent:{parent.id}"}],
        }
    )
    assert parent_decision.rationale.endswith("[REDACTED].")
    inherited = service.record(
        {
            "task_id": child.id,
            "category": "architecture",
            "source": "repository",
            "decision": "Retain the established choice",
            "rationale": "The earlier decision applies.",
            "evidence": [
                {"source": "decision", "ref": f"decision:{parent_decision.id}"}
            ],
        }
    )
    assert inherited.id > parent_decision.id
    with pytest.raises(ValueError, match="parent evidence"):
        service.record(
            {
                "task_id": child.id,
                "category": "architecture",
                "source": "parent",
                "decision": "Invalid parent",
                "rationale": "Wrong parent id.",
                "evidence": [{"source": "parent", "ref": "parent:999"}],
            }
        )
    with pytest.raises(ValueError, match="decision evidence"):
        service.record(
            {
                "task_id": child.id,
                "category": "architecture",
                "source": "repository",
                "decision": "Missing source decision",
                "rationale": "The reference does not exist.",
                "evidence": [{"source": "decision", "ref": "decision:999"}],
            }
        )


def test_decision_event_failure_rolls_back_record(tmp_path):
    store, task = task_store(tmp_path)
    with store.database.connect() as connection:
        connection.execute(
            "CREATE TRIGGER reject_decision_event BEFORE INSERT ON events "
            "WHEN NEW.kind='decision.recorded' "
            "BEGIN SELECT RAISE(ABORT, 'audit unavailable'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="audit unavailable"):
        DecisionService(store).record(
            {
                "task_id": task.id,
                "category": "requirement_resolution",
                "source": "agent",
                "decision": "Resolve goal",
                "rationale": "Supported.",
                "field_names": ["goal"],
                "evidence": [{"source": "task", "ref": f"task:{task.id}"}],
            }
        )
    assert DecisionService(store).list(task.id) == []
