import json

import pytest
from test_evidence_workflow import runtime

from harness.decisions import DecisionService
from harness.domain import Task
from harness.memory_projection import DecisionProjection


def fixture_store(tmp_path):
    store, _ = runtime(tmp_path)
    task = store.create(Task(title="Projection task"))
    DecisionService(store).record(
        {
            "task_id": task.id,
            "category": "architecture",
            "source": "agent",
            "decision": "Use **SQLite** as canonical storage",
            "rationale": "The user chose a stable source of truth.",
            "field_names": ["storage"],
            "evidence": [{"source": "task", "ref": f"task:{task.id}"}],
        }
    )
    return store


def test_decision_projection_writes_deterministic_markdown_and_manifest(tmp_path):
    store = fixture_store(tmp_path)
    projection = DecisionProjection(tmp_path / "vault")

    first = projection.sync(store.decisions.list_all())
    note = tmp_path / "vault" / "_harness" / "decisions" / "1.md"
    content = note.read_text()
    second = projection.sync(store.decisions.list_all())

    assert first == {"created": 1, "updated": 0, "removed": 0, "unchanged": 0}
    assert second == {"created": 0, "updated": 0, "removed": 0, "unchanged": 1}
    assert "projection: harness-decisions-v1" in content
    assert "Use **SQLite** as canonical storage" in content
    assert "task_id: 1" in content
    manifest = json.loads(
        (tmp_path / "vault" / "_harness" / "manifest.json").read_text()
    )
    assert manifest == {"version": 1, "files": ["decisions/1.md"]}


def test_vault_status_reports_root_notes_and_managed_decisions(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Willkommen.md").write_text("hello")
    (vault / ".obsidian").mkdir()
    (vault / ".obsidian" / "hidden.md").write_text("hidden")
    projection = DecisionProjection(vault)
    projection.sync([])

    status = projection.status(obsidian_enabled=True, mcp_enabled=False)

    assert status == {
        "path": str(vault.resolve()),
        "exists": True,
        "obsidian_enabled": True,
        "mcp_enabled": False,
        "markdown_notes": 1,
        "managed_decisions": 0,
        "nested_vaults": [],
    }


def test_vault_status_detects_nested_obsidian_configuration(tmp_path):
    vault = tmp_path / "vault"
    (vault / "nested" / ".obsidian").mkdir(parents=True)

    status = DecisionProjection(vault).status()

    assert status["nested_vaults"] == ["nested/.obsidian"]


def test_vault_status_does_not_create_a_missing_vault(tmp_path):
    vault = tmp_path / "missing"

    status = DecisionProjection(vault).status()

    assert status["exists"] is False
    assert status["markdown_notes"] == 0
    assert not vault.exists()


def test_decision_projection_updates_owned_note_and_preserves_user_files(tmp_path):
    store = fixture_store(tmp_path)
    projection = DecisionProjection(tmp_path / "vault")
    projection.sync(store.decisions.list_all())
    user_note = tmp_path / "vault" / "decisions" / "1.md"
    user_note.parent.mkdir()
    user_note.write_text("User-authored note")
    row = store.decisions.list_all()[0]
    row["decision"] = "Use a revised storage contract"

    result = projection.sync([row])

    assert result["updated"] == 1
    assert (
        "revised storage"
        in (tmp_path / "vault" / "_harness" / "decisions" / "1.md").read_text()
    )
    assert user_note.read_text() == "User-authored note"


def test_decision_projection_removes_only_registered_harness_notes(tmp_path):
    store = fixture_store(tmp_path)
    projection = DecisionProjection(tmp_path / "vault")
    projection.sync(store.decisions.list_all())
    user_note = tmp_path / "vault" / "_harness" / "decisions" / "manual.md"
    user_note.write_text("Do not delete")

    result = projection.sync([])

    assert result["removed"] == 1
    assert user_note.read_text() == "Do not delete"


@pytest.mark.parametrize(
    "name", ["../escape.md", "/tmp/escape.md", "nested/../escape.md"]
)
def test_decision_projection_rejects_unsafe_manifest_entries(tmp_path, name):
    projection = DecisionProjection(tmp_path / "vault")
    projection.manifest_path.parent.mkdir(parents=True)
    projection.manifest_path.write_text(json.dumps({"version": 1, "files": [name]}))

    with pytest.raises(ValueError, match="manifest"):
        projection.sync([])


def test_decision_projection_refuses_to_overwrite_unowned_files(tmp_path):
    projection = DecisionProjection(tmp_path / "vault")
    target = tmp_path / "vault" / "_harness" / "decisions" / "1.md"
    target.parent.mkdir(parents=True)
    target.write_text("User file")
    row = {
        "id": 1,
        "task_id": None,
        "category": "architecture",
        "source": "agent",
        "question_id": None,
        "decision": "Decision",
        "rationale": "Reason",
        "field_names": [],
        "evidence": [],
        "created_at": "2026-09-29T00:00:00Z",
    }

    with pytest.raises(FileExistsError, match="unowned"):
        projection.sync([row])
    assert target.read_text() == "User file"


def test_decision_projection_rejects_symlinked_managed_directory(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (vault / "_harness").symlink_to(outside, target_is_directory=True)

    with pytest.raises(PermissionError, match="symlink"):
        DecisionProjection(vault).sync([])


def test_decision_projection_rejects_symlinked_notes_subdirectory(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "_harness").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (vault / "_harness" / "decisions").symlink_to(outside, target_is_directory=True)

    with pytest.raises(PermissionError, match="symlink"):
        DecisionProjection(vault).sync([])


def test_decision_projection_rejects_path_outside_vault(tmp_path):
    projection = DecisionProjection(tmp_path / "vault")
    with pytest.raises(PermissionError, match="escapes"):
        projection._inside_vault(tmp_path / "outside.md")


def test_decision_projection_rechecks_resolved_path_for_escape(tmp_path, monkeypatch):
    projection = DecisionProjection(tmp_path / "vault")
    candidate = projection.root / "inside.md"
    outside = tmp_path / "outside.md"
    resolve = type(candidate).resolve

    def switched(path, *args, **kwargs):
        if path == candidate:
            return outside
        return resolve(path, *args, **kwargs)

    monkeypatch.setattr(type(candidate), "resolve", switched)
    with pytest.raises(PermissionError, match="escapes"):
        projection._inside_vault(candidate)


@pytest.mark.parametrize(
    "manifest",
    [
        "not json",
        "[]",
        '{"version": 2, "files": []}',
        '{"version": 1, "files": "bad"}',
        '{"version": 1, "files": ["decisions/1.md", "decisions/1.md"]}',
    ],
)
def test_decision_projection_rejects_invalid_manifests(tmp_path, manifest):
    projection = DecisionProjection(tmp_path / "vault")
    projection.manifest_path.parent.mkdir(parents=True)
    projection.manifest_path.write_text(manifest)
    with pytest.raises(ValueError, match="manifest"):
        projection.sync([])


def test_decision_projection_rejects_symlinked_manifest_note(tmp_path):
    projection = DecisionProjection(tmp_path / "vault")
    projection.notes.mkdir(parents=True)
    outside = tmp_path / "outside.md"
    outside.write_text("outside")
    (projection.notes / "1.md").symlink_to(outside)
    projection.manifest_path.write_text(
        json.dumps({"version": 1, "files": ["decisions/1.md"]})
    )
    with pytest.raises(PermissionError, match="symlink"):
        projection.sync([])


def test_decision_projection_rejects_duplicate_and_invalid_ids(tmp_path):
    projection = DecisionProjection(tmp_path / "vault")
    base = {
        "id": 1,
        "task_id": None,
        "category": "architecture",
        "source": "agent",
        "question_id": None,
        "decision": "Decision",
        "rationale": "Reason",
        "field_names": [],
        "evidence": [],
        "created_at": "2026-09-29T00:00:00Z",
    }
    with pytest.raises(ValueError, match="duplicate"):
        projection.sync([base, base])
    with pytest.raises(ValueError, match="positive integer"):
        projection.sync([base | {"id": True}])


def test_decision_projection_refuses_to_delete_modified_registered_note(tmp_path):
    store = fixture_store(tmp_path)
    projection = DecisionProjection(tmp_path / "vault")
    projection.sync(store.decisions.list_all())
    note = tmp_path / "vault" / "_harness" / "decisions" / "1.md"
    note.write_text(
        "User content mentioning projection: harness-decisions-v1, not a header"
    )

    with pytest.raises(FileExistsError, match="unowned"):
        projection.sync([])
    assert "User content" in note.read_text()


def test_projection_preflights_conflicts_before_updating_any_notes(tmp_path):
    store = fixture_store(tmp_path)
    store.decisions.save(None, "Second decision", "Second reason")
    projection = DecisionProjection(tmp_path / "vault")
    rows = store.decisions.list_all()
    projection.sync(rows)
    first_note = projection.notes / "1.md"
    previous = first_note.read_text()
    conflict = projection.notes / "2.md"
    conflict.write_text("User-owned file")
    changed = [rows[0] | {"decision": "Updated decision"}, rows[1]]

    with pytest.raises(FileExistsError, match="unowned"):
        projection.sync(changed)

    assert first_note.read_text() == previous
    assert conflict.read_text() == "User-owned file"


def test_decision_projection_ignores_missing_registered_note(tmp_path):
    store = fixture_store(tmp_path)
    projection = DecisionProjection(tmp_path / "vault")
    projection.sync(store.decisions.list_all())
    (projection.notes / "1.md").unlink()
    result = projection.sync([])
    assert result["removed"] == 0


def test_decision_projection_propagates_temporary_file_creation_failure(
    tmp_path, monkeypatch
):
    import harness.memory_projection as module

    projection = DecisionProjection(tmp_path / "vault")
    monkeypatch.setattr(
        module.tempfile,
        "NamedTemporaryFile",
        lambda **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(OSError, match="disk full"):
        projection._atomic_write(projection.manifest_path, "{}")
