import hashlib
import json
from pathlib import Path

import pytest

from harness.vault_steward import VaultKnowledgeSteward


def setup_steward(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    vault = tmp_path / "vault"
    source_root = tmp_path / "repo"
    vault.mkdir()
    source_root.mkdir()
    source = source_root / "src" / "module.py"
    source.parent.mkdir()
    source.write_text("validated source\n")
    service = VaultKnowledgeSteward(vault, source_root)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    knowledge = vault / "knowledge"
    knowledge.mkdir()
    (knowledge / "Projektwissen.md").write_text(
        "---\ntype: project-knowledge\nlast_reviewed: 2026-10-03\n"
        f"sources: [{{path: src/module.py, sha256: {digest}}}]\n---\n"
        "# Projektwissen\n",
        encoding="utf-8",
    )
    (vault / "Willkommen.md").write_text(
        "---\ntype: vault-home\nlast_reviewed: 2026-10-03\n"
        f"sources: [{{path: src/module.py, sha256: {digest}}}]\n---\n"
        "# Willkommen\n\n[[knowledge/Projektwissen]]\n",
        encoding="utf-8",
    )
    payload = {
        "task_id": "TASK-1",
        "title": "Reviewed source guidance",
        "body": "Keep the behavior deterministic.",
        "category": "lessons-learned",
        "source_path": "src/module.py",
        "source_sha256": digest,
    }
    return service, vault, source, payload


def propose(service, payload):
    return service.propose(**payload)


def test_proposal_is_pending_and_does_not_mutate_curated_notes(tmp_path):
    service, vault, _, payload = setup_steward(tmp_path)
    result = propose(service, payload)

    assert result["status"] == "pending"
    assert not (vault / result["target"]).exists()
    pending = service.pending()[0]
    assert pending["id"] == result["id"]
    assert pending["title"] == payload["title"]
    assert pending["body"] == payload["body"]
    assert pending["category"] == payload["category"]
    assert pending["task_id"] == payload["task_id"]
    assert pending["source_path"] == payload["source_path"]
    assert pending["source_sha256"] == payload["source_sha256"]
    assert pending["source_state"] == "current"
    assert pending["status"] == "pending"
    assert pending["digest"] == result["digest"]
    assert pending["target"] == result["target"]
    assert (
        "Reviewed source guidance"
        not in (vault / "knowledge" / "Projektwissen.md").read_text()
    )
    assert (service.inbox / f"{result['id']}.json").stat().st_mode & 0o777 == 0o600


def test_approved_proposal_creates_source_bound_note_once(tmp_path):
    service, vault, _, payload = setup_steward(tmp_path)
    proposal = propose(service, payload)

    result = service.approve(proposal["id"], proposal["digest"])

    note = (vault / result["target"]).read_text()
    assert result["status"] == "approved"
    assert "type: lessons-learned" in note
    assert 'task_id: "TASK-1"' in note
    assert 'path: "src/module.py"' in note
    assert payload["source_sha256"] in note
    assert note.endswith(
        "# Reviewed source guidance\n\nKeep the behavior deterministic.\n"
    )
    assert service.approve(proposal["id"], proposal["digest"]) == result
    assert service.pending()[0]["status"] == "approved"
    index = (vault / "knowledge" / "Projektwissen.md").read_text()
    assert f"[[{result['target'].removesuffix('.md')}]]" in index


def test_approved_note_passes_curated_vault_governance_audit(tmp_path):
    from harness.vault_audit import audit_vault

    service, vault, _, payload = setup_steward(tmp_path)
    proposal = propose(service, payload)
    service.approve(proposal["id"], proposal["digest"])
    report = audit_vault(
        vault,
        required_notes=(),
        content_governance=True,
        source_root=service.source_root,
    )
    assert report["healthy"] is True, report["findings"]
    assert report["findings"] == []


def test_rejected_proposal_never_writes_note_and_is_single_decision(tmp_path):
    service, vault, _, payload = setup_steward(tmp_path)
    proposal = propose(service, payload)
    rejected = service.reject(proposal["id"], proposal["digest"])
    assert rejected["status"] == "rejected"
    assert not (vault / rejected["target"]).exists()
    assert service.pending()[0]["status"] == "rejected"
    with pytest.raises(ValueError, match="different decision"):
        service.approve(proposal["id"], proposal["digest"])
    assert not (vault / rejected["target"]).exists()


def test_approval_refuses_stale_proposal_digest_and_changed_source(tmp_path):
    service, _, source, payload = setup_steward(tmp_path)
    proposal = propose(service, payload)
    with pytest.raises(ValueError, match="changed"):
        service.approve(proposal["id"], "0" * 64)
    source.write_text("changed source\n")
    assert service.pending()[0]["source_state"] == "stale_or_unavailable"
    with pytest.raises(ValueError, match="source has changed"):
        service.approve(proposal["id"], proposal["digest"])


def test_approval_never_overwrites_existing_or_symlink_notes(tmp_path):
    service, vault, _, payload = setup_steward(tmp_path)
    proposal = propose(service, payload)
    target = vault / proposal["target"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("human note")
    with pytest.raises(FileExistsError, match="overwrite"):
        service.approve(proposal["id"], proposal["digest"])

    service2, vault2, _, payload2 = setup_steward(tmp_path / "second")
    proposal2 = propose(service2, payload2)
    target2 = vault2 / proposal2["target"]
    target2.parent.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.md"
    outside.write_text("outside")
    target2.symlink_to(outside)
    with pytest.raises(ValueError, match="unsafe"):
        service2.approve(proposal2["id"], proposal2["digest"])
    assert outside.read_text() == "outside"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"task_id": True}, "task id"),
        ({"title": ""}, "title"),
        ({"title": "two\nlines"}, "title"),
        ({"body": ""}, "body"),
        ({"body": 3}, "body"),
        ({"category": "architecture"}, "category"),
        ({"source_path": "../outside.py"}, "source reference"),
        ({"source_path": "/tmp/source.py"}, "source reference"),
        ({"source_path": "src/missing.py"}, "source is unavailable"),
        ({"source_sha256": "invalid"}, "source reference"),
    ],
)
def test_proposals_validate_content_and_source_contract(tmp_path, change, message):
    service, _, _, payload = setup_steward(tmp_path)
    with pytest.raises(ValueError, match=message):
        propose(service, payload | change)


def test_proposal_is_deterministic_and_corruption_is_skipped(tmp_path):
    service, _, _, payload = setup_steward(tmp_path)
    first = propose(service, payload)
    second = propose(service, payload)
    assert first == second
    (service.inbox / f"{'f' * 24}.json").write_text("not json")
    (service.inbox / "ignored.json").write_text("{}")
    (service.inbox / "not-a-proposal.json").symlink_to(
        service.inbox / f"{'f' * 24}.json"
    )
    assert [item["id"] for item in service.pending()] == [first["id"]]


def test_invalid_identifiers_missing_records_and_conflicting_decisions(tmp_path):
    service, _, _, payload = setup_steward(tmp_path)
    proposal = propose(service, payload)
    with pytest.raises(ValueError, match="id is invalid"):
        service.reject("../bad", proposal["digest"])
    with pytest.raises(ValueError, match="unavailable"):
        service.approve("0" * 24, "0" * 64)


def test_source_symlink_and_future_mutations_are_blocked(tmp_path):
    service, _, source, payload = setup_steward(tmp_path)
    alias = source.parent / "alias.py"
    alias.symlink_to(source)
    with pytest.raises(ValueError, match="unsafe"):
        propose(service, payload | {"source_path": "src/alias.py"})


def test_proposal_target_has_fallback_for_non_ascii_title(tmp_path):
    service, _, _, payload = setup_steward(tmp_path)
    proposal = propose(service, payload | {"title": "知识管理"})
    assert proposal["target"].startswith("knowledge/knowledge-")


def test_constructor_rejects_missing_or_unsafe_vault_directories(tmp_path):
    with pytest.raises(ValueError, match="must exist"):
        VaultKnowledgeSteward(tmp_path / "missing", tmp_path)
    vault = tmp_path / "vault"
    vault.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (vault / "_harness").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe"):
        VaultKnowledgeSteward(vault, tmp_path)
    other_vault = tmp_path / "other-vault"
    other_vault.mkdir()
    file_root = tmp_path / "not-a-directory"
    file_root.write_text("file")
    with pytest.raises(ValueError, match="directories"):
        VaultKnowledgeSteward(other_vault, file_root)


def test_read_only_steward_listing_does_not_create_inbox(tmp_path):
    vault = tmp_path / "vault"
    root = tmp_path / "repo"
    vault.mkdir()
    root.mkdir()
    service = VaultKnowledgeSteward(vault, root, create_inbox=False)
    assert service.pending() == []
    assert not (vault / "_harness").exists()


def test_read_only_listing_rejects_file_or_symlink_inbox_roots(tmp_path):
    vault = tmp_path / "vault"
    root = tmp_path / "repo"
    vault.mkdir()
    root.mkdir()
    (vault / "_harness").write_text("not a directory")
    with pytest.raises(ValueError, match="unsafe"):
        VaultKnowledgeSteward(vault, root, create_inbox=False)

    (vault / "_harness").unlink()
    service = VaultKnowledgeSteward(vault, root)
    service.inbox.rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    service.inbox.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe"):
        service.pending()


def test_redaction_callback_must_return_text_and_common_credentials_are_masked(
    tmp_path,
):
    service, _vault, _, payload = setup_steward(tmp_path)
    result = service.propose(
        **(
            payload
            | {
                "title": "api_key=visible-secret",
                "body": "Bearer abcdefghijklmnop",
            }
        )
    )
    content = (service.inbox / f"{result['id']}.json").read_text()
    assert "visible-secret" not in content
    assert "abcdefghijklmnop" not in content
    broken, _, _, bad_payload = setup_steward(tmp_path / "broken")
    broken.redactor = lambda _value: None
    with pytest.raises(TypeError, match="must be text"):
        broken.propose(**bad_payload)


def test_proposal_collision_and_write_failure_are_fail_closed(tmp_path, monkeypatch):
    service, _, _, payload = setup_steward(tmp_path)
    proposal = propose(service, payload)
    path = service.inbox / f"{proposal['id']}.json"
    path.write_text("changed")
    with pytest.raises(ValueError, match="collision"):
        propose(service, payload)

    failure_service, _, _, failure_payload = setup_steward(tmp_path / "failure")
    original_fdopen = __import__("os").fdopen
    monkeypatch.setattr(
        "harness.vault_steward.os.fdopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("injected")),
    )
    with pytest.raises(OSError, match="injected"):
        propose(failure_service, failure_payload)
    assert list(failure_service.inbox.glob("*.json")) == []
    monkeypatch.setattr("harness.vault_steward.os.fdopen", original_fdopen)


def test_loader_rejects_symlink_invalid_json_and_malformed_proposal(tmp_path):
    service, _, _, payload = setup_steward(tmp_path)
    proposal = propose(service, payload)
    link = service.inbox / f"{'a' * 24}.json"
    link.symlink_to(service.inbox / f"{proposal['id']}.json")
    with pytest.raises(ValueError, match="unsafe"):
        service.approve("a" * 24, proposal["digest"])

    invalid = service.inbox / f"{'b' * 24}.json"
    invalid.write_text("not json")
    invalid_digest = hashlib.sha256(invalid.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="invalid"):
        service.approve("b" * 24, invalid_digest)

    malformed = service.inbox / f"{'c' * 24}.json"
    malformed.write_text('{"id":"' + "c" * 24 + '","unexpected":true}')
    malformed_digest = hashlib.sha256(malformed.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="invalid"):
        service.approve("c" * 24, malformed_digest)


def test_loader_rejects_rehashed_invalid_category(tmp_path):
    service, _, _, payload = setup_steward(tmp_path)
    proposal = propose(service, payload)
    raw = json.loads((service.inbox / f"{proposal['id']}.json").read_text())
    raw["category"] = "untrusted"
    new_id = hashlib.sha256(
        service._canonical({key: value for key, value in raw.items() if key != "id"})
    ).hexdigest()[:24]
    raw["id"] = new_id
    path = service.inbox / f"{new_id}.json"
    path.write_bytes(service._canonical(raw) + b"\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="invalid"):
        service.approve(new_id, digest)


def test_existing_decision_records_are_idempotent_and_conflicts_fail(tmp_path):
    service, _, _, payload = setup_steward(tmp_path)
    proposal = propose(service, payload)
    service.reject(proposal["id"], proposal["digest"])
    service._write_decision(
        proposal["id"], "rejected", proposal["digest"], proposal["target"]
    )
    with pytest.raises(ValueError, match="different decision"):
        service._write_decision(
            proposal["id"], "rejected", "0" * 64, proposal["target"]
        )
    with pytest.raises(ValueError, match="different decision"):
        service._write_decision(
            proposal["id"], "approved", proposal["digest"], proposal["target"]
        )


def test_symlink_decision_record_is_rejected(tmp_path):
    service, _, _, payload = setup_steward(tmp_path)
    proposal = propose(service, payload)
    record = service.inbox / f"{proposal['id']}.rejected.json"
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    record.symlink_to(outside)
    with pytest.raises(ValueError, match="decision record is unsafe"):
        service.reject(proposal["id"], proposal["digest"])


def test_approval_recovery_accepts_exact_existing_content_after_record_failure(
    tmp_path, monkeypatch
):
    service, vault, _, payload = setup_steward(tmp_path)
    proposal = propose(service, payload)
    monkeypatch.setattr(
        service,
        "_write_decision",
        lambda *_args: (_ for _ in ()).throw(OSError("interrupted")),
    )
    with pytest.raises(OSError, match="interrupted"):
        service.approve(proposal["id"], proposal["digest"])
    monkeypatch.undo()
    result = service.approve(proposal["id"], proposal["digest"])
    assert (vault / result["target"]).is_file()
    assert result["status"] == "approved"


def test_approval_rejects_conflicting_marker_and_target_race(tmp_path, monkeypatch):
    service, vault, _, payload = setup_steward(tmp_path)
    proposal = propose(service, payload)
    (service.inbox / f"{proposal['id']}.rejected.json").write_text("{}")
    with pytest.raises(ValueError, match="different decision"):
        service.approve(proposal["id"], proposal["digest"])
    (service.inbox / f"{proposal['id']}.rejected.json").unlink()
    target = vault / proposal["target"]
    original_open = __import__("os").open

    def race_open(path, *args, **kwargs):
        if Path(path) == target:
            target.write_text("racing writer")
            raise FileExistsError
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr("harness.vault_steward.os.open", race_open)
    with pytest.raises(FileExistsError, match="overwrite"):
        service.approve(proposal["id"], proposal["digest"])
    assert target.read_text() == "racing writer"


def test_approval_detects_inconsistent_approved_marker(tmp_path):
    service, _, _, payload = setup_steward(tmp_path)
    proposal = propose(service, payload)
    marker = service.inbox / f"{proposal['id']}.approved.json"
    marker.write_text(json.dumps({"proposal_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="inconsistent"):
        service.approve(proposal["id"], proposal["digest"])


@pytest.mark.parametrize("index_state", ["missing", "symlink", "no_review_date"])
def test_approval_rejects_unusable_project_knowledge_index(tmp_path, index_state):
    service, vault, _, payload = setup_steward(tmp_path)
    proposal = propose(service, payload)
    index = vault / "knowledge" / "Projektwissen.md"
    if index_state == "missing":
        index.unlink()
    elif index_state == "symlink":
        index.unlink()
        index.symlink_to(vault / "Willkommen.md")
    else:
        index.write_text("# Index without a review date\n")
    with pytest.raises(ValueError, match="index"):
        service.approve(proposal["id"], proposal["digest"])


def test_approval_detects_index_change_during_write_and_cleans_tempfile(
    tmp_path, monkeypatch
):
    service, vault, _, payload = setup_steward(tmp_path)
    proposal = propose(service, payload)
    index = vault / "knowledge" / "Projektwissen.md"
    original_read_text = Path.read_text
    calls = 0

    def changed_read(path, *args, **kwargs):
        nonlocal calls
        if path == index:
            calls += 1
            if calls == 2:
                index.write_text("concurrent edit")
                return "concurrent edit"
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", changed_read)
    with pytest.raises(ValueError, match="changed during approval"):
        service.approve(proposal["id"], proposal["digest"])
    assert list(index.parent.glob(".Projektwissen-*.tmp")) == []


def test_approval_rejects_resolved_external_knowledge_index(tmp_path, monkeypatch):
    service, vault, _, payload = setup_steward(tmp_path)
    proposal = propose(service, payload)
    index = vault / "knowledge" / "Projektwissen.md"
    outside = tmp_path / "external-index.md"
    outside.write_text(index.read_text())
    index.unlink()
    index.symlink_to(outside)
    original = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda path: False if path == index else original(path),
    )
    with pytest.raises(ValueError, match="index is unsafe"):
        service.approve(proposal["id"], proposal["digest"])


def test_directory_symlink_race_is_rejected(tmp_path, monkeypatch):
    from harness.vault_steward import VaultKnowledgeSteward

    path = tmp_path / "new-directory"
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda self: self == path or original(self))
    with pytest.raises(ValueError, match="unsafe"):
        VaultKnowledgeSteward._ensure_directory(path)


def test_read_only_init_rejects_existing_symlink_inbox_path(tmp_path):
    vault = tmp_path / "vault"
    root = tmp_path / "repo"
    vault.mkdir()
    root.mkdir()
    harness = vault / "_harness"
    harness.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe"):
        VaultKnowledgeSteward(vault, root, create_inbox=False)


def test_directory_is_not_accepted_as_a_source_file(tmp_path):
    service, _, _, payload = setup_steward(tmp_path)
    with pytest.raises(ValueError, match="unsafe"):
        service.propose(
            **(
                payload
                | {
                    "source_path": "src",
                    "source_sha256": "0" * 64,
                }
            )
        )


def test_recovery_closes_descriptor_if_stream_creation_fails(tmp_path, monkeypatch):
    import os

    service, _, _, payload = setup_steward(tmp_path)
    original_fdopen = os.fdopen
    original_close = os.close
    monkeypatch.setattr(
        "harness.vault_steward.os.fdopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("fdopen")),
    )
    monkeypatch.setattr(
        "harness.vault_steward.os.close",
        lambda descriptor: (
            original_close(descriptor),
            (_ for _ in ()).throw(OSError("closed")),
        )[1],
    )
    with pytest.raises(OSError, match="fdopen"):
        propose(service, payload)
    monkeypatch.setattr("harness.vault_steward.os.fdopen", original_fdopen)


def test_pending_skips_forged_identifier_filename(tmp_path):
    service, _, _, _ = setup_steward(tmp_path)
    forged = service.inbox / f"{'d' * 24}.json"
    forged.write_text(json.dumps({"id": "e" * 24}))
    assert service.pending() == []


def test_approval_fails_closed_when_resolved_target_escapes_vault(
    tmp_path, monkeypatch
):
    service, vault, _, payload = setup_steward(tmp_path)
    proposal = propose(service, payload)
    outside = tmp_path / "outside"
    outside.mkdir()
    target_directory = (vault / proposal["target"]).parent
    monkeypatch.setattr(
        service,
        "_ensure_directory",
        lambda path: path.mkdir(parents=True, exist_ok=True),
    )
    original_resolve = Path.resolve
    monkeypatch.setattr(
        Path,
        "resolve",
        lambda self, *args, **kwargs: (
            outside
            if self == target_directory
            else original_resolve(self, *args, **kwargs)
        ),
    )
    with pytest.raises(ValueError, match="target is unsafe"):
        service.approve(proposal["id"], proposal["digest"])
