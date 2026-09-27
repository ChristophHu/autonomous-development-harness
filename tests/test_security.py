import subprocess

import pytest

from harness.security import SecretResolver


def test_secret_env_precedes_keychain(monkeypatch):
    resolver = SecretResolver()
    monkeypatch.setenv("KEY", "environment")
    monkeypatch.setattr(
        "harness.security.subprocess.run",
        lambda *a, **k: pytest.fail("keychain should not be read"),
    )
    assert resolver.get("KEY") == "environment"


def test_secret_keychain_read_and_missing(monkeypatch):
    resolver = SecretResolver()
    results = [
        subprocess.CompletedProcess([], 0, "stored\n", ""),
        subprocess.CompletedProcess([], 1, "", "missing"),
    ]
    monkeypatch.setattr(
        "harness.security.subprocess.run", lambda *a, **k: results.pop(0)
    )
    assert resolver.get("KEY") == "stored" and resolver.get("KEY") is None


def test_keychain_operations_and_redaction(monkeypatch):
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, '"acct"<blob>="TOKEN"', "")

    monkeypatch.setattr("harness.security.subprocess.run", run)
    resolver = SecretResolver()
    resolver.set("TOKEN", "secret")
    assert resolver.delete("TOKEN") and resolver.list_names() == ["TOKEN"]
    assert resolver.redact("secret token", ["secret", ""]) == "[REDACTED] token"
    assert len(calls) == 3


def test_keychain_unavailable(monkeypatch):
    resolver = SecretResolver()
    monkeypatch.delenv("NO_KEY", raising=False)
    monkeypatch.setattr(
        "harness.security.subprocess.run",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()),
    )
    assert resolver.get("NO_KEY") is None
    with pytest.raises(FileNotFoundError):
        resolver.set("NO_KEY", "secret")
    monkeypatch.setattr(
        "harness.security.subprocess.run",
        lambda *a, **k: subprocess.CompletedProcess([], 1, "", "missing"),
    )
    assert not resolver.delete("NO_KEY")
