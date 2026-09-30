import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from harness.security import (
    EnvironmentSecretProvider,
    KeychainSecretProvider,
    SecretResolver,
)


def test_secret_providers_are_separated_and_resolver_accepts_injected_providers():
    class Keychain:
        def get(self, name):
            return "from-keychain"

        def set(self, name, value):
            self.saved = (name, value)

        def delete(self, name):
            return True

        def list_names(self):
            return ["KEY"]

    class Environment:
        def get(self, name):
            return "from-env"

    resolver = SecretResolver(
        keychain_provider=Keychain(), environment_provider=Environment()
    )
    assert isinstance(KeychainSecretProvider(), KeychainSecretProvider)
    assert isinstance(EnvironmentSecretProvider(), EnvironmentSecretProvider)
    assert resolver.get("KEY") == "from-keychain"


def test_environment_provider_reads_only_the_requested_name(monkeypatch):
    monkeypatch.setenv("KEY", "from-env")
    provider = EnvironmentSecretProvider()
    assert provider.get("KEY") == "from-env"
    assert provider.get("MISSING") is None


def test_keychain_listing_filters_by_exact_service(monkeypatch):
    dump = """
    "svce"<blob>="other-service"
    "acct"<blob>="OTHER_TOKEN"

    "svce"<blob>="autonomous-development-harness"
    "acct"<blob>="HARNESS_TOKEN"

    "svce"<blob>="autonomous-development-harness-extra"
    "acct"<blob>="PREFIX_TOKEN"

    "svce"<blob>="autonomous-development-harness"
    "acct"<blob>="invalid-name"
    """
    monkeypatch.setattr(
        "harness.security.subprocess.run",
        lambda *a, **k: subprocess.CompletedProcess([], 0, dump, ""),
    )
    assert KeychainSecretProvider().list_names() == ["HARNESS_TOKEN"]


def test_keychain_listing_failure_is_not_reported_as_empty_success(monkeypatch):
    monkeypatch.setattr(
        "harness.security.subprocess.run",
        lambda *a, **k: subprocess.CompletedProcess([], 1, "", "unavailable"),
    )
    with pytest.raises(RuntimeError, match="keychain listing failed"):
        KeychainSecretProvider().list_names()


def test_keychain_set_rejects_empty_secret_before_subprocess(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "harness.security.subprocess.run", lambda *a, **k: calls.append(a)
    )
    with pytest.raises(ValueError, match="must not be empty"):
        KeychainSecretProvider().set("TOKEN", "")
    assert calls == []


@pytest.mark.parametrize("name", ["", "lowercase", "A;bad", "A" * 129, "A B"])
def test_secret_names_are_rejected_before_keychain_access(name, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "harness.security.subprocess.run", lambda *a, **k: calls.append(a)
    )
    with pytest.raises(ValueError, match="invalid secret name"):
        SecretResolver().get(name)
    assert calls == []


def test_secret_exists_includes_keychain_and_environment(monkeypatch):
    monkeypatch.setenv("ENV_ONLY", "available")
    monkeypatch.setattr(
        "harness.security.subprocess.run",
        lambda *a, **k: subprocess.CompletedProcess([], 1, "", "missing"),
    )
    resolver = SecretResolver()
    assert resolver.exists("ENV_ONLY")
    assert not resolver.exists("ABSENT")


def test_secret_keychain_precedes_environment(monkeypatch):
    resolver = SecretResolver()
    monkeypatch.setenv("KEY", "environment")
    monkeypatch.setattr(
        "harness.security.subprocess.run",
        lambda *a, **k: subprocess.CompletedProcess([], 0, "keychain\n", ""),
    )
    assert resolver.get("KEY") == "keychain"


def test_secret_environment_is_fallback_when_keychain_has_no_value(monkeypatch):
    resolver = SecretResolver()
    monkeypatch.setenv("KEY", "environment")
    monkeypatch.setattr(
        "harness.security.subprocess.run",
        lambda *a, **k: subprocess.CompletedProcess([], 1, "", "missing"),
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
        calls.append((args, kwargs))
        if "find-generic-password" in args:
            return subprocess.CompletedProcess(args, 0, "secret\n", "")
        return subprocess.CompletedProcess(
            args,
            0,
            '"svce"<blob>="autonomous-development-harness"\n"acct"<blob>="TOKEN"',
            "",
        )

    monkeypatch.setattr("harness.security.subprocess.run", run)
    resolver = SecretResolver()
    resolver.set("TOKEN", "secret")
    assert resolver.delete("TOKEN") and resolver.list_names() == ["TOKEN"]
    assert resolver.redact("secret token", ["secret", ""]) == "[REDACTED] token"
    assert len(calls) == 4
    assert all("secret" not in args for args, _ in calls)
    assert calls[0][0] == ["security", "-i"]
    assert "736563726574" in calls[0][1]["input"]
    assert "secret" not in calls[0][1]["input"]


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


def test_keychain_provider_targets_only_explicit_isolated_path(tmp_path, monkeypatch):
    path = tmp_path / "isolated keychain.keychain-db"
    calls = []
    current = ["value"]

    def run(args, **kwargs):
        calls.append((args, kwargs))
        if args == ["security", "-i"]:
            current[0] = "private"
            return subprocess.CompletedProcess(args, 0, "", "")
        return subprocess.CompletedProcess(args, 0, current[0] + "\n", "")

    monkeypatch.setattr("harness.security.subprocess.run", run)
    provider = KeychainSecretProvider(keychain_path=path)
    assert provider.get("TOKEN") == "value"
    provider.set("TOKEN", "private")
    assert provider.delete("TOKEN")
    provider.list_names()
    assert all(str(path) in args or args == ["security", "-i"] for args, _ in calls)
    assert str(path) in calls[1][1]["input"]
    assert all("private" not in args for args, _ in calls)
    assert "70726976617465" in calls[1][1]["input"]


def test_keychain_path_must_be_absolute():
    with pytest.raises(ValueError, match="absolute"):
        KeychainSecretProvider(keychain_path=Path("relative.keychain-db"))


@pytest.mark.parametrize("service", ["bad service", "bad;service", "", "a" * 129, None])
def test_keychain_service_name_is_safe(service):
    with pytest.raises(ValueError, match="service name"):
        KeychainSecretProvider(service=service)


def test_keychain_write_verifies_readback_and_preserves_spaces(monkeypatch):
    provider = KeychainSecretProvider()
    monkeypatch.setattr(
        "harness.security.subprocess.run",
        lambda args, **_kwargs: subprocess.CompletedProcess(
            args, 0, "  spaced value  \n", ""
        ),
    )
    provider.set("TOKEN", "  spaced value  ")
    assert provider.get("TOKEN") == "  spaced value  "
    with pytest.raises(RuntimeError, match="write failed"):
        provider.set("TOKEN", "different")


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS Keychain acceptance")
def test_live_isolated_keychain_and_secret_cli(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from harness.cli import app

    path = tmp_path / "isolated keychain.keychain-db"
    service = "harness-test-" + uuid.uuid4().hex
    subprocess.run(
        ["security", "create-keychain", "-p", "", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        subprocess.run(
            ["security", "unlock-keychain", "-p", "", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        provider = KeychainSecretProvider(service, keychain_path=path)
        resolver = SecretResolver(keychain_provider=provider)
        monkeypatch.setattr("harness.cli.SecretResolver", lambda: resolver)
        runner = CliRunner()
        set_result = runner.invoke(
            app, ["secrets", "set", "TEST_TOKEN"], input="first-value\nfirst-value\n"
        )
        assert set_result.exit_code == 0, (
            set_result.stdout,
            repr(set_result.exception),
        )
        assert provider.get("TEST_TOKEN") == "first-value"
        provider.set("TEST_TOKEN", "updated value; $HOME")
        assert provider.get("TEST_TOKEN") == "updated value; $HOME"
        assert provider.list_names() == ["TEST_TOKEN"]
        assert (
            runner.invoke(app, ["secrets", "exists", "TEST_TOKEN"]).stdout.strip()
            == "exists"
        )
        assert "TEST_TOKEN" in runner.invoke(app, ["secrets", "list"]).stdout
        assert runner.invoke(app, ["secrets", "delete", "TEST_TOKEN"]).exit_code == 0
        assert provider.get("TEST_TOKEN") is None
        assert provider.list_names() == []
        monkeypatch.setenv("TEST_TOKEN", "environment-fallback")
        assert resolver.get("TEST_TOKEN") == "environment-fallback"
    finally:
        subprocess.run(
            ["security", "delete-keychain", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    assert not path.exists()
