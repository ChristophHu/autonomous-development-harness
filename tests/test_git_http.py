import http.server
import os
import socket
import ssl
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_git_integration import git_runtime

from harness.core import Config, Permissions
from harness.git_broker import HttpsTransport
from harness.git_http import (
    _copy_stream,
    _ProxyHandler,
    _tunnel,
    host_allowed,
    https_proxy,
    pinned_connection,
    public_addresses,
    validate_https_url,
)
from harness.tools import ToolRegistry


def native(git, cwd, *args):
    return subprocess.run(
        [git, *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


def test_https_url_normalizes_origin_and_host():
    assert validate_https_url("https://EXAMPLE.com/repo.git", ["example.com"]) == (
        "https://example.com/repo.git"
    )
    assert validate_https_url(
        "https://[2001:4860:4860::8888]/repo", ["2001:4860:4860::8888"]
    ) == ("https://[2001:4860:4860::8888]/repo")


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/repo",
        "https://user@example.com/repo",
        "https://example.com:444/repo",
        "https://example.com/repo?token=secret",
        "https://example.com/repo#branch",
        "https://example.com:invalid/repo",
    ],
)
def test_https_url_rejects_unsafe_origin_forms(url):
    with pytest.raises(PermissionError, match="credential-free TLS origin"):
        validate_https_url(url, ["example.com"])


def test_https_url_rejects_host_outside_allowlist():
    with pytest.raises(PermissionError, match="not allowlisted"):
        validate_https_url("https://outside.example/repo", ["example.com"])


def test_host_allowlist_supports_exact_hosts_and_subdomain_patterns():
    assert host_allowed("git.example.com", ["*.example.com"])
    assert host_allowed("git.example.com", ["GIT.EXAMPLE.COM."])
    assert not host_allowed("example.com", ["*.example.com"])
    assert not host_allowed("evil-example.com", ["*.example.com"])


def test_public_addresses_pins_only_global_ips(monkeypatch):
    monkeypatch.setattr(
        "harness.git_http.socket.getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443)),
        ],
    )
    assert public_addresses("git.example.com", 443) == [
        (socket.AF_INET, socket.SOCK_STREAM, 6, ("8.8.8.8", 443))
    ]


def test_public_addresses_fail_closed_for_resolution_and_private_only(monkeypatch):
    monkeypatch.setattr(
        "harness.git_http.socket.getaddrinfo",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("dns unavailable")),
    )
    with pytest.raises(PermissionError, match="could not be resolved"):
        public_addresses("git.example.com", 443)
    monkeypatch.setattr(
        "harness.git_http.socket.getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 443))
        ],
    )
    with pytest.raises(PermissionError, match="non-public"):
        public_addresses("git.example.com", 443)


def test_pinned_connection_closes_failed_addresses_then_uses_next(monkeypatch):
    first = SimpleNamespace(
        settimeout=lambda _: None,
        connect=lambda _: (_ for _ in ()).throw(OSError("unreachable")),
        close=lambda: None,
    )
    second = SimpleNamespace(settimeout=lambda _: None, connect=lambda _: None)
    sockets = iter((first, second))
    monkeypatch.setattr(
        "harness.git_http.public_addresses",
        lambda *_: [(1, 1, 6, ("a", 443)), (1, 1, 6, ("b", 443))],
    )
    monkeypatch.setattr("harness.git_http.socket.socket", lambda *args: next(sockets))
    assert pinned_connection("git.example.com", 443) is second


def test_pinned_connection_fails_when_all_addresses_fail(monkeypatch):
    monkeypatch.setattr(
        "harness.git_http.public_addresses", lambda *_: [(1, 1, 6, ("a", 443))]
    )

    class FailedSocket:
        def settimeout(self, _timeout):
            pass

        def connect(self, _address):
            raise OSError("refused")

        def close(self):
            pass

    monkeypatch.setattr("harness.git_http.socket.socket", lambda *args: FailedSocket())
    with pytest.raises(OSError, match="could not be reached"):
        pinned_connection("git.example.com", 443)


def test_proxy_denies_unlisted_host(monkeypatch):
    with (
        https_proxy(["git.example.com"]) as port,
        socket.create_connection(("127.0.0.1", port), timeout=3) as client,
    ):
        client.sendall(b"CONNECT evil.example:443 HTTP/1.1\r\n\r\n")
        assert b"403 Forbidden" in client.recv(1024)


def test_proxy_denies_non_tls_ports_and_malformed_requests():
    with https_proxy(["git.example.com"]) as port:
        for request in (
            b"CONNECT git.example.com:80 HTTP/1.1\r\n\r\n",
            b"GET https://git.example.com/ HTTP/1.1\r\n\r\n",
            b"not an http request\r\n\r\n",
        ):
            with socket.create_connection(("127.0.0.1", port), timeout=3) as client:
                client.sendall(request)
                assert b"403 Forbidden" in client.recv(1024)


def test_proxy_returns_bad_request_for_oversized_headers():
    with (
        https_proxy(["git.example.com"]) as port,
        socket.create_connection(("127.0.0.1", port), timeout=3) as client,
    ):
        client.sendall(b"CONNECT git.example.com:443 HTTP/1.1\r\nX: " + b"a" * 9000)
        assert b"400 Bad Request" in client.recv(1024)


def test_proxy_returns_bad_gateway_when_pinned_connection_fails(monkeypatch):
    monkeypatch.setattr(
        "harness.git_http.pinned_connection",
        lambda *_args: (_ for _ in ()).throw(OSError("unreachable")),
    )
    with (
        https_proxy(["git.example.com"]) as port,
        socket.create_connection(("127.0.0.1", port), timeout=3) as client,
    ):
        client.sendall(b"CONNECT git.example.com:443 HTTP/1.1\r\n\r\n")
        assert b"502 Bad Gateway" in client.recv(1024)


def test_proxy_rejects_non_ascii_request_line():
    with (
        https_proxy(["git.example.com"]) as port,
        socket.create_connection(("127.0.0.1", port), timeout=3) as client,
    ):
        client.sendall(b"CONNECT \xff:443 HTTP/1.1\r\n\r\n")
        assert b"403 Forbidden" in client.recv(1024)


def test_proxy_closes_when_request_ends_before_headers():
    with https_proxy(["git.example.com"]) as port:
        client = socket.create_connection(("127.0.0.1", port), timeout=3)
        client.close()


def test_copy_stream_stops_cleanly_when_target_fails():
    class Source:
        calls = 0

        def recv(self, _size):
            self.calls += 1
            return b"data" if self.calls == 1 else b""

    class Target:
        def sendall(self, _payload):
            raise BrokenPipeError("closed")

        def shutdown(self, _how):
            raise OSError("already closed")

    _copy_stream(Source(), Target())


def test_tunnel_returns_when_socket_closes():
    left, right = socket.socketpair()
    left.shutdown(socket.SHUT_WR)
    right.shutdown(socket.SHUT_WR)
    try:
        _tunnel(left, right)
    finally:
        left.close()
        right.close()


def test_proxy_tunnels_end_to_end_to_pinned_connection(monkeypatch):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    upstream_port = listener.getsockname()[1]
    accepted = []

    def serve():
        connection, _address = listener.accept()
        with connection:
            accepted.append(connection.recv(1024))
            connection.sendall(b"reply")

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    monkeypatch.setattr(
        "harness.git_http.pinned_connection",
        lambda _host, _port: socket.create_connection(("127.0.0.1", upstream_port)),
    )
    with (
        https_proxy(["git.example.com"]) as proxy_port,
        socket.create_connection(("127.0.0.1", proxy_port), timeout=3) as client,
    ):
        client.sendall(b"CONNECT git.example.com:443 HTTP/1.1\r\n\r\n")
        assert b"200 Connection Established" in client.recv(1024)
        client.sendall(b"hello")
        assert client.recv(1024) == b"reply"
    listener.close()
    thread.join(timeout=3)
    assert accepted == [b"hello"]


@pytest.mark.parametrize(
    "args",
    [
        ["clone", "https://outside.example/repo.git", "copy"],
        ["fetch", "origin", "main"],
    ],
)
def test_https_transport_requires_an_allowlisted_url(tmp_path, args):
    with pytest.raises(PermissionError, match="not allowlisted"):
        HttpsTransport.prepare(
            args,
            tmp_path,
            {},
            "https://outside.example/repo.git",
            ["git.example.com"],
        )


def test_https_transport_requires_explicit_pull_contract(tmp_path):
    with pytest.raises(PermissionError, match="ff-only"):
        HttpsTransport.prepare(
            ["pull", "origin", "main"],
            tmp_path,
            {},
            "https://git.example.com/repo.git",
            ["git.example.com"],
        )


def test_https_transport_requires_named_remote_for_fetch(tmp_path):
    with pytest.raises(PermissionError, match="named remote"):
        HttpsTransport.prepare(
            ["fetch", "https://git.example.com/repo.git", "main"],
            tmp_path,
            {},
            allowed_hosts=["git.example.com"],
        )


def test_https_transport_rejects_ca_directories(tmp_path):
    with pytest.raises(PermissionError, match="CA bundle"):
        HttpsTransport.prepare(
            ["clone", "https://git.example.com/repo.git", "copy"],
            tmp_path,
            {},
            allowed_hosts=["git.example.com"],
            ca_bundle=tmp_path,
        )


def test_https_transport_rejects_git_helper_lookup_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "harness.git_broker.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 70, "", ""),
    )
    with pytest.raises(PermissionError, match="helper lookup"):
        HttpsTransport.prepare(
            ["clone", "https://git.example.com/repo.git", "copy"],
            tmp_path,
            {},
            allowed_hosts=["git.example.com"],
        )


def test_tool_registry_loads_git_host_allowlist_and_ca_bundle(tmp_path):
    config = Config()
    config.data["tools"] = {
        "permissions": {"git": "read"},
        "git": {
            "allowed_hosts": ["git.example.com"],
            "ca_bundle": "/tmp/roots.pem",
            "credentials": {
                "git.example.com": {"username": "oauth2", "secret_name": "GIT_TOKEN"}
            },
            "ssh": {
                "allowed_hosts": ["ssh.example.com"],
                "allowed_ports": [22],
                "host_keys": {"ssh.example.com": ["ssh-ed25519 AAAA"]},
                "credentials": {
                    "ssh.example.com": {
                        "username": "git",
                        "fingerprint": "SHA256:" + "A" * 43,
                    }
                },
            },
        },
    }
    registry = ToolRegistry(Permissions(config), workspace=tmp_path)
    assert registry.executor.git_allow_hosts == {"git.example.com"}
    assert registry.executor.git_ca_bundle == "/tmp/roots.pem"
    assert (
        registry.executor.git_credentials["git.example.com"]["secret_name"]
        == "GIT_TOKEN"
    )
    assert registry.executor.git_ssh_allowed_hosts == {"ssh.example.com"}
    assert registry.executor.git_ssh_allowed_ports == (22,)
    assert registry.executor.git_ssh_credentials["ssh.example.com"]["username"] == "git"


def test_https_transport_rejects_missing_git_helper(tmp_path, monkeypatch):
    helper_dir = subprocess.run(
        ["/opt/homebrew/bin/git", "--exec-path"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    monkeypatch.setattr(
        "harness.git_broker.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, helper_dir, ""
        ),
    )
    original = Path.is_file
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda path: False if path.name == "git-remote-https" else original(path),
    )
    with pytest.raises(PermissionError, match="helper is unavailable"):
        HttpsTransport.prepare(
            ["clone", "https://git.example.com/repo.git", "copy"],
            tmp_path,
            {},
            allowed_hosts=["git.example.com"],
        )


def test_https_transport_fails_closed_when_config_cannot_be_checked(
    tmp_path, monkeypatch
):
    original = subprocess.run
    calls = 0

    def run(command, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return original(command, **kwargs)
        return subprocess.CompletedProcess(command, 70, "", "config failed")

    monkeypatch.setattr("harness.git_broker.subprocess.run", run)
    with pytest.raises(PermissionError, match="configuration could not be verified"):
        HttpsTransport.prepare(
            ["clone", "https://git.example.com/repo.git", "copy"],
            tmp_path,
            {},
            allowed_hosts=["git.example.com"],
        )


def test_https_push_requires_a_secret_credential(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "harness.git_broker.isolated_command", lambda cmd, *_a, **_k: cmd
    )
    monkeypatch.setattr("harness.security.SecretResolver.get", lambda *_: "secret")
    transport = HttpsTransport.prepare(
        ["push", "origin", "main:refs/heads/main"],
        tmp_path,
        {"PATH": "/opt/homebrew/bin:/usr/bin:/bin"},
        "https://git.example.com/repo.git",
        ["git.example.com"],
        credentials={
            "git.example.com": {"username": "oauth2", "secret_name": "GIT_TOKEN"}
        },
    )
    assert transport.push_ref == ("main", False)
    assert transport.auth_username == "oauth2"
    assert transport.auth_secret == "secret"


def test_https_transport_prepares_explicit_bearer_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "harness.git_broker.isolated_command", lambda cmd, *_a, **_k: cmd
    )
    monkeypatch.setattr(
        "harness.security.SecretResolver.get", lambda *_: "bearer-token"
    )
    transport = HttpsTransport.prepare(
        ["fetch", "origin", "main"],
        tmp_path,
        {"PATH": "/opt/homebrew/bin:/usr/bin:/bin"},
        "https://git.example.com/repo.git",
        ["git.example.com"],
        credentials={"git.example.com": {"mode": "bearer", "secret_name": "GIT_TOKEN"}},
    )
    assert transport.auth_mode == "bearer"
    assert transport.auth_username is None
    assert transport.auth_secret == "bearer-token"
    assert "bearer-token" not in repr(transport)
    from dataclasses import replace

    with pytest.raises(PermissionError, match="configuration is invalid"):
        replace(transport, auth_mode="unsupported")._authorization_header()


def test_https_secret_is_resolved_again_for_each_transport_run(tmp_path, monkeypatch):
    tokens = iter(("first-token", "rotated-token"))
    monkeypatch.setattr(
        "harness.git_broker.isolated_command", lambda cmd, *_a, **_k: cmd
    )
    monkeypatch.setattr("harness.security.SecretResolver.get", lambda *_: next(tokens))
    transports = [
        HttpsTransport.prepare(
            ["fetch", "origin", "main"],
            tmp_path,
            {"PATH": "/opt/homebrew/bin:/usr/bin:/bin"},
            "https://git.example.com/repo.git",
            ["git.example.com"],
            credentials={
                "git.example.com": {"mode": "bearer", "secret_name": "GIT_TOKEN"}
            },
        )
        for _ in range(2)
    ]
    assert [item.auth_secret for item in transports] == [
        "first-token",
        "rotated-token",
    ]
    assert [item._authorization_header() for item in transports] == [
        "Authorization: Bearer first-token",
        "Authorization: Bearer rotated-token",
    ]


def test_https_push_rejects_missing_secret_and_bad_credential_config(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("harness.security.SecretResolver.get", lambda *_: None)
    credential = {"username": "oauth2", "secret_name": "GIT_TOKEN"}
    with pytest.raises(PermissionError, match="credential is unavailable"):
        HttpsTransport.prepare(
            ["push", "origin", "main:refs/heads/main"],
            tmp_path,
            {},
            "https://git.example.com/repo.git",
            ["git.example.com"],
            credentials={"git.example.com": credential},
        )
    for invalid in (
        {"username": "bad:user", "secret_name": "GIT_TOKEN"},
        {"username": "oauth2", "secret_name": "bad-name"},
        {"username": "oauth2", "secret_name": "GIT_TOKEN", "password": "x"},
        {"mode": "digest", "username": "oauth2", "secret_name": "GIT_TOKEN"},
        {"mode": "bearer", "username": "oauth2", "secret_name": "GIT_TOKEN"},
        {"mode": "basic", "secret_name": "GIT_TOKEN"},
        {"mode": "bearer", "secret_name": "GIT_TOKEN", "extra": True},
        ["not", "a", "mapping"],
    ):
        with pytest.raises(PermissionError, match="configuration is invalid"):
            HttpsTransport.prepare(
                ["fetch", "origin"],
                tmp_path,
                {},
                "https://git.example.com/repo.git",
                ["git.example.com"],
                credentials={"git.example.com": invalid},
            )
    with pytest.raises(PermissionError, match="requires a configured credential"):
        HttpsTransport.prepare(
            ["push", "origin", "main:refs/heads/main"],
            tmp_path,
            {},
            "https://git.example.com/repo.git",
            ["git.example.com"],
            credentials={
                "other.example.com": {
                    "mode": "bearer",
                    "secret_name": "GIT_TOKEN",
                }
            },
        )


@pytest.mark.parametrize("followup_status", [0, 1])
def test_https_delete_push_reconciles_tracking_and_redacts_secrets(
    tmp_path, monkeypatch, followup_status
):
    import base64
    from contextlib import contextmanager

    from harness.git_broker import HttpsTransport

    token = "unit-secret-token"
    encoded = base64.b64encode(f"oauth2:{token}".encode()).decode()
    header = f"Authorization: Basic {encoded}"
    calls = []

    @contextmanager
    def proxy(_hosts):
        yield 45678

    class Client:
        returncode = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def communicate(self, timeout):
            assert timeout == 120
            return f"{token} {encoded} {header}".encode(), b""

        def poll(self):
            return self.returncode

    def popen(_command, **kwargs):
        assert kwargs["env"]["GIT_CONFIG_VALUE_0"] == header
        calls.append(kwargs)
        return Client()

    def run(command, **kwargs):
        calls.append(command)
        assert command == ["git", "update-ref", "-d", "refs/remotes/origin/topic"]
        return subprocess.CompletedProcess(command, followup_status, "", "")

    monkeypatch.setattr("harness.git_broker.https_proxy", proxy)
    monkeypatch.setattr(
        "harness.git_broker.isolated_command", lambda cmd, *_a, **_k: cmd
    )
    monkeypatch.setattr("harness.git_broker.subprocess.Popen", popen)
    monkeypatch.setattr("harness.git_broker.subprocess.run", run)
    transport = HttpsTransport(
        ("push", "origin", "--delete", "topic"),
        tmp_path,
        "https://git.example.com/repo.git",
        1,
        tmp_path / "git-remote-https",
        {},
        auth_username="oauth2",
        auth_secret=token,
        push_ref=("topic", True),
    )
    assert token not in repr(transport)
    result = transport.run()
    assert token not in result.stdout + result.stderr
    assert encoded not in result.stdout + result.stderr
    assert header not in result.stdout + result.stderr
    assert result.returncode == followup_status
    assert len(calls) == 2


def test_https_bearer_header_is_scoped_to_git_process_and_redacted(
    tmp_path, monkeypatch
):
    from contextlib import contextmanager

    from harness.git_broker import HttpsTransport

    token = "bearer-secret-token"
    header = f"Authorization: Bearer {token}"

    @contextmanager
    def proxy(_hosts):
        yield 45678

    class Client:
        returncode = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def communicate(self, timeout):
            assert timeout == 120
            return f"{token} {header}".encode(), b""

        def poll(self):
            return self.returncode

    captured = {}

    def popen(command, **kwargs):
        captured.update(command=command, env=kwargs["env"])
        return Client()

    monkeypatch.setattr("harness.git_broker.https_proxy", proxy)
    monkeypatch.setattr(
        "harness.git_broker.isolated_command", lambda cmd, *_a, **_k: cmd
    )
    monkeypatch.setattr("harness.git_broker.subprocess.Popen", popen)
    transport = HttpsTransport(
        ("fetch", "origin", "main"),
        tmp_path,
        "https://git.example.com/repo.git",
        1,
        tmp_path / "git-remote-https",
        {},
        auth_secret=token,
        auth_mode="bearer",
    )
    result = transport.run()
    assert captured["env"]["GIT_CONFIG_VALUE_0"] == header
    assert token not in captured["command"]
    assert token not in result.stdout + result.stderr
    assert header not in result.stdout + result.stderr


def test_https_auth_failure_is_normalized_and_secret_safe(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from harness.git_broker import HttpsTransport

    @contextmanager
    def proxy(_hosts):
        yield 45678

    class Client:
        returncode = 128

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def communicate(self, timeout):
            return b"", b"fatal: requested URL returned error: 403 bearer-secret"

        def poll(self):
            return self.returncode

    monkeypatch.setattr("harness.git_broker.https_proxy", proxy)
    monkeypatch.setattr(
        "harness.git_broker.isolated_command", lambda cmd, *_a, **_k: cmd
    )
    monkeypatch.setattr(
        "harness.git_broker.subprocess.Popen", lambda *_a, **_k: Client()
    )
    result = HttpsTransport(
        ("fetch", "origin", "main"),
        tmp_path,
        "https://git.example.com/repo.git",
        1,
        tmp_path / "git-remote-https",
        {},
        auth_secret="bearer-secret",
        auth_mode="bearer",
    ).run()
    assert result.returncode == 128
    assert result.stderr == "HTTPS Git authentication failed (HTTP 403)."
    assert "bearer-secret" not in result.stderr


def test_https_push_without_credentials_keeps_the_approval_unused(
    tmp_path, monkeypatch
):
    store, harness, task, git, repo = git_runtime(tmp_path, monkeypatch)
    remote = tmp_path / "remote.git"
    native(git, tmp_path, "init", "--bare", str(remote))
    native(git, repo, "remote", "add", "origin", str(remote))
    native(
        git,
        repo,
        "remote",
        "set-url",
        "--push",
        "origin",
        "https://git.example.com/remote.git",
    )
    harness.tools.executor.git_allow_hosts = {"git.example.com"}
    task = store.create(task)
    args = ["push", "origin", "main:refs/heads/main"]
    target = harness.tools.git_target(task.id, args)
    question = harness.approvals.request(target)
    store.answer(question, "approve", task.id)
    grant = harness.approvals.issue(task.id, question, target.action, target)
    with pytest.raises(PermissionError, match="requires a configured credential"):
        harness.tools.git(args, approval=grant, task_id=task.id)
    assert grant.permits(target.action, target)


def test_sandbox_profile_allows_only_the_local_proxy_and_trusted_helper(
    tmp_path, monkeypatch
):
    from harness.isolation import isolated_command

    monkeypatch.setattr("harness.isolation.sys.platform", "darwin")
    monkeypatch.setattr(
        "harness.isolation.shutil.which", lambda *args, **kwargs: "/usr/bin/git"
    )
    helper = tmp_path / "git-remote-https"
    helper.touch()
    command = isolated_command(
        ["git", "fetch", "https://git.example.com/repo"],
        tmp_path,
        git=True,
        network_proxy=43127,
        git_helpers=(helper,),
    )
    profile = command[2]
    assert '(allow network-outbound (remote ip "localhost:43127"))' in profile
    assert f'(allow process-exec (literal "{helper}"))' in profile
    assert "(allow network-outbound)" not in profile


@pytest.mark.parametrize("port", [0, -1, 65536, "443"])
def test_sandbox_rejects_invalid_proxy_ports(tmp_path, monkeypatch, port):
    from harness.isolation import isolated_command

    monkeypatch.setattr("harness.isolation.sys.platform", "darwin")
    monkeypatch.setattr(
        "harness.isolation.shutil.which", lambda *args, **kwargs: "/usr/bin/git"
    )
    with pytest.raises(ValueError, match="proxy port"):
        isolated_command(["git", "fetch"], tmp_path, git=True, network_proxy=port)


@pytest.mark.parametrize(
    ("config_key", "config_value"),
    [
        ("http.https://git.example.com/.extraHeader", "Authorization: Bearer secret"),
        ("http.https://git.example.com/.cookieFile", "/tmp/cookies.txt"),
        ("credential.https://git.example.com.helper", "store"),
        ("credential.https://git.example.com.username", "automation"),
        ("url.https://evil.example/.insteadOf", "https://git.example.com/"),
    ],
)
def test_https_transport_rejects_local_auth_and_url_overrides(
    tmp_path, monkeypatch, config_key, config_value
):
    _store, harness, _task, git, repo = git_runtime(tmp_path, monkeypatch)
    native = subprocess.run
    native([git, "config", config_key, config_value], cwd=repo, check=True)
    with pytest.raises(PermissionError, match="credential or URL override"):
        HttpsTransport.prepare(
            ["fetch", "origin", "main"],
            repo,
            harness.tools.executor._environment(),
            "https://git.example.com/repo.git",
            ["git.example.com"],
        )


def _https_git_server(
    git, remote, cert, key, redirect_to=None, credential=None, auth_status=401
):
    class Handler(http.server.BaseHTTPRequestHandler):
        def _serve_git(self):
            if credential:
                import base64

                if len(credential) == 2:
                    scheme = "Basic"
                    principal, secret = credential
                else:
                    scheme, principal, secret = credential
                value = (
                    base64.b64encode(f"{principal}:{secret}".encode()).decode("ascii")
                    if scheme == "Basic"
                    else secret
                )
                expected = f"{scheme} {value}"
                if self.headers.get("Authorization") != expected:
                    self.send_response(auth_status)
                    self.send_header("WWW-Authenticate", f'{scheme} realm="test"')
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
            if redirect_to:
                self.send_response(302)
                self.send_header("Location", redirect_to)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            env = {
                **os.environ,
                "GIT_PROJECT_ROOT": str(remote.parent),
                "GIT_HTTP_EXPORT_ALL": "1",
                "PATH_INFO": self.path.partition("?")[0],
                "QUERY_STRING": self.path.partition("?")[2],
                "REQUEST_METHOD": self.command,
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                "CONTENT_LENGTH": str(len(body)),
                "REMOTE_ADDR": "127.0.0.1",
            }
            if credential:
                env["REMOTE_USER"] = principal
            if self.headers.get("Git-Protocol"):
                env["HTTP_GIT_PROTOCOL"] = self.headers["Git-Protocol"]
            result = subprocess.run(
                [git, "http-backend"],
                input=body,
                capture_output=True,
                env=env,
                check=False,
            )
            headers, separator, content = result.stdout.partition(b"\r\n\r\n")
            if not separator:
                self.send_error(502)
                return
            status = 200
            parsed_headers = []
            for line in headers.decode().split("\r\n"):
                name, colon, value = line.partition(":")
                if not colon:
                    continue
                if name.lower() == "status":
                    status = int(value.strip().split()[0])
                else:
                    parsed_headers.append((name, value.strip()))
            self.send_response(status)
            for name, value in parsed_headers:
                if name.lower() not in {
                    "connection",
                    "transfer-encoding",
                    "date",
                    "server",
                }:
                    self.send_header(name, value)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        do_GET = _serve_git
        do_POST = _serve_git

        def log_message(self, *_args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _test_certificate(tmp_path):
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    subprocess.run(
        [
            "/usr/bin/openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "1",
            "-subj",
            "/CN=127.0.0.1",
            "-addext",
            "subjectAltName=IP:127.0.0.1",
        ],
        check=True,
        capture_output=True,
    )
    return cert, key


def test_https_clone_uses_loopback_proxy_and_trusted_test_ca(tmp_path, monkeypatch):
    _store, harness, _task, git, repo = git_runtime(tmp_path, monkeypatch)
    remote = tmp_path / "remote.git"
    subprocess.run([git, "clone", "--bare", str(repo), str(remote)], check=True)
    cert, key = _test_certificate(tmp_path)
    server, thread = _https_git_server(git, remote, cert, key)
    monkeypatch.setattr(
        "harness.git_http.pinned_connection",
        lambda _host, _port: socket.create_connection(server.server_address),
    )
    harness.tools.executor.git_allow_hosts = {"127.0.0.1"}
    harness.tools.executor.git_ca_bundle = str(cert)
    try:
        result = harness.tools.git(
            ["clone", "https://127.0.0.1/remote.git", "https-copy"]
        )
        assert result.returncode == 0, result.stderr
        result = harness.tools.git(["fetch", "origin"], cwd="https-copy")
        assert result.returncode == 0, result.stderr
        native(git, repo, "branch", "topic/https")
        native(git, repo, "push", str(remote), "topic/https")
        native(git, repo, "tag", "release-https")
        native(git, repo, "push", str(remote), "refs/tags/release-https")
        result = harness.tools.git(
            ["fetch", "--no-tags", "origin", "main", "topic/https"],
            cwd="https-copy",
        )
        assert result.returncode == 0, result.stderr
        assert native(
            git,
            repo / "https-copy",
            "show-ref",
            "--verify",
            "refs/remotes/origin/topic/https",
        )
        result = harness.tools.git(["fetch", "--tags", "origin"], cwd="https-copy")
        assert result.returncode == 0, result.stderr
        assert native(
            git, repo / "https-copy", "show-ref", "--verify", "refs/tags/release-https"
        )
        native(git, remote, "branch", "-D", "topic/https")
        result = harness.tools.git(
            ["fetch", "--prune", "--no-tags", "origin"], cwd="https-copy"
        )
        assert result.returncode == 0, result.stderr
        assert (
            subprocess.run(
                [git, "show-ref", "--verify", "refs/remotes/origin/topic/https"],
                cwd=repo / "https-copy",
                capture_output=True,
                check=False,
            ).returncode
            != 0
        )
        assert native(
            git, repo / "https-copy", "show-ref", "--verify", "refs/tags/release-https"
        )
        native(git, repo, "config", "user.name", "Test")
        native(git, repo, "config", "user.email", "test@example.invalid")
        native(git, repo, "commit", "--allow-empty", "-m", "https update")
        native(git, repo, "push", str(remote), "main")
        result = harness.tools.git(
            ["pull", "--ff-only", "origin", "main"], cwd="https-copy"
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize(
    ("scheme", "credential_config"),
    [
        ("Basic", {"username": "oauth2", "secret_name": "TEST_GIT_TOKEN"}),
        (
            "Bearer",
            {"mode": "bearer", "secret_name": "TEST_GIT_TOKEN"},
        ),
    ],
)
def test_https_credentials_and_approved_push_are_brokered_without_output_leaks(
    tmp_path, monkeypatch, scheme, credential_config
):
    _store, harness, task, git, repo = git_runtime(tmp_path, monkeypatch)
    remote = tmp_path / "remote.git"
    subprocess.run([git, "clone", "--bare", str(repo), str(remote)], check=True)
    cert, key = _test_certificate(tmp_path)
    token = "opaque-test-token-value"
    username = "oauth2"
    server, thread = _https_git_server(
        git, remote, cert, key, credential=(scheme, username, token)
    )
    monkeypatch.setattr(
        "harness.git_http.pinned_connection",
        lambda _host, _port: socket.create_connection(server.server_address),
    )
    executor = harness.tools.executor
    executor.git_allow_hosts = {"127.0.0.1"}
    executor.git_ca_bundle = str(cert)
    executor.git_credentials = {"127.0.0.1": credential_config}
    monkeypatch.setattr("harness.security.SecretResolver.get", lambda *_: token)
    process_envs = []
    original_popen = subprocess.Popen

    def observe_popen(command, **kwargs):
        if kwargs.get("start_new_session"):
            process_envs.append((command, kwargs.get("env", {})))
        return original_popen(command, **kwargs)

    monkeypatch.setattr("harness.git_broker.subprocess.Popen", observe_popen)
    try:
        result = harness.tools.git(
            ["clone", "https://127.0.0.1/remote.git", "https-copy"],
            cwd=repo,
        )
        assert result.returncode == 0, result.stderr
        assert token not in result.stdout + result.stderr
        copy = repo / "https-copy"
        native(git, copy, "config", "user.name", "Test")
        native(
            git,
            copy,
            "config",
            "user.email",
            "test@example.invalid",
        )
        (copy / "credential-push.txt").write_text("pushed\n")
        native(git, copy, "add", "credential-push.txt")
        native(git, copy, "commit", "-m", "credential push")
        task = _store.create(task)
        args = ["push", "origin", "main:refs/heads/main"]
        target = harness.tools.git_target(task.id, args, copy)
        question = harness.approvals.request(target)
        _store.answer(question, "approve", task.id)
        grant = harness.approvals.issue(task.id, question, target.action, target)
        result = harness.tools.git(args, cwd=copy, approval=grant, task_id=task.id)
        assert result.returncode == 0, result.stderr
        assert token not in result.stdout + result.stderr
        assert native(git, remote, "rev-parse", "refs/heads/main") == native(
            git, copy, "rev-parse", "HEAD"
        )
        assert native(git, copy, "rev-parse", "refs/remotes/origin/main") == native(
            git, remote, "rev-parse", "refs/heads/main"
        )
        import base64

        expected_auth = (
            base64.b64encode(f"{username}:{token}".encode()).decode()
            if scheme == "Basic"
            else token
        )
        audit_text = str(_store.events.list(task.id))
        assert token not in audit_text
        assert expected_auth not in audit_text
        assert process_envs
        for command, env in process_envs:
            assert token not in command
            assert env.get("GIT_CONFIG_VALUE_0") == (
                f"Authorization: {scheme} {expected_auth}"
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert result.returncode == 0, result.stderr
    assert (repo / "https-copy/addition.py").exists()
    assert native(git, repo / "https-copy", "rev-parse", "HEAD") == native(
        git, remote, "rev-parse", "refs/heads/main"
    )


@pytest.mark.parametrize("auth_status", [401, 403])
def test_https_bearer_auth_errors_are_clear_and_never_leak_token(
    tmp_path, monkeypatch, auth_status
):
    _store, harness, _task, git, repo = git_runtime(tmp_path, monkeypatch)
    remote = tmp_path / "protected.git"
    subprocess.run([git, "clone", "--bare", str(repo), str(remote)], check=True)
    cert, key = _test_certificate(tmp_path)
    server, thread = _https_git_server(
        git,
        remote,
        cert,
        key,
        credential=("Bearer", "automation", "expected-token"),
        auth_status=auth_status,
    )
    monkeypatch.setattr(
        "harness.git_http.pinned_connection",
        lambda _host, _port: socket.create_connection(server.server_address),
    )
    executor = harness.tools.executor
    executor.git_allow_hosts = {"127.0.0.1"}
    executor.git_ca_bundle = str(cert)
    executor.git_credentials = {
        "127.0.0.1": {"mode": "bearer", "secret_name": "TEST_GIT_TOKEN"}
    }
    monkeypatch.setattr(
        "harness.security.SecretResolver.get", lambda *_: "wrong-secret-token"
    )
    try:
        result = harness.tools.git(
            ["clone", "https://127.0.0.1/protected.git", "protected-copy"],
            cwd=repo,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert result.returncode != 0
    assert "authentication failed" in result.stderr.lower()
    assert "wrong-secret-token" not in result.stdout + result.stderr
    assert "Authorization: Bearer" not in result.stdout + result.stderr


def test_https_clone_without_ca_bundle_keeps_tls_verification(tmp_path, monkeypatch):
    _store, harness, _task, git, repo = git_runtime(tmp_path, monkeypatch)
    remote = tmp_path / "remote.git"
    native(git, tmp_path, "clone", "--bare", str(repo), str(remote))
    cert, key = _test_certificate(tmp_path)
    server, thread = _https_git_server(git, remote, cert, key)
    monkeypatch.setattr(
        "harness.git_http.pinned_connection",
        lambda _host, _port: socket.create_connection(server.server_address),
    )
    harness.tools.executor.git_allow_hosts = {"127.0.0.1"}
    try:
        result = harness.tools.git(
            ["clone", "https://127.0.0.1/remote.git", "untrusted-copy"]
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert result.returncode != 0
    assert "certificate" in result.stderr.lower()


def test_https_redirect_is_not_followed_to_another_host(tmp_path, monkeypatch):
    _store, harness, _task, git, repo = git_runtime(tmp_path, monkeypatch)
    remote = tmp_path / "remote.git"
    native(git, tmp_path, "clone", "--bare", str(repo), str(remote))
    cert, key = _test_certificate(tmp_path)
    server, thread = _https_git_server(
        git, remote, cert, key, "https://redirect.example/other.git"
    )
    connected = []

    def connect(host, _port):
        connected.append(host)
        return socket.create_connection(server.server_address)

    monkeypatch.setattr("harness.git_http.pinned_connection", connect)
    harness.tools.executor.git_allow_hosts = {"127.0.0.1"}
    harness.tools.executor.git_ca_bundle = str(cert)
    try:
        result = harness.tools.git(
            ["clone", "https://127.0.0.1/remote.git", "redirect-copy"]
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert result.returncode != 0
    assert connected == ["127.0.0.1"]


def test_tunnel_stops_on_socket_error():
    class BrokenSocket:
        def settimeout(self, _value):
            pass

        def recv(self, _size):
            raise ConnectionResetError("closed")

        def shutdown(self, _how):
            pass

    client = BrokenSocket()
    upstream = BrokenSocket()
    _tunnel(client, upstream)


def test_proxy_handler_returns_when_client_closes_before_headers():
    handler = _ProxyHandler.__new__(_ProxyHandler)
    handler.request = SimpleNamespace(recv=lambda _size: b"")
    handler.handle()
