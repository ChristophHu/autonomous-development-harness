import base64
import hashlib
import os
import pwd
import signal
import socket
import stat
import subprocess
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness.git_broker import SshTransport, _ssh_config_quote
from harness.git_ssh import (
    _decode_key,
    _SshTunnelHandler,
    agent_identities,
    key_fingerprint,
    parse_ssh_url,
    resolve_ssh_addresses,
    ssh_proxy,
    validate_host_keys,
)


def ssh_key(key_type="ssh-ed25519", body=b"test-public-key"):
    key_type_bytes = key_type.encode()
    blob = len(key_type_bytes).to_bytes(4, "big") + key_type_bytes + body
    return f"{key_type} {base64.b64encode(blob).decode()} test-key"


def native(git, cwd, *args):
    return subprocess.run(
        [git, *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def bind_agent_socket():
    temporary = tempfile.TemporaryDirectory(prefix="ha-", dir="/tmp")
    socket_path = Path(temporary.name) / "agent.sock"
    server = socket.socket(socket.AF_UNIX)
    try:
        server.bind(str(socket_path))
    except BaseException:
        server.close()
        temporary.cleanup()
        raise
    return temporary, server, socket_path


def test_parse_ssh_url_normalizes_scp_and_uri_forms():
    assert parse_ssh_url("git@GIT.EXAMPLE.com:team/repo.git", ["git.example.com"]) == (
        "ssh://git@git.example.com/team/repo.git",
        "git.example.com",
        "git",
        22,
        "team/repo.git",
    )
    assert parse_ssh_url(
        "ssh://git@git.example.com:2222/team/repo.git",
        ["git.example.com"],
        [22, 2222],
    ) == (
        "ssh://git@git.example.com:2222/team/repo.git",
        "git.example.com",
        "git",
        2222,
        "team/repo.git",
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://git.example.com/repo.git",
        "ssh://user:password@git.example.com/repo.git",
        "ssh://git.example.com:2222/repo.git",
        "git@git.example.com:../repo.git",
        "git@git.example.com:repo.git;touch%20/tmp/pwned",
        "git@outside.example:team/repo.git",
        "ssh://git.example.com/team/repo.git?token=secret",
        "git@git.example.com:team//repo.git",
    ],
)
def test_parse_ssh_url_rejects_unsafe_or_unapproved_urls(url):
    with pytest.raises(PermissionError):
        parse_ssh_url(url, ["git.example.com"])


@pytest.mark.parametrize(
    "url",
    [
        None,
        "",
        "https://git.example.com/a",
        "git@example.com:../repo",
        "ssh://u!@git.example.com/repo",
    ],
)
def test_ssh_url_rejects_invalid_scheme_types_username_and_path(url):
    with pytest.raises(PermissionError):
        parse_ssh_url(url, ["git.example.com"])


def test_ssh_url_maps_invalid_ports_and_idna_errors(monkeypatch):
    with pytest.raises(PermissionError, match="port"):
        parse_ssh_url("ssh://git.example.com:bad/repo", ["git.example.com"])

    monkeypatch.setattr(
        "harness.git_ssh.ipaddress.ip_address",
        lambda *_: (_ for _ in ()).throw(ValueError()),
    )
    # An invalid Unicode hostname is rejected before any SSH process starts.
    with pytest.raises(PermissionError, match="host is invalid"):
        parse_ssh_url("ssh://\ud800/repo", ["git.example.com"])


@pytest.mark.parametrize(
    "blob", [b"", b"\0\0", b"\0\0\0\0", b"\0\0\0\x05ab", b"\0\0\0\x01\xff"]
)
def test_decode_key_rejects_truncated_and_non_ascii_blobs(blob):
    encoded = base64.b64encode(blob).decode()
    with pytest.raises(PermissionError):
        _decode_key(f"ssh-ed25519 {encoded}")


def test_decode_key_rejects_embedded_type_mismatch():
    line = ssh_key("ssh-rsa")
    with pytest.raises(PermissionError, match="does not match"):
        _decode_key(line.replace("ssh-rsa", "ssh-ed25519", 1))


def test_key_fingerprint_validates_wire_format_and_key_type():
    line = ssh_key()
    blob = base64.b64decode(line.split()[1])
    expected = "SHA256:" + base64.b64encode(
        hashlib.sha256(blob).digest()
    ).decode().rstrip("=")
    assert key_fingerprint(line) == expected
    with pytest.raises(PermissionError, match="public key"):
        key_fingerprint("not-a-key")
    with pytest.raises(PermissionError, match="key type"):
        key_fingerprint(ssh_key("ssh-dss"))
    with pytest.raises(PermissionError, match="key data"):
        key_fingerprint("ssh-ed25519 invalid")


def test_validate_host_keys_requires_nonempty_supported_keys():
    keys = validate_host_keys("git.example.com", [ssh_key(), ssh_key()])
    assert keys == [ssh_key().split()[:2]] * 2
    for invalid in ([], "ssh-ed25519 bad", ["ssh-dss AAAA"], [ssh_key() + "\nnext"]):
        with pytest.raises(PermissionError, match="host key"):
            validate_host_keys("git.example.com", invalid)


def test_resolve_ssh_addresses_filters_private_and_sorts(monkeypatch):
    monkeypatch.setattr(
        "harness.git_ssh.socket.getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 22)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 22)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", 22)),
        ],
    )
    assert resolve_ssh_addresses("git.example.com", 22) == ["1.1.1.1", "8.8.8.8"]


def test_resolve_ssh_addresses_fails_closed(monkeypatch):
    monkeypatch.setattr(
        "harness.git_ssh.socket.getaddrinfo",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("dns")),
    )
    with pytest.raises(PermissionError, match="could not be resolved"):
        resolve_ssh_addresses("git.example.com", 22)


def test_resolve_ssh_addresses_ignores_malformed_dns_records(monkeypatch):
    monkeypatch.setattr(
        "harness.git_ssh.socket.getaddrinfo",
        lambda *_args, **_kwargs: [(0, 0, 0, "", ("not-an-ip", 22))],
    )
    with pytest.raises(PermissionError, match="non-public"):
        resolve_ssh_addresses("git.example.com", 22)
    monkeypatch.setattr(
        "harness.git_ssh.socket.getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 22))
        ],
    )
    with pytest.raises(PermissionError, match="non-public"):
        resolve_ssh_addresses("git.example.com", 22)


def test_agent_identities_checks_socket_and_lists_keys(tmp_path, monkeypatch):
    temporary, server, socket_path = bind_agent_socket()
    monkeypatch.setattr(
        "harness.git_ssh.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["ssh-add"], 0, ssh_key() + "\n", ""
        ),
    )
    try:
        assert agent_identities(str(socket_path)) == [ssh_key().split()[:2]]
    finally:
        server.close()
        temporary.cleanup()


def test_agent_identities_rejects_bad_or_unavailable_agent(tmp_path, monkeypatch):
    with pytest.raises(PermissionError, match="agent socket"):
        agent_identities(str(tmp_path / "missing.sock"))
    temporary, server, socket_path = bind_agent_socket()
    monkeypatch.setattr(
        "harness.git_ssh.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(["ssh-add"], 1, "", ""),
    )
    with pytest.raises(PermissionError, match="no available identities"):
        agent_identities(str(socket_path))
    server.close()
    temporary.cleanup()


def test_agent_identities_rejects_bad_path_socket_owner_and_empty_agent(
    tmp_path, monkeypatch
):
    with pytest.raises(PermissionError, match="path"):
        agent_identities("relative.sock")
    temporary, sock, path = bind_agent_socket()
    original_lstat = os.lstat
    try:
        monkeypatch.setattr(
            "harness.git_ssh.os.lstat",
            lambda *_: SimpleNamespace(st_mode=stat.S_IFREG, st_uid=os.getuid()),
        )
        with pytest.raises(PermissionError, match="same-user"):
            agent_identities(str(path))
        monkeypatch.setattr(
            "harness.git_ssh.os.lstat",
            lambda *_: SimpleNamespace(st_mode=stat.S_IFSOCK, st_uid=os.getuid() + 1),
        )
        with pytest.raises(PermissionError, match="same-user"):
            agent_identities(str(path))
        monkeypatch.setattr("harness.git_ssh.os.lstat", original_lstat)
        monkeypatch.setattr(
            "harness.git_ssh.subprocess.run",
            lambda *_a, **_k: subprocess.CompletedProcess([], 0, "", ""),
        )
        with pytest.raises(PermissionError, match="no available identities"):
            agent_identities(str(path))
    finally:
        sock.close()
        temporary.cleanup()


def test_agent_identities_wraps_ssh_add_failures(tmp_path, monkeypatch):
    temporary, server, path = bind_agent_socket()
    monkeypatch.setattr(
        "harness.git_ssh.subprocess.run",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("failed")),
    )
    try:
        with pytest.raises(PermissionError, match="could not be read"):
            agent_identities(str(path))
    finally:
        server.close()
        temporary.cleanup()


def test_ssh_tunnel_handler_handles_connect_failure_and_closes_upstream(monkeypatch):
    handler = _SshTunnelHandler.__new__(_SshTunnelHandler)
    handler.server = SimpleNamespace(target=("203.0.113.1", 22))
    handler.request = object()
    monkeypatch.setattr(
        "harness.git_ssh.socket.create_connection",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("offline")),
    )
    handler.handle()

    class Upstream:
        closed = False

        def close(self):
            self.closed = True

    upstream = Upstream()
    monkeypatch.setattr(
        "harness.git_ssh.socket.create_connection", lambda *_a, **_k: upstream
    )
    monkeypatch.setattr("harness.git_ssh._tunnel", lambda *_a: None)
    handler.handle()
    assert upstream.closed


def test_ssh_proxy_yields_loopback_port():
    with ssh_proxy("203.0.113.1", 22) as port:
        assert 1 <= port <= 65535


def test_ssh_public_key_format_does_not_allow_user_host_prefix():
    with pytest.raises(PermissionError, match="host key"):
        validate_host_keys("git.example.com", ["git.example.com " + ssh_key()])


def test_ssh_config_quotes_paths_and_rejects_controls():
    assert _ssh_config_quote('/tmp/a "b"') == '"/tmp/a \\"b\\""'
    for invalid in (None, "a\nb", "a\x00b"):
        with pytest.raises(PermissionError, match="configuration path"):
            _ssh_config_quote(invalid)


def test_ssh_sandbox_allows_only_pinned_host_and_selected_agent_socket(
    tmp_path, monkeypatch
):
    from harness.isolation import isolated_command

    monkeypatch.setattr("harness.isolation.sys.platform", "darwin")
    monkeypatch.setattr(
        "harness.isolation.shutil.which", lambda *_args, **_kwargs: "/usr/bin/git"
    )
    socket_path = tmp_path / "agent.sock"
    socket_path.touch()
    monkeypatch.setattr("harness.isolation.Path.is_socket", lambda _path: True)
    helper = tmp_path / "ssh-wrapper"
    helper.touch()
    command = isolated_command(
        ["git", "fetch", "origin"],
        tmp_path,
        git=True,
        git_helpers=(helper,),
        network_proxy=43129,
        network_remotes=(("8.8.8.8", 2222),),
        unix_sockets=(socket_path,),
        git_shell=True,
    )
    profile = command[2]
    assert '(allow network-outbound (remote ip "8.8.8.8:2222"))' in profile
    assert '(allow network-outbound (remote ip "localhost:43129"))' in profile
    assert '(allow mach-lookup (global-name "com.apple.mDNSResponder"))' not in profile
    assert (
        f'(allow network-outbound (remote unix-socket (literal "{socket_path}")))'
        in profile
    )
    assert "(allow system-socket (socket-domain AF_UNIX))" in profile
    assert "(allow network-outbound)" not in profile
    assert '(allow file-read* (literal "/private/var/select/sh"))' in profile


def test_ssh_sandbox_rejects_invalid_remote_and_missing_agent_socket(
    tmp_path, monkeypatch
):
    from harness.isolation import isolated_command

    monkeypatch.setattr("harness.isolation.sys.platform", "darwin")
    monkeypatch.setattr(
        "harness.isolation.shutil.which", lambda *_args, **_kwargs: "/usr/bin/git"
    )
    with pytest.raises(ValueError, match="network remote"):
        isolated_command(
            ["git", "fetch"], tmp_path, git=True, network_remotes=(("", 22),)
        )
    socket_path = tmp_path / "missing.sock"
    with pytest.raises(PermissionError, match="Unix socket"):
        isolated_command(
            ["git", "fetch"], tmp_path, git=True, unix_sockets=(socket_path,)
        )
    socket_path.touch()
    with pytest.raises(PermissionError, match="Unix socket"):
        isolated_command(
            ["git", "fetch"], tmp_path, git=True, unix_sockets=(socket_path,)
        )


def ssh_setup(tmp_path, monkeypatch):
    from test_git_integration import git_runtime

    _store, harness, task, git, repo = git_runtime(tmp_path, monkeypatch)
    agent_temp, agent_server, agent_socket = bind_agent_socket()
    identity = ssh_key()
    original_run = subprocess.run

    def agent_run(command, *args, **kwargs):
        if command[0] == "/usr/bin/ssh-add":
            return subprocess.CompletedProcess(command, 0, identity + "\n", "")
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr("harness.git_ssh.subprocess.run", agent_run)
    monkeypatch.setattr(
        "harness.git_broker.isolated_command", lambda cmd, *_a, **_k: cmd
    )
    monkeypatch.setattr(
        "harness.git_broker.resolve_ssh_addresses", lambda *_: ["8.8.8.8"]
    )
    configured = {
        "allowed_hosts": ["git.example.com"],
        "allowed_ports": [22, 2222],
        "host_keys": {"git.example.com": [ssh_key("ssh-ed25519", b"host-key")]},
        "credentials": {
            "git.example.com": {
                "username": "git",
                "fingerprint": key_fingerprint(identity),
            }
        },
    }
    return (
        harness,
        task,
        git,
        repo,
        agent_server,
        agent_socket,
        identity,
        configured,
        agent_temp,
    )


def test_ssh_transport_preflight_binds_host_key_agent_identity_and_dns(
    tmp_path, monkeypatch
):
    harness, _task, _git, repo, agent, agent_path, identity, configured, agent_temp = (
        ssh_setup(tmp_path, monkeypatch)
    )
    try:
        transport = SshTransport.prepare(
            ["clone", "git@git.example.com:team/repo.git", "copy"],
            repo,
            harness.tools.executor._environment(),
            allowed_hosts=configured["allowed_hosts"],
            allowed_ports=configured["allowed_ports"],
            host_keys=configured["host_keys"],
            credentials=configured["credentials"],
            agent_socket=str(agent_path),
        )
        assert transport.url == "ssh://git@git.example.com/team/repo.git"
        assert transport.address == "8.8.8.8"
        assert transport.host_keys == tuple(
            tuple(key)
            for key in validate_host_keys(
                "git.example.com", configured["host_keys"]["git.example.com"]
            )
        )
        assert transport.identity == tuple(identity.split()[:2])
        assert transport.agent_socket == str(agent_path)
    finally:
        agent.close()
        agent_temp.cleanup()


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"credentials": {}}, "identity is not configured"),
        (
            {
                "credentials": {
                    "git.example.com": {
                        "username": "git",
                        "fingerprint": "SHA256:" + "A" * 43,
                    }
                }
            },
            "identity is not loaded",
        ),
        (
            {
                "credentials": {
                    "git.example.com": {
                        "username": "other",
                        "fingerprint": "SHA256:" + "A" * 43,
                    }
                }
            },
            "username differs",
        ),
    ],
)
def test_ssh_transport_preflight_rejects_unconfigured_or_unloaded_identities(
    tmp_path, monkeypatch, override, message
):
    harness, _task, _git, repo, agent, agent_path, _identity, configured, agent_temp = (
        ssh_setup(tmp_path, monkeypatch)
    )
    configured.update(override)
    try:
        with pytest.raises(PermissionError, match=message):
            SshTransport.prepare(
                ["fetch", "origin"],
                repo,
                harness.tools.executor._environment(),
                "git@git.example.com:team/repo.git",
                configured["allowed_hosts"],
                configured["allowed_ports"],
                configured["host_keys"],
                configured["credentials"],
                str(agent_path),
            )
    finally:
        agent.close()
        agent_temp.cleanup()


def test_ssh_transport_rejects_url_identity_and_missing_host_pins(
    tmp_path, monkeypatch
):
    harness, _task, _git, repo, agent, agent_path, _identity, configured, agent_temp = (
        ssh_setup(tmp_path, monkeypatch)
    )
    try:
        with pytest.raises(PermissionError, match="username differs"):
            SshTransport.prepare(
                ["fetch", "origin"],
                repo,
                harness.tools.executor._environment(),
                "ssh://intruder@git.example.com/team/repo.git",
                configured["allowed_hosts"],
                configured["allowed_ports"],
                configured["host_keys"],
                configured["credentials"],
                str(agent_path),
            )
        with pytest.raises(PermissionError, match="host key pin is missing"):
            SshTransport.prepare(
                ["fetch", "origin"],
                repo,
                harness.tools.executor._environment(),
                "git@git.example.com:team/repo.git",
                configured["allowed_hosts"],
                configured["allowed_ports"],
                {},
                configured["credentials"],
                str(agent_path),
            )
    finally:
        agent.close()
        agent_temp.cleanup()


def test_ssh_transport_rejects_preconfigured_ssh_command(tmp_path, monkeypatch):
    harness, _task, git, repo, agent, agent_path, _identity, configured, agent_temp = (
        ssh_setup(tmp_path, monkeypatch)
    )
    subprocess.run(
        [git, "config", "core.sshCommand", "/tmp/attacker"], cwd=repo, check=True
    )
    try:
        with pytest.raises(PermissionError, match="SSH command or URL rewrite"):
            SshTransport.prepare(
                ["fetch", "origin"],
                repo,
                harness.tools.executor._environment(),
                "git@git.example.com:team/repo.git",
                configured["allowed_hosts"],
                configured["allowed_ports"],
                configured["host_keys"],
                configured["credentials"],
                str(agent_path),
            )
    finally:
        agent.close()
        agent_temp.cleanup()


def test_ssh_transport_rejects_unsupported_pull_before_preflight(tmp_path, monkeypatch):
    harness, _task, _git, repo, agent, agent_path, _identity, configured, agent_temp = (
        ssh_setup(tmp_path, monkeypatch)
    )
    try:
        with pytest.raises(PermissionError, match="ff-only"):
            SshTransport.prepare(
                ["pull", "origin", "main"],
                repo,
                harness.tools.executor._environment(),
                "git@git.example.com:team/repo.git",
                configured["allowed_hosts"],
                configured["allowed_ports"],
                configured["host_keys"],
                configured["credentials"],
                str(agent_path),
            )
    finally:
        agent.close()
        agent_temp.cleanup()


def test_ssh_transport_rejects_non_named_remote_before_agent_preflight(
    tmp_path, monkeypatch
):
    harness, _task, _git, repo, agent, _path, _identity, _config, agent_temp = (
        ssh_setup(tmp_path, monkeypatch)
    )
    try:
        with pytest.raises(PermissionError, match="named remote"):
            SshTransport.prepare(
                ["fetch", "../remote"], repo, harness.tools.executor._environment()
            )
    finally:
        agent.close()
        agent_temp.cleanup()


@pytest.mark.parametrize(
    "mode", ["config-error", "agent-changed", "agent-missing", "agent-inode-changed"]
)
def test_ssh_transport_closes_failed_preflight_and_run_races(
    tmp_path, monkeypatch, mode
):
    harness, _task, _git, repo, agent, agent_path, _identity, configured, agent_temp = (
        ssh_setup(tmp_path, monkeypatch)
    )
    original_run = subprocess.run
    if mode == "config-error":

        def failing_check(command, *args, **kwargs):
            if command[0] == "/usr/bin/ssh-add":
                return original_run(command, *args, **kwargs)
            return subprocess.CompletedProcess(command, 2, "", "inspection failed")

        monkeypatch.setattr("harness.git_broker.subprocess.run", failing_check)
        try:
            with pytest.raises(PermissionError, match="could not be verified"):
                SshTransport.prepare(
                    ["fetch", "origin"],
                    repo,
                    harness.tools.executor._environment(),
                    "git@git.example.com:team/repo.git",
                    configured["allowed_hosts"],
                    configured["allowed_ports"],
                    configured["host_keys"],
                    configured["credentials"],
                    str(agent_path),
                )
        finally:
            agent.close()
            agent_temp.cleanup()
        return

    transport = SshTransport.prepare(
        ["fetch", "origin"],
        repo,
        harness.tools.executor._environment(),
        "git@git.example.com:team/repo.git",
        configured["allowed_hosts"],
        configured["allowed_ports"],
        configured["host_keys"],
        configured["credentials"],
        str(agent_path),
    )
    original_lstat = os.lstat
    try:
        if mode == "agent-changed":
            calls = 0

            def lstat_changes_after_agent_check(path):
                nonlocal calls
                if path == str(agent_path):
                    calls += 1
                    if calls > 1:
                        raise OSError("replaced")
                return original_lstat(path)

            monkeypatch.setattr(
                "harness.git_broker.os.lstat", lstat_changes_after_agent_check
            )
            with pytest.raises(PermissionError, match="changed during preflight"):
                SshTransport.prepare(
                    ["fetch", "origin"],
                    repo,
                    harness.tools.executor._environment(),
                    "git@git.example.com:team/repo.git",
                    configured["allowed_hosts"],
                    configured["allowed_ports"],
                    configured["host_keys"],
                    configured["credentials"],
                    str(agent_path),
                )
        elif mode == "agent-missing":
            monkeypatch.setattr(
                "harness.git_broker.os.lstat",
                lambda *_: (_ for _ in ()).throw(OSError("gone")),
            )
            with pytest.raises(PermissionError, match="changed before process start"):
                transport.run()
        else:
            monkeypatch.setattr(
                "harness.git_broker.os.lstat",
                lambda *_: SimpleNamespace(st_dev=0, st_ino=0),
            )
            with pytest.raises(PermissionError, match="changed before process start"):
                transport.run()
    finally:
        monkeypatch.setattr("harness.git_broker.os.lstat", original_lstat)
        agent.close()
        agent_temp.cleanup()


def _prepared_transport(tmp_path, monkeypatch, args):
    harness, _task, _git, repo, agent, agent_path, _identity, configured, agent_temp = (
        ssh_setup(tmp_path, monkeypatch)
    )
    transport = SshTransport.prepare(
        args,
        repo,
        harness.tools.executor._environment(),
        "git@git.example.com:team/repo.git",
        configured["allowed_hosts"],
        configured["allowed_ports"],
        configured["host_keys"],
        configured["credentials"],
        str(agent_path),
    )
    monkeypatch.setattr("harness.git_broker.ssh_proxy", lambda *_: nullcontext(43210))
    monkeypatch.setattr(
        "harness.git_broker.isolated_command", lambda command, *_a, **_k: command
    )
    monkeypatch.setattr("harness.git_broker.stop_group", lambda _process: None)
    return transport, repo, agent, agent_temp


def _mock_ssh_process(monkeypatch, returncode=0, error=None):
    class Client:
        pid = 999999

        def __init__(self):
            self.returncode = returncode

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def communicate(self, timeout):
            if error:
                raise error
            return b"stdout", b"stderr"

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(
        "harness.git_broker.subprocess.Popen", lambda *_a, **_k: Client()
    )


@pytest.mark.parametrize(
    ("args", "returncode"),
    [
        (("clone", "ssh://git@git.example.com/team/repo.git", "copy"), 0),
        (("fetch", "origin", "main"), 0),
        (("clone", "ssh://git@git.example.com/team/repo.git", "copy"), 128),
    ],
)
def test_ssh_transport_run_builds_bounded_git_invocations(
    tmp_path, monkeypatch, args, returncode
):
    transport, _repo, agent, agent_temp = _prepared_transport(
        tmp_path, monkeypatch, args
    )
    _mock_ssh_process(monkeypatch, returncode)
    try:
        result = transport.run()
        assert result.returncode == returncode
        assert result.stdout == "stdout"
        assert result.stderr == "stderr"
    finally:
        agent.close()
        agent_temp.cleanup()


def test_ssh_transport_grants_only_generated_ssh_directory_read_access(
    tmp_path, monkeypatch
):
    transport = SshTransport(
        ("clone", "ssh://git@git.example.com/team/repo.git", "copy"),
        tmp_path,
        "ssh://git@git.example.com/team/repo.git",
        1,
        {},
        "git.example.com",
        22,
        "8.8.8.8",
        (("ssh-ed25519", "host-key"),),
        ("ssh-ed25519", "identity-key"),
        str(tmp_path / "agent.sock"),
        (7, 11),
        Path("/usr/bin/ssh"),
    )
    original_lstat = os.lstat

    def lstat(path, *args, **kwargs):
        if str(path) == transport.agent_socket:
            return SimpleNamespace(st_dev=7, st_ino=11)
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr("harness.git_broker.os.lstat", lstat)
    monkeypatch.setattr("harness.git_broker.ssh_proxy", lambda *_: nullcontext(43210))
    monkeypatch.setattr("harness.git_broker.stop_group", lambda _process: None)
    captured = {}

    def isolated(command, *_args, **kwargs):
        captured.update(kwargs)
        return command

    monkeypatch.setattr("harness.git_broker.isolated_command", isolated)
    _mock_ssh_process(monkeypatch)
    transport.run()
    (ssh_root,) = captured["read_roots"]
    assert ssh_root.name == ".ssh"
    assert ssh_root.parent.name.startswith("harness-ssh-")
    assert ssh_root.parent.parent == Path("/private/tmp")


def test_ssh_transport_pull_merges_only_after_success(tmp_path, monkeypatch):
    transport, _repo, agent, agent_temp = _prepared_transport(
        tmp_path, monkeypatch, ("pull", "--ff-only", "origin", "main")
    )
    _mock_ssh_process(monkeypatch)
    merge_calls = []
    monkeypatch.setattr(
        "harness.git_broker.subprocess.run",
        lambda command, **_kwargs: (
            merge_calls.append(command)
            or subprocess.CompletedProcess(command, 0, "merged", "")
        ),
    )
    try:
        result = transport.run()
        assert result.returncode == 0
        assert result.stdout.endswith("merged")
        assert merge_calls[0][3:6] == ["merge", "--ff-only", "FETCH_HEAD"]
    finally:
        agent.close()
        agent_temp.cleanup()


@pytest.mark.parametrize("followup_code", [0, 1])
def test_ssh_push_delete_reconciles_tracking_ref(tmp_path, monkeypatch, followup_code):
    transport, _repo, agent, agent_temp = _prepared_transport(
        tmp_path, monkeypatch, ("push", "origin", "--delete", "main")
    )
    _mock_ssh_process(monkeypatch)
    calls = []
    monkeypatch.setattr(
        "harness.git_broker.subprocess.run",
        lambda command, **_kwargs: (
            calls.append(command)
            or subprocess.CompletedProcess(
                command, followup_code, "", "" if not followup_code else "failure"
            )
        ),
    )
    try:
        result = transport.run()
        assert calls[0][1:4] == ["update-ref", "-d", "refs/remotes/origin/main"]
        assert result.returncode == followup_code
        if followup_code:
            assert "tracking reconciliation failed" in result.stderr
    finally:
        agent.close()
        agent_temp.cleanup()


def test_ssh_push_updates_tracking_through_followup_fetch(tmp_path, monkeypatch):
    transport, _repo, agent, agent_temp = _prepared_transport(
        tmp_path, monkeypatch, ("push", "origin", "main:refs/heads/main")
    )
    _mock_ssh_process(monkeypatch)
    followups = []
    original = SshTransport.run

    def run_with_fetch_spy(self):
        if self is transport:
            return original(self)
        followups.append(self.args)
        return subprocess.CompletedProcess(list(self.args), 0, "reconciled", "")

    monkeypatch.setattr(SshTransport, "run", run_with_fetch_spy)
    try:
        result = run_with_fetch_spy(transport)
        assert result.returncode == 0
        assert result.stdout.endswith("reconciled")
        assert followups == [("fetch", "--no-tags", "origin", "main")]
    finally:
        agent.close()
        agent_temp.cleanup()


def test_ssh_transport_e2e_pinned_clone_and_approved_push(tmp_path, monkeypatch):
    from test_git_integration import git_runtime

    store, harness, task, git, repo = git_runtime(tmp_path, monkeypatch)
    remote_root = tmp_path / "ssh-remote"
    remote_root.mkdir()
    remote = remote_root / "repo.git"
    subprocess.run(
        [git, "init", "--bare", str(remote)], check=True, capture_output=True
    )
    native(git, remote, "symbolic-ref", "HEAD", "refs/heads/main")
    native(git, repo, "push", str(remote), "main")
    client_key = tmp_path / "client_key"
    host_key = tmp_path / "host_key"
    for key_path in (client_key, host_key):
        subprocess.run(
            [
                "/usr/bin/ssh-keygen",
                "-q",
                "-t",
                "ed25519",
                "-N",
                "",
                "-f",
                str(key_path),
            ],
            check=True,
            capture_output=True,
        )
    authorized_keys = tmp_path / "authorized_keys"
    authorized_keys.write_text(client_key.with_suffix(".pub").read_text())
    authorized_keys.chmod(0o600)
    host_public = subprocess.run(
        ["/usr/bin/ssh-keygen", "-y", "-f", str(host_key)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    port_socket = socket.socket()
    try:
        port_socket.bind(("127.0.0.1", 0))
        port = port_socket.getsockname()[1]
    finally:
        port_socket.close()
    short_temp = tempfile.TemporaryDirectory(prefix="ha-", dir="/tmp")
    agent_socket = Path(short_temp.name) / "agent.sock"
    account = pwd.getpwuid(os.getuid()).pw_name
    sshd_config = tmp_path / "sshd_config"
    sshd_config.write_text(
        "\n".join(
            (
                f"Port {port}",
                "ListenAddress 127.0.0.1",
                f"HostKey {host_key}",
                f"PidFile {tmp_path / 'sshd.pid'}",
                f"AuthorizedKeysFile {authorized_keys}",
                "StrictModes no",
                "PubkeyAuthentication yes",
                "PasswordAuthentication no",
                "KbdInteractiveAuthentication no",
                "UsePAM no",
                "AllowAgentForwarding no",
                "AllowTcpForwarding no",
                "X11Forwarding no",
                f"AllowUsers {account}",
                "SetEnv PATH=/opt/homebrew/bin:/usr/bin:/bin",
                "LogLevel ERROR",
                "",
            )
        )
    )
    sshd_config.chmod(0o600)
    subprocess.run(["/usr/sbin/sshd", "-t", "-f", str(sshd_config)], check=True)
    agent = subprocess.Popen(
        ["/usr/bin/ssh-agent", "-D", "-a", str(agent_socket)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    server = subprocess.Popen(
        ["/usr/sbin/sshd", "-D", "-e", "-f", str(sshd_config)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    monkeypatch.setenv("SSH_AUTH_SOCK", str(agent_socket))
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not agent_socket.exists():
            time.sleep(0.02)
        assert agent_socket.exists(), "test ssh-agent did not create its socket"
        subprocess.run(
            ["/usr/bin/ssh-add", str(client_key)],
            env={"PATH": "/usr/bin:/bin", "SSH_AUTH_SOCK": str(agent_socket)},
            check=True,
            capture_output=True,
        )
        monkeypatch.setattr(
            "harness.git_broker.resolve_ssh_addresses", lambda *_: ["127.0.0.1"]
        )
        executor = harness.tools.executor
        executor.git_ssh_allowed_hosts = {"127.0.0.1"}
        executor.git_ssh_allowed_ports = (port,)
        executor.git_ssh_host_keys = {"127.0.0.1": [host_public]}
        executor.git_ssh_credentials = {
            "127.0.0.1": {
                "username": account,
                "fingerprint": key_fingerprint(
                    client_key.with_suffix(".pub").read_text().strip()
                ),
            }
        }
        remote_url = f"ssh://{account}@127.0.0.1:{port}{remote}"
        result = harness.tools.git(["clone", remote_url, "ssh-copy"], cwd=repo)
        stderr = (result.stderr or "").strip()
        sandbox_activation_denied = "sandbox_apply: Operation not permitted" in stderr
        loopback_denied = (
            any(
                f"ssh: connect to host {host} port " in stderr
                for host in ("127.0.0.1", "localhost")
            )
            and ": Operation not permitted" in stderr
        )
        if result.returncode and (sandbox_activation_denied or loopback_denied):
            pytest.skip(
                "the nested SSH sandbox could not reach its loopback proxy "
                "or could not activate; source remains environment-dependent. "
                f"Git stderr: {stderr}"
            )
        assert result.returncode == 0, result.stderr
        copy = repo / "ssh-copy"
        assert (copy / "addition.py").exists()
        native(git, repo, "branch", "topic/ssh")
        native(git, repo, "push", str(remote), "topic/ssh")
        native(git, repo, "tag", "release-ssh")
        native(git, repo, "push", str(remote), "refs/tags/release-ssh")
        result = harness.tools.git(
            ["fetch", "--no-tags", "origin", "main", "topic/ssh"], cwd=copy
        )
        assert result.returncode == 0, result.stderr
        assert native(
            git, copy, "show-ref", "--verify", "refs/remotes/origin/topic/ssh"
        )
        result = harness.tools.git(["fetch", "--tags", "origin"], cwd=copy)
        assert result.returncode == 0, result.stderr
        assert native(git, copy, "show-ref", "--verify", "refs/tags/release-ssh")
        native(git, remote, "branch", "-D", "topic/ssh")
        result = harness.tools.git(
            ["fetch", "--prune", "--no-tags", "origin"], cwd=copy
        )
        assert result.returncode == 0, result.stderr
        assert (
            subprocess.run(
                [git, "show-ref", "--verify", "refs/remotes/origin/topic/ssh"],
                cwd=copy,
                capture_output=True,
                check=False,
            ).returncode
            != 0
        )
        assert native(git, copy, "show-ref", "--verify", "refs/tags/release-ssh")
        native(git, copy, "config", "user.name", "SSH Test")
        native(git, copy, "config", "user.email", "ssh@example.invalid")
        (copy / "ssh-pushed.txt").write_text("ssh push\n")
        native(git, copy, "add", "ssh-pushed.txt")
        native(git, copy, "commit", "-m", "SSH push")
        task = store.create(task)
        args = ["push", "origin", "main:refs/heads/main"]
        target = harness.tools.git_target(task.id, args, copy)
        question = harness.approvals.request(target)
        store.answer(question, "approve", task.id)
        grant = harness.approvals.issue(task.id, question, target.action, target)
        result = harness.tools.git(args, cwd=copy, approval=grant, task_id=task.id)
        assert result.returncode == 0, result.stderr
        assert native(git, remote, "rev-parse", "refs/heads/main") == native(
            git, copy, "rev-parse", "HEAD"
        )
        assert native(git, copy, "rev-parse", "refs/remotes/origin/main") == native(
            git, remote, "rev-parse", "refs/heads/main"
        )
    finally:
        for process in (server, agent):
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=2)
        short_temp.cleanup()
