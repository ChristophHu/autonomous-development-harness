import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness.docker_broker import (
    IMAGE,
    MAX_OUTPUT,
    PROJECT_ROOT,
    DockerComposeBroker,
)


def compose_text(
    image=IMAGE, port='"127.0.0.1:6333:6333"', volume="qdrant_data:/qdrant/storage"
):
    return f"""services:
  qdrant:
    image: {image}
    restart: unless-stopped
    ports: [{port}]
    volumes: [{volume}]
volumes:
  qdrant_data:
"""


@pytest.fixture
def broker(tmp_path, monkeypatch):
    compose = tmp_path / "docker-compose.yml"
    compose.write_text(compose_text())
    socket_path = tmp_path / "docker.sock"
    socket_path.touch()
    monkeypatch.setattr(
        Path,
        "is_socket",
        lambda path: path.resolve() == socket_path.resolve(),
    )
    monkeypatch.setattr(
        "harness.docker_broker.shutil.which", lambda *_a, **_k: "/usr/bin/true"
    )
    return DockerComposeBroker(compose, socket_path), compose, socket_path


@pytest.mark.parametrize(
    ("action", "suffix"),
    [
        ("status", ["ps", "--all", "--format", "json", "qdrant"]),
        ("start", ["up", "--detach", "--no-deps", "--pull", "missing", "qdrant"]),
        ("stop", ["stop", "--timeout", "10", "qdrant"]),
        ("logs", ["logs", "--no-color", "--tail", "100", "qdrant"]),
    ],
)
def test_docker_actions_build_only_fixed_compose_commands(broker, action, suffix):
    service, compose, sock = broker
    command = service.command(action)
    assert command[-len(suffix) :] == suffix
    assert str(compose.resolve()) in command
    assert f"unix://{sock.resolve()}" in command
    assert "--project-name" in command
    assert "--env-file" in command


@pytest.mark.parametrize("tail", [0, 501, True, 1.2, "100"])
def test_docker_logs_require_bounded_integer_tail(broker, tail):
    with pytest.raises(ValueError, match="tail"):
        broker[0].command("logs", tail=tail)


def test_docker_command_rejects_arbitrary_actions(broker):
    with pytest.raises(PermissionError, match="not allowlisted"):
        broker[0].command("exec")


@pytest.mark.parametrize(
    "manifest",
    [
        "not: [valid",
        "services: {}\nvolumes: {}\nextra: true\n",
        compose_text(image="qdrant/qdrant:latest"),
        compose_text(port='"0.0.0.0:6333:6333"'),
        compose_text(volume="/host/path:/qdrant/storage"),
        "services:\n  qdrant:\n    image: qdrant/qdrant:v1.19.0\nvolumes: {}\n",
    ],
)
def test_docker_manifest_rejects_yaml_and_privileged_compose_drift(broker, manifest):
    broker[1].write_text(manifest)
    with pytest.raises(PermissionError, match="Compose"):
        broker[0].command("status")


def test_docker_manifest_rejects_missing_file_and_symlink(tmp_path):
    absent = DockerComposeBroker(tmp_path / "absent.yml", "/tmp/docker.sock")
    with pytest.raises(PermissionError, match="unavailable or invalid"):
        absent._manifest()
    source = tmp_path / "compose.yml"
    source.write_text(compose_text())
    link = tmp_path / "link.yml"
    link.symlink_to(source)
    with pytest.raises(PermissionError, match="symlink"):
        DockerComposeBroker(link, "/tmp/docker.sock")._manifest()


def test_docker_broker_resolves_relative_compose_file_from_project_root():
    broker = DockerComposeBroker("docker-compose.yml", "/tmp/docker.sock")
    assert broker.compose_file == PROJECT_ROOT / "docker-compose.yml"


@pytest.mark.parametrize("home_socket_exists", [True, False])
def test_docker_broker_selects_only_local_unix_socket_defaults(
    tmp_path, monkeypatch, home_socket_exists
):
    home = tmp_path / "home"
    socket_path = home / ".docker/run/docker.sock"
    if home_socket_exists:
        socket_path.parent.mkdir(parents=True)
        socket_path.touch()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    if not home_socket_exists:
        original_exists = Path.exists
        monkeypatch.setattr(
            Path,
            "exists",
            lambda path: False if path == socket_path else original_exists(path),
        )
    broker = DockerComposeBroker()
    assert broker.socket_path == (
        socket_path if home_socket_exists else Path("/var/run/docker.sock")
    )


@pytest.mark.parametrize(
    "executable", [str(PROJECT_ROOT), str(Path(__file__).resolve())]
)
def test_docker_binary_must_be_trusted_host_file(broker, monkeypatch, executable):
    monkeypatch.setattr(
        "harness.docker_broker.shutil.which", lambda *_a, **_k: executable
    )
    with pytest.raises(PermissionError, match="trusted host executable"):
        broker[0]._docker()


def test_docker_binary_must_exist(broker, monkeypatch):
    monkeypatch.setattr("harness.docker_broker.shutil.which", lambda *_a, **_k: None)
    with pytest.raises(PermissionError, match="unavailable"):
        broker[0]._docker()


def test_docker_socket_requires_absolute_existing_unix_owned_endpoint(
    broker, monkeypatch, tmp_path
):
    broker[0].socket_path = Path("relative.sock")
    with pytest.raises(PermissionError, match="absolute"):
        broker[0]._socket()
    broker[0].socket_path = tmp_path / "missing.sock"
    with pytest.raises(PermissionError, match="unavailable"):
        broker[0]._socket()
    broker[0].socket_path = broker[2]
    original_stat = Path.stat

    def foreign_owner(path, *args, **kwargs):
        result = original_stat(path, *args, **kwargs)
        if path == broker[2]:
            return SimpleNamespace(st_uid=os.getuid() + 1)
        return result

    monkeypatch.setattr(Path, "stat", foreign_owner)
    with pytest.raises(PermissionError, match="unexpected owner"):
        broker[0]._socket()


def test_docker_socket_rejects_non_socket(broker, monkeypatch):
    monkeypatch.setattr(Path, "is_socket", lambda _path: False)
    with pytest.raises(PermissionError, match="Unix socket"):
        broker[0]._socket()


@pytest.mark.parametrize(
    ("error", "expected_stdout"),
    [
        (OSError("not installed"), ""),
        (subprocess.TimeoutExpired(["docker"], 1, b"partial", b"late"), "partial"),
    ],
)
def test_docker_run_returns_bounded_structured_process_failures(
    broker, monkeypatch, error, expected_stdout
):
    monkeypatch.setattr(
        "harness.docker_broker.subprocess.run",
        lambda *_a, **_k: (_ for _ in ()).throw(error),
    )
    result = broker[0].run("status")
    assert result.returncode in {124, 127}
    assert result.stdout == expected_stdout
    if result.returncode == 124:
        assert "timed out" in result.stderr
    else:
        assert "not installed" in result.stderr


def test_docker_run_uses_scrubbed_environment_and_caps_output(broker, monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(command, 0, "x" * (MAX_OUTPUT + 5), None)

    monkeypatch.setenv("DOCKER_HOST", "tcp://attacker.invalid:2375")
    monkeypatch.setenv("DOCKER_CONTEXT", "attacker")
    monkeypatch.setattr("harness.docker_broker.subprocess.run", fake_run)
    result = broker[0].run("logs", tail=200)
    assert captured["env"].get("DOCKER_HOST") is None
    assert captured["env"].get("DOCKER_CONTEXT") is None
    assert captured["timeout"] == 120
    assert result.returncode == 0
    assert result.stdout.endswith("[output truncated by Docker broker]")
    assert result.stderr == ""
