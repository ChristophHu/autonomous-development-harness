import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import yaml

import harness.docker_broker as docker_broker_module
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


def test_internal_compose_command_allowlist_and_isolated_lifecycle(broker, monkeypatch):
    service, compose, _sock = broker
    with pytest.raises(PermissionError, match="not allowlisted"):
        service._command(
            "exec",
            compose_file=compose,
            project_directory=compose.parent,
            project_name="adh-test-denied",
        )
    monkeypatch.setattr(service, "_docker", lambda: Path("/usr/bin/true"))
    monkeypatch.setattr(service, "_socket", lambda: Path("/tmp/docker.sock"))
    monkeypatch.setattr(service, "_available_host_port", lambda: 45686)
    project, _port = service.create_isolated_project()
    for action, suffix in (
        ("restart", ["restart", "--timeout", "10", "qdrant"]),
        ("cleanup", ["down", "--volumes", "--remove-orphans", "--timeout", "10"]),
    ):
        command = service._isolated_command(
            action, project_name=project, compose_file=compose
        )
        assert command[-len(suffix) :] == suffix
        assert command[command.index("--project-name") + 1] == project
    with pytest.raises(PermissionError, match="not allowlisted"):
        service._isolated_command("exec", project_name=project, compose_file=compose)
    with pytest.raises(PermissionError, match="not owned"):
        service._isolated_command(
            "cleanup", project_name="adh-test-unowned", compose_file=compose
        )
    with pytest.raises(PermissionError, match="not allowlisted"):
        service.run_isolated("exec", project_name=project)
    with pytest.raises(PermissionError, match="not owned"):
        service.run_isolated("cleanup", project_name="adh-test-unowned")


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
        captured["command"] = command
        captured.update(kwargs)
        return subprocess.CompletedProcess(command, 0, "x" * (MAX_OUTPUT + 5), None)

    monkeypatch.setenv("DOCKER_HOST", "tcp://attacker.invalid:2375")
    monkeypatch.setenv("DOCKER_CONTEXT", "attacker")
    monkeypatch.setattr("harness.docker_broker.subprocess.run", fake_run)
    result = broker[0].run("logs", tail=200)
    command = captured["command"]
    assert command[:2] == ["/usr/bin/sandbox-exec", "-p"]
    assert "(deny process-exec)" in command[2]
    assert "(allow file-read*)\n" not in command[2]
    assert "(allow system-socket (socket-domain AF_UNIX))" in command[2]
    assert captured["env"].get("DOCKER_HOST") is None
    assert captured["env"].get("DOCKER_CONTEXT") is None
    assert Path(captured["env"]["HOME"]).parent == Path("/private/tmp")
    assert captured["env"]["HOME"] != str(Path.home())
    assert captured["timeout"] == 120
    assert result.returncode == 0
    assert result.stdout.endswith("[output truncated by Docker broker]")
    assert result.stderr == ""


def test_isolated_compose_uses_unique_project_port_and_scoped_volume(
    broker, monkeypatch
):
    service, compose, _sock = broker
    monkeypatch.setattr(service, "_available_host_port", lambda: 45678)
    project, port = service.create_isolated_project()
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["manifest"] = yaml.safe_load(
            Path(command[command.index("-f") + 1]).read_text()
        )
        captured.update(kwargs)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("harness.docker_broker.subprocess.run", fake_run)
    result = service.run_isolated("start", project_name=project)
    command = captured["command"]
    assert command[:2] == ["/usr/bin/sandbox-exec", "-p"]
    assert "(deny process-exec)" in command[2]
    assert "(allow file-read*)\n" not in command[2]
    assert result.returncode == 0
    assert command[command.index("--project-name") + 1] == project
    assert (
        project.startswith("adh-test-") and project != "autonomous-development-harness"
    )
    assert captured["manifest"]["services"]["qdrant"]["ports"] == [
        f"127.0.0.1:{port}:6333"
    ]
    assert captured["manifest"]["services"]["qdrant"]["restart"] == "no"
    assert captured["manifest"]["volumes"] == {"qdrant_data": None}
    assert Path(captured["env"]["HOME"]).parent == Path("/private/tmp")
    assert captured["env"]["HOME"] != str(Path.home())
    assert compose.read_text() == compose_text()


def test_isolated_image_override_requires_pinned_official_release(broker):
    service, _compose, _sock = broker
    service._available_host_port = lambda: 45685
    project, _port = service.create_isolated_project()
    manifest = yaml.safe_load(
        service._isolated_manifest(project, "qdrant/qdrant:v1.20.0")
    )
    assert manifest["services"]["qdrant"]["image"] == "qdrant/qdrant:v1.20.0"
    for image in (
        "qdrant/qdrant:latest",
        "other/image:v1.20.0",
        "qdrant/qdrant:v1.20.0;echo unsafe",
    ):
        with pytest.raises(ValueError, match="image must be"):
            service.run_isolated("start", project_name=project, image=image)
    for image in (None, 7):
        with pytest.raises(ValueError, match="image must be"):
            service._validate_pinned_image(image)
    with pytest.raises(ValueError, match="must differ"):
        service.upgrade_smoke_test(IMAGE, IMAGE)


def test_upgrade_smoke_reuses_isolated_volume_and_cleans_up(broker, monkeypatch):
    service, _compose, _sock = broker
    monkeypatch.setattr(service, "_available_host_port", lambda: 45681)
    calls = []
    state = {"marker": None}

    class Response:
        def __init__(self, result=None):
            self._result = result or {}

        def raise_for_status(self):
            return None

        def json(self):
            return self._result

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def put(self, url, *, json):
            if "/points?" in url:
                state["marker"] = json["points"][0]["payload"]["upgrade_marker"]
            return Response()

        def get(self, url):
            if "/points/" in url:
                return Response(
                    {"result": {"payload": {"upgrade_marker": state["marker"]}}}
                )
            return Response({"result": {}})

    monkeypatch.setattr(
        "harness.docker_broker.httpx.Client", lambda **_kwargs: Client()
    )
    monkeypatch.setattr(service, "_wait_for_qdrant", lambda *_args: None)

    def run(action, *, project_name, image=None):
        calls.append((action, project_name, image))
        return subprocess.CompletedProcess(["docker"], 0, "", "")

    monkeypatch.setattr(service, "run_isolated", run)
    result = service.upgrade_smoke_test(
        "qdrant/qdrant:v1.19.0", "qdrant/qdrant:v1.20.0"
    )
    project_names = {call[1] for call in calls}
    assert len(project_names) == 1
    assert [call[0] for call in calls] == ["start", "start", "restart", "cleanup"]
    assert calls[0][2] == "qdrant/qdrant:v1.19.0"
    assert calls[1][2] == calls[2][2] == calls[3][2] == "qdrant/qdrant:v1.20.0"
    assert result == {
        "healthy": True,
        "baseline_image": "qdrant/qdrant:v1.19.0",
        "candidate_image": "qdrant/qdrant:v1.20.0",
        "collection_readable": True,
        "persisted_after_restart": True,
        "cleaned": True,
    }


def test_upgrade_smoke_fails_when_candidate_does_not_preserve_marker(
    broker, monkeypatch
):
    service, _compose, _sock = broker
    monkeypatch.setattr(service, "_available_host_port", lambda: 45682)

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"result": {"payload": {"upgrade_marker": "wrong"}}}

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def put(self, *_args, **_kwargs):
            return Response()

        def get(self, *_args, **_kwargs):
            return Response()

    actions = []
    monkeypatch.setattr(
        "harness.docker_broker.httpx.Client", lambda **_kwargs: Client()
    )
    monkeypatch.setattr(service, "_wait_for_qdrant", lambda *_args: None)
    monkeypatch.setattr(
        service,
        "run_isolated",
        lambda action, *, project_name, image=None: (
            actions.append((action, image))
            or subprocess.CompletedProcess(["docker"], 0, "", "")
        ),
    )
    with pytest.raises(RuntimeError, match="did not preserve"):
        service.upgrade_smoke_test("qdrant/qdrant:v1.19.0", "qdrant/qdrant:v1.20.0")
    assert actions[-1] == ("cleanup", "qdrant/qdrant:v1.20.0")


def test_upgrade_smoke_reports_cleanup_failure(broker, monkeypatch):
    service, _compose, _sock = broker
    monkeypatch.setattr(service, "_available_host_port", lambda: 45683)
    monkeypatch.setattr(
        service,
        "_exercise_upgrade",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        service,
        "run_isolated",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["docker"], 1, "", "cleanup error"
        ),
    )
    with pytest.raises(RuntimeError, match="cleanup failed"):
        service.upgrade_smoke_test("qdrant/qdrant:v1.19.0", "qdrant/qdrant:v1.20.0")


@pytest.mark.parametrize(
    ("failed_action", "failed_image", "message"),
    [
        ("start", "qdrant/qdrant:v1.19.0", "baseline start failed"),
        ("start", "qdrant/qdrant:v1.20.0", "candidate start failed"),
        ("restart", "qdrant/qdrant:v1.20.0", "candidate restart failed"),
    ],
)
def test_upgrade_rehearsal_reports_each_compose_transition_failure(
    broker, monkeypatch, failed_action, failed_image, message
):
    service, _compose, _sock = broker
    service._available_host_port = lambda: 45687
    project, _port = service.create_isolated_project()
    state = {"marker": None}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"result": {"payload": {"upgrade_marker": state["marker"]}}}

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def put(self, url, *, json):
            if "/points?" in url:
                state["marker"] = json["points"][0]["payload"]["upgrade_marker"]
            return Response()

        def get(self, *_args):
            return Response()

    monkeypatch.setattr(
        "harness.docker_broker.httpx.Client", lambda **_kwargs: Client()
    )
    monkeypatch.setattr(service, "_wait_for_qdrant", lambda *_args: None)
    monkeypatch.setattr(
        service,
        "run_isolated",
        lambda action, *, project_name, image=None: subprocess.CompletedProcess(
            ["docker"],
            int((action, image) == (failed_action, failed_image)),
            "",
            "",
        ),
    )
    with pytest.raises(RuntimeError, match=message):
        service._exercise_upgrade(
            project, "qdrant/qdrant:v1.19.0", "qdrant/qdrant:v1.20.0"
        )


def test_isolated_project_must_be_reserved_and_cleanup_is_narrow(broker, monkeypatch):
    service, _compose, _sock = broker
    monkeypatch.setattr(service, "_available_host_port", lambda: 45679)
    with pytest.raises(PermissionError, match="not owned"):
        service.run_isolated("cleanup", project_name="adh-test-000000000000")
    project, _port = service.create_isolated_project()
    captured = {}

    def fake_run(command, **_kwargs):
        captured["command"] = command
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("harness.docker_broker.subprocess.run", fake_run)
    assert service.run_isolated("cleanup", project_name=project).returncode == 0
    command = captured["command"]
    assert command[command.index("--project-name") + 1] == project
    assert command[-5:] == [
        "down",
        "--volumes",
        "--remove-orphans",
        "--timeout",
        "10",
    ]
    assert project not in service._isolated_projects
    with pytest.raises(PermissionError, match="not allowlisted"):
        service.command("cleanup")


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (subprocess.TimeoutExpired(["docker"], 1, b"partial", b"late"), 124),
        (OSError("daemon vanished"), 127),
    ],
)
def test_isolated_run_normalizes_timeout_and_os_errors(
    broker, monkeypatch, error, expected
):
    service, _compose, _sock = broker
    monkeypatch.setattr(service, "_available_host_port", lambda: 45684)
    project, _port = service.create_isolated_project()
    monkeypatch.setattr(
        "harness.docker_broker.subprocess.run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )
    result = service.run_isolated("start", project_name=project)
    assert result.returncode == expected
    assert (
        "timed out" in result.stderr
        if expected == 124
        else "daemon vanished" in result.stderr
    )


def test_isolated_smoke_checks_persistence_across_restart_and_cleans_up(
    broker, monkeypatch
):
    service, _compose, _sock = broker
    monkeypatch.setattr(service, "_available_host_port", lambda: 45680)
    actions = []
    monkeypatch.setattr(
        service,
        "run_isolated",
        lambda action, *, project_name: (
            actions.append((action, project_name))
            or subprocess.CompletedProcess([], 0, "", "")
        ),
    )
    stored = {}

    class Response:
        def __init__(self, status=200, result=None):
            self.status_code = status
            self.is_success = 200 <= status < 300
            self.result = result or {}

        def raise_for_status(self):
            if not self.is_success:
                raise httpx.HTTPStatusError("bad status", request=None, response=None)

        def json(self):
            return {"result": self.result}

    class Client:
        def __init__(self, *args, **kwargs):
            self.health_calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url):
            if url.endswith("/healthz"):
                self.health_calls += 1
                return Response(503 if self.health_calls == 1 else 200)
            return Response(result={"payload": stored})

        def put(self, url, *, json):
            if "/points?" in url:
                stored.update(json["points"][0]["payload"])
            return Response()

    monkeypatch.setattr(docker_broker_module.httpx, "Client", Client)
    monkeypatch.setattr(docker_broker_module.time, "sleep", lambda _delay: None)
    result = service.live_smoke_test()
    assert result == {
        "healthy": True,
        "persisted_after_restart": True,
        "cleaned": True,
    }
    assert [action for action, _project in actions] == ["start", "restart", "cleanup"]
    assert len({project for _action, project in actions}) == 1
    assert stored["smoke_marker"]


def test_isolated_smoke_cleans_up_after_failed_start(broker, monkeypatch):
    service, _compose, _sock = broker
    monkeypatch.setattr(service, "_available_host_port", lambda: 45681)
    actions = []

    def run(action, *, project_name):
        actions.append(action)
        return subprocess.CompletedProcess([], 1 if action == "start" else 0, "", "")

    monkeypatch.setattr(service, "run_isolated", run)
    with pytest.raises(RuntimeError, match="Compose start failed"):
        service.live_smoke_test()
    assert actions == ["start", "cleanup"]


def test_available_port_binds_only_loopback_ephemerally(monkeypatch):
    class PortProbe:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def bind(self, address):
            self.address = address

        def getsockname(self):
            return (*self.address[:1], 45682)

    probe = PortProbe()
    monkeypatch.setattr(docker_broker_module.socket, "socket", lambda: probe)
    assert DockerComposeBroker._available_host_port() == 45682
    assert probe.address == ("127.0.0.1", 0)


def test_qdrant_readiness_timeout_is_bounded(monkeypatch):
    class UnhealthyClient:
        def get(self, _url):
            return SimpleNamespace(is_success=False)

    times = iter([0, 0, 2, 2])
    monkeypatch.setattr(docker_broker_module.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(docker_broker_module.time, "sleep", lambda _delay: None)
    with pytest.raises(RuntimeError, match="did not become healthy"):
        DockerComposeBroker._wait_for_qdrant(UnhealthyClient(), "http://127.0.0.1", 1)


def test_qdrant_readiness_retries_transport_error(monkeypatch):
    class RecoveringClient:
        def __init__(self):
            self.calls = 0

        def get(self, _url):
            self.calls += 1
            if self.calls == 1:
                raise httpx.ConnectError("not ready")
            return SimpleNamespace(is_success=True)

    monkeypatch.setattr(docker_broker_module.time, "sleep", lambda _delay: None)
    client = RecoveringClient()
    monkeypatch.setattr(docker_broker_module.time, "monotonic", lambda: 0)
    DockerComposeBroker._wait_for_qdrant(client, "http://127.0.0.1", 1)
    assert client.calls == 2


def test_isolated_smoke_reports_cleanup_failure(broker, monkeypatch):
    service, _compose, _sock = broker
    monkeypatch.setattr(service, "_available_host_port", lambda: 45683)
    actions = []

    def run(action, *, project_name):
        actions.append(action)
        return subprocess.CompletedProcess([], 1, "", "")

    monkeypatch.setattr(service, "run_isolated", run)
    with pytest.raises(RuntimeError, match="cleanup failed for adh-test-"):
        service.live_smoke_test()
    assert actions == ["start", "cleanup"]


@pytest.mark.parametrize(
    ("failed_action", "message"),
    [
        ("restart", "Compose restart failed"),
        ("persistence", "did not persist after restart"),
    ],
)
def test_isolated_smoke_fails_closed_on_restart_or_persistence_mismatch(
    broker, monkeypatch, failed_action, message
):
    service, _compose, _sock = broker
    monkeypatch.setattr(service, "_available_host_port", lambda: 45685)
    actions = []

    def run(action, *, project_name):
        actions.append(action)
        failed = action == failed_action
        return subprocess.CompletedProcess([], 1 if failed else 0, "", "")

    class Response:
        is_success = True

        def raise_for_status(self):
            return None

        def json(self):
            return {"result": {"payload": {}}}

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, _url):
            return Response()

        def put(self, _url, *, json):
            return Response()

    monkeypatch.setattr(service, "run_isolated", run)
    monkeypatch.setattr(docker_broker_module.httpx, "Client", Client)
    with pytest.raises(RuntimeError, match=message):
        service.live_smoke_test()
    assert actions == ["start", "restart", "cleanup"]
