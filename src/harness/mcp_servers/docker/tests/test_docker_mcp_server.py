"""Unit tests kept beside the standalone Docker MCP server."""

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

SERVER_PATH = Path(__file__).parents[1] / "docker_mcp_server.py"


@pytest.fixture
def server(monkeypatch):
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda _path: None
    fastmcp = types.ModuleType("mcp.server.fastmcp")

    class FakeFastMCP:
        def __init__(self, _name):
            self.tools = []

        def tool(self):
            def register(function):
                self.tools.append(function)
                return function

            return register

        def run(self):
            return None

    fastmcp.FastMCP = FakeFastMCP
    monkeypatch.setitem(sys.modules, "dotenv", dotenv)
    monkeypatch.setitem(sys.modules, "mcp", types.ModuleType("mcp"))
    monkeypatch.setitem(sys.modules, "mcp.server", types.ModuleType("mcp.server"))
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fastmcp)
    spec = importlib.util.spec_from_file_location("docker_mcp_server_test", SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def docker_result(stdout="ok", *, success=True, stderr="", returncode=0):
    return {
        "stdout": stdout,
        "stderr": stderr,
        "success": success,
        "returncode": returncode,
    }


@pytest.mark.parametrize(
    ("exception", "message"),
    [
        (FileNotFoundError(), "docker command not found"),
        (subprocess.TimeoutExpired("docker", 3), "timed out after 3s"),
        (OSError("spawn failed"), "spawn failed"),
    ],
)
def test_run_docker_converts_process_failures(server, monkeypatch, exception, message):
    monkeypatch.setattr(
        server.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(exception)
    )

    result = server._run_docker(["version"], timeout=3)

    assert result["success"] is False
    assert result["returncode"] == -1
    assert message in result["stderr"]


def test_run_docker_builds_argv_and_scrubs_private_environment(server, monkeypatch):
    captured = {}

    def run(command, **kwargs):
        captured.update(command=command, kwargs=kwargs)
        return SimpleNamespace(returncode=0, stdout=" version \n", stderr=" ")

    monkeypatch.setenv("DOCKER_HOST", "unix:///tmp/docker.sock")
    monkeypatch.setenv("_PRIVATE_TEST_VALUE", "omit")
    monkeypatch.setenv("PUBLIC_TEST_VALUE", "keep")
    monkeypatch.setattr(server.subprocess, "run", run)

    result = server._run_docker(["version"], timeout=7)

    assert captured["command"] == ["docker", "version"]
    assert captured["kwargs"]["timeout"] == 7
    assert captured["kwargs"]["check"] is False
    assert captured["kwargs"]["env"]["DOCKER_HOST"] == "unix:///tmp/docker.sock"
    assert captured["kwargs"]["env"]["PUBLIC_TEST_VALUE"] == "keep"
    assert "_PRIVATE_TEST_VALUE" not in captured["kwargs"]["env"]
    assert result == docker_result("version")


@pytest.mark.parametrize(
    ("text", "expected"),
    [("", None), ("  \n", None), ("not json", None), ('{"ok": true}', {"ok": True})],
)
def test_safe_json_load(server, text, expected):
    assert server._safe_json_load(text) == expected


def test_format_result_includes_output_and_failure_status(server):
    assert (
        server._format_result(docker_result("out", stderr="warn"))
        == "✅ OK\n\nout\n[stderr]\nwarn"
    )
    assert "FAILED (exit 7)" in server._format_result(
        docker_result("", success=False, returncode=7)
    )
    assert "(no output)" in server._format_result(docker_result(""))


def test_container_list_handles_empty_failure_and_limit(server, monkeypatch):
    outputs = iter(
        (
            docker_result("one\\ntwo\\nthree"),
            docker_result(""),
            docker_result(success=False, stderr="down"),
        )
    )
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return next(outputs)

    monkeypatch.setattr(server, "_run_docker", run)
    assert "Container (2)" in server.docker_container_list(limit=2)
    assert calls[0][:2] == ["ps", "--format"]
    assert server.docker_container_list() == "(keine Container)"
    assert "down" in server.docker_container_list(all_containers=True)
    assert calls[-1][:2] == ["ps", "-a"]


def test_container_create_constructs_safe_argv_and_reports_failure(server, monkeypatch):
    calls = []
    results = iter(
        (docker_result("1234567890123"), docker_result(success=False, stderr="denied"))
    )
    monkeypatch.setattr(
        server, "_run_docker", lambda args: calls.append(args) or next(results)
    )

    created = server.docker_container_create(
        "busybox:latest",
        name="worker",
        ports="8080:80,443:443",
        volumes="./data:/data",
        env="MODE=test",
        command="sleep 1",
        restart_policy="unless-stopped",
        network="bridge",
        memory="512m",
    )
    failed = server.docker_container_create("bad-image")

    assert "123456789012" in created and "worker" in created
    assert "denied" in failed
    assert calls[0] == [
        "create",
        "-d",
        "--name",
        "worker",
        "-p",
        "8080:80",
        "-p",
        "443:443",
        "-v",
        "./data:/data",
        "-e",
        "MODE=test",
        "--restart",
        "unless-stopped",
        "--network",
        "bridge",
        "--memory",
        "512m",
        "busybox:latest",
        "sleep 1",
    ]


def test_container_lifecycle_logs_exec_and_inspect(server, monkeypatch):
    results = iter(
        [
            docker_result("id"),
            docker_result("id"),
            docker_result("id"),
            docker_result("id"),
            docker_result("logs"),
            docker_result(""),
            docker_result("output"),
            docker_result(""),
            docker_result("error", success=False),
            docker_result("[]"),
            docker_result("invalid"),
            docker_result(
                json.dumps(
                    [
                        {
                            "Name": "/worker",
                            "Id": "123456789012",
                            "State": {"Status": "running"},
                            "NetworkSettings": {
                                "IPAddress": "127.0.0.2",
                                "Ports": {"80/tcp": [{"PublicPort": 8080}]},
                            },
                            "Mounts": [{"Source": "/data", "Destination": "/target"}],
                        }
                    ]
                )
            ),
            docker_result("id"),
            docker_result("id"),
        ]
    )
    calls = []
    monkeypatch.setattr(
        server,
        "_run_docker",
        lambda args, **kwargs: calls.append((args, kwargs)) or next(results),
    )

    assert "gestartet" in server.docker_container_start("c")
    assert "gestoppt" in server.docker_container_stop("c", timeout=4)
    assert "neugestartet" in server.docker_container_restart("c")
    assert "gelöscht" in server.docker_container_delete("c", force=True)
    assert "Logs für" in server.docker_container_logs("c", follow=True, tail=8)
    assert server.docker_container_logs("c") == "ℹ️ Keine Logs vorhanden"
    assert "Befehl ausgeführt" in server.docker_container_exec("c", "echo hi")
    assert "Befehl erfolgreich" in server.docker_container_exec("c", "true")
    assert "fehlgeschlagen" in server.docker_container_exec("c", "false")
    assert "nicht gefunden" in server.docker_container_inspect("c")
    assert "nicht gefunden" in server.docker_container_inspect("c")
    details = server.docker_container_inspect("c")
    assert "worker" in details and "8080" in details and "/data" in details
    assert "pausiert" in server.docker_container_pause("c")
    assert "fortgesetzt" in server.docker_container_unpause("c")
    assert calls[4][1]["timeout"] == 30


def test_container_prune_output_and_error(server, monkeypatch):
    results = iter(
        (
            docker_result("space"),
            docker_result(""),
            docker_result(success=False, stderr="denied"),
        )
    )
    monkeypatch.setattr(server, "_run_docker", lambda *_a, **_k: next(results))
    assert "space" in server.docker_container_prune()
    assert "Keine gestoppten Container" in server.docker_container_prune()
    assert "denied" in server.docker_container_prune()


def test_image_tools_cover_lists_build_and_mutations(server, monkeypatch):
    results = iter(
        [
            docker_result("image"),
            docker_result(""),
            docker_result(success=False, stderr="down"),
            docker_result("pulled"),
            docker_result("pushed"),
            docker_result("built"),
            docker_result("Successfully built abcdef012345"),
            docker_result("other output"),
            docker_result(success=False, stderr="build failed"),
            docker_result("dangling"),
            docker_result(""),
            docker_result(success=False, stderr="denied"),
        ]
    )
    calls = []
    monkeypatch.setattr(
        server,
        "_run_docker",
        lambda args, **kwargs: calls.append((args, kwargs)) or next(results),
    )
    assert "Images (1)" in server.docker_image_list()
    assert server.docker_image_list() == "(keine Images)"
    assert "down" in server.docker_image_list(all_images=True)
    assert "gezogen" in server.docker_image_pull("busybox")
    assert "gepusht" in server.docker_image_push("busybox")
    assert "gelöscht" in server.docker_image_delete("busybox", force=True)
    assert "abcdef012345"[:12] in server.docker_image_build(
        ".", tag="test", dockerfile="Dockerfile"
    )
    assert "other output" in server.docker_image_build(".")
    assert "build failed" in server.docker_image_build(".")
    assert "dangling" in server.docker_image_prune()
    assert "Keine nicht verwendeten Images" in server.docker_image_prune()
    assert "denied" in server.docker_image_prune()
    assert calls[6][0] == ["build", "-f", "Dockerfile", "-t", "test", "."]


def test_network_tools_list_create_delete_and_inspect(server, monkeypatch):
    results = iter(
        [
            docker_result("network"),
            docker_result(""),
            docker_result(success=False, stderr="down"),
            docker_result("id"),
            docker_result("id"),
            docker_result("[]"),
            docker_result(
                json.dumps(
                    [
                        {
                            "Name": "bridge",
                            "Id": "123456789012",
                            "Driver": "bridge",
                            "Scope": "local",
                            "IPAM": {"Config": [{"Subnet": "10.0.0.0/24"}]},
                            "Containers": {"one": {}, "two": {}},
                        }
                    ]
                )
            ),
            docker_result(success=False, stderr="denied"),
        ]
    )
    monkeypatch.setattr(server, "_run_docker", lambda *_a, **_k: next(results))
    assert "Netzwerke (1)" in server.docker_network_list()
    assert server.docker_network_list() == "(keine Netzwerke)"
    assert "down" in server.docker_network_list()
    assert "erstellt" in server.docker_network_create("net")
    assert "gelöscht" in server.docker_network_delete("net")
    assert "nicht gefunden" in server.docker_network_inspect("net")
    assert "Subnet" in server.docker_network_inspect("net")
    assert "denied" in server.docker_network_inspect("net")


def test_volume_tools_cover_empty_success_and_failure(server, monkeypatch):
    results = iter(
        [
            docker_result("volume"),
            docker_result(""),
            docker_result(success=False, stderr="down"),
            docker_result("created"),
            docker_result("created"),
            docker_result("[]"),
            docker_result(
                json.dumps(
                    [
                        {
                            "Name": "data",
                            "Driver": "local",
                            "Mountpoint": "/var/lib/docker",
                            "CreatedAt": "today",
                            "Size": "1GB",
                        }
                    ]
                )
            ),
            docker_result("removed"),
            docker_result("space"),
            docker_result(""),
            docker_result(success=False, stderr="denied"),
        ]
    )
    monkeypatch.setattr(server, "_run_docker", lambda *_a, **_k: next(results))
    assert "Volumes (1)" in server.docker_volume_list()
    assert server.docker_volume_list() == "(keine Volumes)"
    assert "down" in server.docker_volume_list()
    assert "erstellt" in server.docker_volume_create("data")
    assert "erstellt" in server.docker_volume_create("data", driver="custom")
    assert "nicht gefunden" in server.docker_volume_inspect("data")
    assert "Mountpoint" in server.docker_volume_inspect("data")
    assert "gelöscht" in server.docker_volume_delete("data", force=True)
    assert "space" in server.docker_volume_prune()
    assert "Keine nicht verwendeten Volumes" in server.docker_volume_prune()
    assert "denied" in server.docker_volume_prune()


def test_compose_tools_cover_success_empty_and_failure(server, monkeypatch):
    results = iter(
        [
            docker_result("started"),
            docker_result(success=False, stderr="up failed"),
            docker_result("down"),
            docker_result(success=False, stderr="down failed"),
            docker_result("service"),
            docker_result(""),
            docker_result(success=False, stderr="ps failed"),
            docker_result("logs"),
            docker_result(""),
            docker_result(success=False, stderr="logs failed"),
        ]
    )
    calls = []
    monkeypatch.setattr(
        server,
        "_run_docker",
        lambda args, **kwargs: calls.append((args, kwargs)) or next(results),
    )
    assert "Services gestartet" in server.docker_compose_up(
        "/project", services="db,api"
    )
    assert "up failed" in server.docker_compose_up("/project", detach=False)
    assert "gestoppt" in server.docker_compose_down("/project", volumes=True)
    assert "down failed" in server.docker_compose_down("/project")
    assert "Compose Services" in server.docker_compose_ps("/project")
    assert server.docker_compose_ps("/project") == "(keine Services)"
    assert "ps failed" in server.docker_compose_ps("/project")
    assert "Compose Logs" in server.docker_compose_logs("/project", services="db,api")
    assert server.docker_compose_logs("/project") == "ℹ️ Keine Logs vorhanden"
    assert "logs failed" in server.docker_compose_logs("/project")
    assert "-v" in calls[2][0]
    assert calls[7][0][-2:] == ["db", "api"]
