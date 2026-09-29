"""Narrow Docker Compose broker for the Harness-owned Qdrant service.

The Docker daemon is a privileged boundary, so arbitrary Docker CLI arguments,
mounts, images, contexts and TCP daemon endpoints are never forwarded.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml

from .process_control import run_cancellable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPOSE = PROJECT_ROOT / "docker-compose.yml"
PROJECT_NAME = "autonomous-development-harness"
SERVICE = "qdrant"
IMAGE = "qdrant/qdrant:v1.19.0"
MAX_OUTPUT = 1_000_000
_ACTIONS = {"status", "start", "stop", "logs"}


class DockerComposeBroker:
    """Run only lifecycle/diagnostic operations for the fixed Qdrant compose."""

    def __init__(self, compose_file=None, socket_path=None):
        self.compose_file = Path(compose_file or DEFAULT_COMPOSE)
        if not self.compose_file.is_absolute():
            self.compose_file = PROJECT_ROOT / self.compose_file
        if socket_path is None:
            home_socket = Path.home() / ".docker/run/docker.sock"
            socket_path = (
                home_socket if home_socket.exists() else "/var/run/docker.sock"
            )
        self.socket_path = Path(socket_path)

    def _manifest(self):
        if self.compose_file.is_symlink():
            raise PermissionError("Docker Compose file must not be a symlink")
        try:
            path = self.compose_file.resolve(strict=True)
            manifest_text = path.read_text()
            data = yaml.safe_load(manifest_text)
        except (OSError, yaml.YAMLError) as error:
            raise PermissionError(
                "Docker Compose file is unavailable or invalid"
            ) from error
        if not isinstance(data, dict) or set(data) != {"services", "volumes"}:
            raise PermissionError(
                "Docker Compose manifest is outside the broker contract"
            )
        if data["services"] != {
            SERVICE: {
                "image": IMAGE,
                "restart": "unless-stopped",
                "ports": ["127.0.0.1:6333:6333"],
                "volumes": ["qdrant_data:/qdrant/storage"],
            }
        } or data["volumes"] != {"qdrant_data": None}:
            raise PermissionError(
                "Docker Compose service violates the Qdrant broker contract"
            )
        return path, manifest_text

    def _docker(self):
        executable = shutil.which(
            "docker", path="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
        )
        if not executable:
            raise PermissionError("Docker CLI is unavailable")
        resolved = Path(executable).resolve(strict=True)
        if not resolved.is_file() or resolved.is_relative_to(PROJECT_ROOT):
            raise PermissionError("Docker CLI is not a trusted host executable")
        return resolved

    def _socket(self):
        if not self.socket_path.is_absolute():
            raise PermissionError("Docker daemon socket path must be absolute")
        try:
            path = self.socket_path.resolve(strict=True)
            info = path.stat()
        except OSError as error:
            raise PermissionError(
                "local Docker daemon socket is unavailable"
            ) from error
        if not path.is_socket():
            raise PermissionError("Docker daemon endpoint must be a Unix socket")
        if info.st_uid not in {0, os.getuid()}:
            raise PermissionError("Docker daemon socket has an unexpected owner")
        return path

    @staticmethod
    def _decode(value):
        if value is None:
            return ""
        return value.decode(errors="replace") if isinstance(value, bytes) else value

    @classmethod
    def _bounded(cls, result):
        for field in ("stdout", "stderr"):
            value = cls._decode(getattr(result, field, ""))
            if len(value) > MAX_OUTPUT:
                value = value[:MAX_OUTPUT] + "\n[output truncated by Docker broker]"
            setattr(result, field, value)
        return result

    def command(self, action, *, tail=100, compose_file=None, project_directory=None):
        if action not in _ACTIONS:
            raise PermissionError("Docker broker action is not allowlisted")
        if action == "logs" and (
            isinstance(tail, bool) or not isinstance(tail, int) or not 1 <= tail <= 500
        ):
            raise ValueError("Docker log tail must be an integer between 1 and 500")
        if compose_file is None:
            compose_file, _manifest_text = self._manifest()
        else:
            compose_file = Path(compose_file)
        project_directory = Path(project_directory or compose_file.parent)
        docker = self._docker()
        docker_socket = self._socket()
        base = [
            str(docker),
            "--host",
            f"unix://{docker_socket}",
            "compose",
            "--project-directory",
            str(project_directory),
            "--project-name",
            PROJECT_NAME,
            "--env-file",
            "/dev/null",
            "-f",
            str(compose_file),
        ]
        if action == "status":
            return [*base, "ps", "--all", "--format", "json", SERVICE]
        if action == "start":
            return [*base, "up", "--detach", "--no-deps", "--pull", "missing", SERVICE]
        if action == "stop":
            return [*base, "stop", "--timeout", "10", SERVICE]
        return [*base, "logs", "--no-color", "--tail", str(tail), SERVICE]

    def run(self, action, *, tail=100):
        env = {
            "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin",
            "HOME": str(Path.home()),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
        }
        try:
            compose_file, manifest_text = self._manifest()
            with tempfile.TemporaryDirectory(
                prefix="adh-compose-", dir="/private/tmp"
            ) as directory:
                snapshot = Path(directory) / "docker-compose.yml"
                snapshot.write_text(manifest_text)
                snapshot.chmod(0o600)
                command = self.command(
                    action,
                    tail=tail,
                    compose_file=snapshot,
                    project_directory=compose_file.parent,
                )
                result = run_cancellable(
                    subprocess.run,
                    command,
                    cwd=PROJECT_ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
        except subprocess.TimeoutExpired as error:
            result = subprocess.CompletedProcess(
                locals().get("command", []),
                124,
                self._decode(error.stdout),
                self._decode(error.stderr) + "\nDocker broker operation timed out",
            )
        except OSError as error:
            result = subprocess.CompletedProcess(
                locals().get("command", []), 127, "", str(error)
            )
        return self._bounded(result)
