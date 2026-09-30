"""
Docker MCP Server — Container, Images, Networks, Volumes verwalten.

Bietet: Container (list, create, start, stop, restart, delete, logs, exec),
        Images (list, build, pull, push, delete), Networks (list, create, inspect),
        Volumes (list, create, inspect, prune), Compose (up, down, ps, logs)

Authentifizierung: Keine — nutzt Docker CLI direkt.
Optionale Env: DOCKER_HOST für remote Docker.

Verwendung:
    pip install -r requirements.txt
    python docker_mcp_server.py
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# .env relativ zum Server-Verzeichnis laden (nicht CWD)
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("docker-mcp-server")


# ── Helper: Docker CLI aufrufen ───────────────────────────────────────────


def _run_docker(args: list[str], timeout: int = 120) -> dict[str, Any]:
    """Run a docker command and return {success, stdout, stderr, returncode}."""
    cmd = ["docker"] + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env={
                **{k: v for k, v in os.environ.items() if not k.startswith("_")},
                "DOCKER_HOST": os.environ.get("DOCKER_HOST", ""),
            },
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
        }
    except FileNotFoundError:
        return {
            "success": False,
            "stdout": "",
            "stderr": "docker command not found. Is Docker installed?",
            "returncode": -1,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s",
            "returncode": -1,
        }
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        return {"success": False, "stdout": "", "stderr": str(e), "returncode": -1}


def _format_result(result: dict) -> str:
    """Format a docker result dict into a readable string."""
    parts = []
    if result["stdout"]:
        parts.append(result["stdout"])
    if result["stderr"]:
        parts.append(f"[stderr]\n{result['stderr']}")
    if not parts:
        parts.append("(no output)")
    status = (
        "✅ OK" if result["success"] else f"❌ FAILED (exit {result['returncode']})"
    )
    return f"{status}\n\n" + "\n".join(parts)


def _safe_json_load(text: str) -> list[dict] | None:
    """Safely parse JSON, return None on failure."""
    if not text or not text.strip():
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


# ── Tools ──────────────────────────────────────────────────────────────────


@mcp.tool()
def docker_health_check() -> str:
    """
    Prüft ob Docker verfügbar und der Daemon erreichbar ist.
    """
    result = _run_docker(["version", "--format", "{{.Server.Version}}"])
    if result["success"]:
        version = result["stdout"].strip()
        return f"✅ Docker läuft\nVersion: {version}"
    return f"❌ Docker nicht erreichbar\n{result['stderr']}"


@mcp.tool()
def docker_container_list(
    all_containers: bool = False,
    limit: int = 0,
) -> str:
    """
    Listet Container auf.
    all_containers: Auch gestoppte Container anzeigen.
    limit: Maximale Anzahl (0 = alle).
    """
    args = [
        "ps",
        "-a" if all_containers else "",
        "--format",
        "{{.ID}}\\t{{.Names}}\\t{{.Image}}\\t{{.Status}}\\t{{.Ports}}",
    ]
    args = [a for a in args if a]

    result = _run_docker(args)
    if not result["success"]:
        return f"❌ Container-Liste konnte nicht geladen werden\n{result['stderr']}"

    if not result["stdout"]:
        return "(keine Container)"

    lines = result["stdout"].strip().split("\\n")
    if limit > 0:
        lines = lines[:limit]

    lines = [f"🐳 {l}" for l in lines]
    return f"Container ({len(lines)}):\n" + "\n".join(lines)


@mcp.tool()
def docker_container_create(
    image: str,
    name: str | None = None,
    ports: str | None = None,
    volumes: str | None = None,
    env: str | None = None,
    command: str | None = None,
    detach: bool = True,
    restart_policy: str | None = None,
    network: str | None = None,
    memory: str | None = None,
) -> str:
    """
    Erstellt einen neuen Container.
    image: Docker Image (z.B. 'nginx:latest').
    name: Container-Name (optional).
    ports: Port-Mappings (z.B. '8080:80,443:443').
    volumes: Volume-Mounts (z.B. './data:/data').
    env: Umgebungsvariablen (z.B. 'KEY=value').
    command: Override-Command (optional).
    detach: Im Hintergrund starten (Default: True).
    restart_policy: 'always', 'unless-stopped', 'on-failure'.
    network: Netzwerk (z.B. 'bridge').
    memory: Speicher-Limit (z.B. '512m', '1g').
    """
    args = ["create"]
    if detach:
        args.append("-d")
    if name:
        args.extend(["--name", name])
    if ports:
        for p in ports.split(","):
            args.extend(["-p", p.strip()])
    if volumes:
        for v in volumes.split(","):
            args.extend(["-v", v.strip()])
    if env:
        for e in env.split(","):
            args.extend(["-e", e.strip()])
    if restart_policy:
        args.extend(["--restart", restart_policy])
    if network:
        args.extend(["--network", network])
    if memory:
        args.extend(["--memory", memory])

    args.append(image)
    if command:
        args.append(command)

    result = _run_docker(args)
    if result["success"]:
        container_id = result["stdout"].strip()[:12]
        status = f"ID: {container_id}"
        if name:
            status += f"\nName: {name}"
        return f"✅ Container erstellt\n{status}"
    return f"❌ Container konnte nicht erstellt werden\n{result['stderr']}"


@mcp.tool()
def docker_container_start(container_id: str) -> str:
    """Startet einen Container."""
    result = _run_docker(["start", container_id])
    if result["success"]:
        return f"✅ Container '{container_id}' gestartet"
    return f"❌ Container konnte nicht gestartet werden\n{result['stderr']}"


@mcp.tool()
def docker_container_stop(container_id: str, timeout: int = 10) -> str:
    """Stoppt einen Container."""
    result = _run_docker(["stop", f"-t={timeout}", container_id])
    if result["success"]:
        return f"✅ Container '{container_id}' gestoppt"
    return f"❌ Container konnte nicht gestoppt werden\n{result['stderr']}"


@mcp.tool()
def docker_container_restart(container_id: str, timeout: int = 10) -> str:
    """Startet einen Container neu."""
    result = _run_docker(["restart", f"-t={timeout}", container_id])
    if result["success"]:
        return f"✅ Container '{container_id}' neugestartet"
    return f"❌ Container konnte nicht neugestartet werden\n{result['stderr']}"


@mcp.tool()
def docker_container_delete(container_id: str, force: bool = False) -> str:
    """Löscht einen Container."""
    args = ["rm"]
    if force:
        args.append("-f")
    args.append(container_id)
    result = _run_docker(args)
    if result["success"]:
        return f"✅ Container '{container_id}' gelöscht"
    return f"❌ Container konnte nicht gelöscht werden\n{result['stderr']}"


@mcp.tool()
def docker_container_logs(
    container_id: str,
    tail: int = 50,
    follow: bool = False,
) -> str:
    """Zeigt Container-Logs an."""
    args = ["logs"]
    if follow:
        args.append("-f")
    args.extend(["--tail", str(tail)])
    args.append(container_id)

    result = _run_docker(args, timeout=30)
    if not result["success"]:
        return f"❌ Logs konnten nicht geladen werden\n{result['stderr']}"
    if not result["stdout"]:
        return "ℹ️ Keine Logs vorhanden"
    return f"📋 Logs für '{container_id}' (letzte {tail} Zeilen):\n\n{result['stdout']}"


@mcp.tool()
def docker_container_exec(
    container_id: str,
    command: str,
) -> str:
    """Führt einen Befehl im Container aus."""
    result = _run_docker(["exec", container_id, "sh", "-c", command])
    if result["success"]:
        output = result["stdout"]
        if output:
            return f"✅ Befehl ausgeführt in '{container_id}':\n\n{output}"
        return f"✅ Befehl erfolgreich in '{container_id}' ausgeführt"
    return f"❌ Befehl fehlgeschlagen\n{result['stderr']}"


@mcp.tool()
def docker_container_inspect(container_id: str) -> str:
    """Zeigt detaillierte Container-Informationen."""
    result = _run_docker(["inspect", container_id])
    if not result["success"]:
        return f"❌ Container-Informationen konnten nicht geladen werden\n{result['stderr']}"

    data = _safe_json_load(result["stdout"])
    if not data or not data[0]:
        return f"❌ Container nicht gefunden: {container_id}"

    c = data[0]
    state = c.get("State", {})
    network_settings = c.get("NetworkSettings", {})
    mounts = c.get("Mounts", [])

    # Ports extrahieren und formatieren
    ports = network_settings.get("Ports", {})
    if ports:
        ports_display = ", ".join(
            f"{p.get('PublicPort', '?')}" if isinstance(p, dict) else str(p)
            for p in ports.values()
        )
    else:
        ports_display = "keine"

    lines = [
        f"🐳 Container: {c.get('Name', '?')[1:]}",
        f"ID: {c.get('Id', '?')[:12]}",
        f"Image: {c.get('Image', '?')}",
        f"Status: {state.get('Status', 'unknown')} ({state.get('State', 'unknown')})",
        f"Erstellt: {state.get('StartedAt', 'unknown')}",
        f"Ports: {ports_display}",
        f"IP: {network_settings.get('IPAddress', 'unknown')}",
    ]

    if mounts:
        lines.append("\\nVolumes:")
        for m in mounts:
            lines.append(f"  • {m.get('Source', '?')} → {m.get('Destination', '?')}")

    return "\\n".join(lines)


@mcp.tool()
def docker_container_pause(container_id: str) -> str:
    """Pausiert einen Container."""
    result = _run_docker(["pause", container_id])
    if result["success"]:
        return f"✅ Container '{container_id}' pausiert"
    return f"❌ Container konnte nicht pausiert werden\n{result['stderr']}"


@mcp.tool()
def docker_container_unpause(container_id: str) -> str:
    """Setzt einen Container fort."""
    result = _run_docker(["unpause", container_id])
    if result["success"]:
        return f"✅ Container '{container_id}' fortgesetzt"
    return f"❌ Container konnte nicht fortgesetzt werden\n{result['stderr']}"


@mcp.tool()
def docker_container_prune() -> str:
    """Löscht alle gestoppten Container."""
    result = _run_docker(["container", "prune", "-f"])
    if result["success"]:
        lines = result["stdout"].strip().split("\\n")
        lines = [f"  • {l}" for l in lines if l.strip()]
        return (
            "✅ Gestoppte Container gelöscht:\n" + "\n".join(lines)
            if lines
            else "✅ Keine gestoppten Container zum Löschen"
        )
    return f"❌ Cleanup fehlgeschlagen\n{result['stderr']}"


@mcp.tool()
def docker_image_list(all_images: bool = False) -> str:
    """Listet Docker Images auf."""
    args = ["images"]
    if not all_images:
        args.append("-f")
        args.append("dangling=false")
    args.extend(
        [
            "--format",
            "{{.Repository}}:{{.Tag}}\\t{{.ID}}\\t{{.Size}}\\t{{.CreatedSince}}",
        ]
    )

    result = _run_docker(args)
    if not result["success"]:
        return f"❌ Image-Liste konnte nicht geladen werden\n{result['stderr']}"

    if not result["stdout"]:
        return "(keine Images)"

    lines = result["stdout"].strip().split("\\n")
    lines = [f"📦 {l}" for l in lines]
    return f"Images ({len(lines)}):\n" + "\n".join(lines)


@mcp.tool()
def docker_image_pull(image: str) -> str:
    """Zieht ein Docker Image."""
    result = _run_docker(["pull", image])
    if result["success"]:
        return f"✅ Image '{image}' gezogen"
    return f"❌ Image konnte nicht gezogen werden\n{result['stderr']}"


@mcp.tool()
def docker_image_push(image: str) -> str:
    """Pushed ein Docker Image."""
    result = _run_docker(["push", image])
    if result["success"]:
        return f"✅ Image '{image}' gepusht"
    return f"❌ Image konnte nicht gepusht werden\n{result['stderr']}"


@mcp.tool()
def docker_image_delete(image: str, force: bool = False) -> str:
    """Löscht ein Docker Image."""
    args = ["rmi"]
    if force:
        args.append("-f")
    args.append(image)
    result = _run_docker(args)
    if result["success"]:
        return f"✅ Image '{image}' gelöscht"
    return f"❌ Image konnte nicht gelöscht werden\n{result['stderr']}"


@mcp.tool()
def docker_image_build(
    path: str,
    tag: str | None = None,
    dockerfile: str | None = None,
) -> str:
    """Baut ein Docker Image."""
    args = ["build"]
    if dockerfile:
        args.extend(["-f", dockerfile])
    if tag:
        args.extend(["-t", tag])
    args.append(path)

    result = _run_docker(args)
    if result["success"]:
        # Try to extract image ID from build output
        for line in result["stdout"].strip().split("\\n"):
            if "Successfully built" in line:
                image_id = line.split("Successfully built")[-1].strip()[:12]
                return f"✅ Image gebaut: {image_id}"
        return f"✅ Image gebaut\n{result['stdout']}"
    return f"❌ Image konnte nicht gebaut werden\n{result['stderr']}"


@mcp.tool()
def docker_image_prune() -> str:
    """Löscht alle nicht verwendeten Images."""
    result = _run_docker(["image", "prune", "-f"])
    if result["success"]:
        lines = result["stdout"].strip().split("\\n")
        lines = [f"  • {l}" for l in lines if l.strip()]
        return (
            "✅ Nicht verwendete Images gelöscht:\n" + "\n".join(lines)
            if lines
            else "✅ Keine nicht verwendeten Images"
        )
    return f"❌ Cleanup fehlgeschlagen\n{result['stderr']}"


@mcp.tool()
def docker_network_list() -> str:
    """Listet Docker Netzwerke auf."""
    result = _run_docker(
        ["network", "ls", "--format", "{{.ID}}\\t{{.Name}}\\t{{.Driver}}\\t{{.Scope}}"]
    )
    if not result["success"]:
        return f"❌ Netzwerk-Liste konnte nicht geladen werden\n{result['stderr']}"

    if not result["stdout"]:
        return "(keine Netzwerke)"

    lines = result["stdout"].strip().split("\\n")
    lines = [f"🌐 {l}" for l in lines]
    return f"Netzwerke ({len(lines)}):\n" + "\n".join(lines)


@mcp.tool()
def docker_network_create(name: str, driver: str = "bridge") -> str:
    """Erstellt ein Docker Netzwerk."""
    result = _run_docker(["network", "create", driver, name])
    if result["success"]:
        return f"✅ Netzwerk '{name}' ({driver}) erstellt"
    return f"❌ Netzwerk konnte nicht erstellt werden\n{result['stderr']}"


@mcp.tool()
def docker_network_delete(name: str) -> str:
    """Löscht ein Docker Netzwerk."""
    result = _run_docker(["network", "rm", name])
    if result["success"]:
        return f"✅ Netzwerk '{name}' gelöscht"
    return f"❌ Netzwerk konnte nicht gelöscht werden\n{result['stderr']}"


@mcp.tool()
def docker_network_inspect(name: str) -> str:
    """Zeigt detaillierte Netzwerk-Informationen."""
    result = _run_docker(["network", "inspect", name])
    if not result["success"]:
        return f"❌ Netzwerk-Informationen konnten nicht geladen werden\n{result['stderr']}"

    data = _safe_json_load(result["stdout"])
    if not data or not data[0]:
        return f"❌ Netzwerk nicht gefunden: {name}"

    n = data[0]
    containers = list(n.get("Containers", {}).keys())
    lines = [
        f"🌐 Netzwerk: {n.get('Name', '?')}",
        f"ID: {n.get('Id', '?')[:12]}",
        f"Driver: {n.get('Driver', '?')}",
        f"Scope: {n.get('Scope', '?')}",
        f"Subnet: {n.get('IPAM', {}).get('Config', [{}])[0].get('Subnet', 'unknown')}",
    ]
    if containers:
        lines.append(f"Container ({len(containers)}):")
        for c in containers[:10]:
            lines.append(f"  • {c}")

    return "\\n".join(lines)


@mcp.tool()
def docker_volume_list() -> str:
    """Listet Docker Volumes auf."""
    result = _run_docker(
        [
            "volume",
            "ls",
            "--format",
            "{{.Name}}\\t{{.Driver}}\\t{{.Mountpoint}}\\t{{.Size}}",
        ]
    )
    if not result["success"]:
        return f"❌ Volume-Liste konnte nicht geladen werden\n{result['stderr']}"

    if not result["stdout"]:
        return "(keine Volumes)"

    lines = result["stdout"].strip().split("\\n")
    lines = [f"💾 {l}" for l in lines]
    return f"Volumes ({len(lines)}):\n" + "\n".join(lines)


@mcp.tool()
def docker_volume_create(name: str, driver: str = "local") -> str:
    """Erstellt ein Docker Volume."""
    result = _run_docker(["volume", "create", "--driver", driver, name])
    if result["success"]:
        return f"✅ Volume '{name}' ({driver}) erstellt"
    return f"❌ Volume konnte nicht erstellt werden\n{result['stderr']}"


@mcp.tool()
def docker_volume_inspect(name: str) -> str:
    """Zeigt detaillierte Volume-Informationen."""
    result = _run_docker(["volume", "inspect", name])
    if not result["success"]:
        return (
            f"❌ Volume-Informationen konnten nicht geladen werden\n{result['stderr']}"
        )

    data = _safe_json_load(result["stdout"])
    if not data or not data[0]:
        return f"❌ Volume nicht gefunden: {name}"

    v = data[0]
    return (
        f"💾 Volume: {v.get('Name', '?')}\n"
        f"Driver: {v.get('Driver', '?')}\n"
        f"Mountpoint: {v.get('Mountpoint', '?')}\n"
        f"Created: {v.get('CreatedAt', '?')}\n"
        f"Size: {v.get('Size', 'unknown')}"
    )


@mcp.tool()
def docker_volume_delete(name: str, force: bool = False) -> str:
    """Löscht ein Docker Volume."""
    args = ["volume", "rm"]
    if force:
        args.append("-f")
    args.append(name)
    result = _run_docker(args)
    if result["success"]:
        return f"✅ Volume '{name}' gelöscht"
    return f"❌ Volume konnte nicht gelöscht werden\n{result['stderr']}"


@mcp.tool()
def docker_volume_prune() -> str:
    """Löscht alle nicht verwendeten Volumes."""
    result = _run_docker(["volume", "prune", "-f"])
    if result["success"]:
        lines = result["stdout"].strip().split("\\n")
        lines = [f"  • {l}" for l in lines if l.strip()]
        return (
            "✅ Nicht verwendete Volumes gelöscht:\n" + "\n".join(lines)
            if lines
            else "✅ Keine nicht verwendeten Volumes"
        )
    return f"❌ Cleanup fehlgeschlagen\n{result['stderr']}"


@mcp.tool()
def docker_compose_up(
    project_dir: str,
    services: str | None = None,
    detach: bool = True,
) -> str:
    """Startet Docker Compose Services."""
    args = ["compose"]
    if detach:
        args.append("-d")
    args.extend(["-f", f"{project_dir}/docker-compose.yml"])
    if services:
        args.extend(services.split(","))

    result = _run_docker(args)
    if result["success"]:
        return f"✅ Compose Services gestartet\n{result['stdout']}"
    return f"❌ Compose konnte nicht gestartet werden\n{result['stderr']}"


@mcp.tool()
def docker_compose_down(project_dir: str, volumes: bool = False) -> str:
    """Stoppt und entfernt Docker Compose Services."""
    args = ["compose", "-f", f"{project_dir}/docker-compose.yml", "down"]
    if volumes:
        args.append("-v")

    result = _run_docker(args)
    if result["success"]:
        return f"✅ Compose Services gestoppt und entfernt\n{result['stdout']}"
    return f"❌ Compose konnte nicht gestoppt werden\n{result['stderr']}"


@mcp.tool()
def docker_compose_ps(project_dir: str) -> str:
    """Zeigt Docker Compose Services Status."""
    result = _run_docker(
        [
            "compose",
            "-f",
            f"{project_dir}/docker-compose.yml",
            "ps",
            "--format",
            "table {{.Name}}\\t{{.Status}}\\t{{.Ports}}",
        ]
    )
    if not result["success"]:
        return f"❌ Compose-Status konnte nicht geladen werden\n{result['stderr']}"

    if not result["stdout"]:
        return "(keine Services)"

    return f"📋 Compose Services:\n\n{result['stdout']}"


@mcp.tool()
def docker_compose_logs(
    project_dir: str,
    services: str | None = None,
    tail: int = 50,
) -> str:
    """Zeigt Docker Compose Logs."""
    args = [
        "compose",
        "-f",
        f"{project_dir}/docker-compose.yml",
        "logs",
        "--tail",
        str(tail),
    ]
    if services:
        args.extend(services.split(","))

    result = _run_docker(args)
    if not result["success"]:
        return f"❌ Compose-Logs konnten nicht geladen werden\n{result['stderr']}"

    if not result["stdout"]:
        return "ℹ️ Keine Logs vorhanden"

    return f"📋 Compose Logs (letzte {tail} Zeilen):\n\n{result['stdout']}"


# ── Entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
