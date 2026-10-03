"""Per-server lifecycle and health reporting for configured MCP servers."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import ValidationError as PydanticValidationError

from .configuration import MCPServerSettings


@dataclass(frozen=True)
class MCPServerStatus:
    name: str
    enabled: bool
    transport: str
    state: str
    checked_at: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "transport": self.transport,
            "state": self.state,
            "checked_at": self.checked_at,
            "error": self.error,
        }


class MCPServerManager:
    """Validate and probe each MCP server independently with no global startup."""

    def __init__(
        self, raw_settings: object, loader: Callable[[str], None] | None = None
    ):
        if not isinstance(raw_settings, dict):
            raw_settings = {}
        raw_servers = raw_settings.get("servers", {})
        if not isinstance(raw_servers, dict):
            raw_servers = {}
        self._settings: dict[str, MCPServerSettings] = {}
        self._status: dict[str, MCPServerStatus] = {}
        self._loader = loader
        for name, raw in raw_servers.items():
            if not isinstance(name, str) or not re.fullmatch(
                r"[A-Za-z0-9_-]{1,64}", name
            ):
                self._status[str(name)] = MCPServerStatus(
                    str(name),
                    False,
                    "unknown",
                    "invalid_config",
                    error="invalid server name",
                )
                continue
            try:
                settings = MCPServerSettings.model_validate(raw)
            except (PydanticValidationError, TypeError, ValueError):
                self._status[name] = MCPServerStatus(
                    name,
                    False,
                    "unknown",
                    "invalid_config",
                    error="invalid server configuration",
                )
                continue
            self._settings[name] = settings
            self._status[name] = MCPServerStatus(
                name,
                settings.enabled,
                settings.transport,
                "not_started" if settings.enabled else "disabled",
            )

    def names(self) -> tuple[str, ...]:
        return tuple(self._status)

    def settings(self, name: str) -> MCPServerSettings | None:
        return self._settings.get(name)

    def report(self) -> list[dict[str, object]]:
        """Return the current per-server state without starting servers."""
        return [self._status[name].as_dict() for name in self._status]

    def start(self, name: str, *, force: bool = False) -> dict[str, object] | None:
        """Load one enabled server; isolate startup/discovery errors to that server."""
        status = self._status.get(name)
        if status is None:
            return None
        if not status.enabled or status.state == "invalid_config":
            return status.as_dict()
        if self._loader is None or (status.state == "available" and not force):
            return status.as_dict()
        try:
            self._loader(name)
        except Exception as exc:  # noqa: BLE001 — a server failure is server-local
            self._set(name, "unavailable", type(exc).__name__)
        else:
            self._set(name, "available", None)
        return self._status[name].as_dict()

    def mark(self, name: str, state: str, error: str | None = None) -> None:
        if name not in self._status:
            return
        self._set(name, state, error)

    def _set(self, name: str, state: str, error: str | None) -> None:
        previous = self._status[name]
        self._status[name] = MCPServerStatus(
            previous.name,
            previous.enabled,
            previous.transport,
            state,
            datetime.now(UTC).isoformat(),
            error,
        )
