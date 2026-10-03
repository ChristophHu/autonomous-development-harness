"""Typed, secret-safe failure metadata for durable Harness records."""

from __future__ import annotations

import sqlite3
import subprocess
from enum import StrEnum


class FailureCategory(StrEnum):
    CANCELLATION = "cancellation"
    EXECUTION = "execution"
    PERMISSION = "permission"
    PERSISTENCE = "persistence"
    PROVIDER = "provider"
    TIMEOUT = "timeout"
    TRANSPORT = "transport"
    VALIDATION = "validation"


def failure_category(error: BaseException) -> FailureCategory:
    """Map an exception to a stable, non-sensitive operational category."""
    if isinstance(error, PermissionError):
        return FailureCategory.PERMISSION
    if isinstance(error, (TimeoutError, subprocess.TimeoutExpired)):
        return FailureCategory.TIMEOUT
    if isinstance(error, sqlite3.Error):
        return FailureCategory.PERSISTENCE
    if type(error).__name__ in {"TaskCancelled", "CancelledError"}:
        return FailureCategory.CANCELLATION
    module = type(error).__module__
    if module.startswith(("httpx", "httpcore", "urllib")):
        return FailureCategory.TRANSPORT
    if module.startswith("harness.providers") or type(error).__name__.endswith(
        "ProviderError"
    ):
        return FailureCategory.PROVIDER
    if isinstance(error, ValueError) or module.startswith(("pydantic", "jsonschema")):
        return FailureCategory.VALIDATION
    return FailureCategory.EXECUTION


def failure_record(error: BaseException, sanitize=None) -> dict[str, str]:
    """Return persistable failure metadata, sanitizing free-form detail first."""
    message = str(error)
    if sanitize is not None:
        message = sanitize(message)
    return {
        "category": failure_category(error).value,
        "error_type": type(error).__name__,
        "message": message,
    }
