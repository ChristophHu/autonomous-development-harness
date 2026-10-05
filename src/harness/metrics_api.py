"""Narrow, authenticated HTTP surface for external Prometheus scrapes."""

from __future__ import annotations

import os
import re
import secrets
import stat
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse


def read_bearer_token(path):
    if not isinstance(path, str) or not path.strip():
        raise ValueError("metrics token file is unavailable or insecure")
    token_path = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(token_path, flags)
        with os.fdopen(descriptor, "rb") as token_file:
            info = os.fstat(token_file.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or stat.S_IMODE(info.st_mode) & 0o077
                or info.st_size > 4096
            ):
                raise ValueError("metrics token file is unavailable or insecure")
            value = token_file.read(4097).decode("ascii").strip()
    except (OSError, UnicodeError) as error:
        raise ValueError("metrics token file is unavailable or insecure") from error
    if not re.fullmatch(r"[A-Za-z0-9._~-]{32,4096}", value):
        raise ValueError("metrics token file is unavailable or insecure")
    return value


def create_metrics_app(metrics_provider: Callable[[], str], bearer_token: str):
    if not callable(metrics_provider):
        raise TypeError("metrics provider must be callable")
    if not isinstance(bearer_token, str) or not bearer_token:
        raise ValueError("a non-empty metrics bearer token is required")

    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)

    @app.get("/metrics/prometheus", response_class=PlainTextResponse)
    def prometheus_metrics(authorization: str | None = Header(default=None)):
        if not isinstance(authorization, str):
            raise HTTPException(
                status_code=401,
                detail="bearer token required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        scheme, separator, supplied = authorization.partition(" ")
        if (
            not separator
            or scheme.casefold() != "bearer"
            or not supplied
            or not secrets.compare_digest(supplied, bearer_token)
        ):
            raise HTTPException(
                status_code=401,
                detail="bearer token required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return PlainTextResponse(
            metrics_provider(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    return app
