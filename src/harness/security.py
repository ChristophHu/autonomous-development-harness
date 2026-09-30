from __future__ import annotations

import os
import re
import shlex
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

_SECRET_NAME = re.compile(r"[A-Z0-9_]{1,128}\Z")


def _validate_name(name):
    if not isinstance(name, str) or not _SECRET_NAME.fullmatch(name):
        raise ValueError("invalid secret name")


class SecretProvider(Protocol):
    get: Callable[[str], str | None]


class EnvironmentSecretProvider:
    def get(self, name):
        _validate_name(name)
        return os.getenv(name) or None


class KeychainSecretProvider:
    def __init__(self, service="autonomous-development-harness", *, keychain_path=None):
        if not isinstance(service, str) or not re.fullmatch(
            r"[A-Za-z0-9._-]{1,128}", service
        ):
            raise ValueError("invalid keychain service name")
        self.service = service
        if keychain_path is not None and not Path(keychain_path).is_absolute():
            raise ValueError("keychain path must be absolute")
        self.keychain_path = str(keychain_path) if keychain_path is not None else None

    def _target(self):
        return [self.keychain_path] if self.keychain_path else []

    def get(self, name):
        _validate_name(name)
        try:
            result = subprocess.run(
                [
                    "security",
                    "find-generic-password",
                    "-s",
                    self.service,
                    "-a",
                    name,
                    "-w",
                    *self._target(),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return None
        if result.returncode == 0 and result.stdout.rstrip("\r\n"):
            return result.stdout.rstrip("\r\n")
        return None

    def set(self, name, value):
        _validate_name(name)
        if not isinstance(value, str) or not value:
            raise ValueError("secret value must not be empty")
        command = (
            f"add-generic-password -U -s {self.service} -a {name} "
            f"-X {value.encode('utf-8').hex()}"
        )
        if self.keychain_path:
            command += " " + shlex.quote(self.keychain_path)
        subprocess.run(
            ["security", "-i"],
            input=command + "\n",
            check=False,
            capture_output=True,
            text=True,
        )
        if self.get(name) != value:
            raise RuntimeError("keychain write failed")

    def delete(self, name):
        _validate_name(name)
        result = subprocess.run(
            [
                "security",
                "delete-generic-password",
                "-s",
                self.service,
                "-a",
                name,
                *self._target(),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    def list_names(self):
        result = subprocess.run(
            ["security", "dump-keychain", *self._target()],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError("keychain listing failed")
        names = set()
        for item in re.split(r"\n\s*\n", result.stdout):
            service = re.search(r'"svce"<blob>="([^"\n]*)"', item)
            account = re.search(r'"acct"<blob>="([^"\n]*)"', item)
            if service and service.group(1) == self.service and account:
                name = account.group(1)
                if _SECRET_NAME.fullmatch(name):
                    names.add(name)
        return sorted(names)


class SecretResolver:
    def __init__(
        self,
        service="autonomous-development-harness",
        *,
        keychain_provider=None,
        environment_provider=None,
    ):
        self.keychain = keychain_provider or KeychainSecretProvider(service)
        self.environment = environment_provider or EnvironmentSecretProvider()

    def get(self, name):
        _validate_name(name)
        return self.keychain.get(name) or self.environment.get(name)

    def exists(self, name):
        return self.get(name) is not None

    def set(self, name, value):
        self.keychain.set(name, value)

    def delete(self, name):
        return self.keychain.delete(name)

    def list_names(self):
        return self.keychain.list_names()

    def redact(self, text, secrets):
        for secret in secrets:
            if secret:
                text = text.replace(secret, "[REDACTED]")
        return text
