from __future__ import annotations

import os
import subprocess


class SecretResolver:
    def __init__(self, service="autonomous-development-harness"):
        self.service = service

    def get(self, name):
        if os.getenv(name):
            return os.getenv(name)
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
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except OSError:
            return None

    def set(self, name, value):
        subprocess.run(
            [
                "security",
                "add-generic-password",
                "-U",
                "-s",
                self.service,
                "-a",
                name,
                "-w",
                value,
            ],
            check=True,
            capture_output=True,
        )

    def delete(self, name):
        result = subprocess.run(
            ["security", "delete-generic-password", "-s", self.service, "-a", name],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    def list_names(self):
        result = subprocess.run(
            ["security", "dump-keychain"], check=False, capture_output=True, text=True
        )
        return sorted(
            set(__import__("re").findall(r'"acct"<blob>="([^"]+)"', result.stdout))
        )

    def redact(self, text, secrets):
        for secret in secrets:
            if secret:
                text = text.replace(secret, "[REDACTED]")
        return text
