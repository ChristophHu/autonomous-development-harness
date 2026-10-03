"""Explicitly managed macOS LaunchAgent for the opt-in memory watcher."""

from __future__ import annotations

import os
import platform
import plistlib
import subprocess
from pathlib import Path

LABEL = "com.autonomousdevelopmentharness.memory-watch"


class LaunchdMemoryWatchService:
    def __init__(
        self,
        *,
        home: Path,
        root: Path,
        executable: Path,
        runner=None,
        platform_name=None,
        uid=None,
    ):
        self.home = Path(home)
        self.root = Path(root)
        self.executable = Path(executable)
        self.platform_name = platform_name or platform.system()
        get_uid = getattr(os, "getuid", None)
        self.uid = (get_uid() if get_uid is not None else -1) if uid is None else uid
        self.runner = runner or self._run
        self.plist_path = self.home / "Library" / "LaunchAgents" / f"{LABEL}.plist"
        self.log_path = (
            self.home
            / "Library"
            / "Logs"
            / "AutonomousDevelopmentHarness"
            / "memory-watch.log"
        )

    @staticmethod
    def _run(args):
        return subprocess.run(args, capture_output=True, text=True, check=False)

    def _require_macos(self):
        if self.platform_name != "Darwin":
            raise RuntimeError("memory watch service is supported only on macOS")

    def _domain(self):
        if isinstance(self.uid, bool) or not isinstance(self.uid, int) or self.uid < 0:
            raise ValueError("launchd user id is invalid")
        return f"gui/{self.uid}"

    def _plist(self):
        if not self.executable.is_absolute() or not self.executable.is_file():
            raise FileNotFoundError("harness executable is missing or not absolute")
        if not self.root.is_absolute() or not self.root.is_dir():
            raise FileNotFoundError("harness project root is missing or not absolute")
        return {
            "Label": LABEL,
            "ProgramArguments": [str(self.executable), "memory", "watch"],
            "WorkingDirectory": str(self.root),
            "RunAtLoad": True,
            "KeepAlive": True,
            "ThrottleInterval": 30,
            "StandardOutPath": str(self.log_path),
            "StandardErrorPath": str(self.log_path),
        }

    def install(self, *, confirm, monitoring_enabled):
        self._require_macos()
        domain = self._domain()
        if confirm is not True:
            raise ValueError("installation requires explicit confirmation")
        if monitoring_enabled is not True:
            raise ValueError("memory.monitoring.enabled must be true")
        payload = plistlib.dumps(self._plist(), sort_keys=True)
        if self.plist_path.exists():
            raise FileExistsError("LaunchAgent file already exists; refusing overwrite")
        self.plist_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.plist_path.with_name(f".{LABEL}.{os.getpid()}.tmp")
        try:
            temporary.write_bytes(payload)
            os.link(temporary, self.plist_path)
            temporary.unlink()
        except OSError:
            temporary.unlink(missing_ok=True)
            raise
        try:
            result = self.runner(
                ["/bin/launchctl", "bootstrap", domain, str(self.plist_path)]
            )
        except OSError:
            self.plist_path.unlink(missing_ok=True)
            raise RuntimeError("launchctl bootstrap failed") from None
        if result.returncode != 0:
            self.plist_path.unlink(missing_ok=True)
            raise RuntimeError("launchctl bootstrap failed")
        return {"installed": True, "loaded": True, "changed": True}

    def status(self):
        self._require_macos()
        domain = self._domain()
        if not self.plist_path.is_file():
            return {"installed": False, "loaded": False}
        result = self.runner(["/bin/launchctl", "print", f"{domain}/{LABEL}"])
        return {"installed": True, "loaded": result.returncode == 0}

    def uninstall(self, *, confirm):
        self._require_macos()
        domain = self._domain()
        if confirm is not True:
            raise ValueError("uninstall requires explicit confirmation")
        if not self.plist_path.exists():
            return {"installed": False, "removed": False}
        try:
            payload = plistlib.loads(self.plist_path.read_bytes())
        except (OSError, plistlib.InvalidFileException, ValueError):
            raise ValueError(
                "refusing to remove an unverified LaunchAgent file"
            ) from None
        if not isinstance(payload, dict) or payload.get("Label") != LABEL:
            raise ValueError("refusing to remove an unmanaged LaunchAgent file")
        target = f"{domain}/{LABEL}"
        status = self.runner(["/bin/launchctl", "print", target])
        if status.returncode == 0:
            result = self.runner(["/bin/launchctl", "bootout", target])
            if result.returncode != 0:
                raise RuntimeError(
                    "launchctl bootout failed; LaunchAgent file preserved"
                )
        self.plist_path.unlink()
        return {"installed": False, "removed": True}
