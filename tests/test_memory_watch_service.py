import plistlib
import subprocess
from pathlib import Path

import pytest

import harness.memory_watch_service as service_module
from harness.memory_watch_service import LaunchdMemoryWatchService

LABEL = "com.autonomousdevelopmentharness.memory-watch"


def service(tmp_path, runner=None, *, platform_name="Darwin"):
    home = tmp_path / "home"
    root = tmp_path / "project"
    root.mkdir(parents=True)
    executable = root / ".venv" / "bin" / "harness"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    calls = []

    def run(args):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "loaded", "")

    return (
        LaunchdMemoryWatchService(
            home=home,
            root=root,
            executable=executable,
            runner=runner or run,
            platform_name=platform_name,
            uid=501,
        ),
        calls,
        executable,
        root,
    )


def test_install_requires_confirmation_macos_enabled_and_valid_executable(tmp_path):
    instance, calls, _executable, _root = service(tmp_path)
    with pytest.raises(ValueError, match="confirmation"):
        instance.install(confirm=False, monitoring_enabled=True)
    with pytest.raises(ValueError, match="enabled"):
        instance.install(confirm=True, monitoring_enabled=False)
    with pytest.raises(RuntimeError, match="macOS"):
        service(tmp_path / "linux", platform_name="Linux")[0].install(
            confirm=True, monitoring_enabled=True
        )
    assert calls == []


def test_service_defaults_to_native_macos_runner_context_and_validates_uid(tmp_path):
    instance, _calls, _executable, _root = service(tmp_path)
    defaulted = LaunchdMemoryWatchService(
        home=tmp_path / "home",
        root=tmp_path / "project",
        executable=tmp_path / "harness",
    )
    assert defaulted.platform_name == "Darwin"
    assert isinstance(defaulted.uid, int)
    instance.uid = True
    with pytest.raises(ValueError, match="id"):
        instance.status()


def test_install_validates_absolute_project_and_executable_paths(tmp_path):
    instance, _calls, _executable, _root = service(tmp_path)
    instance.executable = Path("relative-harness")
    with pytest.raises(FileNotFoundError, match="executable"):
        instance.install(confirm=True, monitoring_enabled=True)
    instance.executable = tmp_path / "absent"
    with pytest.raises(FileNotFoundError, match="executable"):
        instance.install(confirm=True, monitoring_enabled=True)
    _other, _calls, executable, _root = service(tmp_path / "root")
    instance.executable = executable
    instance.root = tmp_path / "not-a-root"
    with pytest.raises(FileNotFoundError, match="project root"):
        instance.install(confirm=True, monitoring_enabled=True)


def test_install_cleans_temporary_plist_when_atomic_replace_fails(
    tmp_path, monkeypatch
):
    instance, _calls, _executable, _root = service(tmp_path)
    monkeypatch.setattr(
        service_module.os,
        "link",
        lambda *_args: (_ for _ in ()).throw(OSError("link failed")),
    )
    with pytest.raises(OSError, match="link failed"):
        instance.install(confirm=True, monitoring_enabled=True)
    assert list(instance.plist_path.parent.iterdir()) == []


def test_install_creates_managed_plist_and_bootstraps_exact_gui_domain(tmp_path):
    instance, calls, executable, root = service(tmp_path)

    result = instance.install(confirm=True, monitoring_enabled=True)

    assert result == {"installed": True, "loaded": True, "changed": True}
    assert calls == [
        ["/bin/launchctl", "bootstrap", "gui/501", str(instance.plist_path)]
    ]
    with instance.plist_path.open("rb") as stream:
        plist = plistlib.load(stream)
    assert plist["Label"] == LABEL
    assert plist["ProgramArguments"] == [str(executable), "memory", "watch"]
    assert plist["WorkingDirectory"] == str(root)
    assert plist["RunAtLoad"] is True
    assert plist["KeepAlive"] is True
    assert plist["StandardOutPath"] == plist["StandardErrorPath"]


def test_install_refuses_existing_unmanaged_file_and_missing_executable(tmp_path):
    instance, calls, _executable, _root = service(tmp_path)
    instance.plist_path.parent.mkdir(parents=True)
    instance.plist_path.write_text("unmanaged", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        instance.install(confirm=True, monitoring_enabled=True)

    missing, _calls, _executable, _root = service(tmp_path / "missing")
    missing.executable.unlink()
    with pytest.raises(FileNotFoundError, match="executable"):
        missing.install(confirm=True, monitoring_enabled=True)
    assert calls == []


def test_install_rolls_back_its_plist_if_launchctl_fails(tmp_path):
    def failed(args):
        return subprocess.CompletedProcess(args, 5, "", "bootstrap denied")

    instance, _calls, _executable, _root = service(tmp_path, failed)
    with pytest.raises(RuntimeError, match="launchctl bootstrap failed"):
        instance.install(confirm=True, monitoring_enabled=True)
    assert not instance.plist_path.exists()


def test_install_rolls_back_its_plist_when_launchctl_cannot_start(tmp_path):
    def failed(_args):
        raise OSError("launchctl unavailable")

    instance, _calls, _executable, _root = service(tmp_path, failed)
    with pytest.raises(RuntimeError, match="bootstrap failed"):
        instance.install(confirm=True, monitoring_enabled=True)
    assert not instance.plist_path.exists()


def test_status_distinguishes_missing_loaded_and_unloaded(tmp_path):
    instance, calls, _executable, _root = service(tmp_path)
    assert instance.status() == {"installed": False, "loaded": False}
    assert calls == []

    instance.plist_path.parent.mkdir(parents=True)
    instance.plist_path.write_bytes(plistlib.dumps(instance._plist()))
    assert instance.status() == {"installed": True, "loaded": True}

    unloaded_calls = []

    def runner(args):
        unloaded_calls.append(args)
        return subprocess.CompletedProcess(args, 113, "", "not loaded")

    unloaded, _calls, _executable, _root = service(tmp_path / "unloaded", runner)
    unloaded.plist_path.parent.mkdir(parents=True)
    unloaded.plist_path.write_bytes(plistlib.dumps(unloaded._plist()))
    assert unloaded.status() == {"installed": True, "loaded": False}
    assert len(calls) == 1
    assert len(unloaded_calls) == 1


def test_uninstall_requires_confirmation_and_preserves_unknown_or_loaded_service(
    tmp_path,
):
    instance, calls, _executable, _root = service(tmp_path)
    with pytest.raises(ValueError, match="confirmation"):
        instance.uninstall(confirm=False)
    assert instance.uninstall(confirm=True) == {
        "installed": False,
        "removed": False,
    }

    instance.plist_path.parent.mkdir(parents=True)
    instance.plist_path.write_text("not a plist", encoding="utf-8")
    with pytest.raises(ValueError, match="unverified"):
        instance.uninstall(confirm=True)
    assert instance.plist_path.exists()
    instance.plist_path.write_bytes(plistlib.dumps({"Label": "other.service"}))
    with pytest.raises(ValueError, match="unmanaged"):
        instance.uninstall(confirm=True)

    instance.plist_path.write_bytes(plistlib.dumps(instance._plist()))
    denied_calls = []

    def denied_runner(args):
        denied_calls.append(args)
        code = 0 if args[1] == "print" else 5
        return subprocess.CompletedProcess(args, code, "", "bootout denied")

    denied, _calls, _executable, _root = service(tmp_path / "denied", denied_runner)
    denied.plist_path.parent.mkdir(parents=True)
    denied.plist_path.write_bytes(plistlib.dumps(denied._plist()))
    with pytest.raises(RuntimeError, match="launchctl bootout failed"):
        denied.uninstall(confirm=True)
    assert denied.plist_path.exists()
    assert calls == []


def test_uninstall_boots_out_and_removes_only_its_managed_file(tmp_path):
    instance, calls, _executable, _root = service(tmp_path)
    instance.plist_path.parent.mkdir(parents=True)
    instance.plist_path.write_bytes(plistlib.dumps(instance._plist()))

    result = instance.uninstall(confirm=True)

    assert result == {"installed": False, "removed": True}
    assert not instance.plist_path.exists()
    assert calls == [
        ["/bin/launchctl", "print", f"gui/501/{LABEL}"],
        ["/bin/launchctl", "bootout", f"gui/501/{LABEL}"],
    ]


def test_uninstall_removes_managed_plist_without_bootout_when_not_loaded(tmp_path):
    calls = []

    def unloaded(args):
        calls.append(args)
        return subprocess.CompletedProcess(args, 113, "", "not loaded")

    instance, _default_calls, _executable, _root = service(tmp_path, unloaded)
    instance.plist_path.parent.mkdir(parents=True)
    instance.plist_path.write_bytes(plistlib.dumps(instance._plist()))

    assert instance.uninstall(confirm=True) == {"installed": False, "removed": True}
    assert calls == [["/bin/launchctl", "print", f"gui/501/{LABEL}"]]


def test_status_requires_macos(tmp_path):
    instance, _calls, _executable, _root = service(tmp_path, platform_name="Linux")
    with pytest.raises(RuntimeError, match="macOS"):
        instance.status()


def test_subprocess_runner_passes_argument_vector_without_shell(tmp_path, monkeypatch):
    received = []

    def fake_run(args, **kwargs):
        received.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(service_module.subprocess, "run", fake_run)
    result = LaunchdMemoryWatchService._run(["/bin/launchctl", "print", "gui/501"])
    assert result.returncode == 0
    assert received == [
        (
            ["/bin/launchctl", "print", "gui/501"],
            {"capture_output": True, "text": True, "check": False},
        )
    ]
