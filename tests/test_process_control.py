"""Process supervision and task cancellation contracts."""

import asyncio
import signal
import subprocess
import sys
import time
import uuid
from threading import Thread

import pytest

from harness.domain import Task
from harness.process_control import (
    RunControl,
    TaskCancelled,
    run_cancellable,
    supervised_popen,
    use_run_control,
)


def test_without_task_control_delegates_to_subprocess_runner():
    calls = []

    def runner(args, **kwargs):
        calls.append((args, kwargs))
        return "delegated"

    assert run_cancellable(runner, ["fixture"], timeout=3) == "delegated"
    assert calls == [(["fixture"], {"timeout": 3})]


def test_supervised_process_captures_input_output_and_check_contract():
    control = RunControl()
    command = [sys.executable, "-c", "import sys; print(sys.stdin.read().upper())"]
    with use_run_control(control):
        result = run_cancellable(
            subprocess.run, command, input="hello", text=True, capture_output=True
        )
    assert result.returncode == 0 and result.stdout == "HELLO\n"
    with use_run_control(RunControl()), pytest.raises(subprocess.CalledProcessError):
        run_cancellable(
            subprocess.run,
            [sys.executable, "-c", "raise SystemExit(7)"],
            text=True,
            capture_output=True,
            check=True,
        )


def test_supervised_process_enforces_timeout_and_kills_process_group():
    control = RunControl()
    command = [
        sys.executable,
        "-c",
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
    ]
    with use_run_control(control), pytest.raises(subprocess.TimeoutExpired):
        run_cancellable(
            subprocess.run,
            command,
            timeout=0.1,
            text=True,
            capture_output=True,
        )
    assert not control._processes


def test_stopped_control_rejects_process_spawn():
    control = RunControl()
    control.request_stop("aborted")
    with use_run_control(control), pytest.raises(TaskCancelled, match="aborted"):
        run_cancellable(subprocess.run, [sys.executable, "-c", "pass"])


def test_control_signal_tolerates_exited_process(monkeypatch):
    control = RunControl()
    monkeypatch.setattr(
        "harness.process_control.os.killpg",
        lambda pid, sig: (_ for _ in ()).throw(ProcessLookupError()),
    )
    process = type("Process", (), {"pid": 123456})()
    control.register(process)
    assert control.request_stop("lease_lost") is True
    assert control.request_stop("second_reason") is False
    assert control.reason == "lease_lost"
    assert control._signal(process, signal.SIGTERM) is False
    control.unregister(process)
    assert process not in control._processes


def test_signal_permission_error_only_ignored_after_process_exit(monkeypatch):
    monkeypatch.setattr(
        "harness.process_control.os.killpg",
        lambda pid, sig: (_ for _ in ()).throw(PermissionError()),
    )
    monkeypatch.setattr(
        "harness.process_control.os.kill",
        lambda pid, sig: (_ for _ in ()).throw(PermissionError()),
    )
    process = type("Process", (), {"pid": 123, "poll": lambda self: 0})()
    assert RunControl._signal(process, signal.SIGTERM) is False
    process.poll = lambda: None
    with pytest.raises(PermissionError):
        RunControl._signal(process, signal.SIGTERM)


def test_signal_falls_back_to_supervised_pid_when_group_signal_is_denied(monkeypatch):
    sent = []

    def denied_group(_pid, _sig):
        raise PermissionError()

    monkeypatch.setattr("harness.process_control.os.killpg", denied_group)
    monkeypatch.setattr(
        "harness.process_control.os.kill", lambda pid, sig: sent.append((pid, sig))
    )
    process = type("Process", (), {"pid": 456, "poll": lambda self: None})()

    assert RunControl._signal(process, signal.SIGTERM)
    assert sent == [(456, signal.SIGTERM)]


def test_signal_fallback_ignores_disappeared_process(monkeypatch):
    monkeypatch.setattr(
        "harness.process_control.os.killpg",
        lambda *_: (_ for _ in ()).throw(PermissionError()),
    )
    monkeypatch.setattr(
        "harness.process_control.os.kill",
        lambda *_: (_ for _ in ()).throw(ProcessLookupError()),
    )
    process = type("Process", (), {"pid": 457, "poll": lambda self: None})()

    assert RunControl._signal(process, signal.SIGTERM) is False


def test_signal_fallback_ignores_process_exited_during_permission_error(monkeypatch):
    monkeypatch.setattr(
        "harness.process_control.os.killpg",
        lambda *_: (_ for _ in ()).throw(PermissionError()),
    )
    monkeypatch.setattr(
        "harness.process_control.os.kill",
        lambda *_: (_ for _ in ()).throw(PermissionError()),
    )
    polls = iter((None, 0))
    process = type("Process", (), {"pid": 458, "poll": lambda self: next(polls)})()

    assert RunControl._signal(process, signal.SIGTERM) is False


def test_control_signal_reaches_registered_process_group(monkeypatch):
    control = RunControl()
    sent = []
    monkeypatch.setattr(
        "harness.process_control.os.killpg", lambda pid, sig: sent.append((pid, sig))
    )
    process = type("Process", (), {"pid": 321})()
    control.register(process)
    assert control.request_stop("task_aborted")
    assert sent == [(321, signal.SIGTERM)]
    assert control._signal(process, signal.SIGKILL)


def test_register_after_stop_signals_late_process(monkeypatch):
    control = RunControl()
    sent = []
    monkeypatch.setattr(
        "harness.process_control.os.killpg", lambda pid, sig: sent.append((pid, sig))
    )
    control.request_stop("aborted")
    control.register(type("Process", (), {"pid": 654})())
    assert sent == [(654, signal.SIGTERM)]


def test_supervisor_rejects_conflicting_popen_arguments():
    control = RunControl()
    command = [sys.executable, "-c", "pass"]
    with use_run_control(control), pytest.raises(ValueError, match="capture_output"):
        run_cancellable(
            subprocess.run, command, capture_output=True, stdout=subprocess.PIPE
        )
    with use_run_control(control), pytest.raises(ValueError, match="stdin and input"):
        run_cancellable(subprocess.run, command, input="value", stdin=subprocess.PIPE)


def test_supervisor_observes_stop_event_and_kills_ignoring_process():
    control = RunControl()
    errors = []
    command = [
        sys.executable,
        "-c",
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
    ]

    def execute():
        try:
            with use_run_control(control):
                run_cancellable(subprocess.run, command, capture_output=True, text=True)
        except TaskCancelled as error:
            errors.append(error)

    worker = Thread(target=execute)
    worker.start()
    deadline = time.monotonic() + 2
    while not control._processes and time.monotonic() < deadline:
        time.sleep(0.005)
    assert control._processes
    control.stop_event.set()
    worker.join(timeout=3)
    assert not worker.is_alive()
    assert len(errors) == 1 and "interrupted" in str(errors[0])


def test_supervised_popen_registers_and_releases_process():
    control = RunControl()
    with (
        use_run_control(control),
        supervised_popen(
            subprocess.Popen,
            [sys.executable, "-c", "pass"],
            stdout=subprocess.PIPE,
            text=True,
        ) as process,
    ):
        assert process in control._processes
        assert process.wait(timeout=2) == 0
        process.stdout.close()
    assert not control._processes


def test_supervised_popen_without_task_runs_normally():
    with supervised_popen(
        subprocess.Popen,
        [sys.executable, "-c", "pass"],
        stdout=subprocess.PIPE,
        text=True,
    ) as process:
        assert process.wait(timeout=2) == 0
        process.stdout.close()


def test_task_abort_kills_process_tree_and_returns_cancelled_task(tmp_path):
    from test_evidence_workflow import runtime

    store, orchestrator = runtime(tmp_path)
    task = orchestrator.service.create(Task(title="cancel process"))
    started = tmp_path / "started"
    survivor = tmp_path / "survived"
    child_code = (
        "import time; from pathlib import Path; time.sleep(1); "
        f"Path({str(survivor)!r}).write_text('alive')"
    )
    parent_code = (
        "import subprocess,sys,time; from pathlib import Path; "
        f"Path({str(started)!r}).touch(); "
        f"subprocess.Popen([sys.executable,'-c',{child_code!r}]); time.sleep(30)"
    )

    async def executing(task_id, owner):
        store.tasks.transition(task_id, "analyzing", owner)
        return await asyncio.to_thread(
            orchestrator.tools.execute,
            "shell.execute",
            {"command": [sys.executable, "-c", parent_code]},
        )

    orchestrator._run = executing

    async def scenario():
        running = asyncio.create_task(orchestrator.run(task.id))
        for _ in range(300):
            if started.exists():
                break
            await asyncio.sleep(0.01)
        assert started.exists()
        cancelled = orchestrator.service.abort(task.id)
        assert cancelled.status == "cancelled"
        return await asyncio.wait_for(running, timeout=3)

    result = asyncio.run(scenario())
    time.sleep(1.1)
    assert result.status == "cancelled"
    assert not survivor.exists()
    assert not orchestrator.service.active_controls


def test_heartbeat_lease_loss_cancels_worker(tmp_path, monkeypatch):
    from test_evidence_workflow import runtime

    from harness.domain import Task

    store, orchestrator = runtime(tmp_path)
    task = orchestrator.service.create(Task(title="lost lease"))
    signal_seen = asyncio.Event()

    real_sleep = asyncio.sleep

    async def sleep(_delay):
        await real_sleep(0)

    def lose_lease(_task_id, _owner):
        signal_seen.set()
        return False

    async def executing(_task_id, _owner):
        from harness.process_control import _CURRENT_CONTROL

        control = _CURRENT_CONTROL.get()
        await asyncio.to_thread(control.stop_event.wait, 2)
        control.check()

    monkeypatch.setattr("harness.services.asyncio.sleep", sleep)
    monkeypatch.setattr(store.tasks, "renew", lose_lease)
    orchestrator._run = executing
    result = asyncio.run(orchestrator.run(task.id))
    assert signal_seen.is_set()
    assert result.status == "pending"
    assert not orchestrator.service.active_controls


def test_core_transition_without_active_control(tmp_path):
    from test_evidence_workflow import runtime

    store, orchestrator = runtime(tmp_path)
    task = store.create(Task(title="direct runner"))
    owner = uuid.uuid4().hex
    assert store.tasks.claim(task.id, owner)
    try:
        result = asyncio.run(orchestrator._run(task.id, owner))
    finally:
        store.tasks.release(task.id, owner)
    assert result.status == "waiting_human"


def test_core_does_not_mark_lost_lease_failed(tmp_path):
    from test_evidence_workflow import runtime

    store, orchestrator = runtime(tmp_path)
    task = store.create(Task(title="stale lease"))
    owner = uuid.uuid4().hex
    assert store.tasks.claim(task.id, owner)
    control = RunControl()
    control.request_stop("lease_lost")
    with use_run_control(control), pytest.raises(TaskCancelled):
        asyncio.run(orchestrator._run(task.id, owner))
    assert store.get(task.id).status == "pending"
    assert store.events.list(task.id)[-1]["kind"] == "task.lease_lost"
    store.tasks.release(task.id, owner)


def test_core_does_not_fail_task_aborted_during_transition(tmp_path):
    from test_evidence_workflow import runtime

    store, orchestrator = runtime(tmp_path)
    task = store.create(Task(title="aborted transition"))
    owner = uuid.uuid4().hex
    assert store.tasks.claim(task.id, owner)
    control = RunControl()
    control.request_stop("task_aborted")
    with use_run_control(control), pytest.raises(TaskCancelled):
        asyncio.run(orchestrator._run(task.id, owner))
    assert store.get(task.id).status == "pending"
    store.tasks.release(task.id, owner)


def test_task_cleanup_tolerates_replaced_control(tmp_path):
    from test_evidence_workflow import runtime

    store, orchestrator = runtime(tmp_path)
    task = orchestrator.service.create(Task(title="control replacement"))

    async def finish(_task_id, _owner):
        orchestrator.service.active_controls.pop(task.id)
        return store.get(task.id)

    orchestrator._run = finish
    assert asyncio.run(orchestrator.run(task.id)).id == task.id


def test_use_run_control_restores_previous_context():
    from harness.process_control import _CURRENT_CONTROL

    outer, inner = RunControl(), RunControl()
    with use_run_control(outer):
        with use_run_control(inner):
            assert _CURRENT_CONTROL.get() is inner
        assert _CURRENT_CONTROL.get() is outer
    assert _CURRENT_CONTROL.get() is None
