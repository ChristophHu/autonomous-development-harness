"""Cooperative task cancellation and process-group supervision."""

from __future__ import annotations

import contextvars
import os
import signal
import subprocess
import threading
import time
from contextlib import contextmanager


class TaskCancelled(RuntimeError):
    """Raised when an active task was aborted or lost its execution lease."""


_CURRENT_CONTROL = contextvars.ContextVar("harness_run_control", default=None)


def current_run_control():
    return _CURRENT_CONTROL.get()


@contextmanager
def use_run_control(control):
    token = _CURRENT_CONTROL.set(control)
    try:
        yield
    finally:
        _CURRENT_CONTROL.reset(token)


@contextmanager
def supervised_popen(popen, *args, **kwargs):
    """Register a broker-owned Popen with the active task control, if any."""
    control = _CURRENT_CONTROL.get()
    if control is not None:
        control.check()
        kwargs["start_new_session"] = True
    process = popen(*args, **kwargs)
    if control is not None:
        control.register(process)
    try:
        yield process
    finally:
        if control is not None:
            control.unregister(process)


class RunControl:
    def __init__(self):
        self.stop_event = threading.Event()
        self.reason = None
        self._processes = set()
        self._lock = threading.Lock()

    @property
    def stopped(self):
        return self.stop_event.is_set()

    def check(self):
        if self.stopped:
            raise TaskCancelled(self.reason or "task execution was interrupted")

    def register(self, process):
        with self._lock:
            self._processes.add(process)
            stopped = self.stopped
        if stopped:
            self._signal(process, signal.SIGTERM)

    def unregister(self, process):
        with self._lock:
            self._processes.discard(process)

    def request_stop(self, reason):
        with self._lock:
            if self.stopped:
                return False
            self.reason = reason
            self.stop_event.set()
            processes = tuple(self._processes)
        for process in processes:
            self._signal(process, signal.SIGTERM)
        return True

    @staticmethod
    def _signal(process, sig):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return False
        except PermissionError:
            if process.poll() is not None:
                return False
            # Some macOS sandbox combinations deny signaling the whole process
            # group while still allowing the owning parent to signal its child.
            # Preserve group-wide cleanup when available, but don't let that
            # platform restriction prevent termination of the supervised root.
            try:
                os.kill(process.pid, sig)
            except ProcessLookupError:
                return False
            except PermissionError:
                if process.poll() is not None:
                    return False
                raise
        return True


def _terminate(process, grace):
    RunControl._signal(process, signal.SIGTERM)
    try:
        return process.communicate(timeout=grace)
    except subprocess.TimeoutExpired:
        RunControl._signal(process, signal.SIGKILL)
        return process.communicate()


def run_cancellable(run, args, **kwargs):
    """Like ``subprocess.run``; supervise task-owned calls as process groups."""
    control = _CURRENT_CONTROL.get()
    if control is None:
        return run(args, **kwargs)
    control.check()
    timeout = kwargs.pop("timeout", None)
    check = kwargs.pop("check", False)
    input_data = kwargs.pop("input", None)
    capture_output = kwargs.pop("capture_output", False)
    if capture_output:
        if "stdout" in kwargs or "stderr" in kwargs:
            raise ValueError(
                "stdout and stderr arguments may not be used with capture_output"
            )
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    if input_data is not None:
        if "stdin" in kwargs:
            raise ValueError("stdin and input arguments may not both be used")
        kwargs["stdin"] = subprocess.PIPE
    kwargs["start_new_session"] = True
    process = subprocess.Popen(args, **kwargs)
    control.register(process)
    deadline = None if timeout is None else time.monotonic() + timeout
    pending_input = input_data
    try:
        while True:
            if control.stopped:
                _terminate(process, 0.2)
                control.check()
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                output, error = _terminate(process, 0.2)
                raise subprocess.TimeoutExpired(args, timeout, output, error)
            interval = 0.05 if remaining is None else min(0.05, remaining)
            try:
                output, error = process.communicate(
                    input=pending_input, timeout=interval
                )
                break
            except subprocess.TimeoutExpired:
                pending_input = None
        control.check()
        result = subprocess.CompletedProcess(args, process.returncode, output, error)
        if check and result.returncode:
            raise subprocess.CalledProcessError(
                result.returncode, args, output=output, stderr=error
            )
        return result
    finally:
        control.unregister(process)
