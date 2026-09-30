"""One attempt/deadline budget shared by routing and provider retries."""

from __future__ import annotations

import time
from contextvars import ContextVar
from dataclasses import dataclass

from .process_control import current_run_control

_CURRENT: ContextVar[RetryBudget | None] = ContextVar(
    "model_retry_budget", default=None
)


@dataclass
class RetryBudget:
    max_attempts: int
    max_elapsed: float
    started: float
    attempts: int = 0

    @classmethod
    def start(cls, max_attempts, max_elapsed):
        return cls(max_attempts, max_elapsed, time.monotonic())

    @property
    def remaining(self):
        return max(0.0, self.max_elapsed - (time.monotonic() - self.started))

    def claim(self):
        control = current_run_control()
        if control is not None:
            control.check()
        if self.attempts >= self.max_attempts or self.remaining <= 0:
            return False
        self.attempts += 1
        return True

    def wait(self, delay):
        delay = min(max(0.0, delay), self.remaining)
        if delay <= 0:
            return
        control = current_run_control()
        if control is None:
            time.sleep(delay)
        elif control.stop_event.wait(delay):
            control.check()


def current_retry_budget():
    return _CURRENT.get()


def use_retry_budget(budget):
    """Set the budget for one synchronous provider call; always restore it."""

    class Scope:
        def __enter__(self):
            self.token = _CURRENT.set(budget)
            return budget

        def __exit__(self, *_):
            _CURRENT.reset(self.token)

    return Scope()
