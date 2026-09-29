"""Bounded HTTP requests with task-owned asynchronous cancellation."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from contextlib import suppress

import httpx

from .process_control import current_run_control


def request(
    method, url, *, timeout, client=None, transport=None, owned_client=False, **kwargs
):
    """Use the existing sync API outside tasks and cancellable I/O inside them."""
    if isinstance(timeout, dict):
        total = timeout["total"]
        connect = timeout.get("connect", min(5, total))
        read = timeout.get("read", min(30, total))
    else:
        total = timeout
        connect = min(5, total)
        read = min(30, total)
    limits = httpx.Timeout(total, connect=connect, read=read)
    sync_client = client or httpx
    control = current_run_control()
    injected_client = client is not None and client is not httpx and not owned_client
    if control is None or injected_client:
        if control is not None:
            control.check()
        if client is None:
            response = httpx.request(method, url, timeout=limits, **kwargs)
        else:
            response = getattr(sync_client, method.lower())(
                url, timeout=limits, **kwargs
            )
        if control is not None:
            control.check()
        return response
    control.check()
    result = Future()
    ready = threading.Event()
    running = {}

    async def perform():
        async with httpx.AsyncClient(transport=transport) as async_client:
            return await async_client.request(method, url, timeout=limits, **kwargs)

    async def main():
        control.check()
        running["loop"] = asyncio.get_running_loop()
        running["task"] = asyncio.create_task(asyncio.wait_for(perform(), total))
        ready.set()
        control.check()
        return await running["task"]

    def worker():
        try:
            result.set_result(asyncio.run(main()))
        except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001
            # Transfer arbitrary transport failures to the waiting task thread.
            result.set_exception(exc)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    while not result.done():
        if control.stop_event.wait(0.02):
            ready.wait(0.2)
            with suppress(KeyError, RuntimeError):
                running["loop"].call_soon_threadsafe(running["task"].cancel)
            thread.join(1)
            control.check()
    control.check()
    try:
        return result.result()
    except TimeoutError as exc:
        raise httpx.ReadTimeout("HTTP request deadline exceeded") from exc
