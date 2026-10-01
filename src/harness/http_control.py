"""Bounded HTTP requests with task-owned asynchronous cancellation."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from contextlib import suppress

import httpcore
import httpx

from .process_control import current_run_control


def _host_key(host, port):
    if isinstance(host, bytes):
        host = host.decode("ascii")
    return host.casefold(), port


class _PinnedSyncBackend(httpcore.NetworkBackend):
    def __init__(self, addresses):
        self.addresses = addresses
        self.backend = httpcore.SyncBackend()

    def connect_tcp(
        self,
        host,
        port,
        timeout=None,
        local_address=None,
        socket_options=None,
    ):
        address = self.addresses.get(_host_key(host, port))
        if address is None:
            raise httpcore.ConnectError("HTTP destination was not pinned")
        return self.backend.connect_tcp(
            address,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    def connect_unix_socket(self, path, timeout=None, socket_options=None):
        raise httpcore.ConnectError("Unix sockets are not permitted for HTTP tools")

    def sleep(self, seconds):
        return self.backend.sleep(seconds)


class _PinnedAsyncBackend(httpcore.AsyncNetworkBackend):
    def __init__(self, addresses):
        self.addresses = addresses
        self.backend = httpcore.AnyIOBackend()

    async def connect_tcp(
        self,
        host,
        port,
        timeout=None,
        local_address=None,
        socket_options=None,
    ):
        address = self.addresses.get(_host_key(host, port))
        if address is None:
            raise httpcore.ConnectError("HTTP destination was not pinned")
        return await self.backend.connect_tcp(
            address,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):
        raise httpcore.ConnectError("Unix sockets are not permitted for HTTP tools")

    async def sleep(self, seconds):
        return await self.backend.sleep(seconds)


def _pinned_transport(pins, *, asynchronous):
    pins = {
        _host_key(host, port): str(resolved[0])
        for host, port, resolved in pins
        if resolved
    }
    if not pins:
        raise ValueError("at least one pinned HTTP address is required")
    transport = (
        httpx.AsyncHTTPTransport(trust_env=False)
        if asynchronous
        else httpx.HTTPTransport(trust_env=False)
    )
    pool = getattr(transport, "_pool", None)
    if pool is None or not hasattr(pool, "_network_backend"):
        raise RuntimeError("HTTP transport does not support enforced IP pinning")
    pool._network_backend = (
        _PinnedAsyncBackend(pins) if asynchronous else _PinnedSyncBackend(pins)
    )
    return transport


def request(
    method,
    url,
    *,
    timeout,
    client=None,
    transport=None,
    owned_client=False,
    follow_redirects=False,
    pinned_addresses=None,
    **kwargs,
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
            if pinned_addresses:
                pinned_transport = _pinned_transport(
                    pinned_addresses, asynchronous=False
                )
                with httpx.Client(transport=pinned_transport) as pinned_client:
                    return pinned_client.request(
                        method,
                        url,
                        timeout=limits,
                        follow_redirects=follow_redirects,
                        **kwargs,
                    )
            response = httpx.request(
                method, url, timeout=limits, follow_redirects=follow_redirects, **kwargs
            )
        else:
            response = getattr(sync_client, method.lower())(
                url, timeout=limits, follow_redirects=follow_redirects, **kwargs
            )
        if control is not None:
            control.check()
        return response
    control.check()
    result = Future()
    ready = threading.Event()
    running = {}

    async def perform():
        selected_transport = transport
        if selected_transport is None and pinned_addresses:
            selected_transport = _pinned_transport(pinned_addresses, asynchronous=True)
        async with httpx.AsyncClient(
            transport=selected_transport,
            trust_env=not (pinned_addresses and transport is None),
        ) as async_client:
            return await async_client.request(
                method,
                url,
                timeout=limits,
                follow_redirects=follow_redirects,
                **kwargs,
            )

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
