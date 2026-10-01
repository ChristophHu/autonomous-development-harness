"""Real loopback HTTP policy checks without external network dependencies."""

import asyncio
import ipaddress
import ssl
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread
from types import SimpleNamespace

import httpcore
import pytest

from harness.http_control import (
    _host_key,
    _pinned_transport,
    _PinnedAsyncBackend,
    _PinnedSyncBackend,
)
from harness.process_control import RunControl, TaskCancelled, use_run_control
from harness.tools import ToolExecutor


def run_server(handler_type):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_type)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def stop_server(server, thread):
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)
    assert not thread.is_alive()


def test_real_loopback_redirect_is_not_followed_or_forwarded_secret(monkeypatch):
    import harness.security

    monkeypatch.setattr(
        harness.security.SecretResolver,
        "get",
        lambda _self, _name: "loopback-test-secret",
    )
    destination_requests = []
    auth_headers = []

    class DestinationHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            destination_requests.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"destination")

        def log_message(self, *_args):
            return

    destination, destination_thread = run_server(DestinationHandler)

    class RedirectHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            auth_headers.append(self.headers.get("Authorization"))
            self.send_response(302)
            self.send_header(
                "Location", f"http://127.0.0.1:{destination.server_port}/secret"
            )
            self.end_headers()

        def log_message(self, *_args):
            return

    redirect, redirect_thread = run_server(RedirectHandler)
    try:
        tool = ToolExecutor(
            SimpleNamespace(require=lambda *_args, **_kwargs: None),
            http_allow_hosts=["127.0.0.1"],
            http_private_hosts=["127.0.0.1"],
        )
        response = tool.http(
            "GET",
            f"http://127.0.0.1:{redirect.server_port}/start",
            secret_name="TEST_HTTP_SECRET",
        )
        assert response.status_code == 302
        assert auth_headers == ["Bearer loopback-test-secret"]
        assert destination_requests == []
    finally:
        stop_server(redirect, redirect_thread)
        stop_server(destination, destination_thread)


def test_http_connects_to_resolved_ip_but_preserves_original_host():
    seen_hosts = []
    resolutions = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen_hosts.append(self.headers.get("Host"))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"pinned")

        def log_message(self, *_args):
            return

    server, thread = run_server(Handler)

    def resolver(host, port):
        resolutions.append((host, port))
        return [ipaddress.ip_address("127.0.0.1")]

    try:
        tool = ToolExecutor(
            SimpleNamespace(require=lambda *_args, **_kwargs: None),
            http_allow_hosts=["rebind.example.test"],
            http_private_hosts=["rebind.example.test"],
            http_resolver=resolver,
        )
        response = tool.http(
            "GET", f"http://rebind.example.test:{server.server_port}/resource"
        )
        assert response.status_code == 200
        assert response.text == "pinned"
        assert seen_hosts == [f"rebind.example.test:{server.server_port}"]
        assert resolutions == [("rebind.example.test", server.server_port)]
    finally:
        stop_server(server, thread)


def test_task_owned_async_http_uses_the_same_pinned_address():
    seen_hosts = []
    resolutions = []
    finished = Event()
    result = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen_hosts.append(self.headers.get("Host"))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"async-pinned")

        def log_message(self, *_args):
            return

    server, server_thread = run_server(Handler)
    tool = ToolExecutor(
        SimpleNamespace(require=lambda *_args, **_kwargs: None),
        http_allow_hosts=["rebind.example.test"],
        http_private_hosts=["rebind.example.test"],
        http_resolver=lambda host, port: (
            resolutions.append((host, port)) or [ipaddress.ip_address("127.0.0.1")]
        ),
    )

    def run():
        try:
            with use_run_control(RunControl()):
                result.append(
                    tool.http(
                        "GET",
                        f"http://rebind.example.test:{server.server_port}/async",
                    )
                )
        except Exception as error:  # noqa: BLE001
            result.append(error)
        finally:
            finished.set()

    thread = Thread(target=run, daemon=True)
    thread.start()
    try:
        assert finished.wait(3)
        thread.join(timeout=1)
        assert not thread.is_alive()
        assert len(result) == 1
        assert result[0].status_code == 200
        assert result[0].text == "async-pinned"
        assert seen_hosts == [f"rebind.example.test:{server.server_port}"]
        assert resolutions == [("rebind.example.test", server.server_port)]
    finally:
        stop_server(server, server_thread)


def test_https_keeps_original_sni_and_rejects_untrusted_certificate(tmp_path):
    key = tmp_path / "key.pem"
    certificate = tmp_path / "certificate.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(certificate),
            "-days",
            "1",
            "-subj",
            "/CN=rebind.example.test",
            "-addext",
            "subjectAltName=DNS:rebind.example.test",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    names = []
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificate, key)
    context.set_servername_callback(lambda _socket, name, _context: names.append(name))

    class TLSServer(ThreadingHTTPServer):
        def get_request(self):
            connection, address = super().get_request()
            return (
                context.wrap_socket(
                    connection, server_side=True, do_handshake_on_connect=False
                ),
                address,
            )

        def handle_error(self, _request, _client_address):
            return

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def log_message(self, *_args):
            return

    server = TLSServer(("127.0.0.1", 0), Handler)
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    tool = ToolExecutor(
        SimpleNamespace(require=lambda *_args, **_kwargs: None),
        http_allow_hosts=["rebind.example.test"],
        http_private_hosts=["rebind.example.test"],
        http_resolver=lambda _host, _port: [ipaddress.ip_address("127.0.0.1")],
    )
    try:
        with pytest.raises(Exception, match="(certificate|CERTIFICATE|SSL)"):
            tool.http("GET", f"https://rebind.example.test:{server.server_port}/secure")
        with (
            use_run_control(RunControl()),
            pytest.raises(Exception, match="(certificate|CERTIFICATE|SSL)"),
        ):
            tool.http(
                "GET",
                f"https://rebind.example.test:{server.server_port}/async-secure",
            )
        assert names == ["rebind.example.test", "rebind.example.test"]
    finally:
        stop_server(server, server_thread)


def test_task_abort_cancels_an_inflight_async_tls_handshake(tmp_path):
    key = tmp_path / "cancel-key.pem"
    certificate = tmp_path / "cancel-certificate.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(certificate),
            "-days",
            "1",
            "-subj",
            "/CN=rebind.example.test",
            "-addext",
            "subjectAltName=DNS:rebind.example.test",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    entered_tls = Event()
    finish_tls = Event()
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificate, key)

    def observe_sni(_socket, _name, _context):
        entered_tls.set()
        finish_tls.wait(2)

    context.set_servername_callback(observe_sni)

    class TLSServer(ThreadingHTTPServer):
        def get_request(self):
            connection, address = super().get_request()
            return (
                context.wrap_socket(
                    connection, server_side=True, do_handshake_on_connect=False
                ),
                address,
            )

        def handle_error(self, _request, _client_address):
            return

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def log_message(self, *_args):
            return

    server = TLSServer(("127.0.0.1", 0), Handler)
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    control = RunControl()
    errors = []
    tool = ToolExecutor(
        SimpleNamespace(require=lambda *_args, **_kwargs: None),
        http_allow_hosts=["rebind.example.test"],
        http_private_hosts=["rebind.example.test"],
        http_resolver=lambda _host, _port: [ipaddress.ip_address("127.0.0.1")],
    )

    def request():
        try:
            with use_run_control(control):
                tool.http(
                    "GET", f"https://rebind.example.test:{server.server_port}/cancel"
                )
        except TaskCancelled as error:
            errors.append(error)

    request_thread = Thread(target=request, daemon=True)
    request_thread.start()
    try:
        assert entered_tls.wait(3)
        control.request_stop("task_aborted")
        finish_tls.set()
        request_thread.join(3)
        assert not request_thread.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], TaskCancelled)
        assert errors[0].args == ("task_aborted",)
    finally:
        finish_tls.set()
        stop_server(server, server_thread)


def test_pinned_backends_only_connect_resolved_origin_and_forward_options():
    calls = []

    class SyncDelegate:
        def connect_tcp(self, host, port, **kwargs):
            calls.append((host, port, kwargs))
            return "sync-stream"

        def sleep(self, seconds):
            return seconds

    sync = _PinnedSyncBackend({("example.test", 80): "93.184.216.34"})
    sync.backend = SyncDelegate()
    options = {"timeout": 2, "local_address": "127.0.0.1", "socket_options": []}
    assert sync.connect_tcp(b"EXAMPLE.TEST", 80, **options) == "sync-stream"
    assert calls == [("93.184.216.34", 80, options)]
    assert _host_key(b"EXAMPLE.TEST", 80) == ("example.test", 80)
    assert sync.sleep(1) == 1
    with pytest.raises(httpcore.ConnectError, match="not pinned"):
        sync.connect_tcp("other.test", 80)
    with pytest.raises(httpcore.ConnectError, match="Unix sockets"):
        sync.connect_unix_socket("/tmp/socket")

    async_calls = []

    class AsyncDelegate:
        async def connect_tcp(self, host, port, **kwargs):
            async_calls.append((host, port, kwargs))
            return "async-stream"

        async def sleep(self, seconds):
            return seconds

    async_backend = _PinnedAsyncBackend({("example.test", 443): "93.184.216.34"})
    async_backend.backend = AsyncDelegate()
    assert asyncio.run(async_backend.connect_tcp("example.test", 443, **options)) == (
        "async-stream"
    )
    assert async_calls == [("93.184.216.34", 443, options)]
    assert asyncio.run(async_backend.sleep(2)) == 2
    with pytest.raises(httpcore.ConnectError, match="not pinned"):
        asyncio.run(async_backend.connect_tcp("other.test", 443))
    with pytest.raises(httpcore.ConnectError, match="Unix sockets"):
        asyncio.run(async_backend.connect_unix_socket("/tmp/socket"))


@pytest.mark.parametrize("asynchronous", [False, True])
def test_pinned_transport_requires_at_least_one_resolved_address(asynchronous):
    with pytest.raises(ValueError, match="at least one pinned HTTP address"):
        _pinned_transport([("example.test", 80, [])], asynchronous=asynchronous)


def test_pinned_transport_fails_closed_when_httpx_backend_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        "harness.http_control.httpx.HTTPTransport", lambda **_kwargs: object()
    )
    with pytest.raises(RuntimeError, match="does not support enforced IP pinning"):
        _pinned_transport(
            [("example.test", 80, [ipaddress.ip_address("93.184.216.34")])],
            asynchronous=False,
        )
