"""Host-allowlisted HTTPS CONNECT proxy for sandboxed Git clients."""

import ipaddress
import socket
import socketserver
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from urllib.parse import urlsplit


def validate_https_url(url, allowed_hosts):
    parsed = urlsplit(url)
    try:
        port = parsed.port
    except ValueError as error:
        raise PermissionError(
            "Git HTTPS URL must use a credential-free TLS origin"
        ) from error
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or port not in (None, 443)
    ):
        raise PermissionError("Git HTTPS URL must use a credential-free TLS origin")
    host = parsed.hostname.encode("idna").decode("ascii").lower().rstrip(".")
    if not host_allowed(host, allowed_hosts):
        raise PermissionError("Git HTTPS host is not allowlisted")
    netloc = f"[{host}]" if isinstance(_as_ip(host), ipaddress.IPv6Address) else host
    return parsed._replace(netloc=netloc, scheme="https").geturl()


def _as_ip(host):
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def host_allowed(host, allowed_hosts):
    host = host.lower().rstrip(".")
    return any(
        host == item.lower().rstrip(".")
        or (item.startswith("*.") and host.endswith(item[1:].lower()))
        for item in allowed_hosts
    )


def public_addresses(host, port):
    try:
        records = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as error:
        raise PermissionError("Git HTTPS host could not be resolved") from error
    addresses = []
    for family, socktype, protocol, _canonname, sockaddr in records:
        address = ipaddress.ip_address(sockaddr[0])
        if address.is_global:
            addresses.append((family, socktype, protocol, sockaddr))
    if not addresses:
        raise PermissionError("Git HTTPS host resolved only to non-public addresses")
    return addresses


def pinned_connection(host, port, timeout=15):
    failures = []
    for family, socktype, protocol, sockaddr in public_addresses(host, port):
        connection = socket.socket(family, socktype, protocol)
        connection.settimeout(timeout)
        try:
            connection.connect(sockaddr)
            return connection
        except OSError as error:
            failures.append(error)
            connection.close()
    raise OSError("Git HTTPS destination could not be reached") from failures[-1]


def _copy_stream(source, target):
    try:
        while True:
            payload = source.recv(65536)
            if not payload:
                return
            target.sendall(payload)
    except (ConnectionError, OSError):
        return
    finally:
        try:
            target.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def _tunnel(client, upstream):
    client.settimeout(120)
    upstream.settimeout(120)
    with ThreadPoolExecutor(max_workers=2) as workers:
        directions = (
            workers.submit(_copy_stream, client, upstream),
            workers.submit(_copy_stream, upstream, client),
        )
        for direction in directions:
            direction.result()


class _ProxyHandler(socketserver.BaseRequestHandler):
    def handle(self):
        client = self.request
        try:
            header = bytearray()
            while b"\r\n\r\n" not in header and len(header) <= 8192:
                data = client.recv(1024)
                if not data:
                    return
                header.extend(data)
            if b"\r\n\r\n" not in header:
                client.sendall(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n")
                return
            first = bytes(header).split(b"\r\n", 1)[0].decode("ascii")
            method, authority, _version = first.split(" ")
            parsed = urlsplit("//" + authority)
            host = (parsed.hostname or "").encode("idna").decode("ascii").lower()
            port = parsed.port
            if (
                method != "CONNECT"
                or not host
                or port != 443
                or not host_allowed(host, self.server.allowed_hosts)
            ):
                client.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
                return
            upstream = pinned_connection(host, port)
        except (ValueError, UnicodeError, PermissionError):
            client.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
            return
        except OSError:
            client.sendall(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
            return
        try:
            client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            _tunnel(client, upstream)
        finally:
            upstream.close()


class _ThreadedServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


@contextmanager
def https_proxy(allowed_hosts):
    server = _ThreadedServer(("127.0.0.1", 0), _ProxyHandler)
    server.allowed_hosts = tuple(allowed_hosts)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
