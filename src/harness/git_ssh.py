"""Strict URL, key and host-resolution contracts for SSH Git transport."""

import base64
import hashlib
import ipaddress
import os
import re
import socket
import socketserver
import stat
import subprocess
import threading
from contextlib import contextmanager
from urllib.parse import urlsplit

from .git_http import _tunnel
from .process_control import run_cancellable

_USER = re.compile(r"[A-Za-z0-9._-]{1,64}\Z")
_PATH = re.compile(r"/?[A-Za-z0-9._/-]{1,512}\Z")
_KEY_TYPES = {
    "ssh-ed25519",
    "ecdsa-sha2-nistp256",
    "ecdsa-sha2-nistp384",
    "ecdsa-sha2-nistp521",
    "ssh-rsa",
}


def parse_ssh_url(url, allowed_hosts, allowed_ports=(22,)):
    """Return a canonical SSH URL and validated target tuple."""
    if not isinstance(url, str) or not url:
        raise PermissionError("SSH Git URL is invalid")
    username = ""
    try:
        if url.startswith("ssh://"):
            parsed = urlsplit(url)
            port = parsed.port or 22
            host = parsed.hostname or ""
            username = parsed.username or ""
            path = parsed.path.lstrip("/")
            if (
                parsed.scheme != "ssh"
                or parsed.password
                or parsed.query
                or parsed.fragment
                or "%" in url
            ):
                raise PermissionError("SSH Git URL has unsupported components")
        else:
            match = re.fullmatch(
                r"(?:(?P<user>[A-Za-z0-9._-]{1,64})@)?"
                r"(?P<host>\[[0-9A-Fa-f:]+\]|[A-Za-z0-9.-]+):"
                r"(?P<path>[A-Za-z0-9._/-]{1,512})",
                url,
            )
            if not match:
                raise PermissionError("SSH Git URL must use ssh:// or scp syntax")
            username = match.group("user") or ""
            host = match.group("host").strip("[]")
            port = 22
            path = match.group("path")
    except ValueError as error:
        raise PermissionError("SSH Git URL port is invalid") from error
    try:
        canonical_host = ipaddress.ip_address(host).compressed
    except ValueError:
        try:
            canonical_host = host.encode("idna").decode("ascii").lower().rstrip(".")
        except UnicodeError as error:
            raise PermissionError("SSH Git host is invalid") from error
    if not canonical_host or not any(
        canonical_host == item.lower().rstrip(".") for item in allowed_hosts
    ):
        raise PermissionError("SSH Git host is not allowlisted")
    if port not in allowed_ports:
        raise PermissionError("SSH Git port is not allowlisted")
    if username and not _USER.fullmatch(username):
        raise PermissionError("SSH Git username is invalid")
    if (
        not _PATH.fullmatch(path)
        or "//" in path
        or any(part in {".", ".."} for part in path.split("/"))
    ):
        raise PermissionError("SSH Git repository path is invalid")
    authority = f"[{canonical_host}]" if ":" in canonical_host else canonical_host
    normalized = f"ssh://{username + '@' if username else ''}{authority}"
    if port != 22:
        normalized += f":{port}"
    normalized += f"/{path}"
    return normalized, canonical_host, username, port, path


def _decode_key(line):
    if not isinstance(line, str) or "\n" in line or "\r" in line:
        raise PermissionError("SSH public key is invalid")
    parts = line.split()
    if len(parts) < 2 or parts[0] not in _KEY_TYPES:
        raise PermissionError("SSH public key type is not supported")
    try:
        blob = base64.b64decode(parts[1], validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise PermissionError("SSH public key data is invalid") from error
    if len(blob) < 4:
        raise PermissionError("SSH public key data is invalid")
    length = int.from_bytes(blob[:4], "big")
    if length == 0 or 4 + length > len(blob):
        raise PermissionError("SSH public key data is invalid")
    try:
        embedded_type = blob[4 : 4 + length].decode("ascii")
    except UnicodeError as error:
        raise PermissionError("SSH public key data is invalid") from error
    if embedded_type != parts[0]:
        raise PermissionError("SSH public key data does not match its key type")
    return parts[0], parts[1], blob


def key_fingerprint(line):
    """Compute OpenSSH SHA256 fingerprint from a public-key line."""
    _key_type, _encoded, blob = _decode_key(line)
    return "SHA256:" + base64.b64encode(hashlib.sha256(blob).digest()).decode().rstrip(
        "="
    )


def validate_host_keys(host, keys):
    """Validate configured host keys; entries never include host patterns."""
    if not isinstance(keys, list) or not keys:
        raise PermissionError(f"SSH host key pin is missing for {host}")
    decoded = []
    for line in keys:
        try:
            key_type, encoded, _blob = _decode_key(line)
        except PermissionError as error:
            raise PermissionError(f"SSH host key pin is invalid for {host}") from error
        decoded.append([key_type, encoded])
    return decoded


def resolve_ssh_addresses(host, port):
    """Resolve once, then pin SSH to a public address to prevent DNS rebinding."""
    try:
        records = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as error:
        raise PermissionError("SSH Git host could not be resolved") from error
    addresses = set()
    for _family, _socktype, _protocol, _canonname, sockaddr in records:
        try:
            address = ipaddress.ip_address(sockaddr[0])
        except ValueError:
            continue
        if address.is_global:
            addresses.add(address.compressed)
    if not addresses:
        raise PermissionError("SSH Git host resolved only to non-public addresses")
    return sorted(addresses, key=lambda item: (":" in item, item))


def agent_identities(agent_socket):
    """List public keys from one validated, same-user SSH-agent socket."""
    if not isinstance(agent_socket, str) or not os.path.isabs(agent_socket):
        raise PermissionError("SSH agent socket path is invalid")
    try:
        info = os.lstat(agent_socket)
    except OSError as error:
        raise PermissionError("SSH agent socket is unavailable") from error
    if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
        raise PermissionError("SSH agent socket must be a same-user Unix socket")
    try:
        result = run_cancellable(
            subprocess.run,
            ["/usr/bin/ssh-add", "-L"],
            env={"PATH": "/usr/bin:/bin", "SSH_AUTH_SOCK": agent_socket},
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PermissionError("SSH agent identities could not be read") from error
    if result.returncode:
        raise PermissionError("SSH agent has no available identities")
    identities = []
    for line in result.stdout.splitlines():
        key_type, encoded, _blob = _decode_key(line)
        identities.append([key_type, encoded])
    if not identities:
        raise PermissionError("SSH agent has no available identities")
    return identities


class _SshTunnelHandler(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            upstream = socket.create_connection(self.server.target, timeout=15)
        except OSError:
            return
        try:
            _tunnel(self.request, upstream)
        finally:
            upstream.close()


class _ThreadedSshServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


@contextmanager
def ssh_proxy(address, port):
    """Tunnel one pinned SSH destination through a short-lived loopback port."""
    server = _ThreadedSshServer(("127.0.0.1", 0), _SshTunnelHandler)
    server.target = (address, port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
