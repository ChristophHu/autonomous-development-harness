import json

import httpx
import pytest

from harness.mcp import MAX_MESSAGE, MCPError, MCPHTTPClient

URL = "https://mcp.example.test/rpc"
HOSTS = ["mcp.example.test"]
SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}
OUTPUT = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


def response(body, *, headers=None, status=200, content=None):
    return httpx.Response(
        status,
        json=body if content is None else None,
        content=content,
        headers=headers,
        request=httpx.Request("POST", URL),
    )


def tool(name="read", schema=None):
    return {
        "name": name,
        "inputSchema": schema if schema is not None else SCHEMA,
        "outputSchema": OUTPUT,
    }


def install_protocol(
    monkeypatch,
    *,
    result_override=None,
    event_stream=False,
    session=True,
    notify_status=202,
):
    calls = []

    def post(method, url, **kwargs):
        calls.append((method, url, kwargs))
        body = kwargs["json"]
        headers = {"MCP-Session-Id": "session-1"} if session else {}
        if method == "POST" and body.get("method") == "initialize":
            result = {
                "protocolVersion": "2025-11-25",
                "capabilities": {"tools": {}},
            }
        elif body.get("method") == "notifications/initialized":
            return response({}, status=notify_status, headers=headers)
        elif body.get("method") == "tools/list":
            result = {"tools": [tool()]}
        elif body.get("method") == "tools/call":
            result = {"structuredContent": {"ok": True}}
        else:
            result = result_override or {}
        payload = {"jsonrpc": "2.0", "id": body.get("id"), "result": result}
        if event_stream:
            return response(
                {},
                headers=headers | {"content-type": "text/event-stream"},
                content=f"data: {json.dumps(payload)}\n\n".encode(),
            )
        return response(payload, headers=headers)

    monkeypatch.setattr("harness.mcp.http_request", post)
    return calls


def test_mcp_http_transport_initializes_discovers_and_calls_with_bound_auth(
    monkeypatch,
):
    calls = install_protocol(monkeypatch)
    client = MCPHTTPClient(URL, HOSTS, bearer_token="secret-token")

    assert client.discover() == [tool()]
    assert client.call("read", {}, SCHEMA) == {"ok": True}
    assert len(calls) == 5
    assert calls[0][2]["headers"]["Authorization"] == "Bearer secret-token"
    assert calls[2][2]["headers"]["MCP-Session-Id"] == "session-1"
    assert calls[2][2]["headers"]["MCP-Protocol-Version"] == "2025-11-25"


def test_mcp_http_transport_parses_event_stream_and_reuses_initialized_session(
    monkeypatch,
):
    calls = install_protocol(monkeypatch, event_stream=True)
    client = MCPHTTPClient(URL, HOSTS)

    assert client.discover() == [tool()]
    assert client.discover() == [tool()]
    assert [call[2]["json"]["method"] for call in calls].count("initialize") == 1


def test_mcp_http_supports_json_notification_ack_and_servers_without_sessions(
    monkeypatch,
):
    calls = install_protocol(monkeypatch, session=False, notify_status=200)
    client = MCPHTTPClient(URL, HOSTS)
    assert client.discover() == [tool()]
    assert client.session_id is None
    assert len(calls) == 3


@pytest.mark.parametrize(
    "url,hosts",
    [
        ("http://mcp.example.test/rpc", HOSTS),
        ("https://user:pass@mcp.example.test/rpc", HOSTS),
        ("https://other.example.test/rpc", HOSTS),
        ("https://mcp.example.test/rpc?token=x", HOSTS),
        ("https://mcp.example.test:bad/rpc", HOSTS),
    ],
)
def test_mcp_http_transport_rejects_unbound_or_unsafe_endpoints(url, hosts):
    with pytest.raises(ValueError, match="HTTPS and host-allowlisted"):
        MCPHTTPClient(url, hosts)


@pytest.mark.parametrize("timeout", [0, 31, True, "5"])
def test_mcp_http_transport_rejects_invalid_timeout(timeout):
    with pytest.raises(ValueError, match="timeout"):
        MCPHTTPClient(URL, HOSTS, timeout=timeout)


def test_mcp_http_transport_rejects_empty_auth_token():
    with pytest.raises(ValueError, match="bearer token"):
        MCPHTTPClient(URL, HOSTS, bearer_token="")


@pytest.mark.parametrize(
    "fake_response,match",
    [
        (
            response({}, status=302, headers={"location": "https://evil.test"}),
            "redirect",
        ),
        (response({}, status=503), "transport failed"),
        (
            response({}, headers={"content-type": "text/event-stream"}, content=b"bad"),
            "event stream",
        ),
    ],
)
def test_mcp_http_transport_rejects_redirect_http_and_malformed_payload(
    monkeypatch, fake_response, match
):
    monkeypatch.setattr("harness.mcp.http_request", lambda *_a, **_k: fake_response)
    client = MCPHTTPClient(URL, HOSTS)
    with pytest.raises(MCPError, match=match):
        client.discover()


def test_mcp_http_transport_rejects_unexpected_content_type(monkeypatch):
    fake = httpx.Response(
        200,
        text="not json",
        headers={"content-type": "text/plain"},
        request=httpx.Request("POST", URL),
    )
    monkeypatch.setattr("harness.mcp.http_request", lambda *_a, **_k: fake)
    with pytest.raises(MCPError, match="content type"):
        MCPHTTPClient(URL, HOSTS).discover()


def test_mcp_http_transport_enforces_message_limit_session_and_rpc_identity(
    monkeypatch,
):
    client = MCPHTTPClient(URL, HOSTS)
    oversized = response({}, content=b"x" * (MAX_MESSAGE + 1))
    monkeypatch.setattr("harness.mcp.http_request", lambda *_a, **_k: oversized)
    with pytest.raises(MCPError, match="too large"):
        client.discover()

    bad_session = response(
        {"jsonrpc": "2.0", "id": 1, "result": {}},
        headers={"MCP-Session-Id": "bad session"},
    )
    monkeypatch.setattr("harness.mcp.http_request", lambda *_a, **_k: bad_session)
    with pytest.raises(MCPError, match="session identity"):
        MCPHTTPClient(URL, HOSTS).discover()

    bad_id = response({"jsonrpc": "2.0", "id": 999, "result": {}})
    monkeypatch.setattr("harness.mcp.http_request", lambda *_a, **_k: bad_id)
    with pytest.raises(MCPError, match="invalid remote MCP response"):
        MCPHTTPClient(URL, HOSTS).discover()


def test_mcp_http_transport_rejects_bad_server_tools_and_incompatible_protocol(
    monkeypatch,
):
    def incompatible(_method, _url, **kwargs):
        return response(
            {
                "jsonrpc": "2.0",
                "id": kwargs["json"].get("id"),
                "result": {"protocolVersion": "old", "capabilities": {"tools": {}}},
            }
        )

    monkeypatch.setattr("harness.mcp.http_request", incompatible)
    with pytest.raises(MCPError, match="incompatible"):
        MCPHTTPClient(URL, HOSTS).discover()

    client = MCPHTTPClient(URL, HOSTS)
    monkeypatch.setattr(client, "_initialize", lambda: None)
    monkeypatch.setattr(client, "_rpc", lambda *_a: {"tools": [tool("bad name")]})
    with pytest.raises(MCPError, match="tool schema"):
        client.discover()

    monkeypatch.setattr(client, "_rpc", lambda *_a: {"tools": [tool()] * 101})
    with pytest.raises(MCPError, match="tool list"):
        client.discover()
    monkeypatch.setattr(
        client,
        "_rpc",
        lambda *_a: {"tools": [tool(schema={"type": "not-a-json-schema-type"})]},
    )
    with pytest.raises(MCPError, match="tool schema"):
        client.discover()
    monkeypatch.setattr(client, "_rpc", lambda *_a: {"tools": [tool(), tool()]})
    with pytest.raises(MCPError, match="tool schema"):
        client.discover()


def test_mcp_http_call_rejects_changed_schema_remote_error_bad_output_and_text_result(
    monkeypatch,
):
    client = MCPHTTPClient(URL, HOSTS)
    monkeypatch.setattr(client, "_initialize", lambda: None)
    monkeypatch.setattr(
        client,
        "_rpc",
        lambda method, _params: (
            {"tools": [tool()]}
            if method == "tools/list"
            else {"structuredContent": {"ok": "bad"}}
        ),
    )
    with pytest.raises(MCPError, match="invalid output"):
        client.call("read", {}, SCHEMA)
    with pytest.raises(MCPError, match="schema changed"):
        client.call("read", {}, {"type": "object"})
    monkeypatch.setattr(
        client,
        "_rpc",
        lambda method, _params: (
            {"tools": [tool()]} if method == "tools/list" else {"isError": True}
        ),
    )
    with pytest.raises(MCPError, match="reported failure"):
        client.call("read", {}, SCHEMA)
    monkeypatch.setattr(
        client,
        "_rpc",
        lambda method, _params: (
            {"tools": [tool()]}
            if method == "tools/list"
            else {"content": [{"type": "image", "data": "x"}]}
        ),
    )
    with pytest.raises(MCPError, match="no usable result"):
        client.call("read", {}, SCHEMA)
    monkeypatch.setattr(
        client,
        "_rpc",
        lambda method, _params: (
            {"tools": [tool()]}
            if method == "tools/list"
            else {"content": [{"type": "text", "text": "ok"}]}
        ),
    )
    assert client.call("read", {}, SCHEMA) == {"content": "ok"}


def test_mcp_http_accepts_structured_output_without_optional_output_schema(monkeypatch):
    client = MCPHTTPClient(URL, HOSTS)
    monkeypatch.setattr(client, "_initialize", lambda: None)

    def rpc(method, _params):
        if method == "tools/list":
            return {"tools": [{"name": "read", "inputSchema": SCHEMA}]}
        return {"structuredContent": {"arbitrary": "structured"}}

    monkeypatch.setattr(client, "_rpc", rpc)
    assert client.call("read", {}, SCHEMA) == {"arbitrary": "structured"}


def test_tool_registry_pins_remote_server_and_retains_approval_permissions(
    tmp_path, monkeypatch
):
    from harness.core import Config, Permissions
    from harness.tools import ToolRegistry

    instances = []

    class Client:
        def __init__(self, url, allowed_hosts, *, timeout, bearer_token):
            instances.append((url, allowed_hosts, timeout, bearer_token))

        def discover(self):
            return [tool(), tool("not-allowed")]

        def call(self, name, arguments, schema):
            return {"ok": name == "read" and arguments == {} and schema == SCHEMA}

    monkeypatch.setattr("harness.mcp.MCPHTTPClient", Client)
    monkeypatch.setattr(
        "harness.security.SecretResolver",
        lambda: type("Resolver", (), {"get": lambda _self, _name: "resolved-secret"})(),
    )
    config = Config(tmp_path / "missing.yml")
    config.data["paths"]["workspace"] = str(tmp_path)
    config.data["tools"] = {
        "permissions": {"mcp.remote": "write"},
        "mcp": {
            "servers": {
                "remote": {
                    "transport": "streamable_http",
                    "url": URL,
                    "allowed_hosts": HOSTS,
                    "auth_secret": "REMOTE_MCP_TOKEN",
                    "allow_tools": ["read"],
                }
            }
        },
    }

    registry = ToolRegistry(Permissions(config), workspace=tmp_path)

    assert instances == [(URL, HOSTS, 10, "resolved-secret")]
    assert registry.list().count("mcp.remote.read") == 1
    assert "mcp.remote.not-allowed" not in registry.list()
    spec = registry.specs["mcp.remote.read"]
    assert spec.risk_level == "DESTRUCTIVE"
    assert registry.execute("mcp.remote.read", {}) == {"ok": True}


def test_tool_registry_fails_when_remote_auth_reference_is_unavailable(
    tmp_path, monkeypatch
):
    from harness.core import Config, Permissions
    from harness.tools import ToolRegistry

    monkeypatch.setattr(
        "harness.security.SecretResolver",
        lambda: type("Resolver", (), {"get": lambda _self, _name: None})(),
    )
    config = Config(tmp_path / "missing.yml")
    config.data["paths"]["workspace"] = str(tmp_path)
    config.data["tools"] = {
        "mcp": {
            "servers": {
                "remote": {
                    "transport": "streamable_http",
                    "url": URL,
                    "allowed_hosts": HOSTS,
                    "auth_secret": "REMOTE_MCP_TOKEN",
                    "allow_tools": ["read"],
                }
            }
        }
    }
    with pytest.raises(ValueError, match="auth secret is unavailable"):
        ToolRegistry(Permissions(config), workspace=tmp_path)


def test_tool_registry_builds_remote_without_optional_auth(tmp_path, monkeypatch):
    from harness.core import Config, Permissions
    from harness.tools import ToolRegistry

    instances = []

    class Client:
        def __init__(self, url, allowed_hosts, *, timeout, bearer_token):
            instances.append((url, allowed_hosts, timeout, bearer_token))

        def discover(self):
            return []

    monkeypatch.setattr("harness.mcp.MCPHTTPClient", Client)
    config = Config(tmp_path / "missing.yml")
    config.data["paths"]["workspace"] = str(tmp_path)
    config.data["tools"] = {
        "mcp": {
            "servers": {
                "remote": {
                    "transport": "streamable_http",
                    "url": URL,
                    "allowed_hosts": HOSTS,
                    "allow_tools": ["read"],
                }
            }
        }
    }

    registry = ToolRegistry(Permissions(config), workspace=tmp_path)

    assert instances == [(URL, HOSTS, 10, None)]
    assert not any(name.startswith("mcp.remote.") for name in registry.list())
