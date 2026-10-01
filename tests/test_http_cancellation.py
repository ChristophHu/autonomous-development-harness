"""Abort and lease loss interrupt task-owned HTTP without fallback side effects."""

import asyncio
import threading
from types import SimpleNamespace

import httpx
import pytest

from harness.agents import Executor, ModelRegistry, ModelRouter, Subtask
from harness.core import Config, Orchestrator, Store
from harness.http_control import request
from harness.memory import EmbeddingProvider, QdrantMemory
from harness.process_control import RunControl, TaskCancelled, use_run_control
from harness.providers import OpenAICompatibleProvider
from harness.tools import ToolExecutor


def blocking_transport(started, closed):
    async def handle(request):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    return httpx.MockTransport(handle)


def interrupt(call, reason="task_aborted"):
    control = RunControl()
    started = threading.Event()
    closed = threading.Event()
    errors = []

    def worker():
        try:
            with use_run_control(control):
                call(blocking_transport(started, closed))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    assert started.wait(3)
    control.request_stop(reason)
    thread.join(3)
    assert not thread.is_alive()
    assert closed.wait(3)
    assert len(errors) == 1 and isinstance(errors[0], TaskCancelled)
    assert reason in str(errors[0])


def test_provider_request_cancels_network_io():
    interrupt(
        lambda transport: OpenAICompatibleProvider(
            "local", "http://model/v1", model="m", transport=transport
        ).complete("prompt")
    )


def test_router_does_not_fallback_after_lease_loss():
    tried = []

    def call(transport):
        config = SimpleNamespace(
            data={
                "profiles": {"p": {"model": {"primary": "slow", "fallback": ["good"]}}}
            }
        )
        registry = ModelRegistry(config)
        registry.register(
            "slow",
            OpenAICompatibleProvider(
                "slow", "http://model/v1", model="m", transport=transport
            ),
        )
        registry.register(
            "good", SimpleNamespace(complete=lambda *a, **k: tried.append(True))
        )
        ModelRouter(registry, config).complete("p", "prompt")

    interrupt(call, "lease_lost")
    assert tried == []


def test_http_tool_cancels_active_request(monkeypatch):
    async_client = httpx.AsyncClient

    def call(transport):
        monkeypatch.setattr(
            "httpx.AsyncClient", lambda **kwargs: async_client(transport=transport)
        )
        tool = ToolExecutor(
            SimpleNamespace(require=lambda *a, **k: None),
            http_allow_hosts=["localhost"],
            http_private_hosts=["localhost"],
        )
        tool.http("GET", "http://localhost/data")

    interrupt(call)


def test_embedding_and_qdrant_cancel(monkeypatch):
    async_client = httpx.AsyncClient

    def call_embedding(transport):
        monkeypatch.setattr(
            "httpx.AsyncClient", lambda **kwargs: async_client(transport=transport)
        )
        EmbeddingProvider("http://embed/v1", "m").embed("text")

    interrupt(call_embedding)

    def call_qdrant(transport):
        monkeypatch.setattr(
            "httpx.AsyncClient", lambda **kwargs: async_client(transport=transport)
        )
        QdrantMemory("http://qdrant", "c").ensure_collection()

    interrupt(call_qdrant)


def test_cancelled_before_request_never_opens_transport():
    control = RunControl()
    control.request_stop("task_aborted")
    with use_run_control(control), pytest.raises(TaskCancelled):
        OpenAICompatibleProvider("p", "http://model", model="m").complete("x")


def test_active_http_success_and_hard_deadline():
    async def fast(_request):
        return httpx.Response(200, text="done")

    async def slow(_request):
        await asyncio.Event().wait()

    with use_run_control(RunControl()):
        response = request(
            "GET", "http://test", timeout=1, transport=httpx.MockTransport(fast)
        )
        assert response.text == "done"
        with pytest.raises(httpx.ReadTimeout, match="deadline"):
            request(
                "GET",
                "http://test",
                timeout=0.03,
                transport=httpx.MockTransport(slow),
            )


def test_injected_sync_client_checks_control_before_and_after():
    control = RunControl()

    def stop_and_respond(url, **kwargs):
        control.request_stop("lease_lost")
        return httpx.Response(200)

    custom = SimpleNamespace(get=stop_and_respond)
    with use_run_control(control), pytest.raises(TaskCancelled):
        request("GET", "http://test", timeout=1, client=custom)
    with use_run_control(control), pytest.raises(TaskCancelled):
        request("GET", "http://test", timeout=1, client=custom)


def test_injected_httpx_client_retains_its_transport():
    observed = []
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda req: observed.append(req.url.host) or httpx.Response(200)
        )
    )
    with use_run_control(RunControl()):
        assert (
            request("GET", "http://fixture", timeout=1, client=client).status_code
            == 200
        )
    assert observed == ["fixture"]


def test_executor_propagates_cancellation():
    config = SimpleNamespace(data={"profiles": {"coding": {"model": {"primary": "p"}}}})
    router = SimpleNamespace(
        config=config,
        complete=lambda *a, **k: (_ for _ in ()).throw(TaskCancelled("lease_lost")),
    )
    with pytest.raises(TaskCancelled, match="lease_lost"):
        Executor(router).execute(Subtask(id="s", title="s", description="s"))


def test_configured_phase_and_total_deadlines_are_used(tmp_path):
    config = Config()
    config.data["paths"]["database"] = str(tmp_path / "task.db")
    config.data["tools"]["mcp"]["servers"]["vault"]["enabled"] = False
    limits = {"total": 1, "connect": 0.2, "read": 0.3}
    config.data["models"]["providers"]["lmstudio"]["timeout"] = limits
    config.data["memory"]["embeddings"]["timeout"] = limits
    config.data["memory"]["qdrant"]["timeout"] = limits
    config.data["tools"]["http"]["timeout"] = limits
    assert config.validate()
    assert ModelRegistry(config).get("lmstudio").timeout == limits
    runtime = Orchestrator(Store(config), config)
    assert runtime.tools.executor.http_timeout == limits
    assert runtime.qdrant.timeout == limits
    assert runtime.qdrant.embedder.timeout == limits
    observed = []

    def respond(req):
        observed.append(req.extensions["timeout"])
        return httpx.Response(200)

    with use_run_control(RunControl()):
        assert (
            request(
                "GET",
                "http://test",
                timeout=limits,
                transport=httpx.MockTransport(respond),
            ).status_code
            == 200
        )
    assert observed == [{"connect": 0.2, "read": 0.3, "write": 1, "pool": 1}]


@pytest.mark.parametrize(
    "value,match",
    [
        ({"connect": 1}, "requires total"),
        ({"total": 2, "other": 1}, "permits connect/read"),
        ({"total": 2, "read": 3}, "cannot exceed total"),
        ({"total": True}, "between 0 and 300"),
        (0, "between 0 and 300"),
        ("slow", "between 0 and 300"),
        (301, "between 0 and 300"),
    ],
)
def test_invalid_http_timeouts_fail_config_validation(value, match):
    config = Config()
    config.data["memory"]["qdrant"]["timeout"] = value
    with pytest.raises(ValueError, match=match):
        config.validate()


def test_timeout_validation_covers_every_http_section():
    config = Config()
    config.data["memory"]["embeddings"]["timeout"] = -1
    with pytest.raises(ValueError, match="memory.embeddings.timeout"):
        config.validate()
    config.data["memory"]["embeddings"]["timeout"] = 30
    config.data["models"]["providers"]["lmstudio"]["timeout"] = -1
    with pytest.raises(ValueError, match="models.providers.lmstudio.timeout"):
        config.validate()
    config.data["models"]["providers"]["lmstudio"]["timeout"] = 120
    config.data["tools"]["http"]["timeout"] = -1
    with pytest.raises(ValueError, match="tools.http.timeout"):
        config.validate()


def test_http_request_does_not_follow_redirects_by_default():
    def redirect(_request):
        return httpx.Response(302, headers={"Location": "http://foreign.test/"})

    client = httpx.Client(transport=httpx.MockTransport(redirect))
    try:
        response = request("GET", "http://allowed.test/", timeout=1, client=client)
        assert response.status_code == 302
        assert response.history == []
    finally:
        client.close()


def test_http_request_forwards_redirect_policy_to_default_client(monkeypatch):
    response = httpx.Response(200)
    observed = {}

    def fake_request(method, url, **kwargs):
        observed.update(method=method, url=url, **kwargs)
        return response

    monkeypatch.setattr(httpx, "request", fake_request)
    assert request("GET", "https://allowed.test/", timeout=2) is response
    assert observed["follow_redirects"] is False
