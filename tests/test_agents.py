import json
from types import SimpleNamespace

import httpx
import pytest

from harness.agents import (
    AgentProfile,
    Complexity,
    Executor,
    ModelRegistry,
    ModelRouter,
    Planner,
    ProfileRegistry,
    RecoveryAgent,
    Subtask,
    Validator,
)
from harness.domain import Task
from harness.process_control import RunControl, TaskCancelled, use_run_control
from harness.providers import ModelUsage, OpenAICompatibleProvider


class FakeProvider:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    def complete(self, prompt, **kwargs):
        if self.error:
            raise self.error
        return self.result


def config(data):
    return SimpleNamespace(data=data)


def test_registry_profiles_and_selection():
    c = config(
        {
            "models": {"providers": {"disabled": {"enabled": False}}},
            "profiles": {
                "coding": {
                    "model": {"primary": "fake"},
                    "permissions": ["filesystem.read"],
                    "tools": ["filesystem.read"],
                    "max_steps": 2,
                }
            },
        }
    )
    registry = ModelRegistry(c)
    registry.register("fake", FakeProvider("ok"))
    with pytest.raises(ValueError):
        registry.get("missing")
    assert registry.get("fake").complete("x") == "ok"
    assert "disabled" not in registry.providers
    profile = ProfileRegistry(c).get("coding")
    assert profile.permissions == ["filesystem.read"] and profile.max_steps == 2
    assert isinstance(
        AgentProfile(name="p", instructions="i", model="fake"), AgentProfile
    )
    assert ModelRouter(registry, c).select("coding", Complexity.SIMPLE) is registry.get(
        "fake"
    )


def test_discovery_supplies_single_unconfigured_runtime_model_without_mutating_config():
    calls = []
    provider = SimpleNamespace(
        model=None,
        models=lambda: [" local-b ", "local-a", "local-a", "   "],
        complete=lambda prompt, **kwargs: calls.append(kwargs) or "ok",
    )
    cfg = config({"profiles": {"p": {"model": {"primary": "local"}}}})
    registry = ModelRegistry(cfg)
    registry.register("local", provider)
    assert registry.discover("local") == ("local-a", "local-b")
    with pytest.raises(RuntimeError, match="local: ValueError"):
        ModelRouter(registry, cfg).complete("p", "prompt")
    assert calls == []
    provider.models = lambda: ["local-a"]
    assert registry.discover("local") == ("local-a",)
    assert ModelRouter(registry, cfg).complete("p", "prompt") == "ok"
    assert calls == [{"model": "local-a"}]
    with pytest.raises(ValueError, match="capabilities"):
        registry.resolve("local", needs_tools=True)
    assert cfg.data == {"profiles": {"p": {"model": {"primary": "local"}}}}


def test_explicit_model_precedes_discovery_and_failed_refresh_keeps_no_stale_snapshot():
    cfg = config(
        {
            "models": {
                "registry": {
                    "chosen": {
                        "provider": "local",
                        "model": "configured-id",
                        "capabilities": ["tools"],
                    }
                }
            },
            "profiles": {"p": {"model": {"primary": "chosen"}}},
        }
    )
    provider = SimpleNamespace(
        model=None,
        models=lambda: ["discovered-id"],
        complete=lambda prompt, **kwargs: kwargs["model"],
    )
    registry = ModelRegistry(cfg)
    registry.register("local", provider)
    assert registry.discover("local") == ("discovered-id",)
    assert ModelRouter(registry, cfg).complete("p", "prompt") == "configured-id"
    provider.models = lambda: (_ for _ in ()).throw(RuntimeError("offline secret"))
    with pytest.raises(RuntimeError):
        registry.discover("local")
    assert registry.discovered["local"] == ()
    assert ModelRouter(registry, cfg).complete("p", "prompt") == "configured-id"


def test_discovery_rejects_malformed_inventory_and_never_selects_many():
    registry = ModelRegistry(config({}))
    provider = SimpleNamespace(model=None, models=lambda: {"data": ["bad"]})
    registry.register("local", provider)
    with pytest.raises(ValueError, match="inventory"):
        registry.discover("local")
    provider.models = lambda: ["valid", 123]
    with pytest.raises(ValueError, match="inventory"):
        registry.discover("local")
    provider.models = lambda: ["valid\nunsafe"]
    with pytest.raises(ValueError, match="inventory"):
        registry.discover("local")
    assert registry.discovered["local"] == ()


def test_replacing_provider_invalidates_discovery_snapshot():
    registry = ModelRegistry(config({}))
    registry.register("local", SimpleNamespace(model=None, models=lambda: ["old"]))
    assert registry.discover("local") == ("old",)
    registry.register("local", SimpleNamespace(model=None, models=lambda: ["new"]))
    assert registry.resolve("local")[1] == "new"


def test_registry_passes_explicit_lmstudio_kind_to_provider():
    registry = ModelRegistry(
        config(
            {
                "models": {
                    "providers": {
                        "local": {
                            "enabled": True,
                            "kind": "lmstudio",
                            "base_url": "http://127.0.0.1:1234/v1",
                        }
                    }
                }
            }
        )
    )
    assert registry.get("local").kind == "lmstudio"


def test_router_usage_and_callback():
    usage = ModelUsage("provider", "m", 10, 20)
    observed = []
    c = config(
        {
            "profiles": {"p": {"model": {"primary": "fake"}}},
            "models": {"rates": {"m": {"input": 1, "output": 2}}},
        }
    )
    registry = ModelRegistry(c)
    registry.register("fake", FakeProvider(("answer", usage)))
    router = ModelRouter(registry, c, observed.append)
    assert router.complete("p", "prompt") == "answer"
    assert observed == [usage] and router.usage.total() == 0.00005
    router_without_callback = ModelRouter(registry, c)
    assert router_without_callback.complete("p", "prompt") == "answer"


def test_router_fallback_and_exhaustion():
    c = config({"profiles": {"p": {"model": {"primary": "bad", "fallback": ["good"]}}}})
    r = ModelRegistry(c)
    r.register("bad", FakeProvider(error=RuntimeError("down")))
    r.register("good", FakeProvider("recovered"))
    assert ModelRouter(r, c).complete("p", "x") == "recovered"
    r.register("good", FakeProvider(error=OSError("offline")))
    with pytest.raises(RuntimeError, match="all model providers failed"):
        ModelRouter(r, c).complete("p", "x")


def test_router_does_not_fallback_after_provider_backoff_cancellation(monkeypatch):
    c = config({"profiles": {"p": {"model": {"primary": "bad", "fallback": ["good"]}}}})
    registry = ModelRegistry(c)
    calls = []
    registry.register(
        "bad",
        OpenAICompatibleProvider(
            "bad",
            "http://model",
            model="m",
            transport=httpx.MockTransport(
                lambda request: calls.append(request) or httpx.Response(503)
            ),
            retry={"max_attempts": 3, "base_delay": 0.1, "max_delay": 1},
        ),
    )

    class CountingProvider(FakeProvider):
        def __init__(self):
            super().__init__("must not run")
            self.calls = 0

        def complete(self, prompt, **kwargs):
            self.calls += 1
            return self.result

    fallback = CountingProvider()
    registry.register("good", fallback)
    control = RunControl()
    wait = control.stop_event.wait

    def stop_during_backoff(delay):
        if delay > 0.02:
            control.request_stop("test cancellation")
            return True
        return wait(delay)

    monkeypatch.setattr(control.stop_event, "wait", stop_during_backoff)
    with use_run_control(control), pytest.raises(TaskCancelled):
        ModelRouter(registry, c).complete("p", "prompt")
    assert len(calls) == 1 and fallback.calls == 0


def test_planner_structured_and_fallback():
    c = config(
        {
            "models": {},
            "profiles": {
                name: {"model": {"primary": "fake"}} for name in ("planner", "coding")
            },
        }
    )
    r = ModelRegistry(c)
    valid = {
        "summary": "s",
        "complexity": "simple",
        "subtasks": [
            {
                "id": "s1",
                "title": "do",
                "description": "d",
                "expected_result": "done",
                "acceptance_criteria": ["done"],
            }
        ],
    }
    r.register("fake", FakeProvider(json.dumps(valid)))
    planner = Planner(ModelRouter(r, c))
    assert planner.plan(Task(title="t", description="d")).subtasks[0].id == "s1"
    r.register("fake", FakeProvider("{}"))
    with pytest.raises(ValueError):
        planner.plan(Task(title="t", description="d"))
    r.register(
        "fake",
        FakeProvider(
            json.dumps({"summary": "empty", "complexity": "simple", "subtasks": []})
        ),
    )
    with pytest.raises(ValueError):
        planner.plan(Task(title="t", description="d"))


def test_executor_validator_and_recovery():
    c = config({"profiles": {"coding": {"model": {"primary": "fake"}}}, "models": {}})
    registry = ModelRegistry(c)
    registry.register("fake", FakeProvider("done"))
    router = ModelRouter(registry, c)
    result = Executor(router).execute(
        Subtask(id="x", title="write", description="x"), "context"
    )
    assert result.success
    failed = type(
        "Router",
        (),
        {
            "config": c,
            "complete": lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("no provider")
            ),
        },
    )()
    output = Executor(failed).execute(Subtask(id="x", title="write", description="x"))
    assert not output.success
    report = Validator().validate(None, [result, output])
    assert not report.valid and report.errors
    recovery = RecoveryAgent().recover(RuntimeError("crash"))
    assert not recovery.recovered


def test_executor_dispatches_only_profile_granted_tools(tmp_path):
    from harness.core import Config, Permissions
    from harness.providers import ToolCall
    from harness.tools import ToolRegistry

    c = config(
        {
            "models": {},
            "profiles": {
                "coding": {
                    "model": {"primary": "fake"},
                    "permissions": ["filesystem"],
                    "tools": ["filesystem.read"],
                    "max_steps": 2,
                }
            },
        }
    )
    registry = ModelRegistry(c)

    class ToolProvider:
        def __init__(self):
            self.calls = 0

        def complete(self, prompt, tools=None):
            self.calls += 1
            return (
                (
                    "",
                    ModelUsage("fake", "m"),
                    [ToolCall("id", "filesystem.read", {"path": "x"})],
                )
                if self.calls == 1
                else ("done", ModelUsage("fake", "m"), [])
            )

    registry.register("fake", ToolProvider())
    c.data["profiles"]["coding"]["model"] = {"primary": "fake"}
    c.data["tools"] = {"permissions": {"filesystem": "read"}}
    router = ModelRouter(registry, c)
    tool_registry = ToolRegistry(Permissions(Config()), workspace=tmp_path)
    (tmp_path / "x").write_text("contents")
    result = Executor(router, tool_registry).execute(
        Subtask(id="1", title="read", description="read it")
    )
    assert result.success and result.output == "done"
    assert "filesystem.read" in [
        schema["name"]
        for schema in tool_registry.schemas(["filesystem.read", "shell.execute"])
    ]


def test_executor_tracks_move_sources_destinations_and_deduplicates(tmp_path):
    from harness.agents import ModelResponse
    from harness.providers import ToolCall

    c = config(
        {
            "profiles": {
                "coding": {
                    "model": {"primary": "fake"},
                    "permissions": ["filesystem"],
                    "tools": ["filesystem.move"],
                    "max_steps": 2,
                }
            }
        }
    )

    class Router:
        config = c

        def __init__(self):
            self.responses = iter(
                [
                    ModelResponse(
                        text="",
                        tool_calls=[
                            ToolCall(
                                "move-1",
                                "filesystem.move",
                                {"path": "old.py", "destination": "new.py"},
                            ),
                            ToolCall(
                                "move-2",
                                "filesystem.move",
                                {"path": "old.py", "destination": "new.py"},
                            ),
                        ],
                    ),
                    ModelResponse(text="done"),
                ]
            )

        def complete(self, *args, **kwargs):
            return next(self.responses)

    class Tools:
        def schemas(self, names):
            return [{"name": name} for name in names]

        def execute(self, *args, **kwargs):
            return "moved"

    result = Executor(Router(), Tools()).execute(
        Subtask(id="move", title="move", description="move file")
    )
    assert result.success
    assert result.changed_files == ["new.py", "old.py"]
    assert result.tool_evidence[0]["changed_paths"] == ["new.py", "old.py"]


def test_executor_rejects_ungranted_call_and_step_exhaustion():
    from harness.agents import ModelResponse
    from harness.providers import ToolCall

    c = config(
        {
            "profiles": {
                "coding": {
                    "model": {"primary": "fake"},
                    "permissions": ["filesystem"],
                    "tools": ["filesystem.read"],
                    "max_steps": 1,
                }
            }
        }
    )

    class Router:
        config = c

        def __init__(self, responses):
            self.responses = iter(responses)

        def complete(self, *args, **kwargs):
            return next(self.responses)

    class Tools:
        def schemas(self, names):
            return [{"name": name} for name in names]

        def execute(self, *args, **kwargs):
            return "x"

    denied = Executor(
        Router(
            [ModelResponse(text="", tool_calls=[ToolCall("1", "shell.execute", {})])]
        ),
        Tools(),
    )
    assert not denied.execute(Subtask(id="1", title="x", description="x")).success
    loop = Executor(
        Router(
            [
                ModelResponse(
                    text="",
                    tool_calls=[
                        ToolCall("1", "filesystem.read", {"path": "x"}),
                        ToolCall("1b", "filesystem.read", {"path": "x"}),
                    ],
                ),
                ModelResponse(
                    text="",
                    tool_calls=[ToolCall("2", "filesystem.read", {"path": "x"})],
                ),
            ]
        )
    )
    loop.tools = Tools()
    result = loop.execute(Subtask(id="1", title="x", description="x"))
    assert not result.success and "step limit" in result.output


def test_router_wraps_plain_response_for_tool_enabled_request():
    c = config({"profiles": {"p": {"model": {"primary": "fake"}}}, "models": {}})

    class Provider:
        def complete(self, prompt, tools=None):
            return "plain response"

    registry = ModelRegistry(c)
    registry.register("fake", Provider())
    response = ModelRouter(registry, c).complete("p", "prompt", tools=[])
    assert response.text == "plain response" and response.tool_calls == []
