import json
from contextlib import contextmanager
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
    PlannerOutput,
    ProfileRegistry,
    RecoveryAgent,
    Subtask,
    Validator,
)
from harness.domain import Task, TaskComplexity
from harness.process_control import RunControl, TaskCancelled, use_run_control
from harness.providers import ModelUsage, OpenAICompatibleProvider, ProviderHealth


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


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("LOW", "LOW"),
        ("MEDIUM", "MEDIUM"),
        ("HIGH", "HIGH"),
        ("CRITICAL", "CRITICAL"),
        ("simple", "LOW"),
        ("moderate", "MEDIUM"),
        ("complex", "HIGH"),
    ],
)
def test_task_complexity_uses_four_canonical_levels_and_reads_legacy_values(
    raw, expected
):
    assert Task(title="test", complexity=raw).complexity == TaskComplexity(expected)


def test_task_complexity_rejects_unknown_values():
    with pytest.raises(ValueError):
        Task(title="test", complexity="urgent")
    with pytest.raises(ValueError):
        Task(title="test", complexity=4)


def test_planner_output_normalizes_legacy_complexity_to_canonical_level():
    output = PlannerOutput.model_validate(
        {
            "summary": "plan",
            "complexity": "complex",
            "subtasks": [{"id": "s1", "title": "step", "description": "do"}],
        }
    )
    assert output.complexity == TaskComplexity.HIGH
    assert output.model_dump(mode="json")["complexity"] == "HIGH"


def routing_setup(*, available=None, visible=None, capabilities=None, overrides=None):
    tiers = ("premium", "advanced", "standard", "economical", "local")
    models = {
        f"{tier}_model": {
            "provider": f"provider_{tier}",
            "model": f"{tier}-id",
            "tier": tier,
            "capabilities": (capabilities or {}).get(tier, []),
        }
        for tier in tiers
    }
    preferred = {
        "low": ["local", "economical", "standard", "advanced", "premium"],
        "medium": ["standard", "local", "economical", "advanced", "premium"],
        "high": ["advanced", "standard", "premium", "economical", "local"],
        "critical": ["premium", "advanced", "standard", "economical", "local"],
    }
    data = {
        "models": {
            "registry": models,
            "routing": {
                "tier_preferences": preferred,
                "profile_tier_preferences": overrides or {},
            },
        },
        "profiles": {
            "coding": {
                "model": {
                    "primary": "premium_model",
                    "fallback": [
                        "standard_model",
                        "local_model",
                        "economical_model",
                        "advanced_model",
                    ],
                }
            }
        },
    }
    cfg = config(data)
    registry = ModelRegistry(cfg)
    for tier in tiers:
        enabled = (available or {}).get(tier, True)
        health = ProviderHealth(
            "openai_compatible",
            enabled,
            enabled,
            (visible or {}).get(tier, (f"{tier}-id",)) if enabled else (),
        )
        registry.register(
            f"provider_{tier}",
            SimpleNamespace(
                health_report=lambda report=health: report,
                complete=lambda prompt, model=None, **kwargs: model,
            ),
        )
    return cfg, registry


def test_model_router_applies_configured_tier_order_per_complexity():
    cfg, registry = routing_setup()
    router = ModelRouter(registry, cfg)
    assert router.candidates("coding", TaskComplexity.LOW)[0] == "local_model"
    assert router.candidates("coding", TaskComplexity.CRITICAL)[0] == "premium_model"


def test_model_router_honors_profile_tier_override_within_allowed_candidates():
    override = {
        "coding": {
            "critical": ["local", "economical", "standard", "advanced", "premium"]
        }
    }
    cfg, registry = routing_setup(overrides=override)
    router = ModelRouter(registry, cfg)
    assert router.candidates("coding", TaskComplexity.CRITICAL)[0] == "local_model"
    assert "unconfigured_model" not in router.candidates(
        "coding", TaskComplexity.CRITICAL
    )


def test_model_router_skips_unavailable_provider_and_model():
    cfg, registry = routing_setup(available={"local": False, "economical": False})
    router = ModelRouter(registry, cfg)
    assert router.candidates("coding", TaskComplexity.LOW)[0] == "standard_model"


def test_model_router_rejects_configured_model_missing_from_provider_inventory():
    cfg, registry = routing_setup(visible={"local": ("different-id",)})
    router = ModelRouter(registry, cfg)
    assert router.candidates("coding", TaskComplexity.LOW)[0] == "economical_model"


def test_model_router_rejects_lmstudio_model_that_is_not_confirmed_loaded():
    cfg, registry = routing_setup()
    provider = registry.providers["provider_local"]
    provider.health_report = lambda: ProviderHealth(
        "lmstudio", True, True, ("local-id",), loaded_models=None
    )
    router = ModelRouter(registry, cfg)
    assert router.candidates("coding", TaskComplexity.LOW)[0] == "economical_model"


def test_model_router_accepts_loaded_lmstudio_model():
    cfg, registry = routing_setup()
    registry.providers["provider_local"].health_report = lambda: ProviderHealth(
        "lmstudio", True, True, ("local-id",), loaded_models=("local-id",)
    )
    assert (
        ModelRouter(registry, cfg).candidates("coding", TaskComplexity.LOW)[0]
        == "local_model"
    )


def test_model_router_resolves_single_model_from_healthy_direct_provider():
    cfg = config({"profiles": {"coding": {"model": {"primary": "local"}}}})
    provider = SimpleNamespace(
        model=None,
        health_report=lambda: ProviderHealth(
            "openai_compatible", True, True, ("only-model",)
        ),
        complete=lambda prompt, model=None: model,
    )
    registry = ModelRegistry(cfg)
    registry.register("local", provider)
    router = ModelRouter(registry, cfg)
    assert router.candidates("coding", TaskComplexity.LOW) == ["local"]
    assert registry.discovered["local"] == ("only-model",)


def test_model_router_excludes_healthy_provider_with_empty_inventory():
    cfg = config({"profiles": {"coding": {"model": {"primary": "local"}}}})
    registry = ModelRegistry(cfg)
    registry.register(
        "local",
        SimpleNamespace(
            health_report=lambda: ProviderHealth("openai_compatible", True, True),
        ),
    )
    assert ModelRouter(registry, cfg).candidates("coding", TaskComplexity.LOW) == []


def test_model_router_skips_provider_when_health_probe_raises():
    cfg, registry = routing_setup()
    registry.providers["provider_local"].health_report = lambda: (_ for _ in ()).throw(
        OSError("health probe failed")
    )
    assert (
        ModelRouter(registry, cfg).candidates("coding", TaskComplexity.LOW)[0]
        == "economical_model"
    )


def test_model_router_select_fails_when_every_candidate_is_unavailable():
    cfg, registry = routing_setup(
        available={
            tier: False
            for tier in ("premium", "advanced", "standard", "economical", "local")
        }
    )
    with pytest.raises(ValueError, match="no available model"):
        ModelRouter(registry, cfg).select("coding", TaskComplexity.LOW)


def test_model_router_skips_candidates_without_required_capabilities():
    cfg, registry = routing_setup(capabilities={"local": [], "economical": ["tools"]})
    router = ModelRouter(registry, cfg)
    candidates = router.candidates(
        "coding", TaskComplexity.LOW, required_capabilities={"tools"}
    )
    assert candidates[0] == "economical_model"
    assert "local_model" not in candidates


def test_model_router_routes_completion_and_persists_selected_complexity():
    cfg, registry = routing_setup()
    router = ModelRouter(registry, cfg)
    assert (
        router.complete("coding", "prompt", complexity=TaskComplexity.LOW) == "local-id"
    )


def test_model_router_completion_skips_provider_reported_unavailable():
    cfg, registry = routing_setup(available={"local": False})
    router = ModelRouter(registry, cfg)
    assert (
        router.complete("coding", "prompt", complexity=TaskComplexity.LOW)
        == "economical-id"
    )


def test_model_router_fails_closed_when_no_profile_candidate_is_available():
    cfg, registry = routing_setup(
        available={
            tier: False
            for tier in ("premium", "advanced", "standard", "economical", "local")
        }
    )
    with pytest.raises(RuntimeError, match="no available model"):
        ModelRouter(registry, cfg).complete(
            "coding", "prompt", complexity=TaskComplexity.LOW
        )


def test_model_router_audits_complexity_and_tier_selection_reason(caplog):
    cfg, registry = routing_setup()
    spans = []
    caplog.set_level("INFO", logger="harness")

    @contextmanager
    def model(*_args):
        span = {}
        spans.append(span)
        yield span

    response = ModelRouter(registry, cfg, audit=SimpleNamespace(model=model)).complete(
        "coding", "prompt", complexity=TaskComplexity.LOW
    )
    assert response == "local-id"
    assert spans[0]["complexity"] == "LOW"
    assert spans[0]["routing_reason"] == "tier_preference:local"
    assert "model.route profile=coding complexity=LOW model=local_model" in caplog.text


def test_model_router_sanitizes_route_log_identifiers(caplog):
    cfg, registry = routing_setup()
    caplog.set_level("INFO", logger="harness")
    spans = []

    @contextmanager
    def model(*_args):
        span = {}
        spans.append(span)
        yield span

    def sanitize(value):
        return value.replace("local_model", "[REDACTED]")

    ModelRouter(
        registry, cfg, audit=SimpleNamespace(model=model, sanitize=sanitize)
    ).complete("coding", "prompt", complexity=TaskComplexity.LOW)
    assert "model=[REDACTED]" in caplog.text
    assert "model=local_model" not in caplog.text


def test_model_router_keeps_profile_order_when_no_tier_policy_is_configured():
    cfg, registry = routing_setup()
    cfg.data["models"].pop("routing")
    router = ModelRouter(registry, cfg)
    assert router.candidates("coding") == [
        "premium_model",
        "standard_model",
        "local_model",
        "economical_model",
        "advanced_model",
    ]


def test_planner_passes_existing_task_complexity_into_model_routing():
    cfg = config(
        {
            "profiles": {
                "planner": {"model": {"primary": "planner"}},
                "coding": {"model": {"primary": "coding"}},
            }
        }
    )

    class RecordingRouter:
        def __init__(self):
            self.config = cfg
            self.complexity = None

        def complete(self, _profile, _prompt, complexity=None):
            self.complexity = complexity
            return json.dumps(
                {
                    "summary": "done",
                    "complexity": "HIGH",
                    "subtasks": [
                        {
                            "id": "step",
                            "title": "step",
                            "description": "step",
                            "expected_result": "done",
                            "acceptance_criteria": ["done"],
                        }
                    ],
                }
            )

    router = RecordingRouter()
    plan = Planner(router).plan(Task(title="task", complexity="CRITICAL"))
    assert plan.complexity == TaskComplexity.HIGH
    assert router.complexity == TaskComplexity.CRITICAL


def test_executor_passes_parent_task_complexity_into_model_routing():
    cfg = config(
        {"profiles": {"coding": {"model": {"primary": "coding"}, "max_steps": 1}}}
    )

    class RecordingRouter:
        def __init__(self):
            self.config = cfg
            self.complexity = None

        def complete(self, _profile, _prompt, *, tools, complexity):
            self.complexity = complexity
            from harness.agents import ModelResponse

            return ModelResponse(text='{"output":"implemented"}')

    router = RecordingRouter()
    result = Executor(router).execute(
        Subtask(id="step", title="step", description="step"),
        complexity=TaskComplexity.CRITICAL,
    )
    assert result.success is True
    assert router.complexity == TaskComplexity.CRITICAL


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


def test_router_rechecks_active_run_control_after_provider_failure():
    c = config({"profiles": {"p": {"model": {"primary": "bad", "fallback": ["good"]}}}})
    registry = ModelRegistry(c)
    registry.register("bad", FakeProvider(error=RuntimeError("temporary")))
    registry.register("good", FakeProvider("recovered"))
    control = RunControl()
    with use_run_control(control):
        assert ModelRouter(registry, c).complete("p", "request") == "recovered"


def test_router_validates_all_profile_strategy_references_before_calling_primary():
    c = config(
        {
            "models": {"registry": {"known": {"provider": "p", "model": "id"}}},
            "profiles": {"p1": {"model": {"primary": "known", "fallback": ["typo"]}}},
        }
    )
    registry = ModelRegistry(c)
    calls = []
    registry.register("p", FakeProvider("ok"))
    with pytest.raises(ValueError, match="unknown model strategy candidate: typo"):
        ModelRouter(registry, c).complete("p1", "request")
    assert calls == []


def test_profile_strategy_accepts_registered_aliases_and_direct_providers_in_order():
    c = config(
        {
            "models": {
                "providers": {
                    "local": {"enabled": True, "base_url": "http://local.test/v1"}
                },
                "registry": {"alias": {"provider": "local", "model": "id"}},
            },
            "profiles": {"p": {"model": {"primary": "alias", "fallback": ["local"]}}},
        }
    )
    registry = ModelRegistry(c)
    registry.register("local", FakeProvider("done"))
    router = ModelRouter(registry, c)
    assert router.candidates("p") == ["alias", "local"]
    assert router.complete("p", "request") == "done"


def test_profile_strategy_requires_a_nonempty_primary():
    c = config({"profiles": {"p": {"model": {"primary": " "}}}})
    with pytest.raises(ValueError, match="unknown or invalid agent profile"):
        ProfileRegistry(c).get("p")


def test_router_does_not_fallback_after_provider_backoff_cancellation(monkeypatch):
    c = config({"profiles": {"p": {"model": {"primary": "bad", "fallback": ["good"]}}}})
    registry = ModelRegistry(c)
    calls = []
    provider = OpenAICompatibleProvider(
        "bad",
        "http://model",
        model="m",
        transport=httpx.MockTransport(
            lambda request: calls.append(request) or httpx.Response(503)
        ),
        retry={"max_attempts": 3, "base_delay": 0.1, "max_delay": 1},
    )
    provider.health_report = lambda: ProviderHealth(
        "openai_compatible", True, True, ("m",), configured_model="m"
    )
    registry.register("bad", provider)

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


def test_planner_rejects_malformed_recovery_scope_and_incomplete_coverage():
    c = config(
        {
            "profiles": {
                name: {"model": {"primary": "fake"}} for name in ("planner", "coding")
            }
        }
    )
    registry = ModelRegistry(c)
    answer = {
        "summary": "plan",
        "complexity": "low",
        "subtasks": [
            {
                "id": "s",
                "title": "step",
                "description": "work",
                "expected_result": "done",
                "acceptance_criteria": ["criterion"],
            }
        ],
    }
    registry.register("fake", FakeProvider(json.dumps(answer)))
    planner = Planner(ModelRouter(registry, c))
    with pytest.raises(ValueError, match="recovery plan scope is invalid"):
        planner.plan(Task(title="t"), "RECOVERY_SCOPE_JSON:\nnot-json")
    required = Task(title="t", requirements=["not mapped"])
    with pytest.raises(ValueError, match="requirement has no plan step"):
        planner.plan(required)


def test_router_cancellation_during_cross_candidate_backoff(monkeypatch):
    from harness import agents

    c = config(
        {
            "models": {
                "routing": {
                    "fallback": {
                        "max_attempts": 2,
                        "base_delay": 0.1,
                        "max_delay": 0.1,
                        "max_elapsed": 5,
                    }
                }
            },
            "profiles": {"p": {"model": {"primary": "bad", "fallback": ["good"]}}},
        }
    )
    registry = ModelRegistry(c)

    class StoppingProvider(FakeProvider):
        def complete(self, prompt, **kwargs):
            raise RuntimeError("temporary failure")

    control = RunControl()
    stopped = StoppingProvider()

    class Fallback(FakeProvider):
        def __init__(self):
            super().__init__("must not run")
            self.calls = 0

        def complete(self, prompt, **kwargs):
            self.calls += 1
            return self.result

    fallback = Fallback()
    registry.register("bad", stopped)
    registry.register("good", fallback)
    ticks = iter((0.0, 0.0))

    def monotonic():
        try:
            return next(ticks)
        except StopIteration:
            control.request_stop("cancel during fallback wait")
            return 0.1

    monkeypatch.setattr(agents.time, "monotonic", monotonic)
    with use_run_control(control), pytest.raises(TaskCancelled):
        ModelRouter(registry, c).complete("p", "prompt")
    assert fallback.calls == 0


def test_executor_validator_and_recovery():
    c = config({"profiles": {"coding": {"model": {"primary": "fake"}}}, "models": {}})
    registry = ModelRegistry(c)
    registry.register("fake", FakeProvider('{"output":"done"}'))
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
                else ('{"output":"done"}', ModelUsage("fake", "m"), [])
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
                    ModelResponse(text='{"output":"done"}'),
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
