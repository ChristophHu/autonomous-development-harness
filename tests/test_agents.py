import asyncio
import json
from contextlib import contextmanager
from types import SimpleNamespace

import httpx
import pytest
from jsonschema import ValidationError

from harness.agents import (
    AgentProfile,
    Complexity,
    Executor,
    ExecutorOutput,
    ModelRegistry,
    ModelResponse,
    ModelRouter,
    Planner,
    PlannerOutput,
    ProfileRegistry,
    RecoveryAgent,
    Subtask,
    Validator,
    execute_plan_dag,
    normalize_provider_response,
)
from harness.domain import Task, TaskComplexity
from harness.process_control import RunControl, TaskCancelled, use_run_control
from harness.providers import (
    ModelUsage,
    OpenAICompatibleProvider,
    ProviderHealth,
    ToolCall,
)


def test_plan_dag_runs_disjoint_ready_steps_concurrently_and_dependencies_after():
    steps = [
        Subtask(id="a", title="A", description="", write_paths=["src/a.py"]),
        Subtask(id="b", title="B", description="", write_paths=["src/b.py"]),
        Subtask(
            id="c",
            title="C",
            description="",
            dependencies=["a", "b"],
            write_paths=["src/c.py"],
        ),
    ]
    active = maximum = 0
    completed = []

    async def worker(step):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.01)
        active -= 1
        completed.append(step.id)
        return ExecutorOutput(subtask_id=step.id, success=True, output=step.id)

    outputs = asyncio.run(execute_plan_dag(steps, worker, max_parallel_steps=2))
    assert maximum == 2
    assert completed.index("c") > completed.index("a")
    assert completed.index("c") > completed.index("b")
    assert [item.subtask_id for item in outputs] == ["a", "b", "c"]


@pytest.mark.parametrize(
    "result",
    [
        "answer",
        ("answer", ModelUsage("fixture", "m")),
        ("answer", ModelUsage("fixture", "m"), [ToolCall("id", "tool", {})]),
        ModelResponse(text="answer", usage=ModelUsage("fixture", "m")),
    ],
)
def test_provider_response_normalizer_returns_common_typed_fields(result):
    text, usage, calls = normalize_provider_response(result)
    assert text == "answer"
    assert usage is None or isinstance(usage, ModelUsage)
    assert isinstance(calls, list)
    if isinstance(result, tuple) and len(result) == 3:
        assert calls == result[2]


@pytest.mark.parametrize(
    "result",
    [
        None,
        (),
        ("answer",),
        ("answer", "invalid usage"),
        ("answer", ModelUsage("fixture", "m"), "invalid calls"),
        ("answer", ModelUsage("fixture", "m"), ["invalid call"]),
    ],
)
def test_provider_response_normalizer_rejects_malformed_shapes(result):
    with pytest.raises(TypeError):
        normalize_provider_response(result)


def test_model_router_returns_typed_usage_with_tool_response():
    usage = ModelUsage("fixture", "fixture-model", prompt_tokens=2, completion_tokens=3)
    response = ModelResponse(text="ok", usage=usage)
    cfg = config({"profiles": {"p": {"model": {"primary": "fixture"}}}})
    registry = ModelRegistry(cfg)
    registry.register("fixture", FakeProvider(response))

    result = ModelRouter(registry, cfg).complete("p", "prompt", tools=[])

    assert isinstance(result, ModelResponse)
    assert result.usage is usage
    assert result.text == "ok"


def test_model_router_applies_exact_model_token_budget_before_provider_call(
    monkeypatch,
):
    import harness.agents as agents_module

    provider = FakeProvider("ok")
    calls = []
    provider.complete = lambda prompt, **kwargs: calls.append((prompt, kwargs)) or "ok"
    cfg = config(
        {
            "models": {
                "registry": {"alias": {"provider": "fake", "model": "model-id"}},
                "input_token_budgets": {
                    "model-id": {
                        "encoding": "fixture",
                        "max_input_tokens": 1,
                        "framing_tokens": 0,
                    }
                },
            },
            "profiles": {"p": {"model": {"primary": "alias"}}},
        }
    )
    registry = ModelRegistry(cfg)
    registry.register("fake", provider)
    monkeypatch.setattr(
        agents_module, "count_with_tiktoken", lambda _: lambda text: len(text)
    )

    with pytest.raises(RuntimeError, match="all model providers failed"):
        ModelRouter(registry, cfg).complete("p", "too large", tools=[])
    assert calls == []


def test_model_router_enforces_configured_tokenizer_framing_allowance(monkeypatch):
    import harness.agents as agents_module

    cfg = config(
        {
            "models": {
                "registry": {"alias": {"provider": "fake", "model": "model-id"}},
                "input_token_budgets": {
                    "model-id": {
                        "encoding": "fixture",
                        "max_input_tokens": 1000,
                        "framing_tokens": 7,
                    }
                },
            },
            "profiles": {"p": {"model": {"primary": "alias"}}},
        }
    )
    registry = ModelRegistry(cfg)
    registry.register("fake", FakeProvider("ok"))
    monkeypatch.setattr(agents_module, "count_with_tiktoken", lambda _: lambda _text: 0)

    assert ModelRouter(registry, cfg).complete("p", "prompt", tools=[]).text == "ok"


def test_model_router_records_estimate_and_provider_usage_calibration():
    spans = []

    class Audit:
        @staticmethod
        @contextmanager
        def model(*_args):
            span = {}
            yield span
            spans.append(span)

        @staticmethod
        def sanitize(value):
            return value

    usage = ModelUsage("fake", "model-id", prompt_tokens=4, completion_tokens=1)
    cfg = config(
        {
            "models": {
                "registry": {"alias": {"provider": "fake", "model": "model-id"}},
                "input_token_budgets": {
                    "model-id": {
                        "characters_per_token": 100,
                        "max_input_tokens": 1000,
                        "safety_margin_percent": 25,
                    }
                },
            },
            "profiles": {"p": {"model": {"primary": "alias"}}},
        }
    )
    registry = ModelRegistry(cfg)
    registry.register("fake", FakeProvider(("ok", usage)))

    assert (
        ModelRouter(registry, cfg, audit=Audit()).complete("p", "hello", tools=[]).text
        == "ok"
    )
    assert spans[0]["input_token_preflight"]["method"] == "characters_per_token"
    assert spans[0]["input_token_preflight"]["provider_exact"] is False
    assert spans[0]["input_token_calibration"]["provider_prompt_tokens"] == 4
    assert spans[0]["input_token_calibration"]["estimate_minus_provider"] == (
        spans[0]["input_token_preflight"]["estimated_tokens"] - 4
    )


def test_model_router_restricts_allowed_candidates_and_reports_actual_model():
    cfg = config(
        {
            "models": {
                "registry": {
                    "writer": {"provider": "writer-provider", "model": "writer-id"},
                    "reviewer": {
                        "provider": "reviewer-provider",
                        "model": "reviewer-id",
                    },
                }
            },
            "profiles": {
                "review": {"model": {"primary": "writer", "fallback": ["reviewer"]}}
            },
        }
    )
    registry = ModelRegistry(cfg)
    registry.register("writer-provider", FakeProvider("writer"))
    registry.register("reviewer-provider", FakeProvider("reviewed"))
    routed = []
    result = ModelRouter(registry, cfg).complete(
        "review",
        "check",
        allowed_models={"reviewer"},
        route_observer=routed.append,
    )
    assert result == "reviewed"
    assert routed == ["reviewer-id"]


@pytest.mark.parametrize(
    ("allowed_models", "error"),
    [("reviewer", TypeError), ({"unknown"}, RuntimeError)],
)
def test_model_router_rejects_invalid_or_empty_allowed_model_set(allowed_models, error):
    cfg = config({"profiles": {"p": {"model": {"primary": "fake"}}}})
    registry = ModelRegistry(cfg)
    registry.register("fake", FakeProvider("ok"))
    with pytest.raises(error):
        ModelRouter(registry, cfg).complete(
            "p", "prompt", allowed_models=allowed_models
        )


def test_configured_token_budget_requires_every_fallback_model():
    cfg = config(
        {
            "models": {
                "registry": {
                    "first": {"provider": "first-provider", "model": "first-id"},
                    "fallback": {
                        "provider": "fallback-provider",
                        "model": "fallback-id",
                    },
                },
                "input_token_budgets": {
                    "first-id": {
                        "characters_per_token": 4,
                        "max_input_tokens": 1000,
                    }
                },
            },
            "profiles": {
                "p": {"model": {"primary": "first", "fallback": ["fallback"]}}
            },
        }
    )
    registry = ModelRegistry(cfg)
    registry.register("first-provider", FakeProvider(error=RuntimeError("offline")))
    registry.register("fallback-provider", FakeProvider("must not be called"))
    with pytest.raises(RuntimeError, match="all model providers failed"):
        ModelRouter(registry, cfg).complete("p", "prompt")


@pytest.mark.parametrize(
    "paths",
    [
        (["src/shared"], ["src/shared/file.py"]),
        ([], ["src/b.py"]),
        (["src/a.py"], []),
        ([""], ["src/b.py"]),
    ],
)
def test_plan_dag_serializes_conflicting_or_undeclared_writes(paths):
    steps = [
        Subtask(id="a", title="A", description="", write_paths=paths[0]),
        Subtask(id="b", title="B", description="", write_paths=paths[1]),
    ]
    active = maximum = 0

    async def worker(step):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.005)
        active -= 1
        return ExecutorOutput(subtask_id=step.id, success=True, output="ok")

    asyncio.run(execute_plan_dag(steps, worker, max_parallel_steps=4))
    assert maximum == 1


def test_plan_dag_bounds_concurrency_and_skips_failed_dependencies():
    steps = [
        Subtask(id="a", title="A", description="", write_paths=["a.py"]),
        Subtask(
            id="b", title="B", description="", dependencies=["a"], write_paths=["b.py"]
        ),
        Subtask(id="c", title="C", description="", write_paths=["c.py"]),
    ]
    calls = []

    async def worker(step):
        calls.append(step.id)
        return ExecutorOutput(
            subtask_id=step.id, success=step.id != "a", output="result"
        )

    persisted = []
    outputs = asyncio.run(
        execute_plan_dag(
            steps,
            worker,
            max_parallel_steps=1,
            on_complete=lambda step, output: persisted.append(
                (step.id, output.success)
            ),
        )
    )
    assert calls == ["a", "c"]
    assert outputs[1].output == "dependency failed"
    assert [item.subtask_id for item in outputs] == ["a", "b", "c"]
    assert persisted == [("a", False), ("b", False), ("c", True)]


@pytest.mark.parametrize("limit", [0, 33, True, "2"])
def test_plan_dag_rejects_invalid_parallelism(limit):
    async def worker(_step):
        raise AssertionError("invalid limit must fail before scheduling")

    with pytest.raises(ValueError, match="max_parallel_steps"):
        asyncio.run(
            execute_plan_dag(
                [Subtask(id="a", title="A", description="")],
                worker,
                max_parallel_steps=limit,
            )
        )


def test_plan_dag_surfaces_worker_error_after_joining_siblings():
    steps = [
        Subtask(id="a", title="A", description="", write_paths=["a.py"]),
        Subtask(id="b", title="B", description="", write_paths=["b.py"]),
    ]
    finished = []

    async def worker(step):
        await asyncio.sleep(0)
        finished.append(step.id)
        if step.id == "a":
            raise RuntimeError("worker failed")
        return ExecutorOutput(subtask_id=step.id, success=True, output="ok")

    with pytest.raises(RuntimeError, match="worker failed"):
        asyncio.run(execute_plan_dag(steps, worker, max_parallel_steps=2))
    assert sorted(finished) == ["a", "b"]


def test_plan_dag_rejects_graph_that_cannot_progress():
    invalid_step = SimpleNamespace(
        id="blocked", dependencies=["missing"], write_paths=["blocked.py"]
    )

    async def worker(_step):
        raise AssertionError("unreachable step cannot run")

    with pytest.raises(ValueError, match="cannot make progress"):
        asyncio.run(execute_plan_dag([invalid_step], worker))


def test_plan_dag_awaits_async_completion_persistence():
    step = Subtask(id="a", title="A", description="", write_paths=["a.py"])
    persisted = []

    async def worker(current):
        return ExecutorOutput(subtask_id=current.id, success=True, output="ok")

    async def persist(current, output):
        await asyncio.sleep(0)
        persisted.append((current.id, output.output))

    asyncio.run(execute_plan_dag([step], worker, on_complete=persist))
    assert persisted == [("a", "ok")]


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


def test_profile_catalog_lists_only_valid_configured_profiles():
    registry = ProfileRegistry(
        config(
            {
                "profiles": {
                    "coding": {"model": {"primary": "local"}},
                    "planner": {"model": {"primary": "local"}},
                }
            }
        )
    )
    assert tuple(registry.catalog()) == ("coding", "planner")


def test_profile_catalog_rejects_non_mapping_profiles():
    with pytest.raises(TypeError, match="profiles must be a mapping"):
        ProfileRegistry(config({"profiles": []})).catalog()


def test_agent_role_profile_defaults_and_configured_override():
    registry = ProfileRegistry(
        config(
            {
                "profiles": {
                    "planner": {"model": {"primary": "local"}},
                    "coding": {"model": {"primary": "local"}},
                    "validator": {"model": {"primary": "local"}},
                    "test-engineer": {"model": {"primary": "local"}},
                    "review": {"model": {"primary": "local"}},
                },
                "agent_roles": {"independent-review": "review"},
            }
        )
    )
    assert registry.for_role("executor").name == "coding"
    assert registry.for_role("executor").capabilities == ["execute"]
    assert registry.for_role("requirements").capabilities == ["requirements"]
    assert registry.for_role("independent-review").name == "review"
    with pytest.raises(ValueError, match="unknown agent role"):
        registry.for_role("unknown")


def test_agent_dispatch_rejects_explicit_profile_capability_mismatch():
    registry = ProfileRegistry(
        config(
            {
                "profiles": {
                    "coding": {
                        "model": {"primary": "local"},
                        "capabilities": ["test"],
                    }
                }
            }
        )
    )
    with pytest.raises(ValueError, match="lacks required capabilities for executor"):
        registry.for_role("executor")


def test_agent_dispatch_accepts_explicit_minimum_role_capability():
    registry = ProfileRegistry(
        config(
            {
                "profiles": {
                    "coding": {
                        "model": {"primary": "local"},
                        "capabilities": ["execute"],
                    }
                }
            }
        )
    )
    assert registry.for_role("executor").capabilities == ["execute"]


def test_agent_profile_validates_typed_input_and_output_contracts():
    profile = AgentProfile(
        name="typed",
        instructions="follow contract",
        model="local",
        input_schema={
            "type": "object",
            "properties": {"task_id": {"type": "integer"}},
            "required": ["task_id"],
            "additionalProperties": False,
        },
        output_schema={"type": "string", "minLength": 1},
    )
    payload = {"task_id": 7}
    assert profile.validate_input(payload) is payload
    assert profile.validate_output("done") == "done"
    with pytest.raises(ValidationError):
        profile.validate_input({"task_id": "7"})
    with pytest.raises(ValidationError):
        profile.validate_output("")


def test_agent_profile_without_contract_is_backward_compatible():
    profile = AgentProfile(name="legacy", instructions="", model="local")
    payload = {"anything": True}
    assert profile.validate_input(payload) is payload
    assert profile.validate_output(payload) is payload


def test_required_contract_mode_rejects_profile_missing_either_contract():
    registry = ProfileRegistry(
        config(
            {
                "agents": {"contract_mode": "required"},
                "profiles": {"p": {"model": {"primary": "local"}}},
            }
        )
    )
    with pytest.raises(ValueError, match="requires input_schema and output_schema"):
        registry.for_agent("custom", "p")


def test_required_contract_mode_rejects_unknown_role_without_derived_contract():
    assert ProfileRegistry._role_contracts("custom") is None
    registry = ProfileRegistry(
        config(
            {
                "agents": {"contract_mode": "required"},
                "profiles": {"p": {"model": {"primary": "local"}}},
            }
        )
    )
    with pytest.raises(ValueError, match="requires input_schema and output_schema"):
        registry.for_agent("custom", "p")


def test_required_contract_mode_allows_explicit_legacy_profile():
    registry = ProfileRegistry(
        config(
            {
                "agents": {"contract_mode": "required"},
                "profiles": {
                    "p": {"model": {"primary": "local"}, "contract_mode": "legacy"}
                },
            }
        )
    )
    assert registry.for_agent("custom", "p").name == "p"


def test_required_contract_mode_accepts_and_enforces_both_schemas():
    schema = {
        "type": "object",
        "required": ["ok"],
        "properties": {"ok": {"type": "boolean"}},
        "additionalProperties": False,
    }
    registry = ProfileRegistry(
        config(
            {
                "agents": {"contract_mode": "required"},
                "profiles": {
                    "p": {
                        "model": {"primary": "local"},
                        "contract_mode": "strict",
                        "input_schema": schema,
                        "output_schema": schema,
                    }
                },
            }
        )
    )
    profile = registry.for_agent("custom", "p")
    assert profile.validate_input({"ok": True}) == {"ok": True}
    with pytest.raises(ValidationError):
        profile.validate_output({"ok": "yes"})


@pytest.mark.parametrize(
    "role",
    [
        "planner",
        "requirements",
        "executor",
        "tester",
        "validator",
        "recovery-inspector",
        "independent-review",
    ],
)
def test_required_contract_mode_derives_contracts_for_builtin_roles(role):
    names = {"planner", "coding", "test-engineer", "validator"}
    registry = ProfileRegistry(
        config(
            {
                "agents": {"contract_mode": "required"},
                "profiles": {name: {"model": {"primary": "local"}} for name in names},
            }
        )
    )
    profile = registry.for_role(role)
    assert profile.input_schema["type"] == "object"
    assert profile.output_schema["type"] == "object"
    with pytest.raises(ValidationError):
        profile.validate_input(None)
    with pytest.raises(ValidationError):
        profile.validate_output(None)


def test_contract_mode_treats_non_mapping_runtime_settings_as_legacy():
    registry = ProfileRegistry(
        config(
            {
                "agents": [],
                "profiles": {"p": {"model": {"primary": "local"}}},
            }
        )
    )
    assert registry.for_agent("custom", "p").name == "p"


def test_required_mode_rejects_profile_with_only_one_schema():
    registry = ProfileRegistry(
        config(
            {
                "agents": {"contract_mode": "required"},
                "profiles": {
                    "p": {
                        "model": {"primary": "local"},
                        "input_schema": {"type": "object"},
                    }
                },
            }
        )
    )
    with pytest.raises(ValueError, match="requires input_schema and output_schema"):
        registry.for_agent("custom", "p")


def test_profile_strict_mode_requires_contracts_under_optional_global_mode():
    registry = ProfileRegistry(
        config(
            {
                "agents": {"contract_mode": "optional"},
                "profiles": {
                    "p": {
                        "model": {"primary": "local"},
                        "contract_mode": "strict",
                    }
                },
            }
        )
    )
    with pytest.raises(ValueError, match="requires input_schema and output_schema"):
        registry.for_agent("custom", "p")


def test_agent_profile_rejects_malformed_contract_schema():
    registry = ProfileRegistry(
        config(
            {
                "profiles": {
                    "broken": {
                        "model": {"primary": "local"},
                        "input_schema": "not-json-schema",
                    }
                }
            }
        )
    )
    with pytest.raises(ValueError, match="input_schema must be an object"):
        registry.get("broken")


@pytest.mark.parametrize(
    "data,message",
    [
        (
            {"profiles": {"p": {"model": {"primary": "local"}}}, "agent_roles": []},
            "agent_roles must be a mapping",
        ),
        (
            {
                "profiles": {"p": {"model": {"primary": "local"}}},
                "agent_roles": {"review": "missing"},
            },
            "unknown or invalid agent profile: missing",
        ),
        (
            {
                "profiles": {"p": {"model": {"primary": "local"}}},
                "agent_roles": {"review": " "},
            },
            "invalid profile",
        ),
    ],
)
def test_agent_role_catalog_rejects_invalid_configuration(data, message):
    with pytest.raises((ValueError, TypeError), match=message):
        ProfileRegistry(config(data)).role_profiles()


def test_agent_role_catalog_rejects_non_string_role_name():
    with pytest.raises(ValueError, match="role names"):
        ProfileRegistry(
            config({"profiles": {}, "agent_roles": {1: "p"}})
        ).role_profiles()


def test_planner_dispatches_by_role_and_canonicalizes_step_profile():
    plan = {
        "summary": "role-selected plan",
        "complexity": "LOW",
        "subtasks": [
            {
                "id": "s1",
                "title": "implement",
                "description": "do work",
                "assigned_agent": "executor",
                "profile": "coding-specialist",
                "expected_result": "file updated",
                "acceptance_criteria": ["c1"],
            }
        ],
    }

    config_for_test = config(
        {
            "profiles": {
                name: {"model": {"primary": "local"}, "tools": []}
                for name in ("planner-specialist", "coding-specialist")
            },
            "agent_roles": {
                "planner": "planner-specialist",
                "executor": "coding-specialist",
            },
        }
    )
    config_for_test.data["profiles"]["planner-specialist"].update(
        {
            "input_schema": {
                "type": "object",
                "required": ["title"],
                "properties": {"title": {"type": "string"}},
            },
            "output_schema": {
                "type": "object",
                "required": ["summary"],
                "properties": {"summary": {"type": "string"}},
            },
        }
    )

    class Router:
        config = config_for_test

        def complete(self, profile, *_args, **_kwargs):
            self.selected_profile = profile
            return json.dumps(plan)

    router = Router()
    task = Task(
        title="role routing",
        acceptance_criteria=[{"id": "c1", "description": "file updated"}],
    )
    result = Planner(router).plan(task)
    assert router.selected_profile == "planner-specialist"
    assert result.subtasks[0].profile == "coding-specialist"


def test_planner_rejects_profile_that_bypasses_role_assignment():
    class Router:
        config = config(
            {
                "profiles": {
                    name: {"model": {"primary": "local"}, "tools": []}
                    for name in ("planner-specialist", "coding-specialist", "unsafe")
                },
                "agent_roles": {
                    "planner": "planner-specialist",
                    "executor": "coding-specialist",
                },
            }
        )

        def complete(self, *_args, **_kwargs):
            return json.dumps(
                {
                    "summary": "bad plan",
                    "complexity": "LOW",
                    "subtasks": [
                        {
                            "id": "s1",
                            "title": "unsafe",
                            "description": "try escalation",
                            "assigned_agent": "executor",
                            "profile": "unsafe",
                            "expected_result": "none",
                            "acceptance_criteria": ["c1"],
                        }
                    ],
                }
            )

    with pytest.raises(ValueError, match="does not match configured dispatch"):
        Planner(Router()).plan(
            Task(
                title="role boundary",
                acceptance_criteria=[{"id": "c1", "description": "done"}],
            )
        )


def test_agent_assignment_rejects_builtin_role_profile_escalation_and_allows_unknown_role():
    registry = ProfileRegistry(
        config(
            {
                "profiles": {
                    name: {"model": {"primary": "local"}}
                    for name in ("coding", "unsafe", "custom")
                }
            }
        )
    )
    with pytest.raises(ValueError, match="does not match dispatch policy"):
        registry.for_agent("executor", "unsafe")
    assert registry.for_agent("custom-agent", "custom").name == "custom"


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
    isolated_config = Config()
    isolated_config.data["tools"]["mcp"] = {"servers": {}}
    tool_registry = ToolRegistry(Permissions(isolated_config), workspace=tmp_path)
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
