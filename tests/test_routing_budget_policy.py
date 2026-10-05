import pytest

from harness.agents import ModelBudgetExceeded, ModelRegistry, ModelRouter
from harness.audit import AuditRecorder
from harness.core import Config, Store
from harness.database import ModelRunRepository
from harness.domain import Task
from harness.providers import ModelUsage


class FixtureProvider:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = 0

    def complete(self, prompt, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return self.response


def router_config(max_attempts=3):
    config = Config()
    config.data["models"]["routing"] = {
        "fallback": {
            "max_attempts": max_attempts,
            "base_delay": 0,
            "max_delay": 0,
            "max_elapsed": 60,
        }
    }
    config.data["profiles"] = {
        "p": {"model": {"primary": "first", "fallback": ["second", "third"]}}
    }
    return config


def test_task_cost_budget_excess_is_persisted_and_does_not_fallback(tmp_path):
    config = router_config()
    config.data["models"]["rates"] = {"model": {"input": 150000, "output": 150000}}
    config.data["paths"]["database"] = str(tmp_path / "budget.sqlite")
    store = Store(config)
    database = store.database
    audit = AuditRecorder(database)
    registry = ModelRegistry(config)
    first = FixtureProvider(("answer", ModelUsage("first", "model", 2, 2, 0.6)))
    second = FixtureProvider(("fallback", ModelUsage("second", "model", 1, 1, 0.1)))
    registry.register("first", first)
    registry.register("second", second)
    registry.register("third", second)
    task = store.create(Task(title="budget", model_cost_budget=0.5))
    with (
        audit.agent(task, "coder", "p"),
        pytest.raises(ModelBudgetExceeded, match="cost budget"),
    ):
        ModelRouter(registry, config, audit=audit).complete("p", "prompt")
    assert first.calls == 1 and second.calls == 0
    totals = ModelRunRepository(database).usage_report(task_id=task.id)
    with database.connect() as connection:
        row = connection.execute(
            "SELECT status,cost FROM model_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert tuple(row) == ("failed", 0.6)
    assert totals["totals"]["runs"] == 1


def test_token_budget_rejects_unreported_usage_without_fallback(tmp_path):
    config = router_config()
    config.data["paths"]["database"] = str(tmp_path / "tokens.sqlite")
    store = Store(config)
    database = store.database
    audit = AuditRecorder(database)
    registry = ModelRegistry(config)
    first = FixtureProvider(("answer", ModelUsage("first", "model")))
    second = FixtureProvider(("fallback", ModelUsage("second", "model", 1, 1, 0)))
    registry.register("first", first)
    registry.register("second", second)
    registry.register("third", second)
    task = store.create(Task(title="budget", model_token_budget=20))
    with (
        audit.agent(task, "coder", "p"),
        pytest.raises(ModelBudgetExceeded, match="token budget"),
    ):
        ModelRouter(registry, config, audit=audit).complete("p", "prompt")
    assert first.calls == 1 and second.calls == 0


def test_audit_calibration_report_reads_only_matching_numeric_samples(tmp_path):
    config = router_config()
    config.data["paths"]["database"] = str(tmp_path / "calibration.sqlite")
    audit = AuditRecorder(Store(config).database)

    report = audit.model_token_calibration("model", "characters_per_token:20")

    assert report == {"sample_count": 0, "calibrated": False, "multiplier": 1.0}


def test_task_budget_rejects_legacy_response_without_usage(tmp_path):
    config = router_config()
    config.data["paths"]["database"] = str(tmp_path / "legacy.sqlite")
    store = Store(config)
    audit = AuditRecorder(store.database)
    task = store.create(Task(title="budget", model_cost_budget=1.0))
    registry = ModelRegistry(config)
    first = FixtureProvider("answer-without-usage")
    second = FixtureProvider("fallback")
    for name, provider in (("first", first), ("second", second), ("third", second)):
        registry.register(name, provider)
    with (
        audit.agent(task, "coder", "p"),
        pytest.raises(ModelBudgetExceeded, match="reported model usage"),
    ):
        ModelRouter(registry, config, audit=audit).complete("p", "prompt")
    assert first.calls == 1 and second.calls == 0


@pytest.mark.parametrize(
    "budget,usage,error",
    [
        ("model_cost_budget", {"cost": 1.0}, "cost budget"),
        ("model_cost_budget", {"cost": None}, "cost budget"),
        (
            "model_token_budget",
            {"prompt_tokens": 15, "completion_tokens": 5},
            "token budget",
        ),
        (
            "model_token_budget",
            {"prompt_tokens": None, "completion_tokens": 1},
            "unverifiable",
        ),
    ],
)
def test_preexisting_task_usage_stops_model_call(tmp_path, budget, usage, error):
    config = router_config()
    config.data["paths"]["database"] = str(tmp_path / "prior.sqlite")
    store = Store(config)
    audit = AuditRecorder(store.database)
    task = store.create(
        Task(title="budget", **{budget: 20 if "token" in budget else 1.0})
    )
    with store.database.connect() as connection:
        connection.execute(
            "INSERT INTO model_runs(task_id,agent,profile,complexity,provider,model,status,started_at,created_at,fallback_index,prompt_tokens,completion_tokens,cost) VALUES(?,?,?,?,?,?, 'completed',?,?,0,?,?,?)",
            (
                task.id,
                "coder",
                "p",
                None,
                "first",
                "model",
                store.database.now(),
                store.database.now(),
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
                usage.get("cost"),
            ),
        )
    registry = ModelRegistry(config)
    provider = FixtureProvider(("must not run", ModelUsage("first", "model", 1, 1, 0)))
    for name in ("first", "second", "third"):
        registry.register(name, provider)
    with (
        audit.agent(task, "coder", "p"),
        pytest.raises(ModelBudgetExceeded, match=error),
    ):
        ModelRouter(registry, config, audit=audit).complete("p", "prompt")
    assert provider.calls == 0


def test_router_attempt_budget_stops_before_later_candidates():
    config = router_config(max_attempts=1)
    registry = ModelRegistry(config)
    first = FixtureProvider(error=RuntimeError("unavailable"))
    second = FixtureProvider(("unused", ModelUsage("second", "model", 1, 1, 0)))
    registry.register("first", first)
    registry.register("second", second)
    registry.register("third", second)
    with pytest.raises(RuntimeError, match="all model providers failed"):
        ModelRouter(registry, config).complete("p", "prompt")
    assert first.calls == 1 and second.calls == 0


def test_router_backoff_is_exponential_and_capped(monkeypatch):
    from harness import agents

    config = router_config()
    config.data["models"]["routing"]["fallback"].update(
        max_attempts=3, base_delay=0.2, max_delay=0.3
    )
    registry = ModelRegistry(config)
    first = FixtureProvider(error=RuntimeError("unavailable"))
    second = FixtureProvider(error=RuntimeError("unavailable"))
    third = FixtureProvider(("ok", ModelUsage("third", "model", 1, 1, 0)))
    for name, provider in (("first", first), ("second", second), ("third", third)):
        registry.register(name, provider)
    delays = []
    monkeypatch.setattr(agents.time, "sleep", delays.append)
    assert ModelRouter(registry, config).complete("p", "prompt") == "ok"
    assert delays == [0.2, 0.3]


def test_router_total_elapsed_budget_stops_before_next_candidate(monkeypatch):
    from harness import agents

    config = router_config()
    config.data["models"]["routing"]["fallback"].update(max_elapsed=1)
    registry = ModelRegistry(config)
    first = FixtureProvider(error=RuntimeError("unavailable"))
    second = FixtureProvider(("unused", ModelUsage("second", "model", 1, 1, 0)))
    registry.register("first", first)
    registry.register("second", second)
    registry.register("third", second)
    ticks = iter((0.0, 0.1, 2.0))
    monkeypatch.setattr(agents.time, "monotonic", lambda: next(ticks, 2.0))
    with pytest.raises(RuntimeError, match="all model providers failed"):
        ModelRouter(registry, config).complete("p", "prompt")
    assert first.calls == 1 and second.calls == 0


def test_router_elapsed_budget_can_expire_before_first_attempt(monkeypatch):
    from harness import agents

    config = router_config()
    config.data["models"]["routing"]["fallback"]["max_elapsed"] = 1
    registry = ModelRegistry(config)
    first = FixtureProvider(("unused", ModelUsage("first", "model", 1, 1, 0)))
    for name in ("first", "second", "third"):
        registry.register(name, first)
    ticks = iter((0.0, 1.0))
    monkeypatch.setattr(agents.time, "monotonic", lambda: next(ticks))
    with pytest.raises(RuntimeError, match="all model providers failed"):
        ModelRouter(registry, config).complete("p", "prompt")
    assert first.calls == 0
