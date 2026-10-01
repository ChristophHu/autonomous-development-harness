import json
import logging
import time
from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest
from pydantic import BaseModel

from harness.agents import (
    DecisionOutput,
    Executor,
    ExecutorOutput,
    ExecutorReport,
    ModelRegistry,
    ModelResponse,
    ModelRouter,
    RecoveryOutput,
    Subtask,
    TaskClassificationOutput,
    ValidatorOutput,
)
from harness.approvals import (
    ApprovalDenied,
    ApprovalRequired,
    ApprovalService,
    ToolApprovalTarget,
)
from harness.core import Config, Permissions
from harness.process_control import RunControl, TaskCancelled, use_run_control
from harness.providers import (
    OpenAICompatibleProvider,
    ProviderError,
    parse_retry_after,
)
from harness.requirements import RequirementProposal
from harness.retry_budget import RetryBudget, current_retry_budget, use_retry_budget
from harness.structured_output import StructuredOutputError, parse_model_output
from harness.tools import ToolRegistry
from harness.validation import IndependentReviewOutput


class Output(BaseModel):
    value: int


@pytest.mark.parametrize(
    "schema,payload",
    [
        (
            ExecutorOutput,
            {"subtask_id": "s", "success": True, "output": "ok", "unexpected": True},
        ),
        (ValidatorOutput, {"valid": True, "unexpected": True}),
        (RecoveryOutput, {"recovered": False, "action": "stop", "unexpected": True}),
        (DecisionOutput, {"decision": "x", "rationale": "y", "unexpected": True}),
        (
            TaskClassificationOutput,
            {"complexity": "low", "profile": "coding", "unexpected": True},
        ),
    ],
)
def test_agent_output_contracts_reject_undeclared_fields(schema, payload):
    with pytest.raises(ValueError):
        schema.model_validate(payload)


@pytest.mark.parametrize(
    "response",
    [
        "plain completion",
        '{"output":"ok","changed_files":["claimed.py"]}',
        '{"output":7}',
    ],
)
def test_executor_report_requires_only_a_strict_output_summary(response):
    with pytest.raises(ValueError):
        parse_model_output(ExecutorReport, response, agent="executor")


def test_retry_budget_is_shared_and_restored():
    budget = RetryBudget.start(2, 10)
    assert budget.claim()
    with use_retry_budget(budget):
        assert current_retry_budget() is budget
        assert budget.claim()
        assert not budget.claim()
    assert current_retry_budget() is None
    assert budget.remaining > 0


def test_retry_budget_bounds_wait_and_elapsed(monkeypatch):
    ticks = iter((10.0, 10.0, 12.0, 12.0))
    monkeypatch.setattr("harness.retry_budget.time.monotonic", lambda: next(ticks))
    budget = RetryBudget(1, 1.0, 10.0)
    assert budget.remaining == 1
    assert budget.claim()
    assert not budget.claim()
    budget.wait(2)


def test_retry_budget_wait_with_running_task(monkeypatch):
    control = RunControl()
    waits = []
    monkeypatch.setattr(
        control.stop_event, "wait", lambda delay: waits.append(delay) or False
    )
    with use_run_control(control):
        RetryBudget(1, 3, time.monotonic()).wait(0.01)
    assert len(waits) == 1


def test_retry_budget_wait_propagates_task_cancellation():
    control = RunControl()
    control.request_stop("cancel retry")
    with use_run_control(control), pytest.raises(TaskCancelled):
        RetryBudget(1, 3, time.monotonic()).wait(0.01)


def test_provider_retries_consume_router_shared_attempts():
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(503, request=request)

    provider = OpenAICompatibleProvider(
        "test",
        "http://provider/v1",
        model="m",
        transport=httpx.MockTransport(respond),
        retry={"max_attempts": 5, "base_delay": 0, "max_delay": 0},
    )
    budget = RetryBudget.start(2, 5)
    assert budget.claim()
    with use_retry_budget(budget), pytest.raises(ProviderError):
        provider.complete("prompt")
    assert len(calls) == 2
    assert budget.attempts == 2


def test_permanent_http_errors_are_not_provider_retries():
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(401, request=request)

    provider = OpenAICompatibleProvider(
        "test", "http://provider/v1", model="m", transport=httpx.MockTransport(respond)
    )
    with pytest.raises(ProviderError) as error:
        provider.complete("prompt")
    assert error.value.category == "provider_permanent"
    assert not error.value.fallback_allowed
    assert len(calls) == 1


def test_router_stops_on_permanent_provider_error_without_fallback():
    config = SimpleNamespace(
        data={
            "models": {},
            "profiles": {"p": {"model": {"primary": "bad", "fallback": ["good"]}}},
        }
    )
    registry = ModelRegistry(config)
    permanent = OpenAICompatibleProvider(
        "bad",
        "http://provider/v1",
        model="m",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(400, request=request)
        ),
    )

    class Fallback:
        calls = 0

        def complete(self, *_args, **_kwargs):
            self.calls += 1
            return "must not run"

    fallback = Fallback()
    registry.register("bad", permanent)
    registry.register("good", fallback)
    with pytest.raises(ProviderError, match="HTTP 400"):
        ModelRouter(registry, config).complete("p", "prompt")
    assert fallback.calls == 0


def test_retry_after_http_date_is_capped_by_provider_policy(monkeypatch):
    delays = []
    responses = iter((429, 200))

    def respond(request):
        status = next(responses)
        if status == 429:
            return httpx.Response(
                status,
                headers={"Retry-After": "Wed, 21 Oct 2030 07:28:00 GMT"},
                request=request,
            )
        return httpx.Response(
            status,
            json={"choices": [{"message": {"content": "ok"}}]},
            request=request,
        )

    monkeypatch.setattr("harness.providers.time.sleep", delays.append)
    provider = OpenAICompatibleProvider(
        "test",
        "http://provider/v1",
        model="m",
        transport=httpx.MockTransport(respond),
        retry={"max_attempts": 2, "base_delay": 0.1, "max_delay": 1},
    )
    assert provider.complete("prompt")[0] == "ok"
    assert delays == [1]


def test_retry_after_parser_supports_bounded_numeric_and_http_dates(monkeypatch):
    now = datetime(2026, 1, 1, tzinfo=UTC)
    assert parse_retry_after("2.5", now=now) == 2.5
    assert parse_retry_after("Thu, 01 Jan 2026 00:00:05 GMT", now=now) == 5
    assert parse_retry_after("Wed, 31 Dec 2025 23:59:59 GMT", now=now) == 0
    assert parse_retry_after("-1", now=now) is None
    assert parse_retry_after("inf", now=now) is None
    assert parse_retry_after("not-a-date", now=now) is None
    assert parse_retry_after(None, now=now) is None

    from harness import providers

    monkeypatch.setattr(
        providers,
        "parsedate_to_datetime",
        lambda _value: datetime(2026, 1, 1),  # noqa: DTZ001 - exercise naive HTTP dates
    )
    assert parse_retry_after("naive date", now=datetime(2026, 1, 1)) == 0  # noqa: DTZ001


def test_parse_model_output_validates_and_does_not_leak_raw(caplog):
    assert parse_model_output(Output, '{"value":2}', agent="planner").value == 2
    secret = "private-model-response-secret"
    with (
        caplog.at_level(logging.WARNING, logger="harness"),
        pytest.raises(StructuredOutputError, match="planner"),
    ):
        parse_model_output(Output, json.dumps({"value": secret}), agent="planner")
    assert secret not in caplog.text


def test_agent_output_schemas_reject_extra_fields_and_coercion():
    proposal = parse_model_output(
        RequirementProposal,
        json.dumps(
            {
                "fields": {"goal": "reviewed goal"},
                "rationale": "cited source",
                "evidence": {"goal": [{"source": "task", "ref": "task:1"}]},
            }
        ),
        agent="requirements",
    )
    assert proposal.fields["goal"] == "reviewed goal"
    for raw in (
        '{"fields":{"invented":"x"},"rationale":"source","evidence":{}}',
        '{"fields":{},"rationale":" ","evidence":{}}',
    ):
        with pytest.raises(StructuredOutputError):
            parse_model_output(RequirementProposal, raw, agent="requirements")
    review = parse_model_output(
        IndependentReviewOutput,
        '{"requirements":{"R":true},"criteria":{"C":true},"evidence":"checked",'
        '"requirement_evidence":{"R":["file:a.py"]},'
        '"criterion_evidence":{"C":["criterion:C"]}}',
        agent="independent_review",
    )
    assert review.requirements["R"] is True
    for raw in (
        (
            '{"requirements":{"R":1},"criteria":{"C":true},"evidence":"x",'
            '"requirement_evidence":{"R":["file:a.py"]},'
            '"criterion_evidence":{"C":["criterion:C"]}}'
        ),
        (
            '{"requirements":{},"criteria":{},"evidence":"x",'
            '"requirement_evidence":{},"criterion_evidence":{},"extra":1}'
        ),
    ):
        with pytest.raises(StructuredOutputError):
            parse_model_output(IndependentReviewOutput, raw, agent="review")


class Questions:
    def __init__(self):
        self.row = None
        self.consumed = False

    def get(self, question_id):
        return self.row if question_id == 7 else None

    def list(self, task_id):
        return [self.row] if self.row and self.row["task_id"] == task_id else []

    def consume_answer(self, question_id):
        if question_id != 7 or self.consumed:
            return False
        self.consumed = True
        self.row["status"] = "consumed"
        return True

    def consume_approval(self, question_id, reason):
        if question_id != 7 or not self.consumed or self.row["reason"] != reason:
            return False
        self.consumed = False
        self.row["status"] = "executed"
        return True


def test_tool_approval_canonicalizes_target_and_is_single_use():
    target = ToolApprovalTarget.create(
        3, "filesystem.delete", {"path": "notes/a.md"}, ["notes/a.md"]
    )
    same = ToolApprovalTarget.create(
        3, "filesystem.delete", {"path": "notes/a.md"}, ["notes/a.md"]
    )
    assert target == same and target.reason() == same.reason()
    questions = Questions()
    questions.row = {
        "task_id": 3,
        "reason": target.reason(),
        "status": "answered",
        "answer": "approve",
    }
    service = ApprovalService(
        questions, SimpleNamespace(ask=lambda *_args, **_kwargs: 8)
    )
    grant = service.issue_tool(3, 7, target)
    assert grant.consume(target.action, target)
    assert not grant.consume(target.action, target)


def test_tool_approval_rejects_mismatch_and_bad_target():
    target = ToolApprovalTarget.create(3, "http.request", {"method": "POST"})
    with pytest.raises(TypeError):
        ApprovalService(Questions()).request_tool(object())
    with pytest.raises(PermissionError):
        ApprovalService(Questions()).issue_tool(3, 7, object())
    questions = Questions()
    questions.row = {
        "task_id": 4,
        "reason": target.reason(),
        "status": "answered",
        "answer": "approve",
    }
    with pytest.raises(PermissionError):
        ApprovalService(questions).issue_tool(3, 7, target)
    with pytest.raises(ValueError):
        ToolApprovalTarget(0, "", "{}")


def test_tool_approval_question_creation_uses_safe_exact_target():
    observed = []
    store = SimpleNamespace(
        ask=lambda *args, **kwargs: observed.append((args, kwargs)) or 11
    )
    target = ToolApprovalTarget.create(9, "http.request", {"method": "DELETE"})
    assert ApprovalService(Questions(), store).request_tool(target) == 11
    assert observed[0][0][0] == 9
    assert target.reason() in observed[0][0][2]


def test_approval_resume_issues_only_the_exact_answered_action():
    questions = Questions()
    target = ToolApprovalTarget.create(9, "http.request", {"method": "POST"})

    def ask(task_id, question, reason, options, required):
        questions.row = {
            "id": 7,
            "task_id": task_id,
            "question": question,
            "reason": reason,
            "status": "open",
            "answer": None,
        }
        return 7

    service = ApprovalService(questions, SimpleNamespace(ask=ask))
    with pytest.raises(ApprovalRequired) as pending:
        service.grant_tool_for_target(target)
    assert pending.value.question_id == 7
    assert "method" not in questions.row["question"]
    questions.row.update(status="answered", answer="approve")
    grant = service.grant_tool_for_target(target)
    assert grant.consume(target.action, target)
    with pytest.raises(PermissionError, match="already consumed"):
        service.grant_tool_for_target(target)


def test_approval_resume_rejects_denial_and_changed_arguments():
    questions = Questions()
    target = ToolApprovalTarget.create(9, "filesystem.delete", {"path": "a"})
    questions.row = {
        "id": 7,
        "task_id": 9,
        "reason": target.reason(),
        "status": "answered",
        "answer": "deny",
    }
    service = ApprovalService(
        questions, SimpleNamespace(ask=lambda *_args, **_kwargs: 8)
    )
    with pytest.raises(TypeError):
        service.grant_tool_for_target(object())
    with pytest.raises(ApprovalDenied):
        service.grant_tool_for_target(target)
    changed = ToolApprovalTarget.create(9, "filesystem.delete", {"path": "b"})
    with pytest.raises(ApprovalRequired):
        service.grant_tool_for_target(changed)
    questions.row["status"] = "cancelled"
    with pytest.raises(PermissionError, match="not available"):
        service.grant_tool_for_target(target)


def test_registry_consumes_task_bound_tool_approval(tmp_path):
    config = Config()
    config.data["tools"] = {
        "permissions": {"filesystem": "write", "filesystem.delete": "write"}
    }
    registry = ToolRegistry(Permissions(config), workspace=tmp_path)
    questions = Questions()
    registry.approvals = ApprovalService(questions)
    (tmp_path / "remove.txt").write_text("x")
    args = {"path": "remove.txt"}
    target = ToolApprovalTarget.create(
        3, "filesystem.delete", args, [str(tmp_path / "remove.txt")]
    )
    questions.row = {
        "id": 7,
        "task_id": 3,
        "reason": target.reason(),
        "status": "open",
        "answer": None,
    }
    from harness.approvals import ApprovalRequired

    with pytest.raises(ApprovalRequired) as pending:
        registry.execute("filesystem.delete", args, task_id=3)
    assert pending.value.question_id == 7
    questions.row.update(status="answered", answer="approve")
    assert registry.execute("filesystem.delete", args, task_id=3) is True
    assert questions.row["status"] == "executed"


def test_registry_fails_closed_when_task_has_no_approval_service(tmp_path):
    config = Config()
    config.data["tools"] = {"permissions": {"http": "write"}}
    registry = ToolRegistry(Permissions(config), workspace=tmp_path)
    with pytest.raises(PermissionError, match="task-bound human approval"):
        registry.execute(
            "http.request",
            {"method": "POST", "url": "https://example.test"},
            task_id=5,
        )


def test_registry_rejects_grant_bound_to_different_arguments(tmp_path):
    config = Config()
    config.data["tools"] = {
        "permissions": {"filesystem": "write", "filesystem.delete": "write"}
    }
    registry = ToolRegistry(Permissions(config), workspace=tmp_path)
    questions = Questions()
    service = ApprovalService(questions)
    registry.approvals = service
    args = {"path": "target.txt"}
    actual = ToolApprovalTarget.create(4, "filesystem.delete", args)
    different = ToolApprovalTarget.create(4, "filesystem.delete", {"path": "other.txt"})
    questions.row = {
        "id": 7,
        "task_id": 4,
        "reason": different.reason(),
        "status": "answered",
        "answer": "approve",
    }
    mismatched_grant = service.issue_tool(4, 7, different)
    with pytest.raises(PermissionError, match="single-use"):
        registry.execute(
            "filesystem.delete", args, task_id=4, approval=mismatched_grant
        )
    assert actual.reason() != different.reason()


@pytest.mark.parametrize("failure", [ApprovalRequired(2), ApprovalDenied("no")])
def test_executor_propagates_human_approval_control_flow(failure):
    from harness.providers import ToolCall

    config = SimpleNamespace(
        data={
            "profiles": {
                "p": {
                    "model": {"primary": "model"},
                    "tools": ["mcp.demo.delete"],
                }
            }
        }
    )
    router = SimpleNamespace(
        config=config,
        complete=lambda *_args, **_kwargs: ModelResponse(
            text="", tool_calls=[ToolCall("call", "mcp.demo.delete", {})]
        ),
    )

    class Tools:
        def schemas(self, _names):
            return []

        def execute(self, *_args, **_kwargs):
            raise failure

    with pytest.raises(type(failure)):
        Executor(router, Tools()).execute(
            Subtask(
                id="step",
                title="delete",
                description="delete",
                profile="p",
            )
        )
