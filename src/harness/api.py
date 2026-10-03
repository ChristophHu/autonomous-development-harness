import asyncio
import json
from datetime import UTC
from typing import Any, Literal

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictStr

from .core import Event, Task, build
from .decisions import Decision, DecisionService
from .domain import AcceptanceCriterion, Status, TaskComplexity
from .evidence import EvidenceInput
from .services import ConfigurationApplicationService, VerificationEvidenceService


class ErrorResponse(BaseModel):
    detail: str = Field(examples=["task not found"])


class TaskPatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1)
    parent_task_id: int | None = None
    description: str | None = None
    goal: str | None = None
    priority: int | None = None
    requirements: list[str] | None = None
    constraints: list[str] | None = None
    acceptance_criteria: list[AcceptanceCriterion] | None = None
    dependencies: list[int] | None = None
    assigned_agent: str | None = None
    assigned_profile: str | None = None
    complexity: TaskComplexity | None = None
    context: dict[str, Any] | None = None
    decisions: list[dict[str, Any]] | None = None
    test_commands: list[list[str]] | None = None
    lint_commands: list[list[str]] | None = None
    coverage_command: list[str] | None = None
    coverage_report: str | None = None
    coverage_threshold: float | None = Field(default=None, ge=0, le=100)
    workflow: Literal["feature", "bugfix", "hotfix", "release", "other"] | None = None
    release_version: str | None = None


class AnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question_id: int = Field(gt=0, examples=[1])
    answer: str = Field(min_length=1, examples=["Implement the local-only option"])


class QuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(
        min_length=1, examples=["Should the change include a migration?"]
    )
    reason: str = Field(
        min_length=1, examples=["The requirement does not specify data compatibility."]
    )
    options: list[str] = Field(default_factory=list)
    required: bool = True
    purpose: Literal["input", "decision"] = "input"


class QuestionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": 1,
                    "task_id": 7,
                    "question": "Should the change include a migration?",
                    "reason": "Compatibility is unclear.",
                    "options": ["yes", "no"],
                    "required": True,
                    "purpose": "decision",
                    "answer": None,
                    "status": "open",
                    "created_at": "2026-09-28T12:00:00+00:00",
                    "answered_at": None,
                }
            ]
        }
    )

    id: int
    task_id: int
    question: str
    reason: str
    options: list[str]
    required: bool
    purpose: Literal["input", "decision", "approval"] = "input"
    answer: str | None
    status: str
    created_at: str
    answered_at: str | None


class QuestionCreated(BaseModel):
    id: int
    status: str = "open"


class TaskResult(BaseModel):
    task_id: int
    status: Status
    result: str | None


class PlanResponse(BaseModel):
    id: int
    task_id: int
    summary: str
    payload: dict[str, Any]
    created_at: str


class ArtifactCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    key: StrictStr = Field(min_length=1, max_length=200, pattern=r"^[^/]+$")
    content: StrictStr
    expected_version: int | None = Field(default=None, ge=0)


class ArtifactResponse(BaseModel):
    id: int
    task_id: int
    artifact_key: str
    version: int
    content: str
    sha256: str
    created_at: str


class CorrectionResponse(BaseModel):
    id: str
    task_id: int
    plan_id: int | None
    subtask_id: str | None
    category: str
    source: str
    rule: str
    message: str
    affected_paths: list[str]
    evidence: dict[str, Any]
    expected: dict[str, Any]
    status: Literal["open", "in_progress", "resolved"]
    attempts: int
    created_at: str
    updated_at: str
    resolved_at: str | None


class CorrectionStatusRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    status: Literal["open", "in_progress", "resolved"]


class ValidationResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": 2,
                    "task_id": 7,
                    "valid": True,
                    "report": {"tests": {"passed": True}, "coverage": 100},
                    "created_at": "2026-09-28T12:00:00+00:00",
                }
            ]
        }
    )

    id: int
    task_id: int
    valid: bool
    report: dict[str, Any]
    created_at: str


class ModelUsageRow(BaseModel):
    id: int
    task_id: int | None
    agent: str | None
    profile: str | None
    complexity: TaskComplexity | None
    provider: str
    model: str
    status: str | None
    started_at: str | None
    finished_at: str | None
    latency_ms: float | None
    fallback_index: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    cached_tokens: int | None
    reasoning_tokens: int | None
    cost: float | None
    error_type: str | None
    error_category: str | None = None


class ModelUsageTotals(BaseModel):
    runs: int
    prompt_tokens: int | None
    prompt_tokens_reported: int
    prompt_tokens_missing: int
    completion_tokens: int | None
    completion_tokens_reported: int
    completion_tokens_missing: int
    cached_tokens: int | None
    cached_tokens_reported: int
    cached_tokens_missing: int
    reasoning_tokens: int | None
    reasoning_tokens_reported: int
    reasoning_tokens_missing: int
    cost: float | None
    cost_reported: int
    cost_missing: int


class ModelUsageGroup(BaseModel):
    value: int | str | None
    totals: ModelUsageTotals


class UsageReportResponse(BaseModel):
    total: int
    limit: int
    offset: int
    totals: ModelUsageTotals
    items: list[ModelUsageRow]
    group_by: str | None
    groups: list[ModelUsageGroup]


def _service_error(error: ValueError):
    message = str(error)
    if (
        message == "task not found"
        or "question not found" in message
        or message == "correction not found"
    ):
        status = 404
    elif any(
        marker in message
        for marker in (
            "currently running",
            "invalid task transition",
            "version conflict",
            "invalid correction status transition",
        )
    ):
        status = 409
    else:
        status = 422
    return HTTPException(status, message)


def _require_task(task_id):
    try:
        return orchestrator.service.get(task_id)
    except ValueError as exc:
        raise _service_error(exc) from exc


def _event(row):
    return Event(
        id=row["id"],
        task_id=row["task_id"],
        kind=row["kind"],
        payload=store.audit.sanitize(json.loads(row["payload"])),
        created_at=row["created_at"],
    )


def _question(row):
    safe = store.audit.sanitize(
        {
            "question": row["question"],
            "reason": row["reason"],
            "options": json.loads(row["options"]),
            "answer": row["answer"],
        }
    )
    return QuestionResponse(
        id=row["id"],
        task_id=row["task_id"],
        question=safe["question"],
        reason=safe["reason"],
        options=safe["options"],
        required=bool(row["required"]),
        purpose=row["purpose"],
        answer=safe["answer"],
        status=row["status"],
        created_at=row["created_at"],
        answered_at=row["answered_at"],
    )


config, store, orchestrator = build()
app = FastAPI(title="Autonomous Development Harness", version="0.1.0")
router = APIRouter(
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}}
)


@app.exception_handler(RequestValidationError)
async def request_validation_error(_request, error):
    messages = "; ".join(item["msg"] for item in error.errors())
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(detail=messages).model_dump(),
    )


@router.get("/health")
def health():
    return {"status": "ok", "service": "harness"}


@router.get("/configuration/status")
def configuration_status():
    try:
        return ConfigurationApplicationService(config).validate()
    except ValueError as exc:
        raise HTTPException(503, "configuration is invalid") from exc


@router.get("/metrics")
def metrics():
    return orchestrator.observability.metrics()


@router.get("/metrics/prometheus", response_class=PlainTextResponse)
def prometheus_metrics():
    return PlainTextResponse(
        orchestrator.observability.prometheus(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@router.get("/event-kinds")
def event_kinds():
    return {"event_kinds": list(orchestrator.observability.event_kinds)}


@router.get("/models/status")
def model_inventory():
    return {"providers": orchestrator.model_operations.model_inventory()}


@router.get("/decisions", response_model=list[Decision])
def list_decisions(
    task_id: int | None = Query(default=None, gt=0),
    category: str | None = None,
    source: str | None = None,
    tag: str | None = None,
):
    return DecisionService(store).query(task_id, category, source, tag)


@router.get("/decisions/{decision_id}", response_model=Decision)
def get_decision(decision_id: int):
    decision = DecisionService(store).get(decision_id)
    if decision is None:
        raise HTTPException(404, "decision not found")
    return decision


@router.post("/verification/evidence")
def import_verification_evidence(payload: EvidenceInput):
    try:
        return VerificationEvidenceService(store.database).record(payload)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/verification/evidence")
def list_verification_evidence(
    kind: Literal["ci", "provider", "qdrant", "embedding", "http_tls"] | None = None,
    limit: int = Query(default=100, ge=1, le=500),
):
    return VerificationEvidenceService(store.database).list(kind=kind, limit=limit)


@router.get("/verification/evidence/audit")
def audit_verification_evidence(
    subject_sha256: str = Query(pattern=r"^[a-f0-9]{64}$"),
    max_age_hours: int = Query(default=168, ge=1, le=8760),
):
    return VerificationEvidenceService(store.database).audit(
        subject_sha256=subject_sha256, max_age_hours=max_age_hours
    )


@router.post("/tasks", response_model=Task, responses={422: {"model": ErrorResponse}})
def create(task: Task):
    try:
        return orchestrator.service.create(task)
    except ValueError as exc:
        raise _service_error(exc) from exc


@router.get("/tasks", response_model=list[Task])
def list_tasks(status: Status | None = None):
    return orchestrator.service.list(status)


@router.patch(
    "/tasks/{task_id}",
    response_model=Task,
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
def patch_task(task_id: int, payload: TaskPatchRequest):
    try:
        return orchestrator.service.patch(
            task_id, payload.model_dump(exclude_unset=True)
        )
    except ValueError as exc:
        raise _service_error(exc) from exc


@router.delete(
    "/tasks/{task_id}",
    status_code=204,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
def delete_task(task_id: int):
    try:
        orchestrator.service.delete(task_id)
    except ValueError as exc:
        raise _service_error(exc) from exc


@router.get(
    "/tasks/{task_id}", response_model=Task, responses={404: {"model": ErrorResponse}}
)
def get(task_id: int):
    return _require_task(task_id)


@router.post(
    "/tasks/{task_id}/run",
    response_model=Task,
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
async def run(task_id: int):
    try:
        return await orchestrator.run(task_id)
    except ValueError as exc:
        raise _service_error(exc) from exc


@router.post(
    "/tasks/{task_id}/start",
    response_model=Task,
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
async def start_task(task_id: int):
    try:
        return await orchestrator.run(task_id)
    except ValueError as exc:
        raise _service_error(exc) from exc


@router.post(
    "/tasks/{task_id}/abort",
    response_model=Task,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
def abort_task(task_id: int):
    try:
        return orchestrator.service.abort(task_id)
    except ValueError as exc:
        raise _service_error(exc) from exc


@router.get(
    "/tasks/{task_id}/result",
    response_model=TaskResult,
    responses={404: {"model": ErrorResponse}},
)
def task_result(task_id: int):
    task = _require_task(task_id)
    return {"task_id": task_id, "status": task.status, "result": task.result}


@router.get(
    "/tasks/{task_id}/events",
    response_model=list[Event],
    responses={404: {"model": ErrorResponse}},
)
def events(task_id: int):
    _require_task(task_id)
    return [_event(row) for row in orchestrator.service.events(task_id)]


@router.get(
    "/tasks/{task_id}/questions",
    response_model=list[QuestionResponse],
    responses={404: {"model": ErrorResponse}},
)
def questions(task_id: int):
    _require_task(task_id)
    return [_question(row) for row in orchestrator.service.questions(task_id)]


@router.post(
    "/tasks/{task_id}/answers",
    response_model=Task,
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
async def answer(task_id: int, payload: AnswerRequest):
    try:
        return await orchestrator.service.answer(
            task_id, payload.question_id, payload.answer
        )
    except ValueError as exc:
        raise _service_error(exc) from exc


@router.post(
    "/tasks/{task_id}/questions",
    response_model=QuestionCreated,
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
def ask_question(task_id: int, payload: QuestionRequest):
    try:
        question_id = orchestrator.service.ask_question(
            task_id,
            payload.question,
            payload.reason,
            payload.options,
            payload.required,
            payload.purpose,
        )
    except ValueError as exc:
        raise _service_error(exc) from exc
    return {"id": question_id, "status": "open"}


@router.get(
    "/tasks/{task_id}/plan",
    response_model=PlanResponse | None,
    responses={404: {"model": ErrorResponse}},
)
def plan(task_id: int):
    _require_task(task_id)
    return orchestrator.service.plan(task_id)


@router.get(
    "/tasks/{task_id}/validation",
    response_model=ValidationResponse | None,
    responses={404: {"model": ErrorResponse}},
)
def validation(task_id: int):
    _require_task(task_id)
    return orchestrator.service.validation(task_id)


@router.post(
    "/tasks/{task_id}/artifacts",
    response_model=ArtifactResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
def create_artifact(task_id: int, payload: ArtifactCreateRequest):
    try:
        return orchestrator.service.save_artifact(
            task_id, payload.key, payload.content, payload.expected_version
        )
    except ValueError as exc:
        raise _service_error(exc) from exc


@router.get(
    "/tasks/{task_id}/artifacts",
    response_model=list[ArtifactResponse],
    responses={404: {"model": ErrorResponse}},
)
def list_artifacts(task_id: int):
    try:
        return orchestrator.service.artifacts(task_id)
    except ValueError as exc:
        raise _service_error(exc) from exc


@router.get(
    "/tasks/{task_id}/artifacts/{artifact_key}/history",
    response_model=list[ArtifactResponse],
    responses={404: {"model": ErrorResponse}},
)
def artifact_history(task_id: int, artifact_key: str):
    try:
        return orchestrator.service.artifact_history(task_id, artifact_key)
    except ValueError as exc:
        raise _service_error(exc) from exc


@router.get(
    "/tasks/{task_id}/artifacts/{artifact_key}",
    response_model=ArtifactResponse | None,
    responses={404: {"model": ErrorResponse}},
)
def get_artifact(task_id: int, artifact_key: str):
    try:
        return orchestrator.service.artifact(task_id, artifact_key)
    except ValueError as exc:
        raise _service_error(exc) from exc


@router.get(
    "/tasks/{task_id}/corrections",
    response_model=list[CorrectionResponse],
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
def list_corrections(
    task_id: int,
    status: Literal["open", "in_progress", "resolved"] | None = None,
):
    try:
        rows = orchestrator.service.corrections(task_id, status)
        return [store.audit.sanitize(item) for item in rows]
    except ValueError as exc:
        raise _service_error(exc) from exc


@router.patch(
    "/tasks/{task_id}/corrections/{item_id}",
    response_model=CorrectionResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
def update_correction(task_id: int, item_id: str, payload: CorrectionStatusRequest):
    try:
        row = orchestrator.service.update_correction(task_id, item_id, payload.status)
        return store.audit.sanitize(row)
    except ValueError as exc:
        raise _service_error(exc) from exc


@router.get(
    "/models/usage",
    response_model=UsageReportResponse,
    responses={422: {"model": ErrorResponse}},
    description=(
        "Persisted model-call usage. Token and cost totals include reported values; "
        "missing counts make incomplete totals explicit. No prompt content is returned."
    ),
)
def model_usage(
    task_id: int | None = Query(default=None, gt=0),
    agent: str | None = Query(default=None, min_length=1),
    profile: str | None = Query(default=None, min_length=1),
    provider: str | None = Query(default=None, min_length=1),
    model: str | None = Query(default=None, min_length=1),
    group_by: Literal["task_id", "agent", "profile", "provider", "model", "day"]
    | None = None,
    since: AwareDatetime | None = None,
    until: AwareDatetime | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    try:
        return orchestrator.service.model_usage(
            task_id=task_id,
            agent=agent,
            profile=profile,
            provider=provider,
            model=model,
            group_by=group_by,
            since=since,
            until=until,
            limit=limit,
            offset=offset,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/events")
def all_events(
    task_id: int | None = None,
    event_type: str | None = None,
    actor: str | None = None,
    since: AwareDatetime | None = None,
    until: AwareDatetime | None = None,
):
    since_utc = since.astimezone(UTC) if since else None
    until_utc = until.astimezone(UTC) if until else None
    if since_utc and until_utc and since_utc > until_utc:
        raise HTTPException(422, "since must not be later than until")
    rows = orchestrator.service.event_feed(
        task_id,
        event_type,
        since_utc.isoformat() if since_utc else None,
        until_utc.isoformat() if until_utc else None,
        actor,
    )
    return [_event(row) for row in rows]


@router.get(
    "/events/stream",
    responses={
        200: {
            "content": {
                "text/event-stream": {
                    "schema": {
                        "type": "string",
                        "example": "id: 42\\nevent: TASK_COMPLETED\\ndata: {...}\\n\\n",
                    }
                }
            }
        },
        400: {"model": ErrorResponse},
    },
    description="Persisted events as SSE; resumes after the numeric Last-Event-ID cursor.",
)
async def stream(request: Request, task_id: int | None = None):
    if task_id is not None:
        _require_task(task_id)
    raw_cursor = request.headers.get("last-event-id", "0")
    try:
        cursor = int(raw_cursor)
    except ValueError as exc:
        raise HTTPException(
            400, "Last-Event-ID must be a non-negative integer"
        ) from exc
    if cursor < 0:
        raise HTTPException(400, "Last-Event-ID must be a non-negative integer")

    async def gen():
        current = cursor
        while True:
            if await request.is_disconnected():
                return
            rows = orchestrator.service.events_after(current, task_id)
            for row in rows:
                current = row["id"]
                yield f"id: {current}\nevent: {row['kind']}\ndata: {json.dumps(_event(row).model_dump(mode='json'), default=str)}\n\n"
            if not rows:
                yield ": keep-alive\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream")


app.include_router(router)
app.include_router(router, prefix="/api")
