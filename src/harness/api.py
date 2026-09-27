import asyncio
import json

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from .core import Task, build

config, store, orchestrator = build()
app = FastAPI(title="Autonomous Development Harness", version="0.1.0")
router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok", "service": "harness"}


@router.post("/tasks", response_model=Task)
def create(task: Task):
    return orchestrator.service.create(task)


@router.get("/tasks")
def list_tasks(status: str | None = None):
    return orchestrator.service.list(status)


@router.patch("/tasks/{task_id}")
def patch_task(task_id: int, payload: dict):
    task = store.get(task_id)
    if not task:
        raise HTTPException(404, "task not found")
    try:
        return orchestrator.service.patch(task_id, payload)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.delete("/tasks/{task_id}", status_code=204)
def delete_task(task_id: int):
    try:
        orchestrator.service.delete(task_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/tasks/{task_id}", response_model=Task)
def get(task_id: int):
    task = store.get(task_id)
    if not task:
        raise HTTPException(404, "task not found")
    return task


@router.post("/tasks/{task_id}/run", response_model=Task)
async def run(task_id: int):
    return await orchestrator.run(task_id)


@router.post("/tasks/{task_id}/start", response_model=Task)
async def start_task(task_id: int):
    return await orchestrator.run(task_id)


@router.post("/tasks/{task_id}/abort", response_model=Task)
def abort_task(task_id: int):
    return orchestrator.service.abort(task_id)


@router.get("/tasks/{task_id}/result")
def task_result(task_id: int):
    task = store.get(task_id)
    if not task:
        raise HTTPException(404, "task not found")
    return {"task_id": task_id, "status": task.status, "result": task.result}


@router.get("/tasks/{task_id}/events")
def events(task_id: int):
    return [dict(row) for row in store.events.list(task_id)]


@router.get("/tasks/{task_id}/questions")
def questions(task_id: int):
    return [dict(row) for row in store.questions.list(task_id)]


@router.post("/tasks/{task_id}/answers")
async def answer(task_id: int, payload: dict):
    try:
        return await orchestrator.service.answer(
            task_id, payload.get("question_id"), payload.get("answer", "")
        )
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/tasks/{task_id}/questions")
def ask_question(task_id: int, payload: dict):
    if not store.get(task_id):
        raise HTTPException(404, "task not found")
    question_id = store.ask(
        task_id,
        payload.get("question", ""),
        payload.get("reason", ""),
        payload.get("options", []),
        payload.get("required", True),
    )
    return {"id": question_id, "status": "open"}


@router.get("/tasks/{task_id}/plan")
def plan(task_id: int):
    with store.database.connect() as c:
        row = c.execute(
            "SELECT * FROM plans WHERE task_id=? ORDER BY id DESC LIMIT 1", (task_id,)
        ).fetchone()
    return dict(row) if row else None


@router.get("/tasks/{task_id}/validation")
def validation(task_id: int):
    with store.database.connect() as c:
        row = c.execute(
            "SELECT * FROM validations WHERE task_id=? ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
    return dict(row) if row else None


@router.get("/events")
def all_events(
    task_id: int | None = None,
    event_type: str | None = None,
    since: str | None = None,
    until: str | None = None,
):
    return [dict(row) for row in store.events.list(task_id, event_type, since, until)]


@router.get("/events/stream")
async def stream(request: Request, task_id: int | None = None):
    async def gen():
        cursor = int(request.headers.get("last-event-id", "0"))
        while True:
            if await request.is_disconnected():
                return
            rows = store.events.after(cursor, task_id)
            for row in rows:
                cursor = row["id"]
                yield f"id: {cursor}\nevent: {row['kind']}\ndata: {json.dumps(dict(row), default=str)}\n\n"
            if not rows:
                yield ": keep-alive\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream")


app.include_router(router)
app.include_router(router, prefix="/api")
