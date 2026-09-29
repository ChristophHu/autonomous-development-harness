import json
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from harness import api, cli
from harness.core import Config, Store


@pytest.fixture
def usage_runtime(tmp_path, monkeypatch):
    config = Config()
    config.data["paths"]["database"] = str(tmp_path / "usage.db")
    store = Store(config)
    for index in range(8):
        store.tasks.create(f"task-{index}")
    with store.database.connect() as connection:
        connection.executemany(
            """INSERT INTO model_runs(
                task_id,agent,profile,provider,model,status,started_at,finished_at,
                latency_ms,fallback_index,prompt_tokens,completion_tokens,cost,
                cached_tokens,reasoning_tokens,error_type,created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (
                    7,
                    "planner",
                    "planner",
                    "local",
                    "model-a",
                    "completed",
                    "2026-09-28T10:00:00+00:00",
                    "2026-09-28T10:00:01+00:00",
                    1000,
                    0,
                    10,
                    20,
                    0.5,
                    2,
                    3,
                    None,
                    "2026-09-28T10:00:01+00:00",
                ),
                (
                    7,
                    "planner",
                    "planner",
                    "remote",
                    "model-b",
                    "failed",
                    "2026-09-28T11:00:00+00:00",
                    "2026-09-28T11:00:02+00:00",
                    2000,
                    1,
                    None,
                    None,
                    None,
                    None,
                    None,
                    "ProviderError",
                    "2026-09-28T11:00:02+00:00",
                ),
                (
                    8,
                    "coding",
                    "coding",
                    "local",
                    "model-a",
                    "completed",
                    "2026-09-29T10:00:00+00:00",
                    "2026-09-29T10:00:01+00:00",
                    1000,
                    0,
                    4,
                    6,
                    0.25,
                    0,
                    0,
                    None,
                    "2026-09-29T10:00:01+00:00",
                ),
            ],
        )
    monkeypatch.setattr(api, "store", store)
    monkeypatch.setattr(cli, "build", lambda: (config, store, None))
    return store


def test_usage_report_aggregates_filters_and_preserves_unknowns(usage_runtime):
    report = usage_runtime.model_usage.report(
        task_id=7, since="2026-09-28T00:00:00Z", limit=1
    )
    assert report["total"] == 2
    assert report["items"][0]["model"] == "model-b"
    assert report["items"][0]["prompt_tokens"] is None
    assert report["totals"] == {
        "runs": 2,
        "prompt_tokens": 10,
        "prompt_tokens_reported": 1,
        "prompt_tokens_missing": 1,
        "completion_tokens": 20,
        "completion_tokens_reported": 1,
        "completion_tokens_missing": 1,
        "cost": 0.5,
        "cost_reported": 1,
        "cost_missing": 1,
    }


def test_usage_report_filters_and_paginates_stably(usage_runtime):
    report = usage_runtime.model_usage.report(
        provider="local",
        model="model-a",
        since="2026-09-28T00:00:00Z",
        until="2026-09-28T23:59:59Z",
        limit=1,
        offset=0,
    )
    assert report["total"] == 1
    assert [item["task_id"] for item in report["items"]] == [7]
    assert report["offset"] == 0 and report["limit"] == 1


def test_usage_report_empty_has_unknown_totals(usage_runtime):
    report = usage_runtime.model_usage.report(task_id=99)
    assert report["total"] == 0 and report["items"] == []
    assert report["totals"]["prompt_tokens"] is None
    assert report["totals"]["cost"] is None


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"task_id": True}, "task_id"),
        ({"provider": "  "}, "provider"),
        ({"limit": 0}, "limit"),
        ({"limit": True}, "limit"),
        ({"limit": 201}, "limit"),
        ({"offset": -1}, "offset"),
        ({"offset": True}, "offset"),
        ({"since": "2026-09-28T00:00:00"}, "timezone"),
        ({"since": "not-a-date"}, "ISO-8601"),
        ({"since": 12}, "ISO-8601"),
        (
            {
                "since": "2026-09-29T00:00:00Z",
                "until": "2026-09-28T00:00:00Z",
            },
            "since",
        ),
    ],
)
def test_usage_report_rejects_invalid_filters(usage_runtime, values, message):
    with pytest.raises(ValueError, match=message):
        usage_runtime.model_usage.report(**values)


def test_usage_cli_and_rest_share_report_and_validate_query(usage_runtime):
    result = CliRunner().invoke(cli.app, ["models", "usage", "--task-id", "7"])
    assert result.exit_code == 0, result.output
    cli_report = json.loads(result.output)
    response = TestClient(api.app).get("/api/models/usage", params={"task_id": 7})
    assert response.status_code == 200
    assert response.json() == cli_report
    assert (
        TestClient(api.app).get("/models/usage", params={"limit": 0}).status_code == 422
    )
    assert (
        TestClient(api.app)
        .get("/api/models/usage", params={"since": "2026-09-28T00:00:00"})
        .status_code
        == 422
    )
    valid_since = TestClient(api.app).get(
        "/api/models/usage", params={"since": "2026-09-28T00:00:00Z"}
    )
    assert valid_since.status_code == 200
    inverted_dates = TestClient(api.app).get(
        "/api/models/usage",
        params={
            "since": "2026-09-29T00:00:00Z",
            "until": "2026-09-28T00:00:00Z",
        },
    )
    assert inverted_dates.status_code == 422

    invalid_cli = CliRunner().invoke(cli.app, ["models", "usage", "--limit", "0"])
    assert invalid_cli.exit_code == 2
    assert "Usage report failed" in invalid_cli.output


def test_usage_report_accepts_aware_datetime(usage_runtime):
    report = usage_runtime.model_usage.report(
        since=datetime.fromisoformat("2026-09-28T00:00:00+00:00")
    )
    assert report["total"] == 3


def test_usage_report_rejects_non_string_filters(usage_runtime):
    with pytest.raises(TypeError, match="provider"):
        usage_runtime.model_usage.report(provider=3)


def test_usage_endpoint_is_typed_in_openapi(usage_runtime):
    schema = TestClient(api.app).get("/openapi.json").json()
    operation = schema["paths"]["/api/models/usage"]["get"]
    assert operation["parameters"]
    assert "UsageReportResponse" in schema["components"]["schemas"]
