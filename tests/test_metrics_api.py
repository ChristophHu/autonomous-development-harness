import pytest
from fastapi.testclient import TestClient

from harness.metrics_api import create_metrics_app, read_bearer_token


def test_metrics_token_file_is_read_only_when_private_regular_and_bounded(tmp_path):
    token_file = tmp_path / "metrics-token"
    token_file.write_text("s" * 64 + "\n")
    token_file.chmod(0o600)

    assert read_bearer_token(str(token_file)) == "s" * 64


@pytest.mark.parametrize("contents", ["short", "x" * 4097, "not valid spaces"])
def test_metrics_token_file_rejects_invalid_values(tmp_path, contents):
    token_file = tmp_path / "metrics-token"
    token_file.write_text(contents)
    token_file.chmod(0o600)

    with pytest.raises(ValueError, match="unavailable or insecure"):
        read_bearer_token(str(token_file))


def test_metrics_token_file_rejects_public_permissions_symlinks_and_missing_files(
    tmp_path,
):
    token_file = tmp_path / "metrics-token"
    token_file.write_text("s" * 64)
    token_file.chmod(0o644)
    with pytest.raises(ValueError, match="unavailable or insecure"):
        read_bearer_token(str(token_file))

    private = tmp_path / "private-token"
    private.write_text("s" * 64)
    private.chmod(0o600)
    link = tmp_path / "linked-token"
    link.symlink_to(private)
    with pytest.raises(ValueError, match="unavailable or insecure"):
        read_bearer_token(str(link))
    with pytest.raises(ValueError, match="unavailable or insecure"):
        read_bearer_token(str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="unavailable or insecure"):
        read_bearer_token("")


def test_metrics_app_requires_a_provider_and_nonempty_token():
    with pytest.raises(TypeError, match="metrics provider"):
        create_metrics_app(None, "a" * 32)
    with pytest.raises(ValueError, match="bearer token"):
        create_metrics_app(lambda: "metrics", "")


def test_metrics_endpoint_requires_exact_bearer_token_and_does_not_call_provider():
    calls = []
    client = TestClient(
        create_metrics_app(lambda: calls.append(True) or "ok", "s" * 32)
    )

    for authorization in (None, "Basic abc", "Bearer wrong"):
        headers = {} if authorization is None else {"Authorization": authorization}
        response = client.get("/metrics/prometheus", headers=headers)
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
    assert calls == []


def test_metrics_endpoint_returns_only_prometheus_text_for_authorized_get():
    client = TestClient(create_metrics_app(lambda: "harness_tasks_total 3\n", "s" * 32))

    response = client.get(
        "/metrics/prometheus", headers={"Authorization": f"Bearer {'s' * 32}"}
    )
    assert response.status_code == 200
    assert response.text == "harness_tasks_total 3\n"
    assert response.headers["content-type"].startswith("text/plain")


def test_metrics_app_exposes_no_other_api_routes_or_mutating_methods():
    client = TestClient(create_metrics_app(lambda: "metrics", "s" * 32))
    auth = {"Authorization": f"Bearer {'s' * 32}"}

    assert client.get("/health").status_code == 404
    assert client.get("/api/tasks").status_code == 404
    assert client.post("/metrics/prometheus", headers=auth).status_code == 405
