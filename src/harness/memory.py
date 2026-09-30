from __future__ import annotations

import json
import math
import re
import uuid
from pathlib import Path

import httpx

from .http_control import request

EMBEDDING_DIMENSION_DEFAULT = 1024
EMBEDDING_BATCH_SIZE_DEFAULT = 32
EMBEDDING_BATCH_SIZE_MAX = 256


def validate_vector(vector, dimension=None):
    if (
        not isinstance(vector, list)
        or not vector
        or (dimension is not None and len(vector) != dimension)
    ):
        raise ValueError("embedding vector has invalid dimension or values")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in vector
    ):
        raise ValueError("embedding vector has invalid dimension or values")
    return vector


class ObsidianMemory:
    def __init__(self, vault: Path):
        self.vault = vault.resolve()

    def write(self, name: str, content: str):
        p = self.path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return p

    def append(self, name: str, content: str):
        p = self.path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text((p.read_text() + "\n" if p.exists() else "") + content)
        return p

    def search(self, query: str):
        return [
            str(p)
            for p in self.vault.rglob("*.md")
            if p.resolve().is_relative_to(self.vault)
            and query.lower() in p.read_text(errors="ignore").lower()
        ]

    def read(self, name: str):
        p = self.path(name)
        return p.read_text() if p.exists() else None

    def path(self, name):
        if not name or Path(name).is_absolute():
            raise PermissionError("invalid vault-relative note name")
        path = (self.vault / f"{name}.md").resolve()
        if not path.is_relative_to(self.vault):
            raise PermissionError("note path escapes vault")
        return path

    def list_documents(self):
        documents = []
        for path in self.vault.rglob("*.md"):
            relative = path.relative_to(self.vault)
            current = self.vault
            unsafe = False
            for part in relative.parts:
                current = current / part
                if current.is_symlink():
                    unsafe = True
                    break
            if unsafe or not path.resolve().is_relative_to(self.vault):
                continue
            documents.append(relative.with_suffix("").as_posix())
        return sorted(documents)


class ContextBuilder:
    def __init__(self, memory: ObsidianMemory, tools, qdrant=None):
        self.memory = memory
        self.tools = tools
        self.qdrant = qdrant

    def build(self, task, workspace: str):
        fragments = [
            f"Task: {task.title}",
            f"Description: {task.description}",
            f"Workspace: {workspace}",
            f"Memory: {self.memory.read('decisions') or 'none'}",
        ]
        fragments.extend(
            f"Obsidian match: {path}\n{Path(path).read_text()[:16000]}"
            for path in self.memory.search(task.title)[:5]
        )
        if self.qdrant:
            try:
                fragments.extend(
                    f"Related memory: {point.get('payload', {}).get('text', '')}"
                    for point in self.qdrant.search(task.title, limit=3)
                )
            except httpx.HTTPError:
                fragments.append("Related memory: Qdrant unavailable")
        for name in ("README.md", "pyproject.toml", "package.json"):
            path = Path(workspace) / name
            if path.is_file() and path.resolve().is_relative_to(
                Path(workspace).resolve()
            ):
                fragments.append(f"Repository {name}:\n{path.read_text()[:16000]}")
        return "\n".join(fragments)


class QdrantMemory:
    def __init__(
        self,
        url: str,
        collection: str,
        dimension: int = 32,
        embedder=None,
        client=None,
        timeout=5,
        api_key=None,
    ):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", collection) or dimension < 1:
            raise ValueError("invalid collection or embedding dimension")
        self.url = url.rstrip("/")
        self.collection = collection
        self.dimension = dimension
        self.embedder = embedder
        self.client = client or httpx
        self.timeout = timeout
        self.api_key = api_key

    def _request(self, method, url, **kwargs):
        headers = dict(kwargs.pop("headers", {}) or {})
        if self.api_key:
            headers["api-key"] = self.api_key
        return request(
            method,
            url,
            client=self.client,
            timeout=self.timeout,
            headers=headers,
            **kwargs,
        )

    def _vector(self, text):
        if self.embedder is None:
            raise ValueError("semantic memory requires an embedding provider")
        vector = self.embedder.embed(text)
        return validate_vector(vector, self.dimension)

    def health(self):
        try:
            return self._request("GET", f"{self.url}/healthz").is_success
        except httpx.HTTPError:
            return False

    def health_report(self):
        """Return a bounded, secret-free view of service and collection health."""
        report = {
            "healthy": False,
            "service": "unavailable",
            "collection": self.collection,
            "collection_exists": False,
            "collection_status": None,
            "points_count": None,
            "indexed_vectors_count": None,
            "dimension": self.dimension,
            "distance": None,
            "errors": [],
        }
        try:
            health_response = self._request("GET", f"{self.url}/healthz")
            health_response.raise_for_status()
            report["service"] = "available"
            collection_response = self._request(
                "GET", f"{self.url}/collections/{self.collection}"
            )
            if collection_response.status_code == 404:
                report["errors"].append("collection_missing")
                return report
            collection_response.raise_for_status()
            body = collection_response.json()
            result = body.get("result") if isinstance(body, dict) else None
            config = result.get("config") if isinstance(result, dict) else None
            params = config.get("params") if isinstance(config, dict) else None
            vectors = params.get("vectors") if isinstance(params, dict) else None
            if not isinstance(vectors, dict):
                report["errors"].append("collection_contract_invalid")
                return report
            report.update(
                collection_exists=True,
                collection_status=result.get("status"),
                points_count=result.get("points_count"),
                indexed_vectors_count=result.get("indexed_vectors_count"),
                distance=vectors.get("distance"),
            )
            if (
                vectors.get("size") != self.dimension
                or vectors.get("distance") != "Cosine"
            ):
                report["errors"].append("collection_contract_mismatch")
            if result.get("status") not in {"green", "yellow"}:
                report["errors"].append("collection_not_ready")
            report["healthy"] = not report["errors"]
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            report["errors"].append("qdrant_probe_failed")
        return report

    def collection_exists(self):
        try:
            return self._request(
                "GET", f"{self.url}/collections/{self.collection}"
            ).is_success
        except httpx.HTTPError:
            return False

    def ensure_collection(self):
        response = self._request("GET", f"{self.url}/collections/{self.collection}")
        if response.is_success:
            vectors = response.json()["result"]["config"]["params"]["vectors"]
            if (
                vectors.get("size") != self.dimension
                or vectors.get("distance") != "Cosine"
            ):
                raise ValueError("Qdrant collection does not match embedding contract")
            return True
        if response.status_code != 404:
            response.raise_for_status()
        response = self._request(
            "PUT",
            f"{self.url}/collections/{self.collection}",
            json={"vectors": {"size": self.dimension, "distance": "Cosine"}},
        )
        response.raise_for_status()
        return True

    def upsert(self, point_id: str, text: str, payload=None):
        return self.upsert_many([(point_id, text, payload)])

    def upsert_many(self, points):
        points = list(points)
        if not points:
            return True
        if self.embedder is None:
            raise ValueError("semantic memory requires an embedding provider")
        normalized = []
        ids = set()
        for item in points:
            if not isinstance(item, (tuple, list)) or len(item) != 3:
                raise ValueError("each point must contain an id, text, and payload")
            point_id, text, payload = item
            if not isinstance(point_id, str):
                raise TypeError("point id must be a non-empty string")
            if not point_id:
                raise ValueError("point id must be a non-empty string")
            if not isinstance(text, str):
                raise TypeError("point text must be a string")
            try:
                normalized_id = str(uuid.UUID(point_id))
            except ValueError:
                normalized_id = str(uuid.uuid5(uuid.NAMESPACE_URL, point_id))
            if normalized_id in ids:
                raise ValueError("point ids must be unique within a batch")
            ids.add(normalized_id)
            normalized.append((normalized_id, text, payload or {"text": text}))
        texts = [item[1] for item in normalized]
        if callable(getattr(self.embedder, "embed_many", None)):
            vectors = self.embedder.embed_many(texts)
        else:
            vectors = [self.embedder.embed(text) for text in texts]
        if not isinstance(vectors, list) or len(vectors) != len(normalized):
            raise ValueError("embedding batch count does not match input points")
        body_points = [
            {
                "id": point_id,
                "vector": validate_vector(vector, self.dimension),
                "payload": payload,
            }
            for (point_id, _text, payload), vector in zip(
                normalized, vectors, strict=True
            )
        ]
        body = {"points": body_points}
        response = self._request(
            "PUT",
            f"{self.url}/collections/{self.collection}/points",
            json=body,
        )
        response.raise_for_status()
        return True

    def search(self, query: str, limit: int = 5):
        body = {"vector": self._vector(query), "limit": limit, "with_payload": True}
        response = self._request(
            "POST",
            f"{self.url}/collections/{self.collection}/points/search",
            json=body,
        )
        response.raise_for_status()
        return response.json()["result"]

    def delete(self, point_ids):
        response = self._request(
            "POST",
            f"{self.url}/collections/{self.collection}/points/delete",
            json={"points": point_ids},
        )
        response.raise_for_status()
        return True

    def scroll(self, source=None, offset=None, limit=100):
        if source is not None and (not isinstance(source, str) or not source.strip()):
            raise ValueError("scroll source must be a non-empty string")
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= 1000
        ):
            raise ValueError("scroll limit must be between 1 and 1000")
        body = {"limit": limit, "with_payload": True}
        if source is not None:
            body["filter"] = {"must": [{"key": "source", "match": {"value": source}}]}
        if offset is not None:
            body["offset"] = offset
        response = self._request(
            "POST",
            f"{self.url}/collections/{self.collection}/points/scroll",
            json=body,
        )
        response.raise_for_status()
        try:
            result = response.json()["result"]
            points = result["points"]
            next_offset = result.get("next_page_offset")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid Qdrant scroll response") from exc
        if not isinstance(points, list):
            raise TypeError("invalid Qdrant scroll response")
        return points, next_offset

    def scroll_source(self, source=None, limit=100):
        points, offset, seen = [], None, set()
        while True:
            page, next_offset = self.scroll(source, offset, limit)
            points.extend(page)
            if next_offset is None:
                return points
            cursor = json.dumps(next_offset, sort_keys=True)
            if cursor in seen:
                raise ValueError("Qdrant scroll cursor repeated")
            seen.add(cursor)
            offset = next_offset


class EmbeddingProvider:
    def __init__(
        self,
        base_url,
        model,
        api_key=None,
        timeout=30,
        dimension=EMBEDDING_DIMENSION_DEFAULT,
        batch_size=EMBEDDING_BATCH_SIZE_DEFAULT,
    ):
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 1
        ):
            raise ValueError("embedding dimension must be a positive integer")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or not 1 <= batch_size <= EMBEDDING_BATCH_SIZE_MAX
        ):
            raise ValueError("embedding batch size must be between 1 and 256")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.dimension = dimension
        self.batch_size = batch_size

    def embed(self, text):
        return self.embed_many([text])[0]

    def embed_many(self, texts):
        if not isinstance(texts, (list, tuple)) or any(
            not isinstance(text, str) for text in texts
        ):
            raise ValueError("embedding inputs must be a list of strings")
        if not texts:
            return []
        if len(texts) > self.batch_size:
            raise ValueError("embedding batch exceeds configured batch size")
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        response = request(
            "POST",
            f"{self.base_url}/embeddings",
            client=httpx,
            headers=headers,
            json={"model": self.model, "input": list(texts)},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()["data"]
        if not isinstance(data, list) or len(data) != len(texts):
            raise ValueError("embedding response count does not match input batch")
        ordered = [None] * len(texts)
        for item in data:
            if not isinstance(item, dict):
                raise TypeError("embedding response item is invalid")
            index = item.get("index")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < len(texts)
                or ordered[index] is not None
            ):
                raise ValueError("embedding response indexes are invalid")
            ordered[index] = validate_vector(item.get("embedding"), self.dimension)
        return ordered
