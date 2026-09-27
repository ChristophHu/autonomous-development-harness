from __future__ import annotations

import math
import re
import uuid
from pathlib import Path

import httpx


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
        vault.mkdir(parents=True, exist_ok=True)

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
        self, url: str, collection: str, dimension: int = 32, embedder=None, client=None
    ):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", collection) or dimension < 1:
            raise ValueError("invalid collection or embedding dimension")
        self.url = url.rstrip("/")
        self.collection = collection
        self.dimension = dimension
        self.embedder = embedder
        self.client = client or httpx

    def _vector(self, text):
        if self.embedder is None:
            raise ValueError("semantic memory requires an embedding provider")
        vector = self.embedder.embed(text)
        return validate_vector(vector, self.dimension)

    def health(self):
        try:
            return self.client.get(f"{self.url}/healthz", timeout=2).is_success
        except httpx.HTTPError:
            return False

    def collection_exists(self):
        try:
            return self.client.get(
                f"{self.url}/collections/{self.collection}", timeout=5
            ).is_success
        except httpx.HTTPError:
            return False

    def ensure_collection(self):
        response = self.client.get(
            f"{self.url}/collections/{self.collection}", timeout=5
        )
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
        response = self.client.put(
            f"{self.url}/collections/{self.collection}",
            json={"vectors": {"size": self.dimension, "distance": "Cosine"}},
            timeout=5,
        )
        response.raise_for_status()
        return True

    def upsert(self, point_id: str, text: str, payload=None):
        try:
            normalized_id = str(uuid.UUID(point_id))
        except ValueError:
            normalized_id = str(uuid.uuid5(uuid.NAMESPACE_URL, point_id))
        body = {
            "points": [
                {
                    "id": normalized_id,
                    "vector": self._vector(text),
                    "payload": payload or {"text": text},
                }
            ]
        }
        response = self.client.put(
            f"{self.url}/collections/{self.collection}/points", json=body, timeout=5
        )
        response.raise_for_status()
        return True

    def search(self, query: str, limit: int = 5):
        body = {"vector": self._vector(query), "limit": limit, "with_payload": True}
        response = self.client.post(
            f"{self.url}/collections/{self.collection}/points/search",
            json=body,
            timeout=5,
        )
        response.raise_for_status()
        return response.json()["result"]

    def delete(self, point_ids):
        response = self.client.post(
            f"{self.url}/collections/{self.collection}/points/delete",
            json={"points": point_ids},
            timeout=5,
        )
        response.raise_for_status()
        return True


class EmbeddingProvider:
    def __init__(self, base_url, model, api_key=None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def embed(self, text):
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        response = httpx.post(
            f"{self.base_url}/embeddings",
            headers=headers,
            json={"model": self.model, "input": text},
            timeout=30,
        )
        response.raise_for_status()
        return validate_vector(response.json()["data"][0]["embedding"])
