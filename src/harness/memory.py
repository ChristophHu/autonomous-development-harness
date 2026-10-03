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
CONTEXT_CLAIM_FIELDS = {
    "goal",
    "requirements",
    "acceptance_criteria",
    "test_commands",
    "coverage_command",
}
CONTEXT_CLAIMS_MAX = 32
CONTEXT_CLAIM_BYTES_MAX = 4096
CONTEXT_CLAIMS_BYTES_MAX = 8192
CONTEXT_EVIDENCE_FIELDS = (
    "fragments",
    "omitted_sources",
    "truncated_sources",
    "rejected_sources",
    "conflicts",
    "claims",
    "claim_issues",
    "budget_bytes",
)


def context_evidence_envelope(context):
    """Return the exact context evidence object serialized to requirement agents."""
    envelope = {}
    for name in CONTEXT_EVIDENCE_FIELDS:
        if name not in context:
            continue
        if name == "fragments":
            keys = (
                "kind",
                "ref",
                "text",
                "required",
                "truncated",
                "review_state",
                "provenance",
            )
            envelope[name] = [
                {key: fragment[key] for key in keys if key in fragment}
                for fragment in context[name]
                if isinstance(fragment, dict)
            ]
        else:
            envelope[name] = context[name]
    return envelope


def context_evidence_size(context):
    encoded = json.dumps(
        context_evidence_envelope(context),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return len(encoded.encode("utf-8"))


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
            if any(
                part.startswith(".") or part == "_harness" for part in relative.parts
            ):
                continue
            if (
                len(relative.parts) == 3
                and relative.parts[0] == "tasks"
                and relative.parts[1].isdecimal()
                and relative.name == "plan.md"
            ):
                continue
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
    def __init__(self, memory: ObsidianMemory, tools, qdrant=None, max_bytes=65536):
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes < 1
        ):
            raise ValueError("context max_bytes must be a positive integer")
        self.memory = memory
        self.tools = tools
        self.qdrant = qdrant
        self.max_bytes = max_bytes

    def build(self, task, workspace: str):
        return self.build_evidence(task, workspace)["text"]

    @staticmethod
    def _selection_priority(fragment):
        kind = fragment["kind"]
        if kind == "decision_memory":
            return 0 if "Memory: none" not in fragment["text"] else 8
        if kind == "vault":
            if fragment.get("review_state") == "current":
                return (
                    1
                    if fragment.get("provenance", {}).get("source_state") == "current"
                    else 2
                )
            return 6
        return {
            "repository_symbol": 3,
            "qdrant": 4,
            "repository": 5,
            "qdrant_status": 9,
        }.get(kind, 7)

    @classmethod
    def _ordered_fragments(cls, fragments, task):
        terms = set(
            re.findall(r"\w{2,}", f"{task.title} {task.description}".casefold())
        )

        def key(fragment):
            content = fragment["text"].casefold()
            relevance = sum(content.count(term) for term in terms)
            return cls._selection_priority(fragment), -relevance, fragment["ref"]

        return [fragment for fragment in fragments if fragment["required"]] + sorted(
            (fragment for fragment in fragments if not fragment["required"]), key=key
        )

    def build_evidence(self, task, workspace: str):
        fragments = [
            {
                "kind": "task",
                "ref": "context:task/title",
                "text": f"Task: {task.title}",
                "required": True,
            },
            {
                "kind": "task",
                "ref": "context:task/description",
                "text": f"Description: {task.description}",
                "required": True,
            },
            {
                "kind": "task",
                "ref": "context:task/workspace",
                "text": f"Workspace: {workspace}",
                "required": True,
            },
        ]
        decisions = self.memory.read("decisions")
        fragments.append(
            {
                "kind": "decision_memory",
                "ref": "context:vault/decisions.md",
                "text": f"Memory decisions:\n{decisions}"
                if decisions
                else "Memory: none",
                "required": False,
            }
        )
        if task.title.strip() and self.memory.vault.is_dir():
            from .vault_knowledge import VaultKnowledgeService

            service = VaultKnowledgeService(self.memory.vault, source_root=workspace)
            rejected_sources = []
            for match in service.search(task.title, limit=5, context_chars=1000):
                source_state = match["provenance"]["source_state"]
                if source_state not in {"current", "unverified"} or match[
                    "review_state"
                ] in {"stale", "future", "invalid"}:
                    rejected_sources.append(
                        {
                            "ref": match["source_ref"],
                            "status": source_state
                            if source_state not in {"current", "unverified"}
                            else match["review_state"],
                        }
                    )
                    continue
                fragments.append(
                    {
                        "kind": "vault",
                        "ref": match["source_ref"],
                        "text": f"Obsidian match: {match['path']}\n{match['excerpt']}",
                        "required": False,
                        "review_state": match["review_state"],
                        "provenance": match["provenance"],
                        "claims": match["claims"],
                    }
                )
        else:
            rejected_sources = []
        if self.qdrant:
            try:
                for point in self.qdrant.search(task.title, limit=3):
                    payload = point.get("payload", {})
                    point_id = str(point.get("id", len(fragments)))
                    fragments.append(
                        {
                            "kind": "qdrant",
                            "ref": f"context:qdrant/{point_id}",
                            "text": str(payload.get("text", "")),
                            "required": False,
                            "claims": payload.get("claims", {}),
                        }
                    )
            except httpx.HTTPError:
                fragments.append(
                    {
                        "kind": "qdrant_status",
                        "ref": "context:qdrant/unavailable",
                        "text": "Related memory: Qdrant unavailable",
                        "required": False,
                    }
                )
        workspace_root = Path(workspace).resolve()
        for name in ("README.md", "pyproject.toml", "package.json"):
            path = workspace_root / name
            if path.is_file() and path.resolve().is_relative_to(workspace_root):
                fragments.append(
                    {
                        "kind": "repository",
                        "ref": f"context:repository/{name}",
                        "text": path.read_text()[:16000],
                        "required": False,
                    }
                )
        repository_query = f"{task.title} {task.description}"
        if (
            re.search(r"[A-Za-z_][A-Za-z0-9_]{1,63}", repository_query)
            and (workspace_root / "src").is_dir()
        ):
            from .repository_context import RepositoryContextService

            for match in RepositoryContextService(workspace_root).search(
                repository_query, limit=5
            ):
                fragments.append(
                    {
                        "kind": match["kind"],
                        "ref": match["ref"],
                        "text": (
                            f"{match['path']}:{match['line_start']}-"
                            f"{match['line_end']} ({match['symbol']})\n{match['text']}\n"
                            f"Related tests: {', '.join(match['test_paths']) or 'none found'}"
                        ),
                        "required": False,
                        "provenance": match["provenance"],
                    }
                )

        fragments = self._ordered_fragments(fragments, task)
        rendered = []
        included = []
        omitted = []
        truncated = []
        used = 0
        for fragment in fragments:
            prefix = f"[{fragment['ref']}]\n"
            separator = "\n" if rendered else ""
            available = (
                self.max_bytes - used - len((separator + prefix).encode("utf-8"))
            )
            content = fragment["text"]
            if available < 0 or (available == 0 and content):
                if fragment["required"]:
                    raise ValueError("required context exceeds configured byte budget")
                omitted.append(fragment["ref"])
                continue
            if len(content.encode("utf-8")) > available:
                if fragment["required"]:
                    raise ValueError("required context exceeds configured byte budget")
                omitted.append(fragment["ref"])
                continue
            rendered.append(prefix + content)
            used += len((separator + prefix + content).encode("utf-8"))
            included.append({**fragment, "truncated": False})

        while True:
            claims = {}
            claim_issues = {}
            claim_count = 0
            claim_bytes = 0
            for fragment in included:
                values = fragment.get("claims", {})
                if not isinstance(values, dict):
                    continue
                for name, value in values.items():
                    if name not in CONTEXT_CLAIM_FIELDS:
                        continue
                    try:
                        encoded = json.dumps(value, sort_keys=True, allow_nan=False)
                    except (TypeError, ValueError):
                        claim_issues.setdefault(name, []).append(
                            "claim is not valid JSON"
                        )
                        continue
                    encoded_size = len(encoded.encode("utf-8"))
                    if encoded_size > CONTEXT_CLAIM_BYTES_MAX:
                        claim_issues.setdefault(name, []).append(
                            "claim exceeds the evidence size limit"
                        )
                        continue
                    if claim_count >= CONTEXT_CLAIMS_MAX:
                        claim_issues.setdefault(name, []).append(
                            "context claim count exceeds the evidence limit"
                        )
                        continue
                    if claim_bytes + encoded_size > CONTEXT_CLAIMS_BYTES_MAX:
                        claim_issues.setdefault(name, []).append(
                            "total context claim payload exceeds the evidence limit"
                        )
                        continue
                    claims.setdefault(name, []).append(
                        {"ref": fragment["ref"], "value": value}
                    )
                    claim_count += 1
                    claim_bytes += encoded_size
            conflicts = {}
            for name, entries in claims.items():
                distinct = {
                    json.dumps(item["value"], sort_keys=True, default=str)
                    for item in entries
                }
                if len(distinct) > 1:
                    conflicts[name] = {
                        "sources": entries,
                        "values": [item["value"] for item in entries],
                    }
            text = "\n".join(f"[{item['ref']}]\n{item['text']}" for item in included)
            result = {
                "text": text,
                "fragments": included,
                "omitted_sources": list(dict.fromkeys(omitted)),
                "truncated_sources": truncated,
                "rejected_sources": rejected_sources,
                "conflicts": conflicts,
                "claims": claims,
                "claim_issues": claim_issues,
                "budget_bytes": self.max_bytes,
                "used_bytes": 0,
            }
            result["used_bytes"] = context_evidence_size(result)
            if result["used_bytes"] <= self.max_bytes:
                return result
            optional_indices = [
                index
                for index, fragment in enumerate(included)
                if not fragment["required"]
            ]
            if not optional_indices:
                raise ValueError(
                    "required context evidence exceeds configured byte budget"
                )
            optional_index = max(
                optional_indices,
                key=lambda index: (
                    self._selection_priority(included[index]),
                    len(included[index]["text"].encode("utf-8")),
                    included[index]["ref"],
                ),
            )
            removed = included.pop(optional_index)
            omitted.append(removed["ref"])
            truncated = [ref for ref in truncated if ref != removed["ref"]]


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
