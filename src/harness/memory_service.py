"""Markdown is authoritative; vectors only point to current original documents."""

import hashlib
import re
import uuid

from .memory import EMBEDDING_BATCH_SIZE_DEFAULT, EMBEDDING_BATCH_SIZE_MAX


class MemoryService:
    def __init__(
        self,
        notes,
        vectors,
        chunk_size=1200,
        overlap=200,
        batch_size=EMBEDDING_BATCH_SIZE_DEFAULT,
    ):
        if chunk_size < 1 or not 0 <= overlap < chunk_size:
            raise ValueError("invalid memory chunk configuration")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or not 1 <= batch_size <= EMBEDDING_BATCH_SIZE_MAX
        ):
            raise ValueError("invalid memory batch configuration")
        self.notes, self.vectors = notes, vectors
        self.chunk_size, self.overlap = chunk_size, overlap
        self.batch_size = batch_size

    @staticmethod
    def digest(content):
        return hashlib.sha256(content.encode()).hexdigest()

    def chunks(self, content):
        offset = 0
        while offset < len(content):
            yield content[offset : offset + self.chunk_size]
            if offset + self.chunk_size >= len(content):
                return
            offset += self.chunk_size - self.overlap

    @staticmethod
    def _owned_point(point):
        payload = point.get("payload", {})
        if not isinstance(payload, dict):
            return False
        if payload.get("source_type") == "obsidian":
            return True
        source_hash = payload.get("source_hash")
        return (
            isinstance(payload.get("source"), str)
            and isinstance(payload.get("chunk"), int)
            and not isinstance(payload.get("chunk"), bool)
            and payload["chunk"] >= 0
            and isinstance(payload.get("text"), str)
            and isinstance(source_hash, str)
            and re.fullmatch(r"[0-9a-f]{64}", source_hash) is not None
        )

    def _source_points(self, source):
        return self.vectors.scroll_source(source)

    def index(self, name):
        self.notes.path(name)
        content = self.notes.read(name)
        if content is None:
            raise FileNotFoundError(name)
        self.vectors.ensure_collection()
        source_points = self._source_points(name)
        chunks = list(self.chunks(content))
        source_hash = self.digest(content)
        points = [
            (
                f"obsidian:{name}:{index}",
                text,
                {
                    "source": name,
                    "source_type": "obsidian",
                    "source_hash": source_hash,
                    "chunk": index,
                    "text": text,
                },
            )
            for index, text in enumerate(chunks)
        ]
        for start in range(0, len(points), self.batch_size):
            batch = points[start : start + self.batch_size]
            if callable(getattr(self.vectors, "upsert_many", None)):
                self.vectors.upsert_many(batch)
            else:
                for point_id, text, payload in batch:
                    self.vectors.upsert(point_id, text, payload)
        indexed = self._source_points(name)
        keep = set()
        for index, text in enumerate(chunks):
            candidates = [
                point.get("id")
                for point in indexed
                if self._owned_point(point)
                and point.get("payload", {}).get("source_hash") == source_hash
                and point.get("payload", {}).get("chunk") == index
                and point.get("payload", {}).get("text") == text
                and point.get("id") is not None
            ]
            raw_id = f"obsidian:{name}:{index}"
            canonical_id = str(uuid.uuid5(uuid.NAMESPACE_URL, raw_id))
            if candidates:
                keep.add(
                    raw_id
                    if raw_id in candidates
                    else canonical_id
                    if canonical_id in candidates
                    else min(candidates, key=str)
                )
        stale = {
            point.get("id")
            for point in source_points + indexed
            if self._owned_point(point)
            and point.get("id") is not None
            and point.get("id") not in keep
        }
        if stale:
            self.vectors.delete(sorted(stale, key=str))
        return len(chunks)

    def unindex(self, name):
        self.notes.path(name)
        points = self._source_points(name)
        point_ids = sorted(
            {
                point.get("id")
                for point in points
                if self._owned_point(point) and point.get("id") is not None
            },
            key=str,
        )
        if point_ids:
            self.vectors.delete(point_ids)
        return len(point_ids)

    def reconcile(self):
        plan = self.plan_reconciliation()
        for name in plan["needs_index"]:
            self.index(name)
        stale = plan["orphan_points"]
        removed_points = 0
        removed_sources = 0
        for source, point_ids in stale.items():
            # A note created after preview must never be removed from its index.
            try:
                current = self.notes.read(source)
            except (PermissionError, ValueError):
                continue
            if current is not None:
                continue
            self.vectors.delete(point_ids)
            removed_points += len(point_ids)
            removed_sources += 1
        return {
            "notes": plan["notes"],
            "chunks": plan["chunks"],
            "removed_sources": removed_sources,
            "removed_points": removed_points,
        }

    def plan_reconciliation(self):
        """Read-only comparison of authoritative notes with Harness-owned points."""
        documents = self.notes.list_documents()
        present = set(documents)
        indexed = self.vectors.scroll_source()
        by_source = {}
        stale = {}
        for point in indexed:
            payload = point.get("payload", {})
            if not isinstance(payload, dict):
                continue
            source = payload.get("source")
            if not self._owned_point(point) or not isinstance(source, str):
                continue
            if source in present:
                by_source.setdefault(source, []).append(point)
            elif point.get("id") is not None:
                try:
                    self.notes.path(source)
                except PermissionError:
                    continue
                stale.setdefault(source, set()).add(point.get("id"))
        needs_index = []
        total_chunks = 0
        for name in documents:
            content = self.notes.read(name)
            if content is None:
                continue
            chunks = list(self.chunks(content))
            total_chunks += len(chunks)
            digest = self.digest(content)
            actual = by_source.get(name, [])
            expected = {(index, digest, text) for index, text in enumerate(chunks)}
            observed = {
                (
                    point["payload"].get("chunk"),
                    point["payload"].get("source_hash"),
                    point["payload"].get("text"),
                )
                for point in actual
            }
            if len(actual) != len(expected) or observed != expected:
                needs_index.append(name)
        return {
            "notes": len(documents),
            "chunks": total_chunks,
            "needs_index": needs_index,
            "orphan_sources": sorted(stale),
            "orphan_points": {
                source: sorted(point_ids, key=str)
                for source, point_ids in stale.items()
            },
        }

    def search(self, query, limit=5):
        return self.verified_search_points(self.vectors.search(query, limit=limit))

    def verified_search_points(self, points):
        """Resolve only current, Vault-backed hits from one vector search response."""
        if not isinstance(points, list):
            raise TypeError("vector search response must be a list")
        results, seen = [], set()
        for point in points:
            if not isinstance(point, dict):
                raise TypeError("vector search hit must be an object")
            payload = point.get("payload", {})
            if not isinstance(payload, dict):
                raise TypeError("vector search payload must be an object")
            name = payload.get("source")
            if not isinstance(name, str) or not name or name in seen:
                continue
            try:
                content = self.notes.read(name)
            except (PermissionError, ValueError):
                continue
            if content is None or self.digest(content) != payload.get("source_hash"):
                continue
            seen.add(name)
            results.append(
                {"id": point.get("id"), "payload": {"source": name, "text": content}}
            )
        return results
