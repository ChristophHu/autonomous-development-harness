"""Markdown is authoritative; vectors only point to current original documents."""

import hashlib


class MemoryService:
    def __init__(self, notes, vectors, chunk_size=1200, overlap=200):
        if chunk_size < 1 or not 0 <= overlap < chunk_size:
            raise ValueError("invalid memory chunk configuration")
        self.notes, self.vectors = notes, vectors
        self.chunk_size, self.overlap = chunk_size, overlap

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

    def index(self, name):
        content = self.notes.read(name)
        if content is None:
            raise FileNotFoundError(name)
        self.vectors.ensure_collection()
        count = 0
        for index, text in enumerate(self.chunks(content)):
            self.vectors.upsert(
                f"obsidian:{name}:{index}",
                text,
                {
                    "source": name,
                    "source_hash": self.digest(content),
                    "chunk": index,
                    "text": text,
                },
            )
            count += 1
        return count

    def search(self, query, limit=5):
        results, seen = [], set()
        for point in self.vectors.search(query, limit=limit):
            payload = point.get("payload", {})
            name = payload.get("source")
            if not name or name in seen:
                continue
            content = self.notes.read(name)
            if content is None or self.digest(content) != payload.get("source_hash"):
                continue
            seen.add(name)
            results.append(
                {"id": point.get("id"), "payload": {"source": name, "text": content}}
            )
        return results
