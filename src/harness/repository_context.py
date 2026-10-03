"""Bounded, non-executing retrieval of repository symbols and related tests."""

from __future__ import annotations

import ast
import hashlib
import re
import stat
from pathlib import Path

_IGNORED = {".git", ".venv", "venv", "node_modules", "__pycache__", ".tox"}
_TOKEN = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]{1,63}")
_SOURCE_SUFFIXES = {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java"}


class RepositoryContextService:
    """Search Python declarations and test references without importing code."""

    def __init__(self, root, *, max_files=2000, max_file_bytes=512_000):
        try:
            self.root = Path(root).resolve(strict=True)
        except (OSError, TypeError, ValueError) as error:
            raise ValueError(
                "repository context root must be an existing directory"
            ) from error
        if not self.root.is_dir():
            raise ValueError("repository context root must be a directory")
        if (
            isinstance(max_files, bool)
            or not isinstance(max_files, int)
            or max_files < 1
        ):
            raise ValueError("max_files must be a positive integer")
        if (
            isinstance(max_file_bytes, bool)
            or not isinstance(max_file_bytes, int)
            or max_file_bytes < 1
        ):
            raise ValueError("max_file_bytes must be a positive integer")
        self.max_files = max_files
        self.max_file_bytes = max_file_bytes

    def _source_files(self):
        files = []
        for directory in sorted(self.root.rglob("*")):
            relative = directory.relative_to(self.root)
            if any(part in _IGNORED or part.startswith(".") for part in relative.parts):
                continue
            if directory.is_symlink():
                continue
            if directory.is_file() and directory.suffix in _SOURCE_SUFFIXES:
                files.append(directory)
                if len(files) >= self.max_files:
                    break
        return files

    def _read(self, path):
        try:
            if path.is_symlink() or not path.resolve(strict=True).is_relative_to(
                self.root
            ):
                return None
            info = path.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_size > self.max_file_bytes:
                return None
            raw = path.read_bytes()
            return raw.decode("utf-8")
        except (OSError, UnicodeError, ValueError):
            return None

    @staticmethod
    def _declarations(tree):
        found = []

        def visit(nodes, prefix=""):
            for node in nodes:
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    name = f"{prefix}.{node.name}" if prefix else node.name
                    found.append(
                        (name, node.lineno, getattr(node, "end_lineno", node.lineno))
                    )
                    visit(getattr(node, "body", ()), name)

        visit(tree.body)
        return found

    @staticmethod
    def _text_declarations(text, suffix):
        patterns = {
            ".js": r"\b(?:class|function|const|let|var)\s+([A-Za-z_$][\w$]*)",
            ".jsx": r"\b(?:class|function|const|let|var)\s+([A-Za-z_$][\w$]*)",
            ".ts": r"\b(?:class|function|interface|type|enum|const|let|var)\s+([A-Za-z_$][\w$]*)",
            ".tsx": r"\b(?:class|function|interface|type|enum|const|let|var)\s+([A-Za-z_$][\w$]*)",
            ".go": r"\b(?:func\s+(?:\([^)]*\)\s*)?|type\s+)([A-Za-z_]\w*)",
            ".rs": r"\b(?:fn|struct|enum|trait|type)\s+([A-Za-z_]\w*)",
            ".java": r"\b(?:class|interface|enum|record)\s+([A-Za-z_]\w*)",
        }
        pattern = patterns.get(suffix)
        if pattern is None:
            return []
        lines = text.splitlines()
        declarations = []
        for line_number, line in enumerate(lines, start=1):
            for match in re.finditer(pattern, line):
                declarations.append((match.group(1), line_number, line_number))
        return declarations

    def search(self, query, *, changed_paths=(), limit=8, excerpt_lines=18):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 50
        ):
            raise ValueError("limit must be between 1 and 50")
        if (
            isinstance(excerpt_lines, bool)
            or not isinstance(excerpt_lines, int)
            or not 1 <= excerpt_lines <= 80
        ):
            raise ValueError("excerpt_lines must be between 1 and 80")
        changed = set()
        for item in changed_paths:
            if not isinstance(item, str) or not item:
                raise ValueError("changed paths must be safe repository-relative paths")
            path = Path(item)
            if path.is_absolute() or ".." in path.parts or "\\" in item:
                raise ValueError("changed paths must be safe repository-relative paths")
            changed.add(path.as_posix())
        terms = {term.casefold() for term in _TOKEN.findall(query)}
        if not terms:
            raise ValueError("query must contain searchable terms")

        sources = []
        files = self._source_files()
        test_files = [
            item
            for item in files
            if (
                item.relative_to(self.root).as_posix().startswith("tests/")
                or item.stem.startswith("test_")
                or item.name.endswith(
                    (".test.js", ".test.jsx", ".test.ts", ".test.tsx")
                )
            )
        ]
        for path in files:
            text = self._read(path)
            if text is None:
                continue
            if path.suffix == ".py":
                try:
                    declarations = self._declarations(
                        ast.parse(text, filename=path.name)
                    )
                except (SyntaxError, ValueError):
                    continue
            else:
                declarations = self._text_declarations(text, path.suffix)
            relative = path.relative_to(self.root).as_posix()
            lines = text.splitlines()
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            module = relative.removesuffix(".py").replace("/", ".")
            for name, start, end in declarations:
                symbol_terms = {token.casefold() for token in _TOKEN.findall(name)}
                path_terms = {token.casefold() for token in _TOKEN.findall(relative)}
                score = 4 * len(terms & symbol_terms) + 2 * len(terms & path_terms)
                if relative in changed:
                    score += 20
                if not score:
                    continue
                first = max(1, start)
                last = min(end, first + excerpt_lines - 1)
                excerpt = "\n".join(lines[first - 1 : last])
                test_paths = []
                for test_path in test_files:
                    test_relative = test_path.relative_to(self.root).as_posix()
                    if not test_relative.startswith("tests/") or test_path == path:
                        continue
                    test_text = self._read(test_path)
                    if test_text is not None and (
                        module.rsplit(".", 1)[-1] in test_text
                        or name.rsplit(".", 1)[-1] in test_text
                    ):
                        test_paths.append(test_relative)
                sources.append(
                    {
                        "kind": "repository_symbol",
                        "ref": f"context:repository/{relative}#{name}",
                        "path": relative,
                        "symbol": name,
                        "line_start": first,
                        "line_end": last,
                        "text": excerpt,
                        "test_paths": sorted(set(test_paths))[:20],
                        "provenance": {"sha256": digest, "source_state": "current"},
                        "_score": score,
                    }
                )
        sources.sort(key=lambda item: (-item["_score"], item["path"], item["symbol"]))
        return [
            {key: value for key, value in item.items() if key != "_score"}
            for item in sources[:limit]
        ]
