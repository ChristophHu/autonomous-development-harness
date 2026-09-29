"""Bounded repository and workspace search without following symlinks."""

import re
from fnmatch import fnmatch
from pathlib import Path

_DECLARATION = re.compile(
    r"^\s*(?:(?:export|default|async|public|private|protected|static|final)\s+)*"
    r"(?:(?P<class>class|interface|struct|enum|trait)\s+"
    r"|(?P<function>def|function|func|fn)\s+)"
    r"(?P<name>[A-Za-z_][A-Za-z_0-9]*)\b"
)
_SOURCE_SUFFIXES = {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java"}


class RepositorySearch:
    def __init__(self, workspace):
        self.workspace = Path(workspace).resolve(strict=True)

    def _directory(self, directory):
        selected = (self.workspace / directory).resolve(strict=True)
        if not selected.is_relative_to(self.workspace):
            raise PermissionError("search directory escapes configured workspace")
        if not selected.is_dir():
            raise NotADirectoryError(selected)
        return selected

    @staticmethod
    def _limit(limit):
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 500
        ):
            raise ValueError("search limit must be an integer between 1 and 500")

    def _files(self, directory):
        selected = self._directory(directory)
        for path in sorted(selected.rglob("*")):
            if ".git" in path.relative_to(self.workspace).parts or path.is_symlink():
                continue
            if path.is_file():
                yield path

    @staticmethod
    def _lines(path):
        if path.stat().st_size > 1_000_000:
            return ()
        raw = path.read_bytes()
        if b"\0" in raw:
            return ()
        return raw.decode("utf-8", errors="replace").splitlines()

    def file_search(self, pattern, *, directory=".", limit=100):
        self._limit(limit)
        if not pattern:
            raise ValueError("file pattern must not be empty")
        result = []
        for path in self._files(directory):
            relative = path.relative_to(self.workspace).as_posix()
            if fnmatch(path.name, pattern) or fnmatch(relative, pattern):
                result.append(relative)
                if len(result) == limit:
                    break
        return result

    def text_search(self, query, *, directory=".", limit=100):
        self._limit(limit)
        if not query:
            raise ValueError("search query must not be empty")
        result = []
        for path in self._files(directory):
            for number, line in enumerate(self._lines(path), 1):
                if query in line:
                    result.append(
                        {
                            "path": path.relative_to(self.workspace).as_posix(),
                            "line": number,
                            "text": line[:500],
                        }
                    )
                    if len(result) == limit:
                        return result
        return result

    def symbol_search(self, query, *, directory=".", limit=100):
        self._limit(limit)
        if not query:
            raise ValueError("symbol query must not be empty")
        result = []
        for path in self._files(directory):
            if path.suffix not in _SOURCE_SUFFIXES:
                continue
            for number, line in enumerate(self._lines(path), 1):
                match = _DECLARATION.match(line)
                if match and query in match["name"]:
                    result.append(
                        {
                            "path": path.relative_to(self.workspace).as_posix(),
                            "line": number,
                            "name": match["name"],
                            "kind": "class" if match["class"] else "function",
                        }
                    )
                    if len(result) == limit:
                        return result
        return result
