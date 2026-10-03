import hashlib

import pytest

from harness.repository_context import RepositoryContextService


def test_repository_context_retrieves_symbol_tests_and_hash(tmp_path):
    source = tmp_path / "src" / "memory.py"
    source.parent.mkdir()
    source.write_text("class ContextBuilder:\n    def build(self):\n        return 1\n")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_memory.py").write_text("from src.memory import ContextBuilder\n")

    result = RepositoryContextService(tmp_path).search("ContextBuilder")

    assert result[0]["symbol"] == "ContextBuilder"
    assert result[0]["ref"] == "context:repository/src/memory.py#ContextBuilder"
    assert result[0]["test_paths"] == ["tests/test_memory.py"]
    assert (
        result[0]["provenance"]["sha256"]
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    assert "def build" in result[0]["text"]


def test_repository_context_weights_changed_files_and_is_deterministic(tmp_path):
    (tmp_path / "a.py").write_text("def process_task():\n    return 1\n")
    (tmp_path / "b.py").write_text("def process_task():\n    return 2\n")
    service = RepositoryContextService(tmp_path)
    first = service.search("process", changed_paths=("b.py",))
    second = service.search("process", changed_paths=("b.py",))
    assert first == second
    assert first[0]["path"] == "b.py"


def test_repository_context_skips_hidden_invalid_and_oversized_sources(tmp_path):
    (tmp_path / "visible.py").write_text("def task_context():\n    pass\n")
    hidden = tmp_path / ".git"
    hidden.mkdir()
    (hidden / "secret.py").write_text("def task_context():\n    pass\n")
    (tmp_path / "broken.py").write_text("def task_context(:\n")
    (tmp_path / "large.py").write_text("def task_context():\n" + "x" * 100)
    results = RepositoryContextService(tmp_path, max_file_bytes=40).search(
        "task_context"
    )
    assert [item["path"] for item in results] == ["visible.py"]


def test_repository_context_rejects_invalid_configuration_and_query(tmp_path):
    for kwargs in ({"max_files": True}, {"max_files": 0}, {"max_file_bytes": False}):
        with pytest.raises(ValueError):
            RepositoryContextService(tmp_path, **kwargs)
    with pytest.raises(ValueError):
        RepositoryContextService(tmp_path / "missing")
    with pytest.raises(ValueError):
        RepositoryContextService(tmp_path / "file")
    (tmp_path / "file").write_text("x")
    with pytest.raises(ValueError):
        RepositoryContextService(tmp_path / "file")
    service = RepositoryContextService(tmp_path)
    for kwargs in (
        {"query": "  "},
        {"query": "!"},
        {"query": "context", "limit": True},
        {"query": "context", "limit": 0},
        {"query": "context", "excerpt_lines": True},
        {"query": "context", "excerpt_lines": 0},
    ):
        query = kwargs.pop("query")
        with pytest.raises(ValueError):
            service.search(query, **kwargs)


@pytest.mark.parametrize(
    "changed",
    [("../outside.py",), ("/tmp/a.py",), ("bad\\path.py",), (3,)],
)
def test_repository_context_rejects_unsafe_changed_paths(tmp_path, changed):
    with pytest.raises(ValueError):
        RepositoryContextService(tmp_path).search("context", changed_paths=changed)


def test_repository_context_bounds_scan_count_and_does_not_follow_symlinks(tmp_path):
    (tmp_path / "a.py").write_text("def context_task():\n    return True\n")
    (tmp_path / "b.py").write_text("def context_task():\n    return False\n")
    outside = tmp_path.parent / "outside.py"
    outside.write_text("def context_task():\n    return 'secret'\n")
    (tmp_path / "link.py").symlink_to(outside)
    results = RepositoryContextService(tmp_path, max_files=1).search("context_task")
    assert len(results) == 1
    assert results[0]["path"] == "a.py"
    assert "secret" not in results[0]["text"]


def test_repository_context_handles_symlink_directories_and_bad_sources(tmp_path):
    (tmp_path / "source.py").write_text("def context_source():\n    return 1\n")
    outside = tmp_path.parent / "context-outside"
    outside.mkdir()
    (outside / "linked.py").write_text("def context_source():\n    return 2\n")
    (tmp_path / "linked-dir").symlink_to(outside, target_is_directory=True)
    (tmp_path / "bad-encoding.py").write_bytes(b"\xff\xfe")
    (tmp_path / "bad-syntax.py").write_text("def context_source(:\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "bad-test.py").write_bytes(b"\xff")
    (tmp_path / "tests" / "unrelated.py").write_text("def other():\n    pass\n")

    service = RepositoryContextService(tmp_path)
    results = service.search("context_source")
    assert [item["path"] for item in results] == ["source.py"]
    assert service._read(outside / "linked.py") is None
    assert service._read(tmp_path / "missing.py") is None


def test_repository_context_does_not_link_a_test_to_itself(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    test_file = tests / "test_context.py"
    test_file.write_text("def context_test_symbol():\n    pass\n")
    result = RepositoryContextService(tmp_path).search("context_test_symbol")
    own_symbol = next(
        item for item in result if item["path"] == "tests/test_context.py"
    )
    assert own_symbol["test_paths"] == []


@pytest.mark.parametrize(
    ("suffix", "source", "symbol", "test_name", "test_source"),
    [
        (
            ".ts",
            "export class ContextEngine { run() { return true; } }\n",
            "ContextEngine",
            "context-engine.test.ts",
            "import { ContextEngine } from './context-engine';\n",
        ),
        (
            ".go",
            "package core\nfunc ExecuteTask() bool { return true }\n",
            "ExecuteTask",
            "engine_test.go",
            "func TestExecuteTask(t *testing.T) {}\n",
        ),
        (
            ".rs",
            "pub fn validate_context() -> bool { true }\n",
            "validate_context",
            "context_tests.rs",
            "fn validate_context_test() {}\n",
        ),
        (
            ".java",
            "public class RepositoryResolver { void resolve() {} }\n",
            "RepositoryResolver",
            "RepositoryResolverTest.java",
            "class RepositoryResolverTest { void testRepositoryResolver() {} }\n",
        ),
    ],
)
def test_repository_context_supports_static_language_adapters_and_related_tests(
    tmp_path, suffix, source, symbol, test_name, test_source
):
    implementation = tmp_path / "src" / f"implementation{suffix}"
    implementation.parent.mkdir()
    implementation.write_text(source)
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / test_name).write_text(test_source)

    results = RepositoryContextService(tmp_path).search(symbol)

    assert results[0]["symbol"].endswith(symbol)
    assert results[0]["test_paths"] == [f"tests/{test_name}"]
    assert len(results[0]["provenance"]["sha256"]) == 64
    assert symbol in results[0]["text"]


def test_text_declaration_adapter_reports_unsupported_language():
    assert RepositoryContextService._text_declarations("fn task() {}", ".unknown") == []
