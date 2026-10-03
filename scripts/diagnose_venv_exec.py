"""Compare metadata allowances for the real ToolRegistry coverage process.

Run on native macOS with ``.venv/bin/python scripts/diagnose_venv_exec.py``.
Each probe uses a disposable workspace. No production sandbox policy changes.
"""

import json
import sys
import tempfile
from pathlib import Path

import harness.tools as tool_module
from harness import isolation
from harness.core import Config, Permissions
from harness.tools import ToolRegistry


def _subpath(path):
    return f"(allow file-read-metadata (subpath {json.dumps(str(path))}))"


def _literal(path):
    return f"(allow file-read-metadata (literal {json.dumps(str(path))}))"


def main():
    config = Config()
    config.data["tools"] = {
        "permissions": {"shell": "write"},
        "mcp": {"servers": {}},
    }
    original = tool_module.isolated_command
    executable = Path(sys.executable)
    extension = isolation._sqlite_extension_for_interpreter(sys.executable)
    runtime_files = [executable.resolve(strict=True)]
    if extension is not None:
        runtime_files.append(extension)
    libraries = isolation._runtime_dylibs(runtime_files)
    library_entries = {
        entry
        for library in libraries
        for path in (library, library.resolve(strict=True))
        for entry in isolation._metadata_path_entries(path)
    }
    homebrew = Path("/opt/homebrew")
    python_formula = f"python@{sys.version_info.major}.{sys.version_info.minor}"
    cellar_python = homebrew / "Cellar" / python_formula
    cellar_sqlite = homebrew / "Cellar" / "sqlite"
    python_version_root = Path(sys.base_prefix).resolve(strict=True).parents[3]
    sqlite_libraries = tuple(
        path for path in libraries if path.name.startswith("libsqlite")
    )
    sqlite_resolved = next(
        path for path in sqlite_libraries if path.is_relative_to(cellar_sqlite)
    )
    sqlite_version_root = sqlite_resolved.parents[1]
    with tempfile.TemporaryDirectory(prefix="harness-venv-exec-") as directory:
        base = Path(directory).resolve(strict=True)
        registry = ToolRegistry(Permissions(config), workspace=base)
        cases = (
            ("baseline", None),
            ("venv_bin", _subpath(executable.parent)),
            ("resolved_python_bin", _subpath(executable.resolve().parent)),
            ("library_files", "\n".join(_literal(path) for path in libraries)),
            (
                "library_paths",
                "\n".join(_literal(path) for path in sorted(library_entries)),
            ),
            ("homebrew_opt", _subpath(homebrew / "opt")),
            ("homebrew_cellar", _subpath(homebrew / "Cellar")),
            (
                "homebrew_opt_cellar",
                "\n".join((_subpath(homebrew / "opt"), _subpath(homebrew / "Cellar"))),
            ),
            ("opt_python", _subpath(homebrew / "opt" / python_formula)),
            ("cellar_python", _subpath(cellar_python)),
            ("cellar_python_version", _subpath(python_version_root)),
            (
                "python_version_sqlite_files",
                "\n".join(
                    (
                        _subpath(python_version_root),
                        *(_literal(path) for path in sqlite_libraries),
                    )
                ),
            ),
            (
                "python_version_sqlite_lib",
                "\n".join(
                    (_subpath(python_version_root), _subpath(sqlite_resolved.parent))
                ),
            ),
            (
                "python_version_sqlite_version",
                "\n".join(
                    (_subpath(python_version_root), _subpath(sqlite_version_root))
                ),
            ),
            (
                "python_version_sqlite_formula",
                "\n".join((_subpath(python_version_root), _subpath(cellar_sqlite))),
            ),
            ("cellar_python_frameworks", _subpath(python_version_root / "Frameworks")),
            (
                "cellar_python_sqlite",
                "\n".join((_subpath(cellar_python), _subpath(cellar_sqlite))),
            ),
            ("opt_sqlite", _subpath(homebrew / "opt" / "sqlite")),
            ("cellar_sqlite", _subpath(cellar_sqlite)),
            ("homebrew_lib", _subpath(homebrew / "lib")),
            ("homebrew_share", _subpath(homebrew / "share")),
            ("homebrew", _subpath(homebrew)),
            ("users", _subpath(Path("/Users"))),
            ("temporary_root", _subpath(base.parent)),
            ("dev", _subpath(Path("/dev"))),
            ("system", _subpath(Path("/System"))),
            ("usr", _subpath(Path("/usr"))),
            ("global_metadata", "(allow file-read-metadata)"),
        )
        print(json.dumps({"loaded_isolation": str(Path(isolation.__file__).resolve())}))
        for name, extra_rule in cases:
            workspace = base / name
            workspace.mkdir()
            (workspace / "addition.py").write_text("def add(a, b):\n    return a + b\n")
            (workspace / "check.py").write_text(
                "from addition import add\nassert add(2, 3) == 5\n"
            )

            def with_rule(*args, _rule=extra_rule, **kwargs):
                command = original(*args, **kwargs)
                if _rule:
                    command[2] += "\n" + _rule
                return command

            tool_module.isolated_command = with_rule
            try:
                result = registry.shell(
                    [
                        sys.executable,
                        "-m",
                        "coverage",
                        "run",
                        "--branch",
                        "--source=addition",
                        "check.py",
                    ],
                    workspace,
                )
            finally:
                tool_module.isolated_command = original
            stderr_lines = result.stderr.strip().splitlines()
            print(
                json.dumps(
                    {
                        "case": name,
                        "returncode": result.returncode,
                        "stderr": stderr_lines[:2],
                        "stderr_tail": stderr_lines[-4:]
                        if len(stderr_lines) > 2
                        else [],
                    },
                    ensure_ascii=False,
                )
            )


if __name__ == "__main__":
    main()
