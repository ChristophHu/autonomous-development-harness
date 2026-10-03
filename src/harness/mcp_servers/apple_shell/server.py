"""Fixed-operation macOS diagnostics; this server never accepts shell text."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from jsonschema import Draft202012Validator

from ..filesystem import _response as base_response
from ..filesystem import _schema, serve_stdio

EXECUTABLES = {
    "sw_vers": "/usr/bin/sw_vers",
    "uname": "/usr/bin/uname",
    "date": "/bin/date",
}
OPERATIONS = {
    "software_version": ("sw_vers", ()),
    "kernel_info": ("uname", ("-a",)),
    "utc_time": ("date", ("-u", "+%Y-%m-%dT%H:%M:%SZ")),
}
TOOLS = {
    "diagnostic": _schema(
        {"operation": {"type": "string", "enum": sorted(OPERATIONS)}}, ["operation"]
    )
}
OUTPUTS = {
    "diagnostic": _schema(
        {"operation": {"type": "string"}, "output": {"type": "string"}},
        ["operation", "output"],
    )
}
MAX_OUTPUT = 16_384


class AppleShellServer:
    def __init__(self, workspace):
        self.workspace = Path(workspace).resolve(strict=True)
        if not self.workspace.is_dir():
            raise ValueError("workspace must be a directory")

    def call(self, name, arguments):
        if name not in TOOLS:
            raise ValueError("unknown Apple Shell tool")
        Draft202012Validator(TOOLS[name]).validate(arguments)
        operation = arguments["operation"]
        executable_key, argv = OPERATIONS[operation]
        try:
            result = subprocess.run(
                [EXECUTABLES[executable_key], *argv],
                cwd=self.workspace,
                env={"PATH": "/usr/bin:/bin", "LANG": "C"},
                capture_output=True,
                timeout=5,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError("macOS diagnostic command timed out") from exc
        if result.returncode:
            raise ValueError("macOS diagnostic command failed")
        output = result.stdout[: MAX_OUTPUT + 1].decode("utf-8", errors="replace")
        if len(output) > MAX_OUTPUT:
            output = output[:MAX_OUTPUT]
        return {"operation": operation, "output": output}


def _response(server, message):
    return base_response(
        server,
        message,
        tools=TOOLS,
        outputs=OUTPUTS,
        server_name="harness-apple-shell",
        description_prefix="macOS read-only diagnostic",
        error_text="Apple Shell diagnostic failed",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace")
    args = parser.parse_args()
    serve_stdio(AppleShellServer(args.workspace), _response)


if __name__ == "__main__":
    main()
