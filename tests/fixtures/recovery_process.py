"""Controlled model adapter; orchestration, leases, tools and tests are real."""

import asyncio
import json
import os
import signal
import sys

from harness.core import Config, Orchestrator, Store
from harness.providers import ModelUsage, ToolCall


class Provider:
    def __init__(self, phase, workspace):
        self.phase = phase
        self.workspace = workspace

    def complete(self, prompt, **kwargs):
        if isinstance(prompt, list):
            first = prompt[1]["content"]
            step = json.loads(first.split("\nContext:\n")[0].rsplit("\n", 1)[-1])
            if len(prompt) > 2:
                return json.dumps(
                    {"output": "Implemented declared remaining artifact."}
                )
            if step["id"] == "remaining" and self.phase == "crash":
                (self.workspace / "crash.ready").write_text("active execution")
                os.kill(os.getpid(), signal.SIGKILL)
            name = "addition.py" if step["id"] == "addition" else "remainder.py"
            content = (
                "def add(a, b):\n    return a + b\n"
                if name == "addition.py"
                else "def multiply(a, b):\n    return a * b\n"
            )
            return (
                "",
                ModelUsage(provider="fixture", model="controlled"),
                [
                    ToolCall(
                        id="write-" + step["id"],
                        name="filesystem.write",
                        arguments={"path": name, "content": content},
                    )
                ],
            )
        if prompt.startswith("PLAN:"):
            targets = []
            if "RECOVERY_SCOPE_JSON:\n" in prompt:
                targets = json.loads(
                    prompt.split("RECOVERY_SCOPE_JSON:\n")[-1].splitlines()[0]
                )["remaining_targets"]
            names = (
                ["addition", "remaining"] if self.phase == "crash" else ["remaining"]
            )
            return json.dumps(
                {
                    "summary": "initial implementation"
                    if self.phase == "crash"
                    else "remaining work only",
                    "complexity": "simple",
                    "subtasks": [
                        {
                            "id": name,
                            "title": name,
                            "description": "implement only declared file",
                            "expected_result": "working arithmetic",
                            "requirement_ids": (
                                ["add integers", "multiply integers"]
                                if self.phase == "restart"
                                else [
                                    "add integers"
                                    if name == "addition"
                                    else "multiply integers"
                                ]
                            ),
                            "acceptance_criteria": (
                                ["add", "multiply"]
                                if self.phase == "restart"
                                else ["add" if name == "addition" else "multiply"]
                            ),
                            "required_tools": ["filesystem.write"],
                            "dependencies": ["addition"]
                            if name == "remaining" and self.phase == "crash"
                            else [],
                            "recovery_targets": targets,
                            "write_paths": [
                                "addition.py" if name == "addition" else "remainder.py"
                            ],
                        }
                        for name in names
                    ],
                }
            )
        if prompt.startswith("RECOVERY_REVIEW:"):
            return json.dumps(
                {
                    "requirements": {
                        "add integers": {
                            "status": "completed",
                            "criteria": ["add"],
                            "evidence": "Observed addition.py arithmetic implementation; current failing test requires further inspection.",
                        },
                        "multiply integers": {
                            "status": "remaining",
                            "criteria": ["multiply"],
                            "evidence": "remainder.py missing and import fails in freshly executed tests.",
                        },
                    }
                }
            )
        payload = json.loads(prompt.split("\n", 1)[1])
        available = payload["available_evidence_ids"]
        requirements = payload["task"]["requirements"]
        criteria = [item["id"] for item in payload["task"]["acceptance_criteria"]]
        steps = payload["task"].get("plan", {}).get("subtasks", [])

        def references(key, requirement):
            field = "requirement_ids" if requirement else "acceptance_criteria"
            matching = [
                f"plan-step:{step['id']}"
                for step in steps
                if key in step.get(field, [])
            ]
            observed = next(
                (item for item in available if not item.startswith("plan-step:")),
                None,
            )
            return [*matching[:1], *([observed] if observed else [])]

        return json.dumps(
            {
                "requirements": dict.fromkeys(requirements, True),
                "criteria": dict.fromkeys(criteria, True),
                "evidence": "Observed current source artifacts and fresh test reports.",
                "requirement_evidence": {
                    name: references(name, True) for name in requirements
                },
                "criterion_evidence": {
                    name: references(name, False) for name in criteria
                },
            }
        )


def main():
    config = Config()
    config.data = json.loads(sys.argv[1])
    store = Store(config)
    runtime = Orchestrator(store, config)
    runtime.models.register("fixture", Provider(sys.argv[3], runtime.tools.workspace))
    task = asyncio.run(runtime.run(int(sys.argv[2])))
    print(task.model_dump_json())


if __name__ == "__main__":
    main()
