"""Provider-independent agents with validated plans and audited tool conversations."""

from __future__ import annotations

import json
from contextlib import nullcontext
from enum import StrEnum

import httpx
from pydantic import BaseModel, Field, model_validator

from .process_control import TaskCancelled, current_run_control
from .providers import (
    CostCalculator,
    ModelUsage,
    OpenAICompatibleProvider,
    ToolCall,
    UsageTracker,
)


class Complexity(StrEnum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class Subtask(BaseModel):
    id: str = Field(min_length=1)
    title: str
    description: str
    profile: str = "coding"
    assigned_agent: str = "executor"
    dependencies: list[str] = Field(default_factory=list)
    expected_result: str = ""
    acceptance_criteria: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    test_strategy: list[list[str]] = Field(default_factory=list)
    validation_strategy: str = "independent task validation"
    recovery_targets: list[str] = Field(default_factory=list)
    write_paths: list[str] = Field(default_factory=list)


class PlannerOutput(BaseModel):
    summary: str
    complexity: Complexity
    subtasks: list[Subtask] = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def graph(self):
        ids = [step.id for step in self.subtasks]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate plan step IDs")
        known = set(ids)
        for step in self.subtasks:
            if step.id in step.dependencies or not set(step.dependencies) <= known:
                raise ValueError("invalid plan dependencies")
        self.ordered_steps()
        return self

    def ordered_steps(self):
        remaining = list(self.subtasks)
        completed = set()
        ordered = []
        while remaining:
            ready = [step for step in remaining if set(step.dependencies) <= completed]
            if not ready:
                raise ValueError("cyclic plan dependencies")
            for step in ready:
                ordered.append(step)
                completed.add(step.id)
                remaining.remove(step)
        return ordered


class ExecutorOutput(BaseModel):
    subtask_id: str
    success: bool
    output: str
    changed_files: list[str] = Field(default_factory=list)
    tool_evidence: list[dict] = Field(default_factory=list)


class ValidatorOutput(BaseModel):
    valid: bool
    checks: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    required_corrections: list[str] = Field(default_factory=list)


class RecoveryOutput(BaseModel):
    recovered: bool
    action: str
    reason: str = ""


class QuestionOutput(BaseModel):
    question: str
    blocking: bool = True


class DecisionOutput(BaseModel):
    decision: str
    rationale: str


class TaskClassificationOutput(BaseModel):
    complexity: Complexity
    profile: str


class ModelProvider:
    """Minimal provider interface; adapters must implement completion."""

    def complete(self, prompt, **kwargs):
        raise NotImplementedError("provider completion is not implemented")


class ModelResponse(BaseModel):
    text: str
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ModelRegistry:
    def __init__(self, config):
        self.config = config
        self.providers = {}
        self.models = config.data.get("models", {}).get("registry", {})
        for name, data in config.data.get("models", {}).get("providers", {}).items():
            if data.get("enabled"):
                self.providers[name] = OpenAICompatibleProvider(
                    name,
                    data["base_url"],
                    config.data.get("secrets", {}).get(f"{name.upper()}_API_KEY"),
                    model=data.get("model")
                    or config.data.get("secrets", {}).get(f"{name.upper()}_MODEL"),
                    timeout=data.get("timeout", 120),
                )

    def register(self, name: str, provider: ModelProvider):
        self.providers[name] = provider

    def get(self, name: str):
        if name not in self.providers:
            raise ValueError(f"unknown or disabled provider: {name}")
        return self.providers[name]

    def resolve(self, name, needs_tools=False):
        definition = self.models.get(name)
        if definition:
            if needs_tools and "tools" not in definition.get("capabilities", []):
                raise ValueError(f"model {name} does not support tools")
            model = definition.get("model")
            if not model:
                raise ValueError(f"model {name} has no model ID")
            return self.get(definition["provider"]), model
        return self.get(name), None


class AgentProfile(BaseModel):
    name: str
    instructions: str
    model: str
    permissions: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    max_steps: int = Field(default=20, ge=1, le=200)


class ProfileRegistry:
    def __init__(self, config):
        self.config = config

    def get(self, name: str):
        profiles = self.config.data.get("profiles", {})
        if name not in profiles or not profiles[name].get("model", {}).get("primary"):
            raise ValueError(f"unknown or invalid agent profile: {name}")
        data = profiles[name]
        return AgentProfile(
            name=name,
            instructions=data.get("instructions", f"Act as {name}"),
            model=data["model"]["primary"],
            permissions=data.get("permissions", []),
            tools=data.get("tools", []),
            max_steps=data.get("max_steps", 20),
        )


class ModelRouter:
    def __init__(
        self, registry: ModelRegistry, config, usage_callback=None, audit=None
    ):
        self.registry = registry
        self.config = config
        self.usage_callback = usage_callback
        self.audit = audit
        self.usage = UsageTracker()

    def candidates(self, profile):
        selected = ProfileRegistry(self.config).get(profile)
        data = self.config.data["profiles"][profile]["model"]
        return [selected.model, *data.get("fallback", [])]

    def select(self, profile: str, complexity: Complexity = Complexity.MODERATE):
        return self.registry.resolve(self.candidates(profile)[0])[0]

    def complete(self, profile, prompt, tools=None):
        failures = []
        for fallback_index, name in enumerate(self.candidates(profile)):
            control = current_run_control()
            if control is not None:
                control.check()
            try:
                with (
                    self.audit.model(name, name, fallback_index)
                    if self.audit
                    else nullcontext({}) as span
                ):
                    provider, model = self.registry.resolve(name, bool(tools))
                    span["provider"] = getattr(
                        provider,
                        "name",
                        self.registry.models.get(name, {}).get("provider", name),
                    )
                    span["model"] = model or getattr(provider, "model", "") or ""
                    kwargs = {}
                    if tools is not None:
                        kwargs["tools"] = tools
                    if model:
                        kwargs["model"] = model
                    result = provider.complete(prompt, **kwargs)
                    if control is not None:
                        control.check()
                    if isinstance(result, tuple):
                        text, usage = result[:2]
                        calls = result[2] if len(result) > 2 else []
                        if not isinstance(usage, ModelUsage):
                            raise TypeError("invalid usage response")
                        CostCalculator(
                            self.config.data.get("models", {}).get("rates", {})
                        ).calculate(usage)
                        span["usage"] = usage
                        if control is not None:
                            control.check()
                        self.usage.record(usage)
                        if self.usage_callback:
                            self.usage_callback(usage)
                    else:
                        text, calls = result, []
                    if not isinstance(text, str) or (not text.strip() and not calls):
                        raise ValueError("empty or invalid model response")
                    if control is not None:
                        control.check()
                    response = ModelResponse(text=text, tool_calls=calls)
                    if tools is not None:
                        return response
                    if calls:
                        raise ValueError("unexpected tool calls without tool contract")
                    return response.text
            except TaskCancelled:
                raise
            except (
                RuntimeError,
                OSError,
                httpx.HTTPError,
                ValueError,
                TypeError,
            ) as exc:
                if control is not None:
                    control.check()
                failures.append(f"{name}: {type(exc).__name__}")
        raise RuntimeError("all model providers failed: " + "; ".join(failures))


class Planner:
    def __init__(self, router: ModelRouter):
        self.router = router

    def plan(self, task, context="") -> PlannerOutput:
        payload = task.model_dump(mode="json")
        answer = self.router.complete(
            "planner",
            "PLAN: Return JSON matching this schema:\n"
            + json.dumps(PlannerOutput.model_json_schema())
            + "\nTask:\n"
            + json.dumps(payload)
            + "\nContext:\n"
            + context,
        )
        plan = PlannerOutput.model_validate_json(answer)
        for step in plan.subtasks:
            if not step.expected_result or not step.acceptance_criteria:
                raise ValueError(
                    "plan steps require expected results and acceptance criteria"
                )
            profile = ProfileRegistry(self.router.config).get(step.profile)
            if not set(step.required_tools) <= set(profile.tools):
                raise ValueError("plan requests tools not granted to profile")
        return plan


class Executor:
    def __init__(self, router: ModelRouter, tools=None, profiles=None):
        self.router = router
        self.tools = tools
        self.profiles = profiles or ProfileRegistry(router.config)

    def execute(self, subtask: Subtask, context: str = "") -> ExecutorOutput:
        changed, evidence = [], []
        try:
            profile = self.profiles.get(subtask.profile)
            schemas = self.tools.schemas(profile.tools) if self.tools else []
            if not set(subtask.required_tools) <= set(profile.tools):
                raise PermissionError("required tool is not granted to profile")
            messages = [
                {"role": "system", "content": profile.instructions},
                {
                    "role": "user",
                    "content": "EXECUTE: Execute only this step; report performed work.\n"
                    + subtask.model_dump_json()
                    + "\nContext:\n"
                    + context,
                },
            ]
            for _ in range(profile.max_steps):
                response = self.router.complete(
                    subtask.profile, messages, tools=schemas
                )
                if not response.tool_calls:
                    if not set(subtask.required_tools) <= {
                        item["tool"] for item in evidence
                    }:
                        raise ValueError("no tool evidence for required work")
                    return ExecutorOutput(
                        subtask_id=subtask.id,
                        success=True,
                        output=response.text,
                        changed_files=changed,
                        tool_evidence=evidence,
                    )
                messages.append(
                    {
                        "role": "assistant",
                        "content": response.text or None,
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": json.dumps(call.arguments),
                                },
                            }
                            for call in response.tool_calls
                        ],
                    }
                )
                for call in response.tool_calls:
                    if call.name not in profile.tools:
                        raise PermissionError(
                            f"profile {profile.name} may not use {call.name}"
                        )
                    result = self.tools.execute(
                        call.name, call.arguments, profile=profile
                    )
                    if getattr(result, "returncode", 0):
                        raise RuntimeError(
                            f"tool {call.name} failed with exit code {result.returncode}"
                        )
                    observation = str(result)
                    evidence.append(
                        {"call_id": call.id, "tool": call.name, "result": observation}
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": observation,
                        }
                    )
                    if call.name in {
                        "filesystem.write",
                        "filesystem.create",
                        "filesystem.move",
                        "filesystem.copy",
                        "filesystem.delete",
                    }:
                        changed.append(
                            str(
                                call.arguments.get("destination")
                                or call.arguments["path"]
                            )
                        )
            raise RuntimeError("profile tool step limit exceeded")
        except TaskCancelled:
            raise
        except (RuntimeError, OSError, httpx.HTTPError, ValueError, KeyError) as exc:
            return ExecutorOutput(
                subtask_id=subtask.id,
                success=False,
                output=str(exc),
                changed_files=changed,
                tool_evidence=evidence,
            )


class Validator:
    """Full validation is provided by EvidenceValidator in validation.py."""

    def validate(self, task, outputs: list[ExecutorOutput]) -> ValidatorOutput:
        errors = [f"subtask {o.subtask_id} failed" for o in outputs if not o.success]
        errors.append("independent validation evidence is required")
        return ValidatorOutput(
            valid=False, checks=["subtask completion"], errors=errors
        )


class RecoveryAgent:
    def recover(self, error: Exception, attempt: int = 1) -> RecoveryOutput:
        return RecoveryOutput(
            recovered=False, action="reconcile-required", reason=str(error)
        )
