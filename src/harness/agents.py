"""Provider-independent agents with validated plans and audited tool conversations."""

from __future__ import annotations

import json
import logging
import time  # noqa: F401 - compatibility clock hook used by routing tests
from contextlib import nullcontext

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .approvals import ApprovalDenied, ApprovalRequired
from .domain import TaskComplexity
from .process_control import TaskCancelled, current_run_control
from .providers import (
    CostCalculator,
    ModelUsage,
    OpenAICompatibleProvider,
    ProviderError,
    ToolCall,
    UsageTracker,
)
from .retry_budget import RetryBudget, use_retry_budget
from .structured_output import parse_model_output

Complexity = TaskComplexity


class ModelBudgetExceeded(RuntimeError):
    """Raised when a task's configured model budget is spent or unverifiable."""


class Subtask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    title: str
    description: str
    profile: str = "coding"
    assigned_agent: str = "executor"
    dependencies: list[str] = Field(default_factory=list)
    expected_result: str = ""
    acceptance_criteria: list[str] = Field(default_factory=list)
    requirement_ids: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    test_strategy: list[list[str]] = Field(default_factory=list)
    validation_strategy: str = "independent task validation"
    recovery_targets: list[str] = Field(default_factory=list)
    write_paths: list[str] = Field(default_factory=list)


class PlannerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    complexity: Complexity
    subtasks: list[Subtask] = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)

    def validate_task_coverage(
        self, task, required_requirements=None, required_criteria=None
    ):
        errors = []
        requirement_ids = {
            value for step in self.subtasks for value in step.requirement_ids
        }
        criterion_ids = {
            value for step in self.subtasks for value in step.acceptance_criteria
        }
        required_requirements = (
            set(task.requirements)
            if required_requirements is None
            else set(required_requirements)
        )
        required_criteria = (
            {criterion.id for criterion in task.acceptance_criteria}
            if required_criteria is None
            else set(required_criteria)
        )
        for requirement in required_requirements:
            if requirement not in requirement_ids:
                errors.append(f"requirement has no plan step: {requirement}")
                continue
            linked = [
                step for step in self.subtasks if requirement in step.requirement_ids
            ]
            if not any(step.acceptance_criteria for step in linked):
                errors.append(
                    f"requirement has no linked acceptance criterion: {requirement}"
                )
        expected_criteria = required_criteria
        for criterion in expected_criteria - criterion_ids:
            errors.append(f"acceptance criterion has no plan step: {criterion}")
        all_requirements = set(task.requirements)
        if all_requirements:
            unknown_requirements = requirement_ids - all_requirements
            for requirement in sorted(unknown_requirements):
                errors.append(f"plan references unknown requirement: {requirement}")
        all_criteria = {criterion.id for criterion in task.acceptance_criteria}
        if all_criteria:
            for criterion in sorted(criterion_ids - all_criteria):
                errors.append(
                    f"plan references unknown acceptance criterion: {criterion}"
                )
        return errors

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
    model_config = ConfigDict(extra="forbid")

    subtask_id: str
    success: bool
    output: str
    changed_files: list[str] = Field(default_factory=list)
    tool_evidence: list[dict] = Field(default_factory=list)


class ExecutorReport(BaseModel):
    """Model-authored completion summary; execution evidence stays system-owned."""

    model_config = ConfigDict(extra="forbid", strict=True)

    output: str


class CorrectionFinding(BaseModel):
    """Stable, structured validator evidence suitable for persistence and retries."""

    model_config = ConfigDict(extra="forbid")

    category: str
    source: str = "validator"
    rule: str
    message: str
    subtask_id: str | None = None
    affected_paths: list[str] = Field(default_factory=list)
    evidence: dict = Field(default_factory=dict)
    expected: dict = Field(default_factory=dict)


class ValidatorOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: bool
    checks: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    required_corrections: list[str] = Field(default_factory=list)
    findings: list[CorrectionFinding] = Field(default_factory=list)


class RecoveryOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recovered: bool
    action: str
    reason: str = ""


class QuestionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    blocking: bool = True


class DecisionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: str
    rationale: str


class TaskClassificationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    complexity: Complexity
    profile: str


class ModelProvider:
    """Minimal provider interface; adapters must implement completion."""

    def complete(self, prompt, **kwargs):
        raise NotImplementedError("provider completion is not implemented")


class ModelResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ModelRegistry:
    def __init__(self, config):
        self.config = config
        self.providers = {}
        self.models = config.data.get("models", {}).get("registry", {})
        self.discovered = {}
        for name, data in config.data.get("models", {}).get("providers", {}).items():
            if data.get("enabled"):
                self.providers[name] = OpenAICompatibleProvider(
                    name,
                    data["base_url"],
                    config.data.get("secrets", {}).get(f"{name.upper()}_API_KEY"),
                    model=data.get("model")
                    or config.data.get("secrets", {}).get(f"{name.upper()}_MODEL"),
                    timeout=data.get("timeout", 120),
                    retry=data.get("retry"),
                    kind=data.get("kind", "openai_compatible"),
                )

    def register(self, name: str, provider: ModelProvider):
        self.providers[name] = provider
        self.discovered.pop(name, None)

    def get(self, name: str):
        if name not in self.providers:
            raise ValueError(f"unknown or disabled provider: {name}")
        return self.providers[name]

    def discover(self, name: str):
        provider = self.get(name)
        self.discovered[name] = ()
        if not hasattr(provider, "models"):
            return ()
        inventory = provider.models()
        if not isinstance(inventory, (list, tuple)):
            raise ValueError("invalid model inventory")  # noqa: TRY004
        names = set()
        for item in inventory:
            if item is None or item == "":
                continue
            if not isinstance(item, str) or any(char in item for char in "\r\n\t"):
                raise ValueError("invalid model inventory")
            model_id = item.strip()
            if model_id:
                names.add(model_id)
        snapshot = tuple(sorted(names))
        self.discovered[name] = snapshot
        return snapshot

    def resolve(self, name, needs_tools=False, required_capabilities=()):
        definition = self.models.get(name)
        if definition:
            required = set(required_capabilities)
            if needs_tools:
                required.add("tools")
            if not required <= set(definition.get("capabilities", [])):
                missing = ", ".join(
                    sorted(required - set(definition.get("capabilities", [])))
                )
                raise ValueError(
                    f"model {name} does not support required capabilities: {missing}"
                )
            model = definition.get("model")
            if not model:
                raise ValueError(f"model {name} has no model ID")
            return self.get(definition["provider"]), model
        provider = self.get(name)
        if getattr(provider, "model", None):
            return provider, None
        discovered = self.discovered.get(name)
        if discovered is None:
            discovered = self.discover(name)
        if len(discovered) > 1:
            raise ValueError("multiple discovered models need explicit configuration")
        if discovered and needs_tools:
            raise ValueError("discovered model capabilities are unknown")
        return provider, discovered[0] if discovered else None


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
        primary = profiles.get(name, {}).get("model", {}).get("primary")
        if not isinstance(primary, str) or not primary.strip():
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

    def _candidate_available(self, name, required_capabilities, check_availability):
        definition = self.registry.models.get(name)
        provider_name = definition.get("provider") if definition else name
        provider = self.registry.get(provider_name)
        if definition and not set(required_capabilities) <= set(
            definition.get("capabilities", [])
        ):
            return False
        if not check_availability:
            return True
        health_probe = getattr(provider, "health_report", None)
        report = health_probe() if callable(health_probe) else None
        model = (
            definition.get("model") if definition else getattr(provider, "model", None)
        )
        if report is not None:
            if not report.reachable or not report.api_available:
                return False
            discovered = tuple(report.models)
            if report.kind == "lmstudio":
                if report.loaded_models is None:
                    return False
                discovered = tuple(sorted(set(discovered) & set(report.loaded_models)))
            if model:
                if model not in discovered:
                    return False
            elif discovered:
                self.registry.discovered[provider_name] = discovered
            else:
                return False
        return True

    def _tier_preferences(self, profile, complexity):
        if complexity is None:
            return None
        policy = self.config.data.get("models", {}).get("routing", {})
        key = TaskComplexity(complexity).value.lower()
        overrides = policy.get("profile_tier_preferences", {}).get(profile, {})
        return overrides.get(key) or policy.get("tier_preferences", {}).get(key)

    def candidates(
        self,
        profile,
        complexity=None,
        required_capabilities=(),
        check_availability=True,
    ):
        selected = ProfileRegistry(self.config).get(profile)
        data = self.config.data["profiles"][profile]["model"]
        configured = [selected.model, *data.get("fallback", [])]
        models = self.registry.models
        configured_providers = self.config.data.get("models", {}).get("providers", {})
        for name in configured:
            if (
                name not in models
                and name not in configured_providers
                and name not in self.registry.providers
            ):
                raise ValueError(f"unknown model strategy candidate: {name}")
        eligible = []
        for index, name in enumerate(configured):
            try:
                if self._candidate_available(
                    name, required_capabilities, check_availability
                ):
                    eligible.append((index, name))
            except (OSError, RuntimeError, ValueError, TypeError, httpx.HTTPError):
                continue
        preferences = self._tier_preferences(profile, complexity)
        if preferences:
            rank = {tier: index for index, tier in enumerate(preferences)}
            eligible.sort(
                key=lambda item: (
                    rank.get(models.get(item[1], {}).get("tier"), len(rank)),
                    item[0],
                )
            )
        return [name for _, name in eligible]

    def select(
        self,
        profile: str,
        complexity: Complexity | None = None,
        required_capabilities=(),
    ):
        candidates = self.candidates(profile, complexity, required_capabilities)
        if not candidates:
            raise ValueError("no available model satisfies the profile routing policy")
        return self.registry.resolve(
            candidates[0], required_capabilities=required_capabilities
        )[0]

    def complete(
        self, profile, prompt, tools=None, complexity=None, required_capabilities=()
    ):
        required = set(required_capabilities)
        if tools:
            required.add("tools")
        candidates = self.candidates(
            profile,
            complexity,
            required,
            check_availability=complexity is not None,
        )
        if not candidates:
            raise RuntimeError(
                "no available model satisfies the profile routing policy"
            )
        profile_model = ProfileRegistry(self.config).get(profile).model
        profile_order = [
            profile_model,
            *self.config.data["profiles"][profile]["model"].get("fallback", []),
        ]
        failures = []
        routing = self.config.data.get("models", {}).get("routing", {})
        fallback = routing.get("fallback", {})
        max_attempts = fallback.get("max_attempts", len(candidates))
        max_attempts = min(max_attempts, len(candidates))
        base_delay = fallback.get("base_delay", 0)
        max_delay = fallback.get("max_delay", base_delay)
        budget = RetryBudget.start(max_attempts, fallback.get("max_elapsed", 300))
        for attempt, name in enumerate(candidates[:max_attempts]):
            if not budget.claim():
                break
            fallback_index = profile_order.index(name)
            control = current_run_control()
            if control is not None:
                control.check()
            run_context = {}
            prior_usage = None
            if self.audit:
                from .audit import CURRENT_RUN

                run_context = CURRENT_RUN.get() or {}
                task_id = run_context.get("task_id")
                if task_id is not None and (
                    run_context.get("model_cost_budget") is not None
                    or run_context.get("model_token_budget") is not None
                ):
                    usage = self.audit.model_budget_usage(task_id)
                    prior_usage = usage
                    cost_budget = run_context.get("model_cost_budget")
                    token_budget = run_context.get("model_token_budget")
                    if cost_budget is not None and (
                        usage["missing_cost"] or (usage["cost"] or 0) >= cost_budget
                    ):
                        raise ModelBudgetExceeded(
                            "task model cost budget exhausted or unverifiable"
                        )
                    if token_budget is not None and usage["missing_tokens"]:
                        raise ModelBudgetExceeded(
                            "task model token usage is unverifiable"
                        )
                    tokens = (usage["prompt_tokens"] or 0) + (
                        usage["completion_tokens"] or 0
                    )
                    if token_budget is not None and tokens >= token_budget:
                        raise ModelBudgetExceeded("task model token budget exhausted")
            try:
                with (
                    self.audit.model(name, name, fallback_index)
                    if self.audit
                    else nullcontext({}) as span
                ):
                    span["complexity"] = (
                        TaskComplexity(complexity).value
                        if complexity is not None
                        else None
                    )
                    tier = self.registry.models.get(name, {}).get("tier")
                    preferences = self._tier_preferences(profile, complexity)
                    span["routing_reason"] = (
                        f"tier_preference:{tier}"
                        if preferences and tier
                        else "profile_order"
                    )
                    sanitize = getattr(self.audit, "sanitize", None)
                    safe_profile = sanitize(profile) if callable(sanitize) else profile
                    safe_name = sanitize(name) if callable(sanitize) else name
                    logging.getLogger("harness").info(
                        "model.route profile=%s complexity=%s model=%s reason=%s",
                        safe_profile,
                        span["complexity"] or "unspecified",
                        safe_name,
                        span["routing_reason"],
                    )
                    provider, model = self.registry.resolve(
                        name, bool(tools), required_capabilities=required
                    )
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
                    with use_retry_budget(budget):
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
                        if run_context:
                            cost_budget = run_context.get("model_cost_budget")
                            token_budget = run_context.get("model_token_budget")
                            if cost_budget is not None and (
                                usage.cost is None
                                or (
                                    prior_usage
                                    and (prior_usage["cost"] or 0) + usage.cost
                                    > cost_budget
                                )
                            ):
                                raise ModelBudgetExceeded(
                                    "task model cost budget exceeded or unverifiable"
                                )
                            reported_tokens = (
                                usage.prompt_tokens + usage.completion_tokens
                                if usage.prompt_tokens is not None
                                and usage.completion_tokens is not None
                                else None
                            )
                            if token_budget is not None and (
                                reported_tokens is None
                                or (
                                    prior_usage
                                    and (prior_usage["prompt_tokens"] or 0)
                                    + (prior_usage["completion_tokens"] or 0)
                                    + reported_tokens
                                    > token_budget
                                )
                            ):
                                raise ModelBudgetExceeded(
                                    "task model token budget exceeded or unverifiable"
                                )
                        if control is not None:
                            control.check()
                        self.usage.record(usage)
                        if self.usage_callback:
                            self.usage_callback(usage)
                    else:
                        text, calls = result, []
                        if (
                            run_context.get("model_cost_budget") is not None
                            or run_context.get("model_token_budget") is not None
                        ):
                            raise ModelBudgetExceeded(
                                "task budget requires reported model usage"
                            )
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
            except ModelBudgetExceeded:
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
                if isinstance(exc, ProviderError) and not exc.fallback_allowed:
                    raise
                failures.append(f"{name}: {type(exc).__name__}")
            if attempt + 1 < max_attempts and budget.remaining > 0:
                delay = min(base_delay * (2**attempt), max_delay)
                budget.wait(delay)
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
            + "\nFor every step that may change workspace files, declare the narrowest possible write_paths."
            + "\nFor every task requirement, include its exact text in requirement_ids on at least one step; map every acceptance criterion by its ID in that step's acceptance_criteria."
            + "\nTask:\n"
            + json.dumps(payload)
            + "\nContext:\n"
            + context,
            complexity=task.complexity,
        )
        plan = parse_model_output(PlannerOutput, answer, agent="planner")
        for step in plan.subtasks:
            if not step.expected_result or not step.acceptance_criteria:
                raise ValueError(
                    "plan steps require expected results and acceptance criteria"
                )
            profile = ProfileRegistry(self.router.config).get(step.profile)
            if not set(step.required_tools) <= set(profile.tools):
                raise ValueError("plan requests tools not granted to profile")
        required_requirements = required_criteria = None
        marker = "RECOVERY_SCOPE_JSON:\n"
        if marker in context:
            try:
                scope = json.loads(context.split(marker, 1)[1].splitlines()[0])
                targets = scope["remaining_targets"]
                required_requirements = {
                    target.removeprefix("requirement:")
                    for target in targets
                    if target.startswith("requirement:")
                }
                required_criteria = {
                    target.removeprefix("criterion:")
                    for target in targets
                    if target.startswith("criterion:")
                }
            except (KeyError, IndexError, json.JSONDecodeError, TypeError):
                raise ValueError("recovery plan scope is invalid") from None
        coverage_errors = plan.validate_task_coverage(
            task, required_requirements, required_criteria
        )
        if coverage_errors:
            raise ValueError(
                "plan does not cover task contract: " + "; ".join(coverage_errors)
            )
        return plan


class Executor:
    def __init__(self, router: ModelRouter, tools=None, profiles=None):
        self.router = router
        self.tools = tools
        self.profiles = profiles or ProfileRegistry(router.config)

    def execute(
        self,
        subtask: Subtask,
        context: str = "",
        complexity: TaskComplexity | None = None,
    ) -> ExecutorOutput:
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
                    "content": "EXECUTE: Execute only this step. When finished, return JSON matching this schema; do not claim tool evidence or changed paths, which are recorded by the harness:\n"
                    + json.dumps(ExecutorReport.model_json_schema())
                    + "\n"
                    + subtask.model_dump_json()
                    + "\nContext:\n"
                    + context,
                },
            ]
            for _ in range(profile.max_steps):
                response = self.router.complete(
                    subtask.profile,
                    messages,
                    tools=schemas,
                    complexity=complexity,
                )
                if not response.tool_calls:
                    if not set(subtask.required_tools) <= {
                        item["tool"] for item in evidence
                    }:
                        raise ValueError("no tool evidence for required work")
                    report = parse_model_output(
                        ExecutorReport, response.text, agent="executor"
                    )
                    return ExecutorOutput(
                        subtask_id=subtask.id,
                        success=True,
                        output=report.output,
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
                        "filesystem.mkdir",
                    }:
                        changed_paths = [
                            str(
                                call.arguments.get("destination")
                                or call.arguments["path"]
                            )
                        ]
                        if call.name in {"filesystem.move", "filesystem.delete"}:
                            changed_paths.append(str(call.arguments["path"]))
                        for changed_path in changed_paths:
                            if changed_path not in changed:
                                changed.append(changed_path)
                        evidence[-1]["changed_paths"] = changed_paths
                        evidence[-1]["changed_path"] = changed_paths[0]
            raise RuntimeError("profile tool step limit exceeded")
        except TaskCancelled:
            raise
        except (ApprovalRequired, ApprovalDenied):
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
