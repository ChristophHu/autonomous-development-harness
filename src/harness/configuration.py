"""Typed, extensible configuration schemas for the runtime."""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal, TypeAlias
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

Number: TypeAlias = StrictInt | StrictFloat
TimeoutNumber: TypeAlias = Annotated[Number, Field(gt=0, le=300)]


class ExtensibleSettings(BaseModel):
    """Reject undeclared settings; keep custom values in the named extension map."""

    model_config = ConfigDict(extra="forbid", strict=True)
    extensions: dict[StrictStr, Any] = Field(default_factory=dict)


class HarnessSettings(ExtensibleSettings):
    name: StrictStr = "autonomous-development-harness"
    environment: StrictStr = "development"
    max_parallel_steps: StrictInt = Field(default=4, ge=1, le=32)
    max_correction_attempts: StrictInt = Field(default=2, ge=0, le=10)


class PathSettings(ExtensibleSettings):
    workspace: StrictStr = "./workspace"
    obsidian_vault: StrictStr = "./vault"
    logs: StrictStr = "./logs"
    database: StrictStr = "./data/harness.db"


class DatabaseSettings(ExtensibleSettings):
    type: Literal["sqlite"] = "sqlite"


class VerificationSettings(ExtensibleSettings):
    required_evidence: list[
        Literal["ci", "provider", "qdrant", "embedding", "http_tls"]
    ] = Field(default_factory=list)
    max_age_hours: StrictInt = Field(default=168, ge=1, le=8760)


class TimeoutSettings(ExtensibleSettings):
    model_config = ConfigDict(extra="forbid", strict=True)

    total: TimeoutNumber
    connect: TimeoutNumber | None = None
    read: TimeoutNumber | None = None


TimeoutValue: TypeAlias = TimeoutNumber | TimeoutSettings


class RetrySettings(ExtensibleSettings):
    model_config = ConfigDict(extra="forbid", strict=True)

    max_attempts: StrictInt = Field(default=3, ge=1, le=5)
    base_delay: Number = Field(default=0.25, ge=0, le=60)
    max_delay: Number = Field(default=4.0, ge=0, le=60)
    max_elapsed: Number = Field(default=300, gt=0, le=3600)


class ObsidianSettings(ExtensibleSettings):
    enabled: StrictBool = True


class MemoryMonitoringSettings(ExtensibleSettings):
    enabled: StrictBool = False
    interval_seconds: StrictInt = Field(default=60, ge=5, le=3600)
    evidence_max_age_hours: StrictInt = Field(default=168, ge=1, le=8760)


class QdrantSettings(ExtensibleSettings):
    enabled: StrictBool = False
    url: StrictStr = "http://127.0.0.1:6333"
    collection: StrictStr = "harness-memory"
    timeout: TimeoutValue = 5


class EmbeddingSettings(ExtensibleSettings):
    provider: StrictStr = "lmstudio"
    base_url: StrictStr = "http://127.0.0.1:1234/v1"
    model: StrictStr = "text-embedding-model"
    dimensions: StrictInt = Field(default=1024, ge=1)
    batch_size: StrictInt = Field(default=32, ge=1, le=256)
    timeout: TimeoutValue = 30


class MemoryContextSettings(ExtensibleSettings):
    max_bytes: StrictInt = Field(default=65536, ge=256, le=1048576)


class MemorySettings(ExtensibleSettings):
    obsidian: ObsidianSettings = Field(default_factory=ObsidianSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    context: MemoryContextSettings = Field(default_factory=MemoryContextSettings)
    monitoring: MemoryMonitoringSettings = Field(
        default_factory=MemoryMonitoringSettings
    )


class ProviderSettings(ExtensibleSettings):
    enabled: StrictBool = False
    kind: Literal["openai_compatible", "lmstudio"] = "openai_compatible"
    base_url: StrictStr | None = None
    model: StrictStr | None = None
    headers: dict[StrictStr, StrictStr] = Field(default_factory=dict)
    timeout: TimeoutValue = 120
    retry: RetrySettings = Field(default_factory=RetrySettings)

    @field_validator("headers")
    @classmethod
    def validate_provider_headers(cls, value, info):
        allowed = {"http-referer", "x-title"}
        for name, header_value in value.items():
            if name.lower() not in allowed:
                raise ValueError("provider header is not in the safe allowlist")
            if not header_value.strip() or any(c in header_value for c in "\r\n"):
                raise ValueError(
                    "provider header values must be non-empty and single-line"
                )
        if info.data.get("kind") == "lmstudio" and value:
            raise ValueError("custom provider headers are not supported for LM Studio")
        return value


class ModelDefaults(ExtensibleSettings):
    provider: StrictStr = "lmstudio"
    model: StrictStr | None = None


ModelTier = Literal["premium", "advanced", "standard", "economical", "local"]
AgentCapability = Literal[
    "plan", "requirements", "execute", "test", "review", "recovery", "architecture"
]


class ModelDefinition(ExtensibleSettings):
    provider: StrictStr
    model: StrictStr
    tier: ModelTier | None = None
    capabilities: list[StrictStr] = Field(default_factory=list)


class ModelStrategy(ExtensibleSettings):
    provider: StrictStr | None = None
    model: StrictStr | None = None
    profile: StrictStr | None = None


class ModelRate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    input: Annotated[Number, Field(ge=0)]
    output: Annotated[Number, Field(ge=0)]


class ModelInputTokenBudget(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    encoding: Annotated[StrictStr, Field(min_length=1)] | None = None
    tokenizer_file: Annotated[StrictStr, Field(min_length=1)] | None = None
    characters_per_token: Number | None = Field(default=None, gt=0)
    max_input_tokens: StrictInt = Field(ge=1)
    framing_tokens: StrictInt = Field(default=0, ge=0)
    safety_margin_percent: StrictInt = Field(default=20, ge=0, le=100)

    @model_validator(mode="after")
    def exactly_one_tokenizer(self):
        configured_methods = sum(
            value is not None
            for value in (
                self.encoding,
                self.tokenizer_file,
                self.characters_per_token,
            )
        )
        if configured_methods != 1:
            raise ValueError(
                "configure exactly one tokenizer encoding, tokenizer_file, or characters_per_token"
            )
        if self.tokenizer_file is not None and "\x00" in self.tokenizer_file:
            raise ValueError("tokenizer_file contains an invalid path")
        return self


class ModelRoutingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    tier_preferences: dict[
        Literal["low", "medium", "high", "critical"], list[ModelTier]
    ] = Field(default_factory=dict)
    profile_tier_preferences: dict[
        StrictStr, dict[Literal["low", "medium", "high", "critical"], list[ModelTier]]
    ] = Field(default_factory=dict)
    fallback: RetrySettings = Field(default_factory=RetrySettings)

    @staticmethod
    def _validate_orders(preferences):
        tiers = {"premium", "advanced", "standard", "economical", "local"}
        for order in preferences.values():
            if len(order) != len(set(order)) or set(order) != tiers:
                raise ValueError("tier preference must order every tier exactly once")

    @model_validator(mode="after")
    def valid_preferences(self):
        complexities = {"low", "medium", "high", "critical"}
        if self.tier_preferences and set(self.tier_preferences) != complexities:
            raise ValueError("global tier preferences must define every complexity")
        self._validate_orders(self.tier_preferences)
        for preferences in self.profile_tier_preferences.values():
            self._validate_orders(preferences)
        return self


class ModelSettings(ExtensibleSettings):
    defaults: ModelDefaults = Field(default_factory=ModelDefaults)
    providers: dict[StrictStr, ProviderSettings] = Field(default_factory=dict)
    registry: dict[StrictStr, ModelDefinition] = Field(default_factory=dict)
    rates: dict[StrictStr, ModelRate] = Field(default_factory=dict)
    input_token_budgets: dict[StrictStr, ModelInputTokenBudget] = Field(
        default_factory=dict
    )
    strategies: dict[StrictStr, ModelStrategy] = Field(default_factory=dict)
    routing: ModelRoutingSettings = Field(default_factory=ModelRoutingSettings)


class ProfileModelSettings(ExtensibleSettings):
    primary: StrictStr | None = None
    fallback: list[StrictStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_candidates(self):
        candidates = [self.primary, *self.fallback]
        if any(
            candidate is not None and not candidate.strip() for candidate in candidates
        ):
            raise ValueError("model strategy candidates must not be blank")
        configured = [candidate for candidate in candidates if candidate is not None]
        if len(configured) != len(set(configured)):
            raise ValueError("model strategy candidates must be unique")
        return self


class ProfileSettings(ExtensibleSettings):
    model: ProfileModelSettings = Field(default_factory=ProfileModelSettings)
    instructions: StrictStr | None = None
    tools: list[StrictStr] = Field(default_factory=list)
    permissions: list[StrictStr] = Field(default_factory=list)
    capabilities: list[AgentCapability] | None = None
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    max_steps: StrictInt = Field(default=20, ge=1)
    contract_mode: Literal["strict", "legacy"] | None = None


class AgentRuntimeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    contract_mode: Literal["optional", "required"] = "optional"


class GitSettings(ExtensibleSettings):
    enabled: StrictBool = False
    main: StrictStr = "main"
    dev: StrictStr = "dev"
    remote: StrictStr | None = None
    branches: dict[StrictStr, StrictStr] = Field(default_factory=dict)
    workflow: dict[StrictStr, StrictStr] = Field(default_factory=dict)


class DockerSettings(ExtensibleSettings):
    enabled: StrictBool | None = None
    compose_preferred: StrictBool | None = None
    compose_file: StrictStr | None = None
    socket_path: StrictStr | None = None


class APISettings(ExtensibleSettings):
    enabled: StrictBool | None = None
    host: StrictStr = "127.0.0.1"
    port: StrictInt = Field(default=8080, ge=1, le=65535)
    swagger: StrictBool = True


class LoggingSettings(ExtensibleSettings):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    file: StrictStr = "./logs/harness.log"
    max_bytes: StrictInt = Field(default=10_485_760, ge=1024, le=104_857_600)
    backup_count: StrictInt = Field(default=3, ge=0, le=10)


class TestingCoverageSettings(ExtensibleSettings):
    statements: Number = Field(default=100, ge=0, le=100)
    branches: Number = Field(default=100, ge=0, le=100)
    functions: Number = Field(default=100, ge=0, le=100)
    lines: Number = Field(default=100, ge=0, le=100)


class TestingSettings(ExtensibleSettings):
    tdd: StrictBool = True
    coverage: TestingCoverageSettings = Field(default_factory=TestingCoverageSettings)


class HTTPToolSettings(ExtensibleSettings):
    allowed_hosts: list[StrictStr] = Field(default_factory=list)
    private_hosts: list[StrictStr] = Field(default_factory=list)
    timeout: TimeoutValue = 30


class GitSSHCredential(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    username: Annotated[StrictStr, Field(pattern=r"^[A-Za-z0-9._-]{1,64}$")]
    fingerprint: Annotated[StrictStr, Field(pattern=r"^SHA256:[A-Za-z0-9+/]{43}$")]


class GitHTTPSCredential(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    mode: Literal["basic", "bearer"] | None = None
    username: StrictStr | None = None
    secret_name: Annotated[StrictStr, Field(pattern=r"^[A-Z0-9_]{1,128}$")]

    @model_validator(mode="after")
    def valid_mode(self):
        if self.mode != "bearer" and (
            not self.username
            or ":" in self.username
            or "\r" in self.username
            or "\n" in self.username
        ):
            raise ValueError("basic Git credential needs a safe username")
        if self.mode == "bearer" and self.username is not None:
            raise ValueError("bearer Git credential must not have a username")
        return self


class GitSSHSettings(ExtensibleSettings):
    allowed_hosts: list[StrictStr] = Field(default_factory=list)
    allowed_ports: list[Annotated[StrictInt, Field(ge=1, le=65535)]] = Field(
        default_factory=lambda: [22]
    )
    host_keys: dict[StrictStr, list[StrictStr]] = Field(default_factory=dict)
    credentials: dict[StrictStr, GitSSHCredential] = Field(default_factory=dict)


class GitToolSettings(ExtensibleSettings):
    allowed_hosts: list[StrictStr] = Field(default_factory=list)
    ca_bundle: StrictStr | None = None
    credentials: dict[StrictStr, GitHTTPSCredential] = Field(default_factory=dict)
    ssh: GitSSHSettings = Field(default_factory=GitSSHSettings)


class DockerToolSettings(ExtensibleSettings):
    compose_file: StrictStr = "./docker-compose.yml"
    socket_path: StrictStr | None = None


class MCPServerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    enabled: StrictBool = True
    transport: Literal["stdio", "streamable_http"] = "stdio"
    builtin: Literal["filesystem", "obsidian", "apple_shell"] | None = None
    command: list[StrictStr] = Field(default_factory=list)
    url: StrictStr | None = None
    allowed_hosts: list[StrictStr] = Field(default_factory=list)
    auth_secret: StrictStr | None = None
    allow_tools: list[StrictStr] = Field(default_factory=list)
    read_only: StrictBool = True
    allow_delete: StrictBool = False
    trusted_local: StrictBool = False
    read_roots: list[StrictStr] = Field(default_factory=list)
    timeout: Annotated[Number, Field(gt=0, le=30)] = 10

    @model_validator(mode="after")
    def valid_source(self):
        if self.transport == "stdio":
            if (self.builtin is None) == (not self.command):
                raise ValueError("exactly one MCP server source is required")
            if (
                self.url is not None
                or self.allowed_hosts
                or self.auth_secret is not None
            ):
                raise ValueError("stdio MCP server cannot declare remote settings")
        else:
            if (
                self.builtin is not None
                or self.command
                or self.url is None
                or not self.allowed_hosts
            ):
                raise ValueError("remote MCP requires URL and host allowlist only")
            parsed = urlsplit(self.url)
            try:
                port = parsed.port
            except ValueError:
                raise ValueError("remote MCP URL is invalid") from None
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
                or (port is not None and not 1 <= port <= 65535)
                or parsed.hostname.casefold()
                not in {host.casefold() for host in self.allowed_hosts}
                or len({host.casefold() for host in self.allowed_hosts})
                != len(self.allowed_hosts)
            ):
                raise ValueError("remote MCP URL must be HTTPS and host-allowlisted")
            if self.auth_secret is not None and not re.fullmatch(
                r"[A-Z0-9_]{1,128}", self.auth_secret
            ):
                raise ValueError("remote MCP auth_secret must be a secret reference")
            if self.allow_delete or not self.read_only:
                raise ValueError("remote MCP tools cannot bypass approval policy")
        if self.builtin and self.allow_tools:
            raise ValueError("builtin MCP tools are fixed")
        if not self.builtin and not self.allow_tools:
            raise ValueError("external MCP server needs an explicit tool allowlist")
        if (
            self.builtin is None
            and self.transport == "stdio"
            and not self.trusted_local
        ):
            raise ValueError(
                "external MCP server needs explicit trusted_local acknowledgement"
            )
        if self.builtin is None and self.transport == "stdio" and not self.read_roots:
            raise ValueError("external stdio MCP server requires explicit read_roots")
        if any(not root.strip() for root in self.read_roots):
            raise ValueError("MCP read_roots must contain non-empty paths")
        if self.transport == "streamable_http" and self.read_roots:
            raise ValueError("remote MCP servers cannot declare filesystem read_roots")
        if self.builtin and self.read_roots:
            raise ValueError("builtin MCP read_roots are controlled by the harness")
        if self.transport == "streamable_http" and self.trusted_local:
            raise ValueError("remote MCP server cannot use trusted_local")
        if self.builtin and self.trusted_local:
            raise ValueError("builtin MCP server must not use trusted_local")
        if self.read_only and self.allow_delete:
            raise ValueError("read-only MCP server cannot delete")
        if self.builtin == "obsidian" and not self.read_only:
            raise ValueError("Obsidian MCP server is read-only")
        return self


class MCPSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    servers: dict[StrictStr, MCPServerSettings] = Field(default_factory=dict)


class ToolsSettings(ExtensibleSettings):
    permissions: dict[StrictStr, StrictStr] = Field(default_factory=dict)
    http: HTTPToolSettings = Field(default_factory=HTTPToolSettings)
    docker: DockerToolSettings = Field(default_factory=DockerToolSettings)
    git: GitToolSettings = Field(default_factory=GitToolSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)


class HarnessConfig(ExtensibleSettings):
    harness: HarnessSettings = Field(default_factory=HarnessSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    verification: VerificationSettings = Field(default_factory=VerificationSettings)
    agents: AgentRuntimeSettings = Field(default_factory=AgentRuntimeSettings)
    agent_roles: dict[StrictStr, StrictStr] = Field(default_factory=dict)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    git: GitSettings = Field(default_factory=GitSettings)
    docker: DockerSettings = Field(default_factory=DockerSettings)
    api: APISettings = Field(default_factory=APISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    profiles: dict[StrictStr, ProfileSettings] = Field(default_factory=dict)
    testing: TestingSettings = Field(default_factory=TestingSettings)
    tools: ToolsSettings = Field(default_factory=ToolsSettings)
    secrets: dict[StrictStr, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_profile_routing(self):
        unknown = set(self.models.routing.profile_tier_preferences) - set(self.profiles)
        if unknown:
            raise ValueError("routing policy references an unknown profile")
        return self
