"""Typed, extensible configuration schemas for the runtime."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

Number: TypeAlias = StrictInt | StrictFloat
TimeoutNumber: TypeAlias = Annotated[Number, Field(gt=0, le=300)]


class ExtensibleSettings(BaseModel):
    """Validate known fields while preserving explicitly permitted extensions."""

    model_config = ConfigDict(extra="allow", strict=True)


class HarnessSettings(ExtensibleSettings):
    name: StrictStr = "autonomous-development-harness"
    environment: StrictStr = "development"


class PathSettings(ExtensibleSettings):
    workspace: StrictStr = "./workspace"
    obsidian_vault: StrictStr = "./vault"
    logs: StrictStr = "./logs"
    database: StrictStr = "./data/harness.db"


class DatabaseSettings(ExtensibleSettings):
    type: Literal["sqlite"] = "sqlite"


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


class ObsidianSettings(ExtensibleSettings):
    enabled: StrictBool = True


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


class MemorySettings(ExtensibleSettings):
    obsidian: ObsidianSettings = Field(default_factory=ObsidianSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)


class ProviderSettings(ExtensibleSettings):
    enabled: StrictBool = False
    base_url: StrictStr | None = None
    model: StrictStr | None = None
    timeout: TimeoutValue = 120
    retry: RetrySettings = Field(default_factory=RetrySettings)


class ModelDefaults(ExtensibleSettings):
    provider: StrictStr = "lmstudio"
    model: StrictStr | None = None


class ModelDefinition(ExtensibleSettings):
    provider: StrictStr
    model: StrictStr
    tier: StrictStr | None = None
    capabilities: list[StrictStr] = Field(default_factory=list)


class ModelStrategy(ExtensibleSettings):
    provider: StrictStr | None = None
    model: StrictStr | None = None
    profile: StrictStr | None = None


class ModelRate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    input: Annotated[Number, Field(ge=0)]
    output: Annotated[Number, Field(ge=0)]


class ModelSettings(ExtensibleSettings):
    defaults: ModelDefaults = Field(default_factory=ModelDefaults)
    providers: dict[StrictStr, ProviderSettings] = Field(default_factory=dict)
    registry: dict[StrictStr, ModelDefinition] = Field(default_factory=dict)
    rates: dict[StrictStr, ModelRate] = Field(default_factory=dict)
    strategies: dict[StrictStr, ModelStrategy] = Field(default_factory=dict)


class ProfileModelSettings(ExtensibleSettings):
    primary: StrictStr | None = None
    fallback: list[StrictStr] = Field(default_factory=list)


class ProfileSettings(ExtensibleSettings):
    model: ProfileModelSettings = Field(default_factory=ProfileModelSettings)
    instructions: StrictStr | None = None
    tools: list[StrictStr] = Field(default_factory=list)
    permissions: list[StrictStr] = Field(default_factory=list)
    max_steps: StrictInt = Field(default=20, ge=1)


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
    builtin: Literal["filesystem"] | None = None
    command: list[StrictStr] = Field(default_factory=list)
    allow_tools: list[StrictStr] = Field(default_factory=list)
    read_only: StrictBool = True
    allow_delete: StrictBool = False
    timeout: Annotated[Number, Field(gt=0, le=30)] = 10

    @model_validator(mode="after")
    def valid_source(self):
        if (self.builtin is None) == (not self.command):
            raise ValueError("exactly one MCP server source is required")
        if self.builtin and self.allow_tools:
            raise ValueError("builtin MCP tools are fixed")
        if not self.builtin and not self.allow_tools:
            raise ValueError("external MCP server needs an explicit tool allowlist")
        if self.read_only and self.allow_delete:
            raise ValueError("read-only MCP server cannot delete")
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
