"""Pydantic schemas for Overmind HTTP requests/responses."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


RunStatus = Literal[
    "queued",
    "running",
    "awaiting_input",
    "succeeded",
    "failed",
    "canceled",
]


class AgentCreate(BaseModel):
    """Request schema for creating an agent.

    Attributes:
        name: Agent display name.
        role: Free-form role/description.
        model: Model identifier string.
        tools_allowed: List of tool names the agent can invoke.
    """

    name: str = Field(min_length=1)
    role: str = Field(min_length=1)
    model: str = Field(min_length=1)
    tools_allowed: list[str] = Field(default_factory=list)


class AgentUpdate(BaseModel):
    """Request schema for updating an agent.

    Attributes:
        name: Optional updated name.
        role: Optional updated role.
        model: Optional updated model identifier.
        tools_allowed: Optional replacement tool allowlist.
        status: Optional status transition.
    """

    name: str | None = None
    role: str | None = None
    model: str | None = None
    tools_allowed: list[str] | None = None
    status: Literal["active", "disabled"] | None = None


class RunCreate(BaseModel):
    """Request schema for creating a run.

    Attributes:
        agent_id: Agent ID that will execute the run.
        task: Task string.
        step_limit: Maximum steps allowed for this run, or 0 for no limit.
    """

    agent_id: str
    task: str
    step_limit: int = Field(default=8, ge=0)


class MemoryStoreRequest(BaseModel):
    """Request schema for storing a memory item.

    Attributes:
        collection: Collection name.
        text: Text to store.
        metadata: Optional metadata payload.
    """

    collection: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemorySearchRequest(BaseModel):
    """Request schema for searching memory.

    Attributes:
        collection: Collection name.
        query: Query string.
        top_k: Max number of results.
    """

    collection: str
    query: str
    top_k: int = Field(default=5, ge=1, le=50)


class ToolCallRequest(BaseModel):
    """Request schema for invoking a tool.

    Attributes:
        tool_name: Tool name to invoke.
        args: Tool arguments.
    """

    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)


class McpLocalServerRequest(BaseModel):
    """Request schema for creating/updating a local MCP server config."""

    id: str = Field(min_length=1)
    command: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True


class RunInputRequest(BaseModel):
    """Request schema for supplying user input to a paused run.

    Attributes:
        message: User-provided message used to resume planning.
    """

    message: str = Field(min_length=1)
