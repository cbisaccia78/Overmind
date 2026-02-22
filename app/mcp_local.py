"""Local stdio MCP client helpers.

Provides minimal support for discovering and calling tools exposed by local
Model Context Protocol (MCP) servers started as subprocesses.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StdioTransport


@dataclass(frozen=True)
class LocalMcpServerConfig:
    """Configuration for a local stdio MCP server."""

    id: str
    command: str
    args: list[str]
    env: dict[str, str]
    enabled: bool = True


@dataclass(frozen=True)
class LocalMcpTool:
    """Discovered MCP tool metadata bound to a local server."""

    local_name: str
    remote_name: str
    server_id: str
    description: str
    input_schema: dict[str, Any]


def parse_local_mcp_servers(raw_json: str | None) -> list[LocalMcpServerConfig]:
    """Parse local MCP server config from app settings JSON."""
    if not isinstance(raw_json, str) or not raw_json:
        return []
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []

    configs: list[LocalMcpServerConfig] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        server_id = str(item.get("id") or "").strip()
        command = str(item.get("command") or "").strip()
        if not server_id or not command:
            continue
        args_raw = item.get("args")
        env_raw = item.get("env")
        args = [str(arg) for arg in args_raw] if isinstance(args_raw, list) else []
        env = (
            {str(k): str(v) for k, v in env_raw.items()}
            if isinstance(env_raw, dict)
            else {}
        )
        enabled = bool(item.get("enabled", True))
        configs.append(
            LocalMcpServerConfig(
                id=server_id,
                command=command,
                args=args,
                env=env,
                enabled=enabled,
            )
        )
    return configs


def dump_local_mcp_servers(configs: list[LocalMcpServerConfig]) -> str:
    """Serialize local MCP server configs to JSON for app settings."""
    payload = [
        {
            "id": cfg.id,
            "command": cfg.command,
            "args": cfg.args,
            "env": cfg.env,
            "enabled": cfg.enabled,
        }
        for cfg in configs
    ]
    return json.dumps(payload)


def discover_tools(
    config: LocalMcpServerConfig,
    timeout_s: int = 8,
) -> list[LocalMcpTool]:
    """Discover tools from a local MCP server."""
    try:
        tools_payload = _run_coro(_discover_tools_async(config, timeout_s=timeout_s))
    except Exception:
        return []

    tools: list[LocalMcpTool] = []
    for item in tools_payload:
        remote_name = str(item.get("name") or "").strip()
        if not remote_name:
            continue
        description = str(item.get("description") or f"MCP tool {remote_name}")
        input_schema = item.get("inputSchema")
        if not isinstance(input_schema, dict):
            input_schema = {
                "type": "object",
                "properties": {},
                "additionalProperties": True,
            }
        local_name = f"mcp.{config.id}.{remote_name}"
        tools.append(
            LocalMcpTool(
                local_name=local_name,
                remote_name=remote_name,
                server_id=config.id,
                description=description,
                input_schema=input_schema,
            )
        )
    return tools


def call_tool(
    config: LocalMcpServerConfig,
    remote_name: str,
    args: dict[str, Any],
    timeout_s: int = 20,
) -> dict[str, Any]:
    """Invoke one MCP tool on a local server."""
    try:
        result = _run_coro(
            _call_tool_async(
                config,
                remote_name=remote_name,
                args=args,
                timeout_s=timeout_s,
            )
        )
    except Exception as exc:
        return {
            "ok": False,
            "error": {
                "code": "mcp_transport_error",
                "message": str(exc),
            },
        }

    is_error = bool(result.get("isError"))
    return {
        "ok": not is_error,
        "mcp": {
            "server_id": config.id,
            "tool_name": remote_name,
            "result": result,
        },
        **(
            {
                "error": {
                    "code": "mcp_tool_error",
                    "message": f"MCP tool '{remote_name}' returned isError",
                }
            }
            if is_error
            else {}
        ),
    }


def _run_coro(coro: Any) -> Any:
    """Run an async coroutine from synchronous contexts."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001
            error["exc"] = exc

    import threading

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if "exc" in error:
        raise error["exc"]
    return result.get("value")


def _build_client(config: LocalMcpServerConfig, timeout_s: int) -> Client:
    transport = StdioTransport(
        command=config.command,
        args=config.args,
        env={**os.environ, **config.env},
    )
    return Client(transport=transport, timeout=timeout_s)


async def _discover_tools_async(
    config: LocalMcpServerConfig,
    timeout_s: int,
) -> list[dict[str, Any]]:
    client = _build_client(config, timeout_s=timeout_s)
    async with client:
        tools = await client.list_tools()
    return [tool.model_dump(by_alias=True, exclude_none=True) for tool in tools]


async def _call_tool_async(
    config: LocalMcpServerConfig,
    *,
    remote_name: str,
    args: dict[str, Any],
    timeout_s: int,
) -> dict[str, Any]:
    client = _build_client(config, timeout_s=timeout_s)
    async with client:
        result = await client.call_tool(
            name=remote_name,
            arguments=args,
            timeout=timeout_s,
            raise_on_error=False,
        )
    if hasattr(result, "model_dump"):
        dumped = result.model_dump(by_alias=True, exclude_none=True)
        if isinstance(dumped, dict):
            return dumped
    if isinstance(result, dict):
        return result
    return {"content": [str(result)], "isError": False}
