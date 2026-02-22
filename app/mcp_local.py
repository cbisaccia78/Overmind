"""Local stdio MCP client helpers.

Provides minimal support for discovering and calling tools exposed by local
Model Context Protocol (MCP) servers started as subprocesses.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
import threading
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


class _PersistentMcpSession:
    """Thread-hosted persistent MCP client session."""

    def __init__(self, config: LocalMcpServerConfig, timeout_s: int):
        self.config = config
        self.timeout_s = timeout_s
        self.loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._startup_error: Exception | None = None
        self._client: Client | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=max(1, timeout_s + 2))
        if self._startup_error is not None:
            raise self._startup_error
        if not self._ready.is_set() or self._client is None:
            raise RuntimeError("failed to start persistent MCP session")

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self._client = _build_client(self.config, timeout_s=self.timeout_s)
        try:
            self.loop.run_until_complete(self._client.__aenter__())
        except Exception as exc:  # noqa: BLE001
            self._startup_error = exc
            self._ready.set()
            return

        self._ready.set()
        self.loop.run_forever()

        try:
            self.loop.run_until_complete(self._client.__aexit__(None, None, None))
        finally:
            self.loop.close()
            self._closed.set()

    def list_tools(self, timeout_s: int) -> list[dict[str, Any]]:
        if self._client is None:
            raise RuntimeError("persistent session not initialized")

        async def _list() -> list[dict[str, Any]]:
            tools = await self._client.list_tools()
            return [tool.model_dump(by_alias=True, exclude_none=True) for tool in tools]

        future = asyncio.run_coroutine_threadsafe(_list(), self.loop)
        return future.result(timeout=max(1, timeout_s + 1))

    def call_tool(
        self,
        *,
        remote_name: str,
        args: dict[str, Any],
        timeout_s: int,
    ) -> dict[str, Any]:
        if self._client is None:
            raise RuntimeError("persistent session not initialized")

        async def _call() -> dict[str, Any]:
            result = await self._client.call_tool(
                name=remote_name,
                arguments=args,
                timeout=timeout_s,
                raise_on_error=False,
            )
            return _normalize_tool_result(result)

        future = asyncio.run_coroutine_threadsafe(_call(), self.loop)
        return future.result(timeout=max(1, timeout_s + 1))

    def close(self) -> None:
        if self._closed.is_set():
            return
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=max(1, self.timeout_s + 2))
        self._closed.set()


_SESSIONS_LOCK = threading.Lock()
_SESSIONS: dict[tuple[str, str], _PersistentMcpSession] = {}


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
    session_key: str | None = None,
) -> dict[str, Any]:
    """Invoke one MCP tool on a local server."""
    try:
        if session_key:
            session = _get_or_create_session(
                config=config,
                session_key=session_key,
                timeout_s=timeout_s,
            )
            result = session.call_tool(
                remote_name=remote_name,
                args=args,
                timeout_s=timeout_s,
            )
        else:
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

    is_error = _is_mcp_error_result(result)
    error_message = _extract_mcp_error_message(result)
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
                    "message": error_message
                    or f"MCP tool '{remote_name}' returned an error",
                }
            }
            if is_error
            else {}
        ),
    }


def _is_mcp_error_result(result: dict[str, Any]) -> bool:
    """Determine whether an MCP tool result represents an error."""
    if bool(result.get("isError")) or bool(result.get("is_error")):
        return True
    for text in _extract_mcp_content_texts(result):
        lowered = _strip_ansi(text).lower()
        if "### error" in lowered:
            return True
        if "is_error=true" in lowered:
            return True
    return False


def _extract_mcp_error_message(result: dict[str, Any]) -> str:
    """Extract a user-facing error message from MCP tool result content."""
    for text in _extract_mcp_content_texts(result):
        cleaned = _strip_ansi(text).strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if "### error" in lowered or "is_error=true" in lowered:
            return cleaned
        if lowered.startswith("error:"):
            return cleaned
    return ""


def _extract_mcp_content_texts(result: dict[str, Any]) -> list[str]:
    """Return normalized text fragments from MCP result content."""
    payload = result.get("content")
    if not isinstance(payload, list):
        return []
    texts: list[str] = []
    for item in payload:
        if isinstance(item, str):
            texts.append(item)
            continue
        if isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str):
                texts.append(text)
            continue
        texts.append(str(item))
    return texts


def _strip_ansi(value: str) -> str:
    """Remove ANSI escape sequences from a text string."""
    return "".join(ch for ch in value if ch == "\n" or ch == "\t" or ord(ch) >= 32)


def close_sessions(session_key: str) -> int:
    """Close all persistent sessions associated with one session key."""
    keys_to_close: list[tuple[str, str]] = []
    with _SESSIONS_LOCK:
        for key in list(_SESSIONS.keys()):
            if key[1] == session_key:
                keys_to_close.append(key)

    closed = 0
    for key in keys_to_close:
        session: _PersistentMcpSession | None = None
        with _SESSIONS_LOCK:
            session = _SESSIONS.pop(key, None)
        if session is None:
            continue
        session.close()
        closed += 1
    return closed


def _get_or_create_session(
    *,
    config: LocalMcpServerConfig,
    session_key: str,
    timeout_s: int,
) -> _PersistentMcpSession:
    cache_key = (config.id, session_key)
    with _SESSIONS_LOCK:
        session = _SESSIONS.get(cache_key)
    if session is not None:
        return session

    created = _PersistentMcpSession(config=config, timeout_s=timeout_s)
    with _SESSIONS_LOCK:
        existing = _SESSIONS.get(cache_key)
        if existing is not None:
            created.close()
            return existing
        _SESSIONS[cache_key] = created
    return created


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
    return _normalize_tool_result(result)


def _normalize_tool_result(result: Any) -> dict[str, Any]:
    if hasattr(result, "model_dump"):
        dumped = result.model_dump(by_alias=True, exclude_none=True)
        if isinstance(dumped, dict):
            return dumped
    if isinstance(result, dict):
        return result
    return {"content": [str(result)], "isError": False}
