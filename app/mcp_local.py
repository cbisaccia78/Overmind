"""Local stdio MCP client helpers.

Provides minimal support for discovering and calling tools exposed by local
Model Context Protocol (MCP) servers started as subprocesses.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
import json
import os
import re
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

    def __init__(
        self,
        config: LocalMcpServerConfig,
        timeout_s: int,
        cwd: str | None = None,
    ):
        self.config = config
        self.timeout_s = timeout_s
        self.cwd = _normalize_cwd(cwd)
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
        self._client = _build_client(
            self.config,
            timeout_s=self.timeout_s,
            cwd=self.cwd,
        )
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
        try:
            return future.result(timeout=max(1, timeout_s + 1))
        except FuturesTimeoutError:
            future.cancel()
            return _run_coro(_discover_tools_async(self.config, timeout_s=timeout_s))

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
        try:
            return future.result(timeout=max(1, timeout_s + 1))
        except FuturesTimeoutError:
            future.cancel()
            return _run_coro(
                _call_tool_async(
                    self.config,
                    remote_name=remote_name,
                    args=args,
                    timeout_s=timeout_s,
                )
            )

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
    session_cwd: str | None = None,
) -> dict[str, Any]:
    """Invoke one MCP tool on a local server."""
    normalized_cwd = _normalize_cwd(session_cwd)
    try:
        if session_key:
            session = _get_or_create_session(
                config=config,
                session_key=session_key,
                timeout_s=timeout_s,
                cwd=normalized_cwd,
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
                    cwd=normalized_cwd,
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
    observation = _build_structured_observation(result, remote_name=remote_name)
    return {
        "ok": not is_error,
        "mcp": {
            "server_id": config.id,
            "tool_name": remote_name,
            "result": result,
        },
        **({"observation": observation} if observation else {}),
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


def _build_structured_observation(
    result: dict[str, Any], *, remote_name: str
) -> dict[str, Any]:
    """Build compact structured observation from MCP result text."""
    merged = _merge_mcp_text(result)
    normalized = _normalize_mcp_text(merged)
    if not normalized:
        return {}

    sections = _extract_markdown_sections(normalized)
    page_url = _extract_page_field(normalized, "page url")
    page_title = _extract_page_field(normalized, "page title")
    console_errors, console_warnings = _extract_console_counts(normalized)
    actions = _extract_action_candidates(normalized)
    text = _compact_observation_text(normalized)

    summary_bits = [f"MCP tool `{remote_name}` completed."]
    if page_title or page_url:
        label = page_title or page_url
        if page_title and page_url:
            label = f"{page_title} ({page_url})"
        summary_bits.append(f"Page: {label}.")
    if console_errors is not None or console_warnings is not None:
        summary_bits.append(
            "Console: "
            f"{console_errors if console_errors is not None else '?'} errors, "
            f"{console_warnings if console_warnings is not None else '?'} warnings."
        )
    if actions:
        shown = ", ".join(actions[:4])
        suffix = "..." if len(actions) > 4 else ""
        summary_bits.append(f"Interactive targets: {shown}{suffix}.")
    elif sections:
        summary_bits.append("Sections: " + ", ".join(sections[:4]) + ".")
    elif text:
        summary_bits.append(text.splitlines()[0][:180])

    observation: dict[str, Any] = {
        "source": "mcp",
        "tool_name": remote_name,
        "summary": " ".join(part for part in summary_bits if part).strip(),
    }
    if page_url:
        observation["page_url"] = page_url
    if page_title:
        observation["page_title"] = page_title
    if console_errors is not None:
        observation["console_errors"] = console_errors
    if console_warnings is not None:
        observation["console_warnings"] = console_warnings
    if sections:
        observation["sections"] = sections[:8]
    if actions:
        observation["action_candidates"] = actions[:12]
    if text:
        observation["text"] = text
    return observation


def _merge_mcp_text(result: dict[str, Any]) -> str:
    """Join content texts into one string for parsing."""
    texts = _extract_mcp_content_texts(result)
    if not texts:
        return ""
    merged = "\n".join(text.strip() for text in texts if str(text).strip()).strip()
    if not merged:
        return ""
    return _strip_ansi(merged)


def _normalize_mcp_text(text: str) -> str:
    """Normalize wrapped MCP text payloads into readable text."""
    value = str(text or "").strip()
    if not value:
        return ""

    wrapped = _extract_textcontent_body(value)
    if wrapped:
        value = wrapped

    return _unescape_common_sequences(value).strip()


def _extract_textcontent_body(value: str) -> str:
    """Extract `text=` payload from wrapped `TextContent(...)` strings."""
    text = str(value or "")
    marker = "TextContent("
    start = text.find(marker)
    if start < 0:
        return ""
    text_idx = text.find("text=", start)
    if text_idx < 0:
        return ""

    quote_idx = text_idx + len("text=")
    if quote_idx >= len(text):
        return ""
    quote = text[quote_idx]
    if quote not in {"'", '"'}:
        return ""

    chars: list[str] = []
    escaped = False
    for ch in text[quote_idx + 1 :]:
        if escaped:
            chars.append(ch)
            escaped = False
            continue
        if ch == "\\":
            chars.append(ch)
            escaped = True
            continue
        if ch == quote:
            return "".join(chars)
        chars.append(ch)
    return ""


def _unescape_common_sequences(value: str) -> str:
    """Decode common escaped sequences without full eval."""
    decoded = str(value or "")
    for old, new in [
        ("\\r\\n", "\n"),
        ("\\n", "\n"),
        ("\\t", "\t"),
        ("\\r", "\n"),
        ("\\'", "'"),
        ('\\"', '"'),
        ("\\\\", "\\"),
    ]:
        decoded = decoded.replace(old, new)
    return decoded


def _extract_markdown_sections(text: str) -> list[str]:
    """Extract markdown section headings (### Heading)."""
    sections: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped.startswith("###"):
            continue
        heading = stripped.lstrip("#").strip()
        if heading:
            sections.append(heading)
    return _unique_preserve_order(sections)


def _extract_page_field(text: str, field: str) -> str:
    """Extract a page metadata field from markdown bullet lines."""
    wanted = field.lower().strip()
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("- ") or ":" not in line:
            continue
        key, value = line[2:].split(":", 1)
        if key.strip().lower() == wanted:
            return value.strip()
    return ""


def _extract_console_counts(text: str) -> tuple[int | None, int | None]:
    """Extract console error/warning counts when present."""
    console_line = _extract_page_field(text, "console")
    if not console_line:
        return None, None
    errors_match = re.search(r"(\d+)\s+errors?", console_line, flags=re.IGNORECASE)
    warnings_match = re.search(
        r"(\d+)\s+warnings?", console_line, flags=re.IGNORECASE
    )
    errors = int(errors_match.group(1)) if errors_match else None
    warnings = int(warnings_match.group(1)) if warnings_match else None
    return errors, warnings


def _extract_action_candidates(text: str) -> list[str]:
    """Extract likely interactive labels from snapshot text."""
    labels: list[str] = []
    pattern = re.compile(
        r'\b(?:button|link|textbox|tab|menuitem)\s+"([^"]+)"', flags=re.IGNORECASE
    )
    for match in pattern.finditer(text):
        label = match.group(1).strip()
        if not label:
            continue
        if len(label) > 80:
            continue
        labels.append(label)
    return _unique_preserve_order(labels)


def _compact_observation_text(text: str, max_chars: int = 1200) -> str:
    """Compact verbose snapshot output into short, high-signal text."""
    lines: list[str] = []
    in_code_block = False
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        lines.append(stripped)
        if len(lines) >= 40:
            break

    compact = "\n".join(lines).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars] + "..."


def _unique_preserve_order(values: list[str]) -> list[str]:
    """Return unique values while preserving first-seen order."""
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


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
    cwd: str | None = None,
) -> _PersistentMcpSession:
    cache_key = (config.id, session_key)
    with _SESSIONS_LOCK:
        session = _SESSIONS.get(cache_key)
    normalized_cwd = _normalize_cwd(cwd)
    if session is not None:
        if (str(getattr(session, "cwd", "")).strip() or None) == normalized_cwd:
            return session
        with _SESSIONS_LOCK:
            _SESSIONS.pop(cache_key, None)
        session.close()

    created = _PersistentMcpSession(
        config=config,
        timeout_s=timeout_s,
        cwd=normalized_cwd,
    )
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


def _normalize_cwd(cwd: str | None) -> str | None:
    """Normalize optional cwd to a non-empty string or None."""
    if cwd is None:
        return None
    value = str(cwd).strip()
    return value or None


def _build_client(
    config: LocalMcpServerConfig,
    timeout_s: int,
    cwd: str | None = None,
) -> Client:
    transport = StdioTransport(
        command=config.command,
        args=config.args,
        env={**os.environ, **config.env},
        cwd=cwd,
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
    cwd: str | None = None,
) -> dict[str, Any]:
    client = _build_client(config, timeout_s=timeout_s, cwd=cwd)
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
