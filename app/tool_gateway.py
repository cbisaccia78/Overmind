"""Tool execution gateway.

Enforces an agent allowlist, validates tool arguments against per-tool schemas,
dispatches supported tools (files, memory, host shell), and records tool call
audit events.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from pathlib import Path
from typing import Any, Callable

from .mcp_local import (
    LocalMcpServerConfig,
    call_tool as call_local_mcp_tool,
    discover_tools as discover_local_mcp_tools,
    parse_local_mcp_servers,
)
from .memory import LocalVectorMemory
from .repository import Repository
from .shell_runner import ShellRunner


@dataclass(frozen=True)
class ToolSpec:
    """Registry entry for a supported tool."""

    name: str
    args_schema: dict[str, dict[str, Any]] | None
    handler: Callable[[dict[str, Any]], dict[str, Any]]
    json_schema: dict[str, Any] | None = None
    validate_args: bool = True


class ToolGateway:
    """Authorize and dispatch tool calls, then persist audit records."""

    _TOOL_DESCRIPTIONS: dict[str, str] = {
        "run_shell": "Run a shell command on the host OS within the workspace.",
        "read_file": "Read a UTF-8 text file from the workspace.",
        "write_file": "Write UTF-8 text content to a file in the workspace.",
        "store_memory": "Store a memory item in a named collection.",
        "search_memory": "Search memory items in a named collection.",
        "final_answer": "Finish the run by returning a final answer message (no side effects).",
    }
    _MCP_SETTINGS_KEY = "mcp_local_servers"

    def __init__(
        self,
        repo: Repository,
        memory: LocalVectorMemory,
        shell_runner: ShellRunner,
        workspace_root: str,
        max_file_bytes: int = 100_000,
    ):
        """Create a tool gateway for dispatching and auditing tool calls.

        Args:
            repo: Repository used to persist tool call audit records and events.
            memory: Memory subsystem used by memory tools.
            shell_runner: Host shell runner used by `run_shell`.
            workspace_root: Root directory used to constrain file access.
            max_file_bytes: Maximum bytes for read/write operations.
        """
        self.repo = repo
        self.memory = memory
        self.shell_runner = shell_runner
        self.workspace_root = Path(workspace_root).resolve()
        self.max_file_bytes = max_file_bytes
        self._tool_registry: dict[str, ToolSpec] = self._build_registry()
        self._refresh_mcp_registry()

    def _build_registry(self) -> dict[str, ToolSpec]:
        """Create the registry of supported tools and schemas."""
        return {
            "run_shell": ToolSpec(
                name="run_shell",
                args_schema={
                    "command": {"type": str, "required": True, "min_length": 1},
                    "timeout_s": {
                        "type": int,
                        "required": False,
                        "default": 20,
                        "min": 1,
                        "max": 120,
                    },
                    "allow_write": {
                        "type": bool,
                        "required": False,
                        "default": False,
                    },
                    "write_subdir": {
                        "type": (str, type(None)),
                        "required": False,
                        "default": None,
                    },
                },
                handler=self._handle_run_shell,
            ),
            "read_file": ToolSpec(
                name="read_file",
                args_schema={
                    "path": {"type": str, "required": True, "min_length": 1},
                },
                handler=self._handle_read_file,
            ),
            "write_file": ToolSpec(
                name="write_file",
                args_schema={
                    "path": {"type": str, "required": True, "min_length": 1},
                    "content": {"type": str, "required": True},
                },
                handler=self._handle_write_file,
            ),
            "store_memory": ToolSpec(
                name="store_memory",
                args_schema={
                    "collection": {
                        "type": str,
                        "required": False,
                        "default": "default",
                        "min_length": 1,
                    },
                    "text": {"type": str, "required": True, "min_length": 1},
                    "metadata": {
                        "type": dict,
                        "required": False,
                        "default": {},
                    },
                },
                handler=self._handle_store_memory,
            ),
            "search_memory": ToolSpec(
                name="search_memory",
                args_schema={
                    "collection": {
                        "type": str,
                        "required": False,
                        "default": "default",
                        "min_length": 1,
                    },
                    "query": {"type": str, "required": True, "min_length": 1},
                    "top_k": {
                        "type": int,
                        "required": False,
                        "default": 5,
                        "min": 1,
                        "max": 50,
                    },
                },
                handler=self._handle_search_memory,
            ),
            "final_answer": ToolSpec(
                name="final_answer",
                args_schema={
                    "message": {"type": str, "required": True, "min_length": 1},
                },
                handler=self._handle_final_answer,
            ),
        }

    def _refresh_mcp_registry(self) -> None:
        """Load enabled local MCP servers from settings and register their tools."""
        for tool_name in [
            name for name in self._tool_registry if name.startswith("mcp.")
        ]:
            self._tool_registry.pop(tool_name, None)
            self._TOOL_DESCRIPTIONS.pop(tool_name, None)

        for config in self.list_local_mcp_servers():
            if not config.enabled:
                continue
            try:
                tools = discover_local_mcp_tools(config)
            except Exception:
                continue
            for tool in tools:
                self._TOOL_DESCRIPTIONS[tool.local_name] = tool.description
                self._tool_registry[tool.local_name] = ToolSpec(
                    name=tool.local_name,
                    args_schema=None,
                    json_schema=tool.input_schema,
                    validate_args=False,
                    handler=self._make_mcp_handler(
                        config=config,
                        remote_tool_name=tool.remote_name,
                    ),
                )

    def _make_mcp_handler(
        self,
        *,
        config: LocalMcpServerConfig,
        remote_tool_name: str,
    ) -> Callable[[dict[str, Any]], dict[str, Any]]:
        def _handler(args: dict[str, Any]) -> dict[str, Any]:
            return call_local_mcp_tool(
                config=config,
                remote_name=remote_tool_name,
                args=args,
            )

        return _handler

    def list_local_mcp_servers(self) -> list[LocalMcpServerConfig]:
        """Return local MCP server configs from app settings."""
        raw = self.repo.get_setting(self._MCP_SETTINGS_KEY)
        return parse_local_mcp_servers(raw)

    def set_local_mcp_servers(self, configs: list[LocalMcpServerConfig]) -> None:
        """Persist local MCP server configs and refresh registered MCP tools."""
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
        self.repo.set_setting(self._MCP_SETTINGS_KEY, json.dumps(payload))
        self._refresh_mcp_registry()

    def upsert_local_mcp_server(self, config: LocalMcpServerConfig) -> None:
        """Create or replace one local MCP server config by id."""
        existing = [cfg for cfg in self.list_local_mcp_servers() if cfg.id != config.id]
        existing.append(config)
        existing.sort(key=lambda cfg: cfg.id)
        self.set_local_mcp_servers(existing)

    def remove_local_mcp_server(self, server_id: str) -> bool:
        """Remove one local MCP server config by id."""
        existing = self.list_local_mcp_servers()
        filtered = [cfg for cfg in existing if cfg.id != server_id]
        if len(filtered) == len(existing):
            return False
        self.set_local_mcp_servers(filtered)
        return True

    def list_tool_names(self) -> list[str]:
        """Return all currently registered tool names."""
        return list(self._tool_registry.keys())

    def list_tool_specs(self, tool_names: list[str] | None = None) -> list[ToolSpec]:
        """List registered tool specs, optionally filtered by name.

        Args:
            tool_names: Optional list of tool names to include.

        Returns:
            List of tool specs in registry order.
        """
        if tool_names is None:
            return list(self._tool_registry.values())

        allowed = set(tool_names)
        return [spec for name, spec in self._tool_registry.items() if name in allowed]

    def list_openai_tools(
        self, tool_names: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Render registered tools as OpenAI function-calling definitions.

        Args:
            tool_names: Optional list of tool names to include.

        Returns:
            List of OpenAI-compatible tool definitions.
        """
        tools: list[dict[str, Any]] = []
        for spec in self.list_tool_specs(tool_names):
            if spec.json_schema is not None:
                parameters = spec.json_schema
            elif spec.args_schema is not None:
                parameters = self._args_schema_to_json_schema(spec.args_schema)
            else:
                parameters = {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": True,
                }
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": self._TOOL_DESCRIPTIONS.get(
                            spec.name,
                            f"Invoke tool {spec.name}",
                        ),
                        "parameters": parameters,
                    },
                }
            )
        return tools

    def _args_schema_to_json_schema(
        self, args_schema: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Convert internal arg schema to JSON Schema for function calling.

        Args:
            args_schema: Internal args schema mapping.

        Returns:
            JSON Schema object for tool parameters.
        """
        properties: dict[str, Any] = {}
        required: list[str] = []

        for arg_name, rules in args_schema.items():
            prop = self._rule_to_json_schema(rules)
            if "default" in rules:
                prop["default"] = rules["default"]
            properties[arg_name] = prop
            if rules.get("required"):
                required.append(arg_name)

        schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        }
        if required:
            schema["required"] = required
        return schema

    @staticmethod
    def _rule_to_json_schema(rules: dict[str, Any]) -> dict[str, Any]:
        """Convert one internal argument rule into JSON Schema.

        Args:
            rules: Rule mapping containing `type`, bounds, and defaults.

        Returns:
            JSON Schema snippet for one argument.
        """
        py_type = rules.get("type")
        schema: dict[str, Any] = {}

        if py_type is str:
            schema["type"] = "string"
            if "min_length" in rules:
                schema["minLength"] = int(rules["min_length"])
        elif py_type is int:
            schema["type"] = "integer"
            if "min" in rules:
                schema["minimum"] = int(rules["min"])
            if "max" in rules:
                schema["maximum"] = int(rules["max"])
        elif py_type is bool:
            schema["type"] = "boolean"
        elif py_type is dict:
            schema["type"] = "object"
        elif py_type == (str, type(None)):
            schema["anyOf"] = [{"type": "string"}, {"type": "null"}]
        else:
            schema["type"] = "string"

        return schema

    def _safe_path(self, rel_path: str) -> Path:
        """Resolve a relative path within the workspace root.

        Args:
            rel_path: Path relative to `workspace_root`.

        Returns:
            Resolved `Path` inside the workspace.

        Raises:
            ValueError: If the resolved path escapes the workspace.
        """
        target = (self.workspace_root / rel_path).resolve()
        if not str(target).startswith(str(self.workspace_root)):
            raise ValueError("Path escapes workspace")
        return target

    def call(
        self,
        *,
        run_id: str,
        step_id: str | None,
        agent: dict[str, Any],
        tool_name: str,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Authorize, dispatch, and audit a tool call.

        Args:
            run_id: Run ID.
            step_id: Optional step ID associated with the call.
            agent: Agent row dict (must include `tools_allowed` and `id`).
            tool_name: Tool name.
            args: Tool arguments.

        Returns:
            Tool result dict. Always includes `ok`.
        """
        allowed_tools = set(agent.get("tools_allowed") or [])
        start = time.monotonic()
        if tool_name not in allowed_tools or tool_name not in self._tool_registry:
            if tool_name not in self._tool_registry:
                message = f"Tool '{tool_name}' is not registered"
            else:
                message = f"Tool '{tool_name}' is not allowed for agent {agent['id']}"
            result = {
                "ok": False,
                "error": {
                    "code": "tool_not_allowed",
                    "message": message,
                },
            }
            latency_ms = int((time.monotonic() - start) * 1000)
            self.repo.create_tool_call(
                run_id, step_id, tool_name, args, result, False, latency_ms
            )
            self.repo.create_event(
                run_id, "tool.denied", {"tool_name": tool_name, "args": args}
            )
            return result

        try:
            result = self._dispatch(tool_name, args)
        except Exception as exc:  # noqa: BLE001
            result = {
                "ok": False,
                "error": {
                    "code": "tool_error",
                    "message": str(exc),
                },
            }

        latency_ms = int((time.monotonic() - start) * 1000)
        self.repo.create_tool_call(
            run_id, step_id, tool_name, args, result, True, latency_ms
        )
        self.repo.create_event(
            run_id,
            "tool.called",
            {
                "tool_name": tool_name,
                "args": args,
                "ok": result.get("ok", False),
                "latency_ms": latency_ms,
            },
        )
        return result

    def _dispatch(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Dispatch the tool implementation for a supported tool name.

        Args:
            tool_name: Tool name.
            args: Tool arguments.

        Returns:
            Tool result dict.
        """
        spec = self._tool_registry.get(tool_name)
        if not spec:
            return {
                "ok": False,
                "error": {"code": "unknown_tool", "message": tool_name},
            }

        if not isinstance(args, dict):
            return {
                "ok": False,
                "error": {"code": "bad_args", "message": "args must be an object"},
            }

        if spec.validate_args:
            validated = self._validate_args(spec, args)
            if isinstance(validated, dict) and validated.get("ok") is False:
                return validated
            return spec.handler(validated)

        return spec.handler(args)

    def _validate_args(self, spec: ToolSpec, args: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalize args against a tool schema."""
        if spec.args_schema is None:
            return args
        unknown = sorted(set(args.keys()) - set(spec.args_schema.keys()))
        if unknown:
            return {
                "ok": False,
                "error": {
                    "code": "bad_args",
                    "message": f"unknown arg(s): {', '.join(unknown)}",
                },
            }

        normalized: dict[str, Any] = {}
        for name, rules in spec.args_schema.items():
            required = bool(rules.get("required", False))
            has_value = name in args
            if not has_value:
                if required and "default" not in rules:
                    return {
                        "ok": False,
                        "error": {
                            "code": "bad_args",
                            "message": f"{name} is required",
                        },
                    }
                if "default" in rules:
                    normalized[name] = rules["default"]
                continue

            value = args[name]
            expected_type = rules.get("type", object)
            if not isinstance(value, expected_type):
                expected = (
                    ", ".join(t.__name__ for t in expected_type)
                    if isinstance(expected_type, tuple)
                    else expected_type.__name__
                )
                return {
                    "ok": False,
                    "error": {
                        "code": "bad_args",
                        "message": f"{name} must be of type {expected}",
                    },
                }

            if isinstance(value, str):
                min_length = rules.get("min_length")
                if min_length is not None and len(value.strip()) < int(min_length):
                    return {
                        "ok": False,
                        "error": {
                            "code": "bad_args",
                            "message": f"{name} must be a non-empty string",
                        },
                    }
                if rules.get("strip", True):
                    value = value.strip()

            if isinstance(value, int):
                min_value = rules.get("min")
                max_value = rules.get("max")
                if min_value is not None and value < int(min_value):
                    return {
                        "ok": False,
                        "error": {
                            "code": "bad_args",
                            "message": f"{name} must be >= {min_value}",
                        },
                    }
                if max_value is not None and value > int(max_value):
                    return {
                        "ok": False,
                        "error": {
                            "code": "bad_args",
                            "message": f"{name} must be <= {max_value}",
                        },
                    }

            normalized[name] = value

        return normalized

    def _handle_run_shell(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.shell_runner.run_shell(
            command=args["command"],
            timeout_s=args["timeout_s"],
            allow_write=args["allow_write"],
            write_subdir=args["write_subdir"],
        )

    def _handle_read_file(self, args: dict[str, Any]) -> dict[str, Any]:
        path = self._safe_path(args["path"])
        if not path.exists() or not path.is_file():
            return {
                "ok": False,
                "error": {"code": "not_found", "message": "file not found"},
            }
        content = path.read_text(encoding="utf-8")
        if len(content.encode("utf-8")) > self.max_file_bytes:
            return {
                "ok": False,
                "error": {"code": "size_limit", "message": "file exceeds max size"},
            }
        return {
            "ok": True,
            "path": str(path.relative_to(self.workspace_root)),
            "content": content,
        }

    def _handle_write_file(self, args: dict[str, Any]) -> dict[str, Any]:
        path = self._safe_path(args["path"])
        content = args["content"]
        size = len(content.encode("utf-8"))
        if size > self.max_file_bytes:
            return {
                "ok": False,
                "error": {
                    "code": "size_limit",
                    "message": "content exceeds max size",
                },
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {
            "ok": True,
            "path": str(path.relative_to(self.workspace_root)),
            "bytes": size,
        }

    def _handle_store_memory(self, args: dict[str, Any]) -> dict[str, Any]:
        item = self.memory.store(
            text=args["text"],
            collection=args["collection"],
            metadata=args["metadata"],
        )
        return {"ok": True, "item": item}

    def _handle_search_memory(self, args: dict[str, Any]) -> dict[str, Any]:
        results = self.memory.search(
            query=args["query"],
            collection=args["collection"],
            top_k=args["top_k"],
        )
        return {"ok": True, "results": results}

    @staticmethod
    def _handle_final_answer(args: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, "message": str(args["message"])}
