"""Tool execution gateway.

Enforces an agent allowlist, validates tool arguments against per-tool schemas,
dispatches supported tools (files, memory, Docker shell), and records tool call
audit events.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from pathlib import Path
from typing import Any, Callable

from .docker_runner import DockerRunner
from .memory import LocalVectorMemory
from .repository import Repository


@dataclass(frozen=True)
class ToolSpec:
    """Registry entry for a supported tool."""

    name: str
    args_schema: dict[str, dict[str, Any]]
    handler: Callable[[dict[str, Any]], dict[str, Any]]


class ToolGateway:
    """Authorize and dispatch tool calls, then persist audit records."""

    def __init__(
        self,
        repo: Repository,
        memory: LocalVectorMemory,
        docker_runner: DockerRunner,
        workspace_root: str,
        max_file_bytes: int = 100_000,
    ):
        """Create a tool gateway for dispatching and auditing tool calls.

        Args:
            repo: Repository used to persist tool call audit records and events.
            memory: Memory subsystem used by memory tools.
            docker_runner: Docker sandbox used by `run_shell`.
            workspace_root: Root directory used to constrain file access.
            max_file_bytes: Maximum bytes for read/write operations.
        """
        self.repo = repo
        self.memory = memory
        self.docker_runner = docker_runner
        self.workspace_root = Path(workspace_root).resolve()
        self.max_file_bytes = max_file_bytes
        self._tool_registry: dict[str, ToolSpec] = self._build_registry()

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
        }

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

        validated = self._validate_args(spec, args)
        if isinstance(validated, dict) and validated.get("ok") is False:
            return validated
        return spec.handler(validated)

    def _validate_args(
        self, spec: ToolSpec, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate and normalize args against a tool schema."""
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
        return self.docker_runner.run_shell(
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
